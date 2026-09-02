"""Build topic wiki pages from a concept map, transcript excerpts, and final gates.

Pages are synthesized from transcript context around evidence quotes, while claims form
the structure and anchors; claims alone preserve too little source material. Page length
scales with its claim count rather than a percentage of input. The map groups concepts,
not sources: singletons merge, uncovered claims move to the closest tagged page, and the
plan records how many lessons appear on multiple pages. Completion requires clean
llmwiki validate and audit results rather than merely written files.

State lives in work/pages-plan.json and final/wiki/*.md. A repeated run continues from
the first page whose status is not done.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from llmwiki.ids import normalize_text

from .adapters.base import ROLE_THINK, HarnessAdapter, RunResult
from .observability import RoleHeartbeat
from .orchestrator import RunConfig, _structured_answer_policy

EXIT_OK = 0
EXIT_QUOTA = 75
EXIT_GATE_REFUSED = 77
EXIT_TRANSIENT = 111
EXIT_FAIL = 1
_STOP_EXIT = {"quota": EXIT_QUOTA, "transient": EXIT_TRANSIENT, "fatal": EXIT_FAIL}

MIN_WORDS_PER_CLAIM = 60.0
MAX_WORDS_PER_CLAIM = 250.0
TARGET_WORDS_PER_CLAIM = (100, 200)
MAX_MAP_CLAIM_CHARS = 160
MAX_PAGE_CLAIMS = 30
MIN_MAP_COVERAGE_PERCENT = 60
EXCERPT_RADIUS = 2_500
MAX_PAGE_MATERIAL_CHARS = 120_000
_MAX_BAD_ANSWER = 256 * 1024
_SLUG = re.compile(r"^[a-z0-9а-яё][a-z0-9а-яё-]{1,79}$")
_ANCHOR = re.compile(r"\(clm_[0-9a-f]{12}\)")
_CLAIM_ID = re.compile(r"^clm_[0-9a-f]{12}$")
_H2 = re.compile(r"^## ", re.MULTILINE)


class PagesError(ValueError):
    pass


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".",
        suffix=".tmp", delete=False,
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _words(text: str) -> int:
    return len(text.split())


def _normalized_title(page: object) -> str | None:
    if type(page) is not dict:
        return None
    title = page.get("title")
    if type(title) is not str or not title.strip():
        return None
    return normalize_text(title)


# ------------------------------------------------------------------- page map

def _map_prompt(claims: list[dict], sources: dict[str, dict]) -> str:
    lines = [
        "You are building the map of wiki pages for a course from its verified claims. Do not "
        "use tools, files or the network - everything you need is below.",
        "",
        "Map rules (a violation means a rejected answer):",
        "- Pages follow the CONCEPTS of the course, not its sources. Pages named after a "
        "lesson, a day, a webinar or office hours are forbidden ('odds and ends from lesson 3', "
        "'patterns from day 2').",
        "- One lesson usually feeds several pages: points on different subjects go to different concepts.",
        "- Every claim belongs to EXACTLY one page. Do not leave any claim unused.",
        f"- A page holds 2-{MAX_PAGE_CLAIMS} claims. Attach a lone claim to a related page; "
        "split a subject larger than the ceiling into subtopics.",
        "- slug - lowercase Latin/Cyrillic letters, digits and hyphens, 2-80 characters, unique.",
        "- title - a short Russian name for the concept.",
        "",
        'Return exactly the JSON object {"pages":[{"slug":string,"title":string,"claims":[clm_id,...]}]}.',
        "",
        "SOURCES (id: title):",
    ]
    for source_id, row in sources.items():
        lines.append(f"- {source_id}: {row.get('title', '')}")
    lines += ["", "CLAIMS (id | sources | tags | text):"]
    for claim in claims:
        text = " ".join(str(claim["text"]).split())
        if len(text) > MAX_MAP_CLAIM_CHARS:
            text = text[: MAX_MAP_CLAIM_CHARS - 1] + "…"
        lines.append(
            f"{claim['id']} | {','.join(claim['sources'])} | "
            f"{','.join(claim.get('tags') or [])} | {text}"
        )
    return "\n".join(lines)


def sanitize_map(value: dict, known: set[str]) -> list[dict]:
    """Repair a page map before validation and report every discarded item.

    Retrying the model is ineffective when a large otherwise-valid map repeatedly
    contains one invented claim id or assigns one claim to two pages. Repair those
    cases deterministically: discard unknown ids, keep the first placement of a
    duplicate claim, and remove pages that become empty. ``_validate_map`` still rejects
    fundamentally broken results such as an empty page list or inadequate coverage.
    Returned records contain ``reason``, ``page``, and optionally ``claim_id``.
    """
    pages = value.get("pages")
    if type(pages) is not list:
        return []
    removed: list[dict] = []
    seen: set[str] = set()
    cleaned: list[dict] = []
    for page in pages:
        if type(page) is not dict or type(page.get("claims")) is not list:
            # A structurally broken page is not our business: _validate_map judges it.
            cleaned.append(page)
            continue
        slug = page.get("slug")
        claims: list[str] = []
        for claim_id in page["claims"]:
            if type(claim_id) is not str or claim_id not in known:
                removed.append(
                    {"reason": "unknown_claim", "page": slug, "claim_id": claim_id}
                )
                continue
            if claim_id in seen:
                removed.append(
                    {"reason": "claim_on_two_pages", "page": slug, "claim_id": claim_id}
                )
                continue
            seen.add(claim_id)
            claims.append(claim_id)
        page["claims"] = claims
        cleaned.append(page)

    # One title = one llm-wiki id = one concept. Merge after cleaning the claims but
    # before dropping empty pages: the first page of a same-title pair may be filled with
    # the claims of the second and must keep its own slug/title.
    by_title: dict[str, dict] = {}
    merged: list[dict] = []
    for page in cleaned:
        title_key = _normalized_title(page)
        if title_key is None or type(page.get("claims")) is not list:
            merged.append(page)
            continue
        target = by_title.get(title_key)
        if target is None:
            by_title[title_key] = page
            merged.append(page)
            continue
        present = set(target["claims"])
        target["claims"].extend(
            claim_id for claim_id in page["claims"] if claim_id not in present
        )
        removed.append({
            "reason": "same_title_merged",
            "page": page.get("slug"),
            "into": target.get("slug"),
            "claim_count": len(target["claims"]),
        })

    kept: list[dict] = []
    for page in merged:
        if type(page) is dict and type(page.get("claims")) is list and not page["claims"]:
            removed.append({"reason": "page_without_claims", "page": page.get("slug")})
            continue
        kept.append(page)
    value["pages"] = kept
    return removed


def _validate_map(value: dict, known: set[str]) -> None:
    pages = value.get("pages")
    if type(pages) is not list or not pages:
        raise ValueError("pages: a non-empty list is required")
    seen_slug: set[str] = set()
    seen_title: dict[str, str] = {}
    seen_claim: set[str] = set()
    for page in pages:
        if type(page) is not dict or set(page) != {"slug", "title", "claims"}:
            raise ValueError("page: exactly the fields slug, title, claims are required")
        slug, title, claims = page["slug"], page["title"], page["claims"]
        if type(slug) is not str or not _SLUG.fullmatch(slug):
            raise ValueError(f"slug {slug!r}: lowercase, letters/digits/hyphen, 2-80 characters")
        if slug in seen_slug:
            raise ValueError(f"slug {slug!r} is repeated")
        seen_slug.add(slug)
        if type(title) is not str or not title.strip() or len(title) > 120:
            raise ValueError(f"title of page {slug!r} is empty or longer than 120 characters")
        title_key = normalize_text(title)
        if title_key in seen_title:
            raise ValueError(
                f"title of page {slug!r} repeats the title of page {seen_title[title_key]!r}"
            )
        seen_title[title_key] = slug
        if type(claims) is not list or not claims:
            raise ValueError(f"page {slug!r} has no claims")
        for claim_id in claims:
            if type(claim_id) is not str or claim_id not in known:
                raise ValueError(f"page {slug!r}: unknown claim {claim_id!r}")
            if claim_id in seen_claim:
                raise ValueError(f"claim {claim_id} lies on two pages")
            seen_claim.add(claim_id)
    minimum = max(1, (len(known) * MIN_MAP_COVERAGE_PERCENT + 99) // 100)
    if len(seen_claim) < minimum:
        raise ValueError(
            f"covered {len(seen_claim)} claims of {len(known)}: at least "
            f"{MIN_MAP_COVERAGE_PERCENT}% required, "
            "spread the rest across pages"
        )


def _nearest_page(
    tags: set[str], pages: list[dict], by_id: dict[str, dict], *,
    max_claims: int | None = None,
) -> dict:
    candidates = pages
    if max_claims is not None:
        with_room = [page for page in pages if len(page["claims"]) < max_claims]
        if with_room:
            candidates = with_room
    best, best_score = candidates[0], -1.0
    for page in candidates:
        page_tags: set[str] = set()
        for claim_id in page["claims"]:
            page_tags.update(by_id[claim_id].get("tags") or [])
        union = len(tags | page_tags) or 1
        score = len(tags & page_tags) / union + len(page["claims"]) / 10_000
        if score > best_score:
            best, best_score = page, score
    return best


def build_pages_plan(value: dict, claims: list[dict]) -> list[dict]:
    """Turn a role response into a full-coverage map by merging singletons and orphans."""
    by_id = {claim["id"]: claim for claim in claims}
    pages = [
        {"page": page["slug"], "title": page["title"].strip(), "claims": list(page["claims"])}
        for page in value["pages"]
    ]
    covered = {claim_id for page in pages for claim_id in page["claims"]}
    for claim_id in by_id:
        if claim_id not in covered:
            target = _nearest_page(
                set(by_id[claim_id].get("tags") or []), pages, by_id,
                max_claims=MAX_PAGE_CLAIMS,
            )
            target["claims"].append(claim_id)
    while len(pages) > 1:
        single = next((page for page in pages if len(page["claims"]) < 2), None)
        if single is None:
            break
        pages.remove(single)
        tags = set(by_id[single["claims"][0]].get("tags") or [])
        _nearest_page(tags, pages, by_id)["claims"].extend(single["claims"])
    plan = []
    for page in pages:
        lessons = sorted({src for claim_id in page["claims"] for src in by_id[claim_id]["sources"]})
        plan.append({**page, "lessons": lessons, "status": "todo"})
    return plan


def map_stats(plan: list[dict], *, model_claims: int | None = None) -> dict:
    lesson_pages: dict[str, int] = {}
    for page in plan:
        for lesson in page["lessons"]:
            lesson_pages[lesson] = lesson_pages.get(lesson, 0) + 1
    shared = sum(1 for count in lesson_pages.values() if count >= 2)
    total = len(lesson_pages)
    result = {
        "pages": len(plan),
        "claims": sum(len(page["claims"]) for page in plan),
        "lessons": total,
        "lessons_in_two_or_more_pages": shared,
        "share_percent": round(100.0 * shared / total, 1) if total else 0.0,
    }
    if model_claims is not None:
        result["model_claims"] = model_claims
        result["code_assigned_claims"] = max(0, result["claims"] - model_claims)
        result["model_coverage_percent"] = (
            round(100.0 * model_claims / result["claims"], 1) if result["claims"] else 0.0
        )
    return result


# --------------------------------------------------------- transcript excerpts

def _read_source_text(topic: Path, row: dict) -> str:
    content_path = row.get("content_path")
    if not isinstance(content_path, str) or not content_path:
        return ""
    path = Path(content_path)
    if not path.is_absolute():
        path = topic / path
    try:
        return path.read_text("utf-8")
    except (OSError, UnicodeError):
        return ""


def _locate(text: str, quote: str, locator: object) -> tuple[int, int] | None:
    quote = quote.strip()
    if quote:
        pos = text.find(quote)
        if pos < 0 and len(quote) > 80:
            pos = text.find(quote[:80])
        if pos >= 0:
            return pos, pos + min(len(quote), max(80, len(quote)))
    if isinstance(locator, dict) and locator.get("type") == "line":
        match = re.fullmatch(r"(\d+)-(\d+)", str(locator.get("value", "")))
        if match:
            lines = text.split("\n")
            first = max(1, int(match.group(1)))
            last = max(first, int(match.group(2)))
            start = sum(len(line) + 1 for line in lines[: first - 1])
            end = sum(len(line) + 1 for line in lines[:last])
            return start, min(end, len(text))
    return None


def page_material(
    topic: Path, page: dict, by_id: dict[str, dict], sources: dict[str, dict],
    *, radius: int = EXCERPT_RADIUS, budget: int = MAX_PAGE_MATERIAL_CHARS,
) -> list[dict]:
    """Merge transcript excerpts around a page's claim evidence by source."""
    spans: dict[str, list[tuple[int, int]]] = {}
    texts: dict[str, str] = {}
    for claim_id in page["claims"]:
        for item in by_id[claim_id].get("evidence") or []:
            source_id = item.get("source_id")
            if source_id not in sources:
                continue
            if source_id not in texts:
                texts[source_id] = _read_source_text(topic, sources[source_id])
            text = texts[source_id]
            if not text:
                continue
            located = _locate(text, str(item.get("quote", "")), item.get("locator"))
            if located is None:
                continue
            spans.setdefault(source_id, []).append(located)
    while True:
        excerpts = []
        for source_id, items in spans.items():
            text = texts[source_id]
            merged: list[list[int]] = []
            for start, end in sorted(items):
                start, end = max(0, start - radius), min(len(text), end + radius)
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            for start, end in merged:
                excerpts.append({
                    "source_id": source_id,
                    "title": sources[source_id].get("title", source_id),
                    "start": start,
                    "text": text[start:end],
                })
        total = sum(len(item["text"]) for item in excerpts)
        if total <= budget or radius <= 200:
            break
        radius //= 2
    if sum(len(item["text"]) for item in excerpts) > budget:
        excerpts.sort(key=lambda item: -len(item["text"]))
        kept, used = [], 0
        for item in excerpts:
            if used + len(item["text"]) > budget:
                item["text"] = item["text"][: max(0, budget - used)]
            if item["text"]:
                kept.append(item)
                used += len(item["text"])
        excerpts = kept
    excerpts.sort(key=lambda item: (item["title"], item["start"]))
    return excerpts


def _page_prompt(page: dict, by_id: dict[str, dict], excerpts: list[dict]) -> str:
    count = len(page["claims"])
    low, high = TARGET_WORDS_PER_CLAIM[0] * count, TARGET_WORDS_PER_CLAIM[1] * count
    lines = [
        f"You are writing a domain wiki page for a course: \"{page['title']}\". Do not use tools, "
        "files or the network - the material is below. The page must stand on its own: the reader "
        "has not seen the course.",
        "",
        "Rules (a violation means a rejected answer):",
        "- Write from the TRANSCRIPT EXCERPTS. The claims are a verified skeleton and anchors, not "
        "the only material: state the mechanics of 'how it works', 'why and when', the specifics of "
        "the course (commands, thresholds, numbers, examples), trade-offs and contradictions explicitly.",
        "- Structure: a short introductory paragraph, then `## ...` sections. Every section must carry "
        "at least one anchor `(clm_...)` at the end of a paragraph whose point the claim covers. A "
        "section without an anchor is your own construction and must not exist. Use only anchors from "
        "the page's claim list; every claim of the page at least once.",
        f"- Length: from {low} to {high} words ({count} claims x {TARGET_WORDS_PER_CLAIM[0]}-"
        f"{TARGET_WORDS_PER_CLAIM[1]}). Less is a skeleton, more is filler.",
        "- No narration and no filler; preserve modality (the author's position != a fact). Do not "
        "invent schemes, lists or checklists that are absent from the transcript.",
        "- Language: Russian; English words only as terms (`plan mode`, `goroutine`).",
        "- The body is markdown without an H1 heading (the title already exists).",
        "",
        'Return exactly the JSON object {"body": string}.',
        "",
        "PAGE CLAIMS (id: text):",
    ]
    for claim_id in page["claims"]:
        lines.append(f"- {claim_id}: {' '.join(str(by_id[claim_id]['text']).split())}")
    lines += ["", "TRANSCRIPT EXCERPTS:"]
    for item in excerpts:
        lines += ["", f"### {item['title']} (from offset {item['start']})", item["text"]]
    return "\n".join(lines)


def _validate_body(value: dict, page: dict) -> None:
    body = value.get("body")
    if type(body) is not str or not body.strip():
        raise ValueError("body: non-empty markdown is required")
    if body.lstrip().startswith("# "):
        raise ValueError("body: must have no H1 heading")
    allowed = set(page["claims"])
    anchors = {anchor[1:-1] for anchor in _ANCHOR.findall(body)}
    unknown = sorted(anchors - allowed)
    if unknown:
        raise ValueError(f"anchors not from the page claims: {', '.join(unknown[:5])}")
    missing = sorted(allowed - anchors)
    if len(missing) > max(1, len(allowed) // 5):
        raise ValueError(
            f"{len(missing)} page claims have no anchor (at most "
            f"{max(1, len(allowed) // 5)}): {', '.join(missing[:5])}"
        )
    sections = _H2.split(body)[1:]
    bare = [section.split("\n", 1)[0].strip() for section in sections if not _ANCHOR.search(section)]
    if bare:
        raise ValueError(f"sections without an anchor (clm_...): {'; '.join(bare[:3])}")
    count = len(allowed)
    words = _words(body)
    if words < MIN_WORDS_PER_CLAIM * count:
        raise ValueError(
            f"{words} words for {count} claims - a skeleton, at least "
            f"{int(MIN_WORDS_PER_CLAIM * count)}"
        )
    if words > MAX_WORDS_PER_CLAIM * count:
        raise ValueError(
            f"{words} words for {count} claims - filler, ceiling {int(MAX_WORDS_PER_CLAIM * count)}"
        )


# ----------------------------------------------------------------- runner

def _short_count(value: object) -> str:
    """104213 -> 104k, 992 -> 992, None -> ?."""
    if type(value) is not int:
        return "?"
    return f"{round(value / 1000)}k" if value >= 1000 else str(value)


def _short_duration(minutes: int) -> str:
    """150 -> 2h30, 45 -> 45m."""
    hours, rest = divmod(max(0, int(minutes)), 60)
    return f"{hours}h{rest:02d}" if hours else f"{rest}m"


def _merge_plan_same_titles(plan: list[dict]) -> tuple[list[dict], list[dict]]:
    """A legacy pages-plan with one concept under several slugs -> a compatible plan."""
    repaired = deepcopy(plan)
    by_title: dict[str, dict] = {}
    kept: list[dict] = []
    merged: list[dict] = []
    for page in repaired:
        title_key = _normalized_title(page)
        if (
            title_key is None
            or type(page.get("page")) is not str
            or type(page.get("claims")) is not list
        ):
            kept.append(page)
            continue
        target = by_title.get(title_key)
        if target is None:
            by_title[title_key] = page
            kept.append(page)
            continue
        present = set(target["claims"])
        for claim_id in page["claims"]:
            if claim_id not in present:
                target["claims"].append(claim_id)
                present.add(claim_id)
        if type(target.get("lessons")) is list and type(page.get("lessons")) is list:
            target["lessons"] = sorted(set(target["lessons"]) | set(page["lessons"]))
        target["status"] = "todo"
        target.pop("path", None)
        target.pop("words", None)
        merged.append({
            "into": target["page"],
            "page": page["page"],
            "claim_count": len(target["claims"]),
        })
    return kept, merged


class PagesRunner:
    """Run page mapping, synthesis, and gates for any llmwiki topic with claims."""

    def __init__(
        self, topic_dir: str | Path, *, wiki: object, adapter: HarnessAdapter,
        config: RunConfig, heartbeat: RoleHeartbeat | None = None,
        on_call: Optional[Callable[[str, RunResult], None]] = None,
        notifier=None, log: Optional[Callable[[str], None]] = None,
    ):
        self.topic = Path(topic_dir)
        self.wiki = wiki
        self.adapter = adapter
        self.cfg = config
        self.heartbeat = heartbeat or RoleHeartbeat(self.topic)
        self.on_call = on_call
        self.notifier = notifier
        self.log = log or (lambda text: None)
        self.stop_kind: str | None = None
        self.stop_reason: str | None = None
        self.plan_path = self.topic / "work" / "pages-plan.json"
        self._failures = 0
        self._last_error: str | None = None

    # --- llmwiki ---
    def _claims(self) -> list[dict]:
        rows = self.wiki.list_claims(self.topic, zone="all", fields=["*"])
        claims = []
        for row in rows:
            if row.get("status") == "retracted":
                continue
            if row.get("zone") == "staging" and row.get("status") != "verified":
                continue
            sources = sorted({
                item.get("source_id") for item in row.get("evidence") or []
                if isinstance(item, dict) and isinstance(item.get("source_id"), str)
            })
            claims.append({**row, "sources": sources})
        return claims

    def _sources(self) -> dict[str, dict]:
        return {row["id"]: row for row in self.wiki.read_sources(self.topic) if isinstance(row.get("id"), str)}

    # --- role call ---
    def _bad_answer(self, role: str, text: str, error: Exception) -> str:
        self._failures += 1
        path = self.topic / "work" / "bad-answers" / f"pages-{role}-attempt-{self._failures}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {error}\n{text[:_MAX_BAD_ANSWER]}", "utf-8")
        self._last_error = str(error)
        return f"pages_{role}_structural_failure: {error}"

    def _call(self, role: str, prompt: str, schema: dict, validator) -> tuple[int | None, dict | None]:
        self._last_error = None

        def call(call_prompt: str) -> RunResult:
            # Retry after a rejected answer: extend the generic protocol reminder with the
            # concrete reason, otherwise the model repeats the same length/structure and all
            # three attempts are wasted.
            if self._last_error and call_prompt != prompt:
                call_prompt += f"\nPrevious answer rejected: {self._last_error}"
            model, effort = self.cfg.for_role(
                ROLE_THINK, harness_default=self.adapter.default_model_for_role(ROLE_THINK)
            )
            token = self.heartbeat.start(
                phase="import:pages", role="plan" if role == "map" else "synth", cycle=0
            )
            try:
                result = self.adapter.run(
                    call_prompt, cwd=str(self.topic), timeout=self.cfg.timeout,
                    retries=self.cfg.retries, retry_backoff=self.cfg.retry_backoff,
                    model=model, effort=effort, tools="", allowed_tools="", network=False,
                    on_harness_pid=lambda pid: self.heartbeat.harness_pid(token, pid),
                )
            finally:
                self.heartbeat.finish(token)
            if self.on_call is not None:
                self.on_call(role, result)
            return result

        result, value, diagnostic = _structured_answer_policy(
            prompt, call=call, bad_answer=lambda text, error: self._bad_answer(role, text, error),
            retries=self.cfg.retries, retry_backoff=self.cfg.retry_backoff,
            schema=schema, validator=validator, role_label=f"pages {role}",
        )
        if result.stop != "done":
            self.stop_kind, self.stop_reason = result.stop, f"pages_{role}_role"
            return _STOP_EXIT.get(result.stop, EXIT_FAIL), None
        if diagnostic is not None:
            self.stop_kind, self.stop_reason = "fatal", diagnostic
            return EXIT_FAIL, None
        return None, value

    # --- phases ---
    def _load_plan(self) -> list[dict] | None:
        if not self.plan_path.is_file():
            return None
        plan = json.loads(self.plan_path.read_text("utf-8"))
        if type(plan) is not list:
            raise PagesError("pages_plan_invalid")
        return plan

    def _repair_same_title_pages(self, plan: list[dict]) -> list[dict]:
        repaired, merged = _merge_plan_same_titles(plan)
        if not merged:
            return plan
        delete_page = getattr(self.wiki, "delete_wiki_page", None)
        if not callable(delete_page):
            raise PagesError("pages: llmwiki without delete_wiki_page")
        for row in merged:
            deleted = delete_page(self.topic, slug=row["page"])
            if (
                type(deleted) is not dict
                or not isinstance(deleted.get("path"), str)
                or deleted.get("result") not in {"deleted", "missing"}
            ):
                raise PagesError("invalid_llmwiki_result")
        _atomic_json(self.plan_path, repaired)
        for row in merged:
            self.log(
                f"pages: merged same-title pages: {row['into']} <- {row['page']} "
                f"({row['claim_count']} claims)"
            )
            if row["claim_count"] > MAX_PAGE_CLAIMS:
                self.log(
                    f"pages: after the merge {row['into']} holds {row['claim_count']} "
                    f"claims - above the usual ceiling {MAX_PAGE_CLAIMS}, which is allowed"
                )
        return repaired

    def _record_map_fixes(self, removed: list[dict]) -> None:
        """What the sanitizer dropped from the map - to stderr and to disk, line by line."""
        for row in removed:
            _append_jsonl(
                self.topic / "work" / "dropped-claims.jsonl", {"stage": "pages-map", **row}
            )
            if row.get("reason") == "same_title_merged":
                self.log(
                    f"map: merged same-title pages: {row.get('into')} <- "
                    f"{row.get('page')} ({row.get('claim_count')} claims)"
                )
                if row.get("claim_count", 0) > MAX_PAGE_CLAIMS:
                    self.log(
                        f"map: after the merge {row.get('into')} holds "
                        f"{row.get('claim_count')} claims - above the usual ceiling "
                        f"{MAX_PAGE_CLAIMS}, which is allowed"
                    )
        counts: dict[str, int] = {}
        for row in removed:
            counts[str(row.get("reason"))] = counts.get(str(row.get("reason")), 0) + 1
        detail = ", ".join(f"{reason} x{count}" for reason, count in sorted(counts.items()))
        self.log(
            f"map cleaned before validation: {detail} "
            "(evidence in work/dropped-claims.jsonl)"
        )

    def _emit(self, event: str, text: str, *, kind: str, title: str, lines=None) -> None:
        """Pages notice: title without caps, facts in English (canon v2)."""
        if self.notifier is not None:
            self.notifier.emit(event, text, kind=kind, title=title, lines=lines)

    def run(self) -> int:
        started = time.monotonic()
        claims = self._claims()
        if not claims:
            raise PagesError("pages_no_claims")
        by_id = {claim["id"]: claim for claim in claims}
        sources = self._sources()
        plan = self._load_plan()
        if plan is not None:
            plan = self._repair_same_title_pages(plan)
        if plan is None:
            self.log(f"page map: {len(claims)} claims, {len(sources)} sources")
            cleaned: list[dict] = []
            model_claims = 0

            def check_map(data: dict) -> None:
                nonlocal model_claims
                # Sanitizer before validator: repair what is repairable, reject only what is broken.
                cleaned[:] = sanitize_map(data, set(by_id))
                _validate_map(data, set(by_id))
                model_claims = sum(len(page["claims"]) for page in data["pages"])

            code, value = self._call(
                "map", _map_prompt(claims, sources), {"pages": list}, check_map,
            )
            if code is not None:
                return code
            if cleaned:
                self._record_map_fixes(cleaned)
            plan = build_pages_plan(value, claims)
            stats = map_stats(plan, model_claims=model_claims)
            _atomic_json(self.plan_path, plan)
            _atomic_json(self.topic / "work" / "pages-map-stats.json", stats)
            self.log(
                f"map: {stats['pages']} pages, assigned by the model "
                f"{stats['model_claims']}/{stats['claims']} "
                f"({stats['model_coverage_percent']}%), completed by code "
                f"{stats['code_assigned_claims']}; lessons in 2+ pages "
                f"{stats['lessons_in_two_or_more_pages']}/{stats['lessons']} ({stats['share_percent']}%)"
            )
            self._emit(
                "pages_planned", f"pages map: {stats['pages']}", kind="progress",
                title="pages map",
                lines=[
                    f"pages {stats['pages']} · claims {stats['claims']}",
                    f"lessons in 2+ pages {stats['lessons_in_two_or_more_pages']}/{stats['lessons']}",
                ],
            )
        known = set(by_id)
        for page in plan:
            if page.get("status") == "done":
                continue
            page["claims"] = [claim_id for claim_id in page["claims"] if claim_id in known]
            if not page["claims"]:
                page["status"] = "done"
                _atomic_json(self.plan_path, plan)
                continue
            excerpts = page_material(self.topic, page, by_id, sources)
            if not excerpts:
                raise PagesError(f"pages_no_material:{page['page']}")
            self.log(
                f"page {page['page']}: {len(page['claims'])} claims, "
                f"{sum(len(item['text']) for item in excerpts)} characters of excerpts"
            )
            code, value = self._call(
                "page", _page_prompt(page, by_id, excerpts), {"body": str},
                lambda data, current=page: _validate_body(data, current),
            )
            if code is not None:
                return code
            # Promote only what is still in staging: llmwiki rejects the whole batch when an
            # id is already in final (a repeat / --rebuild / an archived topic with final claims).
            staging = [claim_id for claim_id in page["claims"] if by_id[claim_id].get("zone") == "staging"]
            if staging:
                promoted = self.wiki.promote_claims(self.topic, staging)
                if type(promoted) is not list:
                    raise PagesError("invalid_llmwiki_result")
                for claim_id in staging:
                    by_id[claim_id]["zone"] = "final"
            written = self.wiki.put_wiki_page(
                self.topic, title=page["title"], claims=list(page["claims"]),
                body=value["body"].strip() + "\n", slug=page["page"],
            )
            if type(written) is not dict or not isinstance(written.get("path"), str):
                raise PagesError("invalid_llmwiki_result")
            page["status"] = "done"
            page["path"] = written["path"]
            page["words"] = _words(value["body"])
            _atomic_json(self.plan_path, plan)
        return self._gate(plan, started)

    def _gate(self, plan: list[dict], started: float) -> int:
        validation = self.wiki.validate_topic(self.topic)
        findings = self.wiki.audit(
            self.topic, min_words_per_claim=MIN_WORDS_PER_CLAIM,
            max_words_per_claim=MAX_WORDS_PER_CLAIM,
        )
        if type(validation) is not list or type(findings) is not list:
            raise PagesError("invalid_llmwiki_result")
        _atomic_json(self.topic / "work" / "audit-report.json", findings)
        stats = self.wiki.stats(self.topic)
        if isinstance(stats, dict):
            _atomic_json(self.topic / "work" / "stats.json", stats)
        errors = [row for row in findings if isinstance(row, dict) and row.get("level") == "error"]
        summary = (
            f"pages {len(plan)}, final claims {stats.get('claims_final')}, "
            f"wiki words {stats.get('wiki_words')} ({stats.get('words_per_claim')} per claim)"
            if isinstance(stats, dict) else f"pages {len(plan)}"
        )
        notice_summary = (
            f"pages {len(plan)} · claims {stats.get('claims_final')} · "
            f"words {_short_count(stats.get('wiki_words'))}"
            if isinstance(stats, dict) else f"pages {len(plan)}"
        )
        minutes = int((time.monotonic() - started) // 60)
        if validation or errors:
            codes: dict[str, int] = {}
            for row in errors:
                codes[str(row.get("code"))] = codes.get(str(row.get("code")), 0) + 1
            detail = ", ".join(f"{code} x{count}" for code, count in sorted(codes.items()))
            if validation:
                detail = f"validate: {validation[0]}" + (f"; {detail}" if detail else "")
            self.stop_kind, self.stop_reason = "gate_refused", "audit" if errors else "validate"
            self.log(f"PAGES GATE REFUSED: {detail}")
            self._emit(
                "pages_gate_refused", f"pages gate refused: {detail}", kind="fail",
                title="pages gate refused",
                lines=[detail, notice_summary, "report: work/audit-report.json"],
            )
            return EXIT_GATE_REFUSED
        self.log(f"pages ready: {summary}")
        self._emit(
            "pages_done", f"pages done: {notice_summary}", kind="done",
            title="pages done", lines=[notice_summary, _short_duration(minutes)],
        )
        return EXIT_OK


__all__ = [
    "PagesRunner", "PagesError", "build_pages_plan", "map_stats", "page_material",
    "sanitize_map",
    "MIN_WORDS_PER_CLAIM", "MAX_WORDS_PER_CLAIM", "MIN_MAP_COVERAGE_PERCENT",
]
