"""Durable CLI import of a bounded course dump into one llm-wiki topic."""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote

from . import checkpoint
from .adapters.base import ROLE_THINK, HarnessAdapter, RunResult
from .observability import RoleHeartbeat
from .orchestrator import RunConfig, _structured_answer_policy
from .sources import (
    ImportPlan,
    ImportReceipt,
    SourceImportError,
    SourceWindow,
    iter_source_windows,
    normalize_extract_result,
    normalize_verify_result,
    plan_import,
    verified_claim_payload,
    write_verified_claims,
)

EXIT_OK = 0
EXIT_QUOTA = 75
EXIT_PLAN_REVIEW = 76
EXIT_GATE_REFUSED = 77
EXIT_TRANSIENT = 111
EXIT_FAIL = 1
_STOP_EXIT = {"quota": EXIT_QUOTA, "transient": EXIT_TRANSIENT, "fatal": EXIT_FAIL}
_DECISIONS = {"KEEP", "SKIM", "DROP"}
_PLAN_ROW = re.compile(
    r"^\|\s*(\d{3})\s*\|.*?\|.*?\|.*?\|\s*(KEEP|SKIM|DROP)\s*\|.*\|\s*$"
)
_APPROVAL = re.compile(r"^approved: (plan_[0-9a-f]{24})$")
_MAX_ROLE_TEXT = 256 * 1024
# A milestone is printed after every lesson: the scheduler watchdog calls a run hung when
# the job log stops moving, and extract/verify of a 122-lesson course takes hours.
# The milestone carries no notice (canon 25.08): course progress is collected by the status
# snapshot. The windows of one lesson are extracted/verified by a pool (<= 3, like the search
# collector); state and the checkpoint are touched from threads only under this lock.
_STATE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ImportRunOptions:
    dump: Path
    base: Path
    slug: str
    extract_adapter: str
    verify_adapter: str
    auto_plan: bool
    link_dump: bool
    extract_config: RunConfig
    verify_config: RunConfig
    allow_same_adapter_verify: bool = False


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".",
        suffix=".tmp", delete=False
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".",
        suffix=".tmp", delete=False
    ) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _safe_cell(value: str) -> str:
    return " ".join(value.replace("|", "/").split())


def _index_metadata(dump: Path) -> dict[str, tuple[str, str, bool]]:
    """Map decoded coursedump targets to (duration, words, no-text), best effort."""
    path = dump / "INDEX.md"
    if not path.is_file():
        return {}
    result: dict[str, tuple[str, str, bool]] = {}
    # Cyrillic on purpose: this parses the INDEX.md written by the coursedump tool, whose
    # line format ("... - NNN слов", the "ТЕКСТА НЕТ" marker) is an external contract.
    link = re.compile(r"\[[^]]*\]\(([^)]+)\)(?:\s+\(([^)]+)\))?\s*-\s*(\d+)\s+слов")
    try:
        lines = path.read_text("utf-8").splitlines()
    except (OSError, UnicodeError):
        return {}
    for line_text in lines:
        match = link.search(line_text)
        if match is None:
            continue
        relative = unquote(match.group(1)).removeprefix("text/")
        result[relative] = (
            match.group(2) or "-",
            match.group(3),
            "ТЕКСТА НЕТ" in line_text.upper(),
        )
    return result


def _plan_rows(plan: ImportPlan, dump: Path) -> list[dict[str, object]]:
    metadata = _index_metadata(dump)
    rows = []
    planned = [(document.relative_path, document, None) for document in plan.documents]
    planned += [(document.relative_path, None, document) for document in plan.skipped]
    planned.sort(key=lambda item: item[0].casefold())
    for number, (relative_path, document, skipped) in enumerate(planned, 1):
        duration, words, no_text = metadata.get(relative_path, ("-", "-", False))
        forced_drop = skipped is not None
        reason = "included by the deterministic plan"
        if skipped is not None:
            reason = {
                "document_too_large": "DROP: document above the bounded import limit",
                "non_transcript": "DROP: non-transcript attachment",
            }[skipped.reason]
        elif no_text or (words.isdigit() and int(words) <= 2):
            forced_drop = True
            reason = "DROP: empty extraction (near-zero words / no-text marker)"
        rows.append(
            {
                "nnn": f"{number:03d}",
                "document_id": document.document_id if document is not None else None,
                "relative_path": relative_path,
                "title": document.title if document is not None else skipped.title,
                "duration": duration,
                "words": words,
                "decision": "DROP" if forced_drop else "KEEP",
                "reason": reason,
                "forced_drop": forced_drop,
            }
        )
    return rows


def _render_plan(plan: ImportPlan, rows: list[dict[str, object]]) -> str:
    lines = [
        "# Course import map",
        "",
        f"plan_id: {plan.plan_id}",
        "",
        "| NNN | Title | Duration | Words | Decision | Why |",
        "|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['nnn']} | {_safe_cell(str(row['title']))} | "
            f"{row['duration']} | {row['words']} | {row['decision']} | "
            f"{_safe_cell(str(row['reason']))} |"
        )
    lines += [
        "",
        "Allowed decisions: KEEP, SKIM, DROP. KEEP and SKIM are imported; DROP is skipped.",
        f"To approve, add a separate exact line: `approved: {plan.plan_id}`",
        "",
    ]
    return "\n".join(lines)


def _parse_plan(path: Path, rows: list[dict[str, object]], plan_id: str) -> dict[str, str] | None:
    try:
        lines = path.read_text("utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError("import_plan_unreadable") from error
    approval_lines = [line for line in lines if line.startswith("approved:")]
    if not approval_lines:
        return None
    if len(approval_lines) != 1:
        raise ValueError("import_plan_approval_conflict")
    approval = _APPROVAL.fullmatch(approval_lines[0])
    if approval is not None and approval.group(1) != plan_id:
        raise ValueError("import_plan_approval_conflict")
    expected_approval = f"approved: {plan_id}"
    if approval_lines[0] != expected_approval:
        raise ValueError(
            f"import_plan_approval_invalid: expected {expected_approval}"
        )
    decisions: dict[str, str] = {}
    for line in lines:
        match = _PLAN_ROW.fullmatch(line)
        if match:
            if match.group(1) in decisions:
                raise ValueError("import_plan_duplicate_row")
            decisions[match.group(1)] = match.group(2)
    expected = {str(row["nnn"]) for row in rows}
    if set(decisions) != expected:
        raise ValueError("import_plan_rows_conflict")
    forced_drop = {
        str(row["nnn"]) for row in rows if row.get("forced_drop") is True
    }
    if any(decisions[nnn] != "DROP" for nnn in forced_drop):
        raise ValueError("import_plan_forced_drop_conflict")
    return decisions


def _selection_digest(decisions: Mapping[str, str]) -> str:
    raw = json.dumps(decisions, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _dump_content_path(dump: Path, plan: ImportPlan, relative: str) -> Path:
    root = dump / "text" if plan.source_kind == "coursedump-manifest" else dump
    if dump.is_file():
        candidate = dump
    else:
        candidate = root / relative
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"import_source_document_missing:{relative}")
    return resolved


def _document_text(windows: list[SourceWindow], expected_sha256: str) -> str:
    text = ""
    for window in sorted(windows, key=lambda item: item.start_offset):
        if window.start_offset > len(text):
            raise ValueError("import_source_window_gap")
        overlap = len(text) - window.start_offset
        if overlap > len(window.text) or text[window.start_offset:] != window.text[:overlap]:
            raise ValueError("import_source_window_conflict")
        text += window.text[overlap:]
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected_sha256:
        raise ValueError("import_source_document_hash_conflict")
    return text


def _copied_content_path(topic: Path, nnn: str, text: str) -> str:
    relative = Path("sources") / "content" / f"{nnn}.md"
    destination = topic / relative
    if destination.is_file():
        try:
            existing = destination.read_text("utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError("import_content_copy_unreadable") from error
        if existing != text:
            raise ValueError("import_content_copy_conflict")
    else:
        _atomic_text(destination, text)
    return relative.as_posix()


def _planned_document_ids(rows: list[dict[str, object]]) -> list[str]:
    return [
        row["document_id"]
        for row in rows
        if isinstance(row.get("document_id"), str)
    ]


def _state_metrics(state: dict) -> dict:
    metrics = state.setdefault(
        "metrics", {"role_calls": {"extract": 0, "verify": 0, "pages-map": 0, "pages-page": 0}, "usage": {},
                    "phase_seconds": {}, "cost_usd": 0.0}
    )
    return metrics


def _merge_numbers(target: dict, value: object) -> None:
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if type(item) in {int, float}:
            target[key] = target.get(key, 0) + item
        elif isinstance(item, dict):
            nested = target.setdefault(key, {})
            if isinstance(nested, dict):
                _merge_numbers(nested, item)


def _record_call(state: dict, role: str, result: RunResult) -> None:
    metrics = _state_metrics(state)
    calls = metrics["role_calls"]
    calls[role] = calls.get(role, 0) + 1
    usage = result.raw.get("usage") if isinstance(result.raw, dict) else None
    _merge_numbers(metrics["usage"], usage)
    if type(result.cost_usd) in {int, float}:
        metrics["cost_usd"] += result.cost_usd


def _record_phase_time(state: dict, phase: str, started: float) -> None:
    elapsed = max(0.0, time.monotonic() - started)
    times = _state_metrics(state)["phase_seconds"]
    times[phase] = round(times.get(phase, 0.0) + elapsed, 6)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _STATE_LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _bad_answer(topic: Path, state: dict, role: str, text: str, error: Exception) -> str:
    with _STATE_LOCK:
        attempt = state.get("structural_failures", 0)
        attempt = attempt + 1 if type(attempt) is int and attempt >= 0 else 1
        state["structural_failures"] = attempt
        path = topic / "work" / "bad-answers" / f"import-{role}-attempt-{attempt}.txt"
        _atomic_text(path, f"# {error}\n{text[:_MAX_ROLE_TEXT]}")
        checkpoint.save(topic, state)
    return f"import_{role}_structural_failure: {error}"


def _call_role(
    topic: Path, state: dict, *, adapter: HarnessAdapter, cfg: RunConfig,
    role_name: str, prompt: str, schema: dict[str, type], heartbeat: RoleHeartbeat,
    validator=None,
) -> tuple[int | None, dict | None]:
    def call(call_prompt: str) -> RunResult:
        model, effort = cfg.for_role(
            ROLE_THINK, harness_default=adapter.default_model_for_role(ROLE_THINK)
        )
        heartbeat_role = {"extract": "collect", "verify": "verify"}.get(role_name, "synth")
        token = heartbeat.start(
            phase=str(state.get("phase")), role=heartbeat_role,
            cycle=state.get("cycles") if isinstance(state.get("cycles"), int) else 0,
        )
        try:
            result = adapter.run(
                call_prompt, cwd=str(topic), timeout=cfg.timeout, retries=cfg.retries,
                retry_backoff=cfg.retry_backoff, model=model, effort=effort,
                # Empty physical tool set closes Claude; Codex has no equivalent switch and
                # remains bounded by its no-network sandbox plus the supplied source window.
                tools="", allowed_tools="", network=False,
                on_harness_pid=lambda pid: heartbeat.harness_pid(token, pid),
            )
        finally:
            heartbeat.finish(token)
        with _STATE_LOCK:
            _record_call(state, role_name, result)
            checkpoint.save(topic, state)
        return result

    result, value, diagnostic = _structured_answer_policy(
        prompt,
        call=call,
        bad_answer=lambda text, error: _bad_answer(
            topic, state, role_name, text, error
        ),
        retries=cfg.retries,
        retry_backoff=cfg.retry_backoff,
        schema=schema,
        validator=validator,
        role_label=f"import {role_name}",
    )
    if result.stop != "done":
        with _STATE_LOCK:
            state["stop_kind"] = result.stop
            state["stop"] = {"reason": f"import_{role_name}_role"}
            checkpoint.save(topic, state)
        return _STOP_EXIT.get(result.stop, EXIT_FAIL), None
    if diagnostic is not None:
        return EXIT_FAIL, None
    return None, value


def _pool_map(tasks: list, pool_size: int) -> list:
    """Run the thunks in a pool and return the results in the original order."""
    workers = max(1, min(3, int(pool_size or 1), len(tasks)))
    if workers == 1:
        return [task() for task in tasks]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda task: task(), tasks))


def _extract_prompt(window: SourceWindow, title: str) -> str:
    payload = window.to_dict()
    return (
        "You extract verifiable knowledge from a single bounded window of a course lesson. "
        "Do not use tools, files, the network or any knowledge outside WINDOW.\n"
        "What counts as a claim: an atomic substantive statement of the course - a mechanism "
        "(how it works), a rule, a threshold, a number, an example, a recommendation, a "
        "trade-off, the author's position. One claim is one thought, written in Russian and "
        "understandable without the context of the lesson; English words only as terms. "
        "Preserve modality: state the author's opinion as their position, not as a fact. "
        "NOT claims: organizational matters, greetings, announcements, exercise instructions, "
        "repetitions of what was already said in this window, retelling without substance.\n"
        "Density: usually 3-8 claims per window, never more than 12. This is a skeleton of "
        "knowledge, not a summary: the steps of one technique or the walkthrough of one example "
        "are ONE claim carrying the essence of the technique, not a claim per step; take the "
        "statements a reader will want to remember and apply.\n"
        "evidence - 1-2 verbatim substrings of WINDOW.text supporting the claim; copy the quote "
        "character by character (offsets need not be counted: start_offset/end_offset may be "
        "omitted, they are computed from the quote). The text is a raw transcript with "
        "recognition errors: quote it AS IS, including typos, clipped words and odd "
        "punctuation, correcting and completing nothing. "
        "confidence: high - said directly, medium - follows from what was said, low - a hint. "
        "tags - 1-4 short lowercase concept tags (latin-kebab).\n"
        "Return exactly the JSON object "
        "{\"claims\":[{\"text\":string,\"confidence\":\"high|medium|low\","
        "\"evidence\":[{\"window_id\":string,\"quote\":a verbatim substring of WINDOW.text}],"
        "\"tags\":[string]}]}. If there is no knowledge, claims is empty.\n"
        f"LESSON: {title}\nWINDOW:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _verify_prompt(candidates: list[dict[str, object]], window: SourceWindow) -> str:
    numbered = [{"index": index, **candidate} for index, candidate in enumerate(candidates)]
    return (
        "You are an independent critic. For EVERY candidate check that the statement follows "
        "entirely from WINDOW and its verbatim quote: the quote is in WINDOW.text character by "
        "character, the statement adds nothing beyond it and the window, and modality is not "
        "distorted (an opinion is not passed off as a fact). Do not use tools, files or the "
        "network. Reject inventions, generalizations beyond what was said, and retellings of "
        "organizational matters. Return exactly "
        "{\"verdicts\":[{\"index\":int,\"accepted\":boolean,\"reason\":a non-empty string}]} "
        "- one verdict per index, with no gaps.\n"
        f"CANDIDATES:\n{json.dumps(numbered, ensure_ascii=False)}\n"
        f"WINDOW:\n{json.dumps(window.to_dict(), ensure_ascii=False)}"
    )


def normalize_verify_batch(value: object, count: int) -> list[bool]:
    """Validate a critic batch strictly: exactly ``count`` gap-free verdicts."""
    if type(value) is not dict or set(value) != {"verdicts"}:
        raise SourceImportError("invalid_verify_result")
    verdicts = value["verdicts"]
    if type(verdicts) is not list or len(verdicts) != count:
        raise SourceImportError("invalid_verify_result")
    accepted: dict[int, bool] = {}
    for row in verdicts:
        if type(row) is not dict or set(row) != {"index", "accepted", "reason"}:
            raise SourceImportError("invalid_verify_result")
        index = row["index"]
        if type(index) is not int or index < 0 or index >= count or index in accepted:
            raise SourceImportError("invalid_verify_result")
        accepted[index] = normalize_verify_result(
            {"accepted": row["accepted"], "reason": row["reason"]}
        )
    return [accepted[index] for index in range(count)]


def _document_windows(windows: tuple[SourceWindow, ...]) -> dict[str, list[SourceWindow]]:
    result: dict[str, list[SourceWindow]] = {}
    for window in windows:
        result.setdefault(window.document_id, []).append(window)
    return result


def _assert_source_row(wiki: object, topic: Path, source_id: str, expected: dict) -> None:
    reader = getattr(wiki, "read_sources", None)
    if callable(reader):
        rows = reader(topic)
    else:
        rows = [json.loads(line) for line in (topic / "sources" / "sources.jsonl").read_text("utf-8").splitlines() if line]
    row = next((item for item in rows if item.get("id") == source_id), None)
    if not isinstance(row, dict) or any(row.get(key) != value for key, value in expected.items()):
        raise SourceImportError("source_replay_conflict")


def _verify_quotes(topic: Path, wiki: object) -> list[dict]:
    verifier = getattr(wiki, "verify_quotes", None)
    if callable(verifier):
        return verifier(topic, zone="staging")
    from llmwiki.ext.evidence import verify_quotes
    return verify_quotes(topic, zone="staging")


def _read_staging_claims(topic: Path, wiki: object) -> list[dict]:
    reader = getattr(wiki, "read_claims", None)
    if not callable(reader):
        from llmwiki import read_claims
        reader = read_claims
    rows = reader(topic, zone="staging")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("read_claims_invalid_result")
    return rows


def _retract_quote_not_found(
    topic: Path, wiki: object, findings: list[dict],
) -> dict[str, int]:
    """Defuse quote-not-found only through the public llm-wiki contract.

    The current llmwiki has no way to delete a single evidence item:
    merge_claim_evidence is append-only. So even when verbatim support remains, the one
    safe fallback is retract_claim on the whole claim. We never write claims.jsonl directly.
    """
    bad_by_claim: dict[str, set[int]] = {}
    error_by_claim: dict[str, set[int]] = {}
    for row in findings:
        if not isinstance(row, dict) or row.get("level") != "error":
            continue
        claim_id = row.get("claim")
        evidence_index = row.get("evidence")
        if not isinstance(claim_id, str) or type(evidence_index) is not int:
            raise ValueError("verify_quotes_invalid_result")
        error_by_claim.setdefault(claim_id, set()).add(evidence_index)
        if row.get("code") == "quote-not-found":
            bad_by_claim.setdefault(claim_id, set()).add(evidence_index)
    if not bad_by_claim:
        return {"healed_quotes": 0, "retracted_claims": 0}

    claims = {
        row.get("id"): row
        for row in _read_staging_claims(topic, wiki)
        if isinstance(row.get("id"), str) and row.get("status") != "retracted"
    }
    retract = getattr(wiki, "retract_claim", None)
    if not callable(retract):
        from llmwiki import retract_claim
        retract = retract_claim

    healed_quotes = 0
    retracted_claims = 0
    for claim_id in sorted(bad_by_claim):
        claim = claims.get(claim_id)
        evidence = claim.get("evidence") if isinstance(claim, dict) else None
        bad = sorted(bad_by_claim[claim_id])
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(index < 0 or index >= len(evidence) for index in bad)
        ):
            raise ValueError("verify_quotes_invalid_result")
        remaining = sum(
            1 for index in range(len(evidence))
            if index not in error_by_claim.get(claim_id, set())
        )
        retract(topic, "staging", claim_id)
        _append_jsonl(
            topic / "work" / "dropped-claims.jsonl",
            {
                "reason": "quote_not_found_at_gate",
                "claim_id": claim_id,
                "zone": "staging",
                "evidence_indexes": bad,
                "remaining_verbatim_evidence": remaining,
                "action": "retracted",
                "detail": "llmwiki has no API for deleting a single evidence item",
            },
        )
        healed_quotes += len(bad)
        retracted_claims += 1
    print(
        f"gate: healed {healed_quotes} quotes, "
        f"retracted {retracted_claims} claims",
        file=sys.stderr,
        flush=True,
    )
    return {
        "healed_quotes": healed_quotes,
        "retracted_claims": retracted_claims,
    }


def _quote_gate_repair_total(state: dict, delta: dict[str, int]) -> dict[str, int]:
    prior = state.get("quote_gate_repair")
    total = {
        "healed_quotes": (
            prior.get("healed_quotes", 0) if isinstance(prior, dict) else 0
        ),
        "retracted_claims": (
            prior.get("retracted_claims", 0) if isinstance(prior, dict) else 0
        ),
    }
    for key in total:
        if type(total[key]) is not int or total[key] < 0:
            total[key] = 0
        total[key] += delta.get(key, 0)
    state["quote_gate_repair"] = total
    return total


def _quote_gate_summary(repair: Mapping[str, int]) -> str:
    return (
        f"gate: healed {repair.get('healed_quotes', 0)} quotes, "
        f"retracted {repair.get('retracted_claims', 0)} claims"
    )


class CourseImportRunner:
    def __init__(
        self, topic_dir: str | Path, *, wiki: object, extract_adapter: HarnessAdapter,
        verify_adapter: HarnessAdapter, extract_config: RunConfig,
        verify_config: RunConfig, pages_config: RunConfig | None = None,
        notifier=None, allow_same_adapter_verify: bool = False,
    ):
        self.topic = Path(topic_dir)
        self.wiki = wiki
        self.extract_adapter = extract_adapter
        self.verify_adapter = verify_adapter
        self.extract_cfg = extract_config
        self.verify_cfg = verify_config
        self.allow_same_adapter_verify = allow_same_adapter_verify
        # Pages are written by the extraction harness (the first harness, as in the archive
        # flow: synthesis is its job, criticism belongs to the second one); the synthesis
        # model/effort can be set separately.
        self.pages_adapter = extract_adapter
        self.pages_cfg = pages_config if pages_config is not None else extract_config
        self.notifier = notifier
        self._heartbeat = RoleHeartbeat(self.topic)

    @contextmanager
    def _lock(self):
        path = self.topic / "work" / "resume.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("import_topic_already_running") from error
            lock.seek(0)
            lock.truncate()
            lock.write(f"pid={os.getpid()}\n")
            lock.flush()
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _state(self) -> dict:
        state = checkpoint.load(self.topic)
        if state is None or state.get("run_mode") != "import":
            raise ValueError("not_an_import_topic")
        if (
            self.extract_adapter.name == self.verify_adapter.name
            and not self.allow_same_adapter_verify
        ):
            raise ValueError("verification_adapter_not_independent")
        configured = state.get("adapters")
        if (
            not isinstance(configured, dict)
            or set(configured) != {"extract", "verify"}
            or not all(isinstance(value, str) and value for value in configured.values())
        ):
            raise ValueError("invalid_import_adapters")
        current = {"extract": self.extract_adapter.name, "verify": self.verify_adapter.name}
        if configured != current:
            extracted_dir = self.topic / "work" / "extracted"
            has_extracted = extracted_dir.is_dir() and any(extracted_dir.glob("*.json"))
            if (
                has_extracted and isinstance(configured, dict)
                and configured.get("extract") != current["extract"]
            ):
                raise ValueError("import_extract_adapter_change_after_extract")
            state["adapters"] = current
            checkpoint.save(self.topic, state)
        return state

    def run(self) -> int:
        with self._lock():
            while True:
                state = self._state()
                phase = state.get("phase")
                if phase == "import:done":
                    try:
                        self._snapshot(state)
                    except (SourceImportError, OSError, TypeError, ValueError) as error:
                        raise ValueError(
                            "import_corpus_changed_after_done: use extend or "
                            "rebuild the topic"
                        ) from error
                    return EXIT_OK
                handler = {
                    "import:plan": self._plan,
                    "import:register": self._register,
                    "import:extract": self._extract,
                    "import:verify": self._verify,
                    "import:pages": self._pages,
                }.get(phase)
                if handler is None:
                    raise ValueError("invalid_import_phase")
                started = time.monotonic()
                code = handler(state)
                latest = self._state()
                _record_phase_time(latest, phase, started)
                checkpoint.save(self.topic, latest)
                receipt_path = self.topic / "work" / "import-receipt.json"
                if phase in {"import:verify", "import:pages"} and receipt_path.is_file():
                    receipt = json.loads(receipt_path.read_text("utf-8"))
                    times = _state_metrics(latest)["phase_seconds"]
                    receipt["phase_seconds"] = times
                    receipt["total_seconds"] = sum(times.values())
                    _atomic_json(receipt_path, receipt)
                if code is not None:
                    return code

    def _snapshot(self, state: dict) -> tuple[ImportPlan, tuple[SourceWindow, ...]]:
        plan = ImportPlan.from_dict(json.loads((self.topic / "work" / "import-plan.json").read_text("utf-8")))
        if (
            plan.plan_id != state.get("plan_id")
            or plan.corpus_sha256 != state.get("corpus_sha256")
            or [document.document_id for document in plan.documents]
            != _planned_document_ids(state.get("documents", []))
        ):
            raise ValueError("import_plan_artifact_conflict")
        windows = iter_source_windows(
            state["dump"], plan=plan, approval=plan.plan_id
        )
        decisions = _parse_plan(
            self.topic / "work" / "plan.md", state["documents"], plan.plan_id
        )
        if decisions is None or _selection_digest(decisions) != state.get("selection_digest"):
            raise ValueError("import_plan_changed_after_approval")
        return plan, windows

    def _plan(self, state: dict) -> int | None:
        plan = ImportPlan.from_dict(json.loads((self.topic / "work" / "import-plan.json").read_text("utf-8")))
        if (
            plan.plan_id != state.get("plan_id")
            or plan.corpus_sha256 != state.get("corpus_sha256")
            or [document.document_id for document in plan.documents]
            != _planned_document_ids(state.get("documents", []))
        ):
            raise ValueError("import_plan_artifact_conflict")
        path = self.topic / "work" / "plan.md"
        if state.get("auto_plan") and _parse_plan(path, state["documents"], plan.plan_id) is None:
            _atomic_text(path, path.read_text("utf-8") + f"approved: {plan.plan_id}\n")
        decisions = _parse_plan(path, state["documents"], plan.plan_id)
        if decisions is None:
            state["stop_kind"] = "waiting_human"
            state["stop"] = {"reason": "import_plan_review"}
            checkpoint.save(self.topic, state)
            return EXIT_PLAN_REVIEW
        if not any(value != "DROP" for value in decisions.values()):
            raise ValueError("import_plan_selects_nothing")
        # Reconcile the fd-anchored snapshot before advancing the durable phase.
        iter_source_windows(state["dump"], plan=plan, approval=plan.plan_id)
        state["decisions"] = decisions
        state["selection_digest"] = _selection_digest(decisions)
        state["stop_kind"] = None
        state["stop"] = None
        state["phase"] = "import:register"
        checkpoint.save(self.topic, state)
        self._snapshot(state)
        return None

    def _register(self, state: dict) -> int | None:
        plan, windows = self._snapshot(state)
        registered = state.setdefault("registered", {})
        sources_map: dict[str, dict[str, str]] = {}
        add_source = getattr(self.wiki, "add_source", None)
        if not callable(add_source):
            raise ValueError("invalid_llmwiki_socket")
        by_id = {row.document_id: row for row in plan.documents}
        windows_by_document = _document_windows(windows)
        for row in state["documents"]:
            nnn = row["nnn"]
            if state["decisions"][nnn] == "DROP":
                continue
            document = by_id[row["document_id"]]
            if state.get("link_dump") is True:
                content_path = str(
                    _dump_content_path(
                        Path(state["dump"]), plan, document.relative_path
                    )
                )
            else:
                content_path = _copied_content_path(
                    self.topic,
                    nnn,
                    _document_text(
                        windows_by_document[document.document_id], document.sha256
                    ),
                )
            expected = {
                "content_path": content_path,
                "meta": {
                    "document_id": document.document_id,
                    "relative_path": document.relative_path,
                    "sha256": document.sha256,
                    "source_kind": plan.source_kind,
                    "order": nnn,
                    "decision": state["decisions"][nnn],
                },
            }
            source_id = registered.get(document.document_id)
            if source_id is None:
                source_id = add_source(
                    self.topic, kind="local", title=document.title,
                    url=f"urn:researcher-source:{document.document_id}",
                    tool="researcher-course-import-v1", content_path=content_path,
                    meta=expected["meta"],
                )
                if not isinstance(source_id, str) or not source_id.startswith("src_"):
                    raise ValueError("invalid_llmwiki_result")
                registered[document.document_id] = source_id
                checkpoint.save(self.topic, state)
            _assert_source_row(self.wiki, self.topic, source_id, expected)
            sources_map[nnn] = {
                "source_id": source_id,
                "content_path": content_path,
            }
        _atomic_json(self.topic / "work" / "sources-map.json", sources_map)
        state["phase"] = "import:extract"
        checkpoint.save(self.topic, state)
        return None

    def _milestone(
        self, label: str, done: int, total: int, *, started: float, processed: int,
    ) -> None:
        """Lesson milestone: one stderr line per lesson, with the course slug; no notice.

        The ETA is computed over the lessons of THIS run: on resume the finished ones are
        skipped instantly and an overall rate would be off by a wide margin.
        """
        percent = round(done * 100 / total) if total else 0
        elapsed = time.monotonic() - started
        left = ""
        if processed and done < total:
            remaining = elapsed / processed * (total - done)
            left = f", ~{int(remaining // 60)} min left"
        head = f"{label} {self.topic.name}: {done}/{total} ({percent}%)"
        # To stderr, like the pages lines: it is line-buffered, so the milestone reaches the
        # job log immediately instead of arriving in a batch at the end like buffered stdout.
        print(f"{head}, {int(elapsed // 60)} min elapsed{left}", file=sys.stderr, flush=True)

    def _extract(self, state: dict) -> int | None:
        plan, windows = self._snapshot(state)
        grouped = _document_windows(windows)
        selected = [row for row in state["documents"] if state["decisions"][row["nnn"]] != "DROP"]
        extracted_dir = self.topic / "work" / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        started, processed, done = time.monotonic(), 0, 0
        for row in selected:
            path = extracted_dir / f"{row['nnn']}.json"
            if path.is_file():
                done += 1
                continue
            candidates: list[dict[str, object]] = []
            document_windows = grouped[row["document_id"]]
            answers = _pool_map(
                [
                    lambda current=window: _call_role(
                        self.topic, state, adapter=self.extract_adapter,
                        cfg=self.extract_cfg,
                        role_name="extract",
                        heartbeat=self._heartbeat,
                        prompt=_extract_prompt(current, str(row["title"])),
                        schema={"claims": list},
                        validator=lambda payload, current=current: normalize_extract_result(
                            payload["claims"], current, on_drop=lambda _c: None
                        ),
                    )
                    for window in document_windows
                ],
                self.extract_cfg.pool_size,
            )
            for window, (code, value) in zip(document_windows, answers):
                if code is not None:
                    return code
                try:
                    if not isinstance(value, dict) or set(value) != {"claims"}:
                        raise SourceImportError("invalid_extract_result")
                    window_drops: list[dict[str, object]] = []
                    normalized = normalize_extract_result(
                        value["claims"], window, on_drop=window_drops.append
                    )
                    for lost in window_drops:
                        _append_jsonl(
                            self.topic / "work" / "dropped-claims.jsonl",
                            {"nnn": row["nnn"], "window_id": window.window_id, "claim": lost},
                        )
                    if window_drops:
                        print(
                            f"extract: {row['nnn']} window {window.window_id}: "
                            f"dropped {len(window_drops)} claim(s) without a supporting quote",
                            file=sys.stderr,
                        )
                except SourceImportError as error:
                    _bad_answer(self.topic, state, "extract", json.dumps(value, ensure_ascii=False), error)
                    return EXIT_FAIL
                candidates.extend({**claim, "window_id": window.window_id} for claim in normalized)
            unique: dict[str, dict[str, object]] = {}
            for claim in candidates:
                key = str(claim["text"])
                prior = unique.get(key)
                if prior is None:
                    unique[key] = claim
                    continue
                if prior == claim:
                    continue
                # The same statement in two windows of one lesson is the same class as in two
                # lessons (claim id = hash(text)): the lecturer repeated the thought and the
                # evidence and tags diverged. Take the first occurrence and record the evidence
                # instead of failing the course.
                _append_jsonl(
                    self.topic / "work" / "dropped-claims.jsonl",
                    {
                        "nnn": row["nnn"],
                        "reason": "duplicate_text_in_document",
                        "kept_window": prior.get("window_id"),
                        "dropped_window": claim.get("window_id"),
                        "claim": claim,
                    },
                )
                print(
                    f"extraction: {row['nnn']} window {claim.get('window_id')}: "
                    f"statement repeated from window {prior.get('window_id')}, duplicate dropped",
                    file=sys.stderr,
                )
            _atomic_json(path, {
                "schema_version": 1,
                "document_id": row["document_id"],
                "adapter": self.extract_adapter.name,
                "claims": list(unique.values()),
            })
            state.setdefault("done", []).append(f"extract:{row['nnn']}")
            checkpoint.save(self.topic, state)
            processed += 1
            done += 1
            self._milestone(
                "extraction", done, len(selected),
                started=started, processed=processed,
            )
        state["phase"] = "import:verify"
        checkpoint.save(self.topic, state)
        return None

    def _claim_conflict(self, nnn: str, report: dict) -> None:
        """Evidence for a claim that did not land as a new line: to disk and to stderr.

        Before 26.08 such a refusal reached the log as the SINGLE word
        `claim_replay_conflict` - without the claim, the lesson or the two versions, and both
        sides had to be dug out of the topic by hand. Now every case has a line in
        work/dropped-claims.jsonl.
        """
        _append_jsonl(self.topic / "work" / "dropped-claims.jsonl", {"nnn": nnn, **report})
        incoming = report.get("incoming") or {}
        existing = report.get("existing") or {}
        print(
            f"verification: {nnn} claim {report.get('claim_id')}: {report.get('reason')}"
            f" (window {incoming.get('source_window')} / sources "
            f"{','.join(incoming.get('source_ids') or []) or '-'}"
            f" against window {existing.get('source_window')} / sources "
            f"{','.join(existing.get('source_ids') or []) or '-'}"
            f"; diverged: {', '.join(report.get('diverged') or []) or '-'}"
            f"{'; merge ' + str(report['merge']) if 'merge' in report else ''})",
            file=sys.stderr, flush=True,
        )

    def _verify(self, state: dict) -> int | None:
        plan, windows = self._snapshot(state)
        by_window = {window.window_id: window for window in windows}
        verified_dir = self.topic / "work" / "verified"
        verified_dir.mkdir(parents=True, exist_ok=True)
        selected = [row for row in state["documents"] if state["decisions"][row["nnn"]] != "DROP"]
        started, processed, done = time.monotonic(), 0, 0
        for row in selected:
            path = verified_dir / f"{row['nnn']}.json"
            fresh = not path.is_file()
            if path.is_file():
                payload = json.loads(path.read_text("utf-8"))
            else:
                extracted = json.loads((self.topic / "work" / "extracted" / f"{row['nnn']}.json").read_text("utf-8"))
                if extracted.get("adapter") != state["adapters"]["extract"]:
                    raise ValueError("extracted_artifact_adapter_conflict")
                if (
                    extracted["adapter"] == self.verify_adapter.name
                    and not self.allow_same_adapter_verify
                ):
                    raise ValueError("verification_adapter_not_independent")
                accepted: list[dict[str, object]] = []
                # One batch per window: a single critic call for all candidates of the window
                # instead of one per claim (13 lessons = 500-1000 harness processes in sequence,
                # hours wasted). Harness independence and a per-claim verdict are preserved.
                by_window_candidates: dict[str, list[dict[str, object]]] = {}
                for candidate in extracted["claims"]:
                    by_window_candidates.setdefault(candidate["window_id"], []).append(
                        {key: candidate[key] for key in ("text", "confidence", "evidence", "tags")}
                    )
                batches = list(by_window_candidates.items())
                answers = _pool_map(
                    [
                        lambda window_id=window_id, candidates=candidates: _call_role(
                            self.topic, state, adapter=self.verify_adapter,
                            cfg=self.verify_cfg,
                            role_name="verify",
                            heartbeat=self._heartbeat,
                            prompt=_verify_prompt(candidates, by_window[window_id]),
                            schema={"verdicts": list},
                            validator=lambda payload, n=len(candidates): normalize_verify_batch(payload, n),
                        )
                        for window_id, candidates in batches
                    ],
                    self.verify_cfg.pool_size,
                )
                for (window_id, candidates), (code, verdict) in zip(batches, answers):
                    window = by_window[window_id]
                    if code is not None:
                        return code
                    try:
                        keeps = normalize_verify_batch(verdict, len(candidates))
                    except SourceImportError as error:
                        _bad_answer(self.topic, state, "verify", json.dumps(verdict, ensure_ascii=False), error)
                        return EXIT_FAIL
                    for normalized, keep in zip(candidates, keeps):
                        if keep:
                            accepted.append(verified_claim_payload(
                                normalized, window, source_id=state["registered"][row["document_id"]]
                            ))
                payload = {"schema_version": 1, "document_id": row["document_id"], "claims": accepted, "applied": False}
                _atomic_json(path, payload)
            if not payload.get("applied"):
                written = duplicates = 0
                if payload["claims"]:
                    written, duplicates = write_verified_claims(
                        self.topic, payload["claims"], wiki=self.wiki,
                        on_conflict=lambda report, nnn=row["nnn"]: self._claim_conflict(
                            nnn, report
                        ),
                    )
                payload["written"] = written
                payload["duplicates"] = duplicates
                payload["applied"] = True
                _atomic_json(path, payload)
                state.setdefault("done", []).append(f"verify:{row['nnn']}")
                checkpoint.save(self.topic, state)
            done += 1
            if fresh:
                processed += 1
                self._milestone(
                    "verification", done, len(selected),
                    started=started, processed=processed,
                )

        findings = _verify_quotes(self.topic, self.wiki)
        if not isinstance(findings, list):
            raise ValueError("verify_quotes_invalid_result")
        quote_errors = [
            row for row in findings
            if isinstance(row, dict)
            and row.get("level") == "error"
            and row.get("code") == "quote-not-found"
        ]
        source_unreadable = [
            row for row in findings
            if isinstance(row, dict)
            and row.get("level") == "error"
            and row.get("code") == "source-unreadable"
        ]
        if quote_errors and not source_unreadable:
            repair = _retract_quote_not_found(self.topic, self.wiki, findings)
            _quote_gate_repair_total(state, repair)
            checkpoint.save(self.topic, state)
            # Self-healing counts as finished only after a repeated clean gate.
            findings = _verify_quotes(self.topic, self.wiki)
            if not isinstance(findings, list):
                raise ValueError("verify_quotes_invalid_result")
        _atomic_json(self.topic / "work" / "quotes-report.json", findings)
        errors = [row for row in findings if isinstance(row, dict) and row.get("level") == "error"]
        validation = self.wiki.validate_topic(self.topic)
        if not isinstance(validation, list):
            raise ValueError("validate_invalid_result")
        if errors or validation:
            state["stop_kind"] = "gate_refused"
            state["stop"] = {"reason": "verify_quotes" if errors else "validate"}
            checkpoint.save(self.topic, state)
            return EXIT_GATE_REFUSED

        extracted = [json.loads((self.topic / "work" / "extracted" / f"{row['nnn']}.json").read_text("utf-8")) for row in selected]
        verified = [json.loads((self.topic / "work" / "verified" / f"{row['nnn']}.json").read_text("utf-8")) for row in selected]
        metrics = _state_metrics(state)
        selected_document_ids = {row["document_id"] for row in selected}
        receipt = ImportReceipt(
            schema_version=1, plan_id=plan.plan_id, status="imported",
            documents=len(selected),
            windows=sum(
                document.window_count
                for document in plan.documents
                if document.document_id in selected_document_ids
            ),
            candidates=sum(len(row["claims"]) for row in extracted),
            verified=sum(row.get("written", 0) for row in verified),
            duplicates=sum(row.get("duplicates", 0) for row in verified),
            role_calls=metrics["role_calls"], usage={**metrics["usage"], "cost_usd": metrics["cost_usd"]},
            phase_seconds=metrics["phase_seconds"], total_seconds=sum(metrics["phase_seconds"].values()),
        )
        receipt_payload = receipt.to_dict()
        receipt_payload["content_mode"] = (
            "linked" if state.get("link_dump") is True else "copied"
        )
        receipt_payload["verification_independent"] = (
            self.extract_adapter.name != self.verify_adapter.name
        )
        repair = _quote_gate_repair_total(
            state, {"healed_quotes": 0, "retracted_claims": 0}
        )
        receipt_payload["quote_gate_repair"] = repair
        receipt_payload["quote_gate_summary"] = _quote_gate_summary(repair)
        receipt_payload["dropped"] = [
            {
                "nnn": row["nnn"],
                "relative_path": row["relative_path"],
                "reason": row["reason"],
            }
            for row in state["documents"]
            if state["decisions"][row["nnn"]] == "DROP"
        ]
        _atomic_json(self.topic / "work" / "import-receipt.json", receipt_payload)
        state["phase"] = "import:pages"
        state["stop_kind"] = None
        state["stop"] = None
        checkpoint.save(self.topic, state)
        return None

    def _pages(self, state: dict) -> int | None:
        from .pages import PagesRunner

        def on_call(role: str, result: RunResult) -> None:
            _record_call(state, f"pages-{role}", result)
            checkpoint.save(self.topic, state)

        runner = PagesRunner(
            self.topic, wiki=self.wiki, adapter=self.pages_adapter, config=self.pages_cfg,
            heartbeat=self._heartbeat, on_call=on_call, notifier=self.notifier,
            log=lambda text: print(f"pages: {text}", file=sys.stderr),
        )
        code = runner.run()
        receipt_path = self.topic / "work" / "import-receipt.json"
        receipt = (
            json.loads(receipt_path.read_text("utf-8"))
            if receipt_path.is_file() else None
        )
        if isinstance(receipt, dict):
            map_stats_path = self.topic / "work" / "pages-map-stats.json"
            map_stats = (
                json.loads(map_stats_path.read_text("utf-8"))
                if map_stats_path.is_file() else {}
            )
            for key in (
                "model_claims", "code_assigned_claims", "model_coverage_percent"
            ):
                if type(map_stats.get(key)) in {int, float}:
                    receipt[f"pages_map_{key}"] = map_stats[key]
            receipt["role_calls"] = _state_metrics(state)["role_calls"]
            _atomic_json(receipt_path, receipt)
        if code != EXIT_OK:
            state["stop_kind"] = runner.stop_kind or "fatal"
            state["stop"] = {"reason": runner.stop_reason or "pages"}
            checkpoint.save(self.topic, state)
            return code
        if isinstance(receipt, dict):
            stats_path = self.topic / "work" / "stats.json"
            stats = json.loads(stats_path.read_text("utf-8")) if stats_path.is_file() else {}
            receipt["pages"] = stats.get("pages")
            receipt["wiki_words"] = stats.get("wiki_words")
            receipt["words_per_claim"] = stats.get("words_per_claim")
            _atomic_json(receipt_path, receipt)
        state["phase"] = "import:done"
        state["stop_kind"] = None
        state["stop"] = None
        checkpoint.save(self.topic, state)
        return EXIT_OK


@contextmanager
def _initialization_lock(base: Path, slug: str):
    """Serializes the check/create of the initial manifest for one slug across processes."""
    base.mkdir(parents=True, exist_ok=True)
    path = base / f".{slug}.researcher-import-init.lock"
    with path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _initialize_import_locked(
    wiki: object, options: ImportRunOptions, dump: Path, plan: ImportPlan,
) -> Path:
    topic = options.base / options.slug
    if topic.exists():
        state = checkpoint.load(topic)
        if isinstance(state, dict):
            if state.get("run_mode") != "import" or state.get("dump") != str(dump):
                raise ValueError("import_topic_conflict")
            if options.auto_plan and state.get("phase") == "import:plan":
                state["auto_plan"] = True
                checkpoint.save(topic, state)
            return topic
        manifest = wiki.find_topic(topic)
        if not isinstance(manifest, dict) or manifest.get("topic") != f"Course: {options.slug}":
            raise ValueError("import_topic_conflict")
    else:
        topic = Path(wiki.init_topic(options.base, f"Course: {options.slug}", slug=options.slug))
    rows = _plan_rows(plan, dump)
    _atomic_json(topic / "work" / "import-plan.json", plan.to_dict())
    _atomic_text(topic / "work" / "plan.md", _render_plan(plan, rows))
    state = {
        "phase": "import:plan", "run_mode": "import", "topic": f"Course: {options.slug}",
        "slug": options.slug, "dump": str(dump), "source_kind": plan.source_kind,
        "plan_id": plan.plan_id, "corpus_sha256": plan.corpus_sha256,
        "documents": rows, "done": [], "structural_failures": 0,
        "registered": {}, "adapters": {"extract": options.extract_adapter, "verify": options.verify_adapter},
        "auto_plan": options.auto_plan, "link_dump": options.link_dump,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    _state_metrics(state)
    checkpoint.save(topic, state)
    return topic


def initialize_import(wiki: object, options: ImportRunOptions) -> Path:
    # Absolute but not resolve(): SourceImportTool must still see and reject every
    # symlink component instead of having it normalized away before its fd checks.
    dump = Path(os.path.abspath(os.fspath(options.dump.expanduser())))
    if not dump.exists():
        raise ValueError("import_dump_not_found")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", options.slug):
        raise ValueError("invalid_import_slug")
    if (
        options.extract_adapter == options.verify_adapter
        and not options.allow_same_adapter_verify
    ):
        raise ValueError("verification_adapter_not_independent")
    plan = plan_import(dump)
    with _initialization_lock(options.base, options.slug):
        return _initialize_import_locked(wiki, options, dump, plan)


__all__ = ["CourseImportRunner", "ImportRunOptions", "initialize_import", "normalize_verify_batch"]
