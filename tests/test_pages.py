"""Pages: the concept map, transcript material, validators, the pages CLI and course-queue."""

from __future__ import annotations

import fcntl
import json
import re
import sys
from pathlib import Path

import pytest

from researcher import checkpoint, cli, course_import, course_queue, pages
from researcher.adapters.base import RunResult
from researcher.cli import EXIT_FAIL, EXIT_GATE_REFUSED, EXIT_OK, main
from researcher.orchestrator import RunConfig
from tests.test_course_import_cli import _ExtractAdapter, _VerifyAdapter, import_env  # noqa: F401

_LLMWIKI = Path(__file__).resolve().parents[2] / "tool-llm-wiki"


def _claim(cid: str, tags: list[str], sources: list[str]) -> dict:
    return {"id": cid, "text": f"text {cid}", "tags": tags, "sources": sources, "evidence": []}


def test_validate_map_rejects_duplicates_unknown_and_low_coverage():
    known = {f"clm_{i:012x}" for i in range(10)}
    ids = sorted(known)
    pages._validate_map({"pages": [{"slug": "aa", "title": "A", "claims": ids}]}, known)
    with pytest.raises(ValueError, match="lies on two pages"):
        pages._validate_map(
            {"pages": [{"slug": "aa", "title": "A", "claims": ids}, {"slug": "bb", "title": "B", "claims": ids[:1]}]},
            known,
        )
    with pytest.raises(ValueError, match="unknown claim"):
        pages._validate_map({"pages": [{"slug": "aa", "title": "A", "claims": ["clm_ffffffffffff"]}]}, known)
    with pytest.raises(ValueError, match="covered 5 claims of 10: at least 60% required"):
        pages._validate_map({"pages": [{"slug": "aa", "title": "A", "claims": ids[:5]}]}, known)
    with pytest.raises(ValueError, match="slug"):
        # Cyrillic on purpose: _SLUG allows lowercase Cyrillic, so the rejection of an
        # uppercase Cyrillic slug with a space is exactly what is under test here.
        pages._validate_map({"pages": [{"slug": "Урок 3", "title": "A", "claims": ids}]}, known)


def test_live_size_map_with_1709_of_2351_claims_is_accepted_and_completed_by_code():
    """Regression for the iz-captainbuilders live stop: 72.7% is a usable map, not fatal."""
    claims = [
        _claim(
            f"clm_{index:012x}",
            [f"topic-{index % 19}"],
            [f"src_{index % 162:03d}"],
        )
        for index in range(2351)
    ]
    mapped = claims[:1709]
    answer = {
        "pages": [
            {
                "slug": f"concept-{offset // 20:03d}",
                "title": f"Concept {offset // 20:03d}",
                "claims": [claim["id"] for claim in mapped[offset: offset + 20]],
            }
            for offset in range(0, len(mapped), 20)
        ]
    }

    pages._validate_map(answer, {claim["id"] for claim in claims})
    plan = pages.build_pages_plan(answer, claims)
    assigned = [claim_id for page in plan for claim_id in page["claims"]]
    assert len(assigned) == len(set(assigned)) == 2351
    assert max(len(page["claims"]) for page in plan) <= pages.MAX_PAGE_CLAIMS
    stats = pages.map_stats(plan, model_claims=1709)
    assert stats["claims"] == 2351
    assert stats["model_claims"] == 1709
    assert stats["code_assigned_claims"] == 642
    assert stats["model_coverage_percent"] == 72.7


def test_sanitize_map_fixes_one_bad_claim_instead_of_dropping_the_whole_map():
    """Incident ok-msthirdedition 26.08: a 50 KB map was rejected three times over a single
    claim. Repair what is repairable; rejection stays for a genuinely broken answer."""
    known = {f"clm_{i:012x}" for i in range(10)}
    ids = sorted(known)
    value = {"pages": [
        {"slug": "aa", "title": "A", "claims": ids[:5] + ["clm_6fad6daa2ef0"]},
        {"slug": "bb", "title": "B", "claims": ids[5:] + [ids[0]]},
        {"slug": "cc", "title": "C", "claims": ["clm_ffffffffffff"]},
    ]}
    removed = pages.sanitize_map(value, known)
    assert [page["slug"] for page in value["pages"]] == ["aa", "bb"]
    assert value["pages"][0]["claims"] == ids[:5]
    assert value["pages"][1]["claims"] == ids[5:]
    assert [(row["reason"], row["page"]) for row in removed] == [
        ("unknown_claim", "aa"),
        ("claim_on_two_pages", "bb"),
        ("unknown_claim", "cc"),
        ("page_without_claims", "cc"),
    ]
    # After the cleanup the map passes the validator without repeating the role call.
    pages._validate_map(value, known)

    # A broken answer stays broken: every id is invented - no pages are left.
    broken = {"pages": [{"slug": "aa", "title": "A", "claims": ["clm_ffffffffffff"]}]}
    assert len(pages.sanitize_map(broken, known)) == 2
    with pytest.raises(ValueError, match="non-empty list"):
        pages._validate_map(broken, known)


def test_sanitize_map_merges_normalized_same_title_with_full_coverage():
    # Cyrillic on purpose: the two titles differ only by case and whitespace, so this
    # exercises non-ASCII case folding in normalize_text.
    claims = [
        _claim(f"clm_{index:012x}", ["keys"], [f"src_{index % 2}"])
        for index in range(1, 5)
    ]
    known = {claim["id"] for claim in claims}
    value = {"pages": [
        {
            "slug": "chastota-klyuchei",
            "title": " Ключи   в кратком описании ",
            "claims": [claims[0]["id"], claims[1]["id"]],
        },
        {
            "slug": "klyuchi-v-opisanii",
            "title": "ключи в КРАТКОМ описании",
            "claims": [claims[2]["id"], claims[3]["id"]],
        },
    ]}

    removed = pages.sanitize_map(value, known)

    assert [page["slug"] for page in value["pages"]] == ["chastota-klyuchei"]
    assert value["pages"][0]["title"] == " Ключи   в кратком описании "
    assert value["pages"][0]["claims"] == [claim["id"] for claim in claims]
    assert removed == [{
        "reason": "same_title_merged",
        "page": "klyuchi-v-opisanii",
        "into": "chastota-klyuchei",
        "claim_count": 4,
    }]
    pages._validate_map(value, known)
    plan = pages.build_pages_plan(value, claims)
    assert len(plan) == 1
    assert set(plan[0]["claims"]) == known

    # Validator backstop: if the sanitizer is bypassed, a same-title map does not pass.
    # Cyrillic on purpose: the titles differ only by case and whitespace, so this checks
    # non-ASCII case folding in normalize_text.
    duplicate = {"pages": [
        {"slug": "first", "title": "Один  title", "claims": [claims[0]["id"], claims[1]["id"]]},
        {"slug": "second", "title": " один title ", "claims": [claims[2]["id"], claims[3]["id"]]},
    ]}
    with pytest.raises(ValueError, match="title of page .* repeats the title"):
        pages._validate_map(duplicate, known)


def test_build_pages_plan_assigns_orphans_and_merges_singletons():
    claims = [
        _claim("clm_000000000001", ["go", "channels"], ["src_a"]),
        _claim("clm_000000000002", ["go", "channels"], ["src_a"]),
        _claim("clm_000000000003", ["sql"], ["src_b"]),
        _claim("clm_000000000004", ["sql", "index"], ["src_b"]),
        _claim("clm_000000000005", ["channels"], ["src_c"]),  # orphan -> to channels
        _claim("clm_000000000006", ["misc"], ["src_a"]),       # singleton -> merge
    ]
    answer = {"pages": [
        {"slug": "channels", "title": "Channels", "claims": ["clm_000000000001", "clm_000000000002"]},
        {"slug": "sql", "title": "SQL", "claims": ["clm_000000000003", "clm_000000000004"]},
        {"slug": "misc", "title": "Misc", "claims": ["clm_000000000006"]},
    ]}
    plan = pages.build_pages_plan(answer, claims)
    by_slug = {page["page"]: page for page in plan}
    assert set(by_slug) == {"channels", "sql"}
    assert "clm_000000000005" in by_slug["channels"]["claims"]
    assert "clm_000000000006" in by_slug["channels"]["claims"] + by_slug["sql"]["claims"]
    assert all(page["status"] == "todo" for page in plan)
    assert by_slug["channels"]["lessons"] == ["src_a", "src_c"]
    stats = pages.map_stats(plan)
    assert stats["pages"] == 2 and stats["claims"] == 6
    assert stats["lessons_in_two_or_more_pages"] in {0, 1}


class _ResumeWikiWithoutDelete:
    def __init__(self, topic: Path, claims: list[dict]):
        self.topic = topic
        self.claims = claims
        self.put_calls: list[dict] = []

    def list_claims(self, *args, **kwargs):
        return self.claims

    def read_sources(self, *args, **kwargs):
        return [{
            "id": "src_resume",
            "title": "Lesson",
            "content_path": "sources/content/001.md",
        }]

    def promote_claims(self, *args, **kwargs):
        raise AssertionError("final claims must not be promoted again")

    def put_wiki_page(self, topic, **payload):
        self.put_calls.append(payload)
        return {"path": str(Path(topic) / "final" / "wiki" / f"{payload['slug']}.md")}

    def validate_topic(self, *args, **kwargs):
        return []

    def audit(self, *args, **kwargs):
        return []

    def stats(self, *args, **kwargs):
        return {
            "claims_final": len(self.claims),
            "wiki_words": 280,
            "words_per_claim": 70,
        }


class _ResumeWiki(_ResumeWikiWithoutDelete):
    def __init__(self, topic: Path, claims: list[dict]):
        super().__init__(topic, claims)
        self.delete_calls: list[str] = []

    def delete_wiki_page(self, topic, *, slug: str):
        self.delete_calls.append(slug)
        return {
            "path": str(Path(topic) / "final" / "wiki" / f"{slug}.md"),
            "result": "deleted",
        }


class _ResumePageAdapter:
    name = "fake"

    def __init__(self):
        self.prompts: list[str] = []

    def default_model_for_role(self, role):
        return None

    def run(self, prompt, **kwargs):
        self.prompts.append(prompt)
        ids = re.findall(r"^- (clm_[0-9a-f]{12}):", prompt, re.MULTILINE)
        filler = " ".join(["word"] * 70)
        body = "Intro paragraph.\n\n" + "\n\n".join(
            f"## Section {index}\n\n{filler} ({claim_id})"
            for index, claim_id in enumerate(ids, 1)
        )
        return RunResult(True, json.dumps({"body": body}, ensure_ascii=False), 0, stop="done")


def _resume_pages_fixture(tmp_path: Path, wiki_type):
    # Cyrillic on purpose: the two plan titles differ only by case and whitespace - the
    # resume merge relies on non-ASCII case folding in normalize_text.
    topic = tmp_path / "topic"
    (topic / "work").mkdir(parents=True)
    (topic / "sources" / "content").mkdir(parents=True)
    quotes = [f"Fact number {index} is confirmed by the transcript." for index in range(1, 5)]
    (topic / "sources" / "content" / "001.md").write_text("\n".join(quotes), "utf-8")
    claims = [
        {
            "id": f"clm_{index:012x}",
            "text": quote,
            "status": "verified",
            "zone": "final",
            "tags": ["keys"],
            "evidence": [{"source_id": "src_resume", "quote": quote}],
        }
        for index, quote in enumerate(quotes, 1)
    ]
    plan = [
        {
            "page": "first-page",
            "title": "Ключи в кратком описании",
            "claims": [claim["id"] for claim in claims[:2]],
            "lessons": ["src_resume"],
            "status": "done",
            "path": "/old/first-page.md",
            "words": 140,
        },
        {
            "page": "second-page",
            "title": "  ключи  В кратком описании ",
            "claims": [claim["id"] for claim in claims[2:]],
            "lessons": ["src_resume"],
            "status": "done",
            "path": "/old/second-page.md",
            "words": 140,
        },
    ]
    plan_path = topic / "work" / "pages-plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return topic, plan_path, claims, wiki_type(topic, claims)


def test_resume_merges_done_same_title_deletes_loser_and_rewrites_receiver(tmp_path):
    topic, plan_path, claims, wiki = _resume_pages_fixture(tmp_path, _ResumeWiki)
    adapter = _ResumePageAdapter()
    logs: list[str] = []
    runner = pages.PagesRunner(
        topic, wiki=wiki, adapter=adapter, config=RunConfig(retries=0), log=logs.append,
    )

    assert runner.run() == pages.EXIT_OK

    assert wiki.delete_calls == ["second-page"]
    assert len(wiki.put_calls) == 1
    assert wiki.put_calls[0]["slug"] == "first-page"
    assert wiki.put_calls[0]["claims"] == [claim["id"] for claim in claims]
    saved = json.loads(plan_path.read_text("utf-8"))
    assert len(saved) == 1
    assert saved[0]["page"] == "first-page"
    assert saved[0]["status"] == "done"
    assert saved[0]["claims"] == [claim["id"] for claim in claims]
    assert any(
        line == "pages: merged same-title pages: first-page <- second-page (4 claims)"
        for line in logs
    )


def test_resume_duplicate_title_without_delete_socket_keeps_plan_intact(tmp_path):
    topic, plan_path, _, wiki = _resume_pages_fixture(tmp_path, _ResumeWikiWithoutDelete)
    adapter = _ResumePageAdapter()
    original = plan_path.read_bytes()
    runner = pages.PagesRunner(
        topic, wiki=wiki, adapter=adapter, config=RunConfig(retries=0),
    )

    with pytest.raises(ValueError, match="llmwiki without delete_wiki_page"):
        runner.run()

    assert plan_path.read_bytes() == original
    assert wiki.put_calls == []
    assert adapter.prompts == []


def test_validate_body_enforces_anchors_and_word_budget():
    page = {"claims": ["clm_000000000001", "clm_000000000002"]}
    filler = " ".join(["word"] * 80)
    good = f"Intro.\n\n## One\n\n{filler} (clm_000000000001)\n\n## Two\n\n{filler} (clm_000000000002)"
    pages._validate_body({"body": good}, page)
    with pytest.raises(ValueError, match="sections without an anchor"):
        pages._validate_body({"body": good + "\n\n## Three\n\nno anchor"}, page)
    with pytest.raises(ValueError, match="not from the page claims"):
        pages._validate_body({"body": good.replace("clm_000000000002", "clm_0000000000ff")}, page)
    with pytest.raises(ValueError, match="a skeleton"):
        pages._validate_body({"body": "## One\n\nshort (clm_000000000001) (clm_000000000002)"}, page)
    with pytest.raises(ValueError, match="filler"):
        pages._validate_body({"body": good + "\n\n" + " ".join(["water"] * 600)}, page)
    with pytest.raises(ValueError, match="H1"):
        pages._validate_body({"body": "# Heading\n" + good}, page)


def test_page_material_merges_windows_and_respects_budget(tmp_path):
    topic = tmp_path / "topic"
    (topic / "sources" / "content").mkdir(parents=True)
    text = "".join(f"line {i} about channels and goroutines\n" for i in range(2000))
    (topic / "sources" / "content" / "001.md").write_text(text, "utf-8")
    sources = {"src_a": {"id": "src_a", "title": "Lesson 1", "content_path": "sources/content/001.md"}}
    quotes = ["line 100 about channels", "line 120 about channels", "line 1500 about channels"]
    by_id = {
        f"clm_{i:012x}": {"id": f"clm_{i:012x}", "text": q, "evidence": [{"source_id": "src_a", "quote": q}]}
        for i, q in enumerate(quotes)
    }
    page = {"page": "kanaly", "claims": list(by_id)}
    excerpts = pages.page_material(topic, page, by_id, sources, radius=400)
    assert len(excerpts) == 2  # 100 and 120 merged, 1500 stays separate
    assert all("channels" in item["text"] for item in excerpts)
    small = pages.page_material(topic, page, by_id, sources, radius=3000, budget=1500)
    assert sum(len(item["text"]) for item in small) <= 1500
    # evidence without a quote but with a line locator is found as well
    by_id["clm_0000000000aa"] = {
        "id": "clm_0000000000aa", "text": "x",
        "evidence": [{"source_id": "src_a", "quote": "no such quote", "locator": {"type": "line", "value": "5-6"}}],
    }
    located = pages.page_material(topic, {"page": "p", "claims": ["clm_0000000000aa"]}, by_id, sources, radius=10)
    assert located and "line 4" in located[0]["text"]


def _topic_with_claims(base: Path, llmwiki, slug: str = "kurs") -> Path:
    topic = Path(llmwiki.init_topic(base, f"Course: {slug}", slug=slug))
    content = topic / "sources" / "content"
    content.mkdir(parents=True, exist_ok=True)
    text = "A goroutine is cheaper than a thread. A channel synchronizes goroutines. Select waits on several channels.\n"
    (content / "001.md").write_text(text * 20, "utf-8")
    source_id = llmwiki.add_source(
        topic, kind="local", title="001 Channels", url="urn:test:1",
        tool="test", content_path="sources/content/001.md", meta={"order": "001"},
    )
    rows = [
        {"text": t, "confidence": "high", "status": "verified", "tags": ["go"],
         "evidence": [{"source_id": source_id, "quote": q, "stance": "supports",
                       "locator": {"type": "line", "value": "1-1"}}], "meta": {}}
        for t, q in [
            ("A goroutine is cheaper than an OS thread.", "A goroutine is cheaper than a thread."),
            ("A channel synchronizes goroutines.", "A channel synchronizes goroutines."),
            ("Select waits on several channels at once.", "Select waits on several channels."),
        ]
    ]
    llmwiki.add_claims(topic, "staging", rows)
    return topic


def test_cli_pages_sanitizes_map_without_a_second_model_call(import_env, capfd):
    """One invented id and one claim on two pages are not worth repeating the role call:
    the map is repaired in place, the evidence goes to stderr and work/dropped-claims.jsonl."""
    _, base, llmwiki = import_env
    topic = _topic_with_claims(base, llmwiki, slug="karta")
    ids = [row["id"] for row in llmwiki.read_claims(topic, zone="staging")]
    _ExtractAdapter.scripted_texts = [json.dumps({"pages": [
        {"slug": "osnovy", "title": "Basics", "claims": ids[:2] + ["clm_6fad6daa2ef0"]},
        {"slug": "prodolzhenie", "title": "Continuation", "claims": [ids[2], ids[0]]},
    ]}, ensure_ascii=False)]

    assert main(["pages", str(topic), "--adapter", "claude"]) == EXIT_OK
    captured = capfd.readouterr()
    # The role was not repeated: bad-answers is empty, exactly two calls (map + page).
    assert not (topic / "work" / "bad-answers").exists()
    assert len(_ExtractAdapter.prompts) == 2
    assert "map cleaned before validation" in captured.err
    assert "claim_on_two_pages x1" in captured.err and "unknown_claim x1" in captured.err
    dropped = [
        json.loads(line)
        for line in (topic / "work" / "dropped-claims.jsonl").read_text("utf-8").splitlines()
    ]
    assert [(row["stage"], row["reason"], row.get("claim_id")) for row in dropped] == [
        ("pages-map", "unknown_claim", "clm_6fad6daa2ef0"),
        ("pages-map", "claim_on_two_pages", ids[0]),
    ]
    # No real claim was lost: all three stayed in the map.
    plan = json.loads((topic / "work" / "pages-plan.json").read_text("utf-8"))
    assert sorted(cid for page in plan for cid in page["claims"]) == sorted(ids)


def test_cli_pages_standalone_builds_pages_and_rebuild(import_env, capfd):
    _, base, llmwiki = import_env
    topic = _topic_with_claims(base, llmwiki)
    assert main(["pages", str(topic), "--adapter", "claude", "--json"]) == EXIT_OK
    out = json.loads(capfd.readouterr().out)
    assert out["exit_code"] == 0 and out["stop_kind"] is None
    plan = json.loads((topic / "work" / "pages-plan.json").read_text("utf-8"))
    assert [page["status"] for page in plan] == ["done"]
    assert len(llmwiki.read_pages(topic)) == 1
    assert llmwiki.read_claims(topic, zone="staging") == []
    assert (topic / "work" / "audit-report.json").is_file()
    assert (topic / "work" / "pages-map-stats.json").is_file()
    # a repeat without --rebuild: everything is done, no harness calls
    calls = len(_ExtractAdapter.prompts)
    assert main(["pages", str(topic), "--adapter", "claude"]) == EXIT_OK
    capfd.readouterr()
    assert len(_ExtractAdapter.prompts) == calls
    assert main(["pages", str(topic), "--adapter", "claude", "--rebuild"]) == EXIT_OK
    capfd.readouterr()
    assert len(_ExtractAdapter.prompts) == calls + 2
    assert len(llmwiki.read_pages(topic)) == 1


def test_cli_pages_gate_refused_when_audit_red(import_env, capfd, monkeypatch):
    _, base, llmwiki = import_env
    topic = _topic_with_claims(base, llmwiki, slug="kurs2")
    monkeypatch.setattr(
        llmwiki, "audit",
        lambda *args, **kwargs: [{"level": "error", "code": "page-too-thin", "page": "osnovy"}],
    )
    assert main(["pages", str(topic), "--adapter", "claude"]) == EXIT_GATE_REFUSED
    captured = capfd.readouterr()
    assert "PAGES GATE REFUSED" in captured.err and "page-too-thin x1" in captured.err
    assert "Traceback" not in captured.err


def test_cli_pages_structural_failure_feeds_reason_into_retry(import_env, capfd):
    _, base, llmwiki = import_env
    topic = _topic_with_claims(base, llmwiki, slug="kurs3")
    ids = [row["id"] for row in llmwiki.read_claims(topic, zone="staging")]
    good_map = json.dumps({"pages": [{"slug": "osnovy", "title": "Basics", "claims": ids}]})
    _ExtractAdapter.scripted_texts = [json.dumps({"pages": []}), good_map]
    assert main(["pages", str(topic), "--adapter", "claude"]) == EXIT_OK
    capfd.readouterr()
    retry_prompt = _ExtractAdapter.prompts[1]
    assert "Previous answer rejected" in retry_prompt and "non-empty list" in retry_prompt
    assert (topic / "work" / "bad-answers" / "pages-map-attempt-1.txt").is_file()


def test_cli_pages_refuses_busy_or_missing_topic(import_env, capfd):
    _, base, llmwiki = import_env
    assert main(["pages", str(base / "net")]) == EXIT_FAIL
    assert "topic does not exist" in capfd.readouterr().err

    topic = _topic_with_claims(base, llmwiki, slug="busy")
    lock_path = topic / "work" / "resume.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        assert main(["pages", str(topic)]) == EXIT_FAIL
    assert "topic is locked by a live run" in capfd.readouterr().err
    assert not (topic / "work" / "pages-plan.json").exists()


def test_course_queue_processes_next_pending_course(import_env, capfd):
    course, base, llmwiki = import_env
    dumps = course.parent
    queue = dumps / "queue.txt"
    queue.write_text(
        "# queue\nnet-takogo\n" + course.name + "   # first\n\nchuzhaya\n", "utf-8"
    )
    foreign = base / "chuzhaya"
    foreign.mkdir(parents=True)
    (dumps / "chuzhaya").mkdir()
    (dumps / "chuzhaya" / "INDEX.md").write_text("# x\n", "utf-8")
    args = ["course-queue", "--queue", str(queue), "--dumps", str(dumps), "--base", str(base)]
    assert main(args + ["--dry-run"]) == EXIT_OK
    table = capfd.readouterr().out
    assert "net-takogo | no transcript" in table
    assert f"{course.name} | needs a run" in table
    assert "chuzhaya | topic not from import" in table
    assert main(args + ["--json"]) == EXIT_OK
    summary = json.loads(capfd.readouterr().out)
    assert summary["course"] == course.name and summary["exit_code"] == 0
    assert summary["remaining"] == 0
    assert checkpoint.load(base / course.name)["phase"] == "import:done"
    receipt = json.loads(
        (base / course.name / "work" / "import-receipt.json").read_text("utf-8")
    )
    assert receipt["pages_map_model_claims"] == 2
    assert receipt["pages_map_code_assigned_claims"] == 0
    assert receipt["pages_map_model_coverage_percent"] == 100.0
    assert len(llmwiki.read_pages(base / course.name)) == 1
    assert main(args) == EXIT_OK
    assert "course queue is empty" in capfd.readouterr().out
    assert main(args + ["--json"]) == EXIT_OK
    empty = json.loads(capfd.readouterr().out)
    assert empty["queue_empty"] is True
    assert empty["course"] is None and empty["exit_code"] is None


def test_course_queue_passes_through_import_plan_review(import_env, capfd, monkeypatch):
    course, base, _ = import_env
    dumps = course.parent
    queue = dumps / "queue.txt"
    queue.write_text(course.name + "\n", "utf-8")

    def wait_for_plan(runner):
        state = checkpoint.load(runner.topic)
        state["stop_kind"] = "waiting_human"
        state["stop"] = {"reason": "import_plan_review"}
        checkpoint.save(runner.topic, state)
        return course_import.EXIT_PLAN_REVIEW

    monkeypatch.setattr(course_import.CourseImportRunner, "run", wait_for_plan)
    args = ["course-queue", "--queue", str(queue), "--dumps", str(dumps), "--base", str(base)]

    assert main(args) == course_import.EXIT_PLAN_REVIEW
    captured = capfd.readouterr()
    assert (
            f"course {course.name}: import plan is waiting for review (exit 76)"
        in captured.err
    )


def test_course_queue_stops_on_failed_course_and_retries_it(import_env, capfd):
    course, base, llmwiki = import_env
    dumps = course.parent
    queue = dumps / "queue.txt"
    queue.write_text(course.name + "\n", "utf-8")
    args = ["course-queue", "--queue", str(queue), "--dumps", str(dumps), "--base", str(base)]
    _ExtractAdapter.quota_on_call = 1
    assert main(args) == 75
    capfd.readouterr()
    _ExtractAdapter.quota_on_call = None
    assert main(args + ["--dry-run"]) == EXIT_OK
    assert f"{course.name} | needs a run" in capfd.readouterr().out
    assert main(args) == EXIT_OK
    capfd.readouterr()
    assert checkpoint.load(base / course.name)["phase"] == "import:done"


def test_course_queue_skips_course_running_in_another_worker(import_env, capfd):
    import fcntl

    course, base, llmwiki = import_env
    dumps = course.parent
    queue = dumps / "queue.txt"
    queue.write_text("zanyatoy\n" + course.name + "\n", "utf-8")
    busy = base / "zanyatoy"
    (busy / "work").mkdir(parents=True)
    (dumps / "zanyatoy").mkdir()
    (dumps / "zanyatoy" / "INDEX.md").write_text("# x\n", "utf-8")
    checkpoint.save(busy, {"run_mode": "import", "phase": "import:extract"})
    args = ["course-queue", "--queue", str(queue), "--dumps", str(dumps), "--base", str(base)]
    with (busy / "work" / "resume.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert main(args + ["--dry-run"]) == EXIT_OK
        out = capfd.readouterr().out
        assert "zanyatoy | running (another worker) - skipped" in out
        assert f"{course.name} | needs a run" in out
        # The free course is taken, the busy one is not touched.
        assert main(args) == EXIT_OK
        capfd.readouterr()
        assert checkpoint.load(base / course.name)["phase"] == "import:done"
        # Only the busy one is left: nothing free, but the job does not claim "all done".
        assert main(args) == EXIT_OK
    assert "no available courses: remaining courses run in other workers" in capfd.readouterr().out
    # The lock is released (the worker died or finished) - the course is available again.
    assert main(args + ["--dry-run"]) == EXIT_OK
    assert "zanyatoy | needs a run" in capfd.readouterr().out


def test_course_queue_reservation_sends_parallel_worker_to_next_slug(tmp_path):
    queue = tmp_path / "queue.txt"
    dumps = tmp_path / "dumps"
    base = tmp_path / "topics"
    queue.write_text("first\nsecond\n", "utf-8")
    for slug in ("first", "second"):
        (dumps / slug).mkdir(parents=True)
        (dumps / slug / "INDEX.md").write_text("# transcript\n", "utf-8")

    with course_queue.reserve_next_course(queue, base, dumps) as first:
        assert first.slug == "first"
        with course_queue.reserve_next_course(queue, base, dumps) as second:
            assert second.slug == "second"
            assert {row["slug"]: row["state"] for row in second.rows}["first"] == "running"
