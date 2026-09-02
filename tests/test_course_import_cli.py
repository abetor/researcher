"""CLI course import: durable phases, independent roles, gates and status."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from researcher import checkpoint, cli, course_import
from researcher.adapters.base import RunResult
from researcher.cli import (
    EXIT_FAIL,
    EXIT_GATE_REFUSED,
    EXIT_OK,
    EXIT_PLAN_REVIEW,
    EXIT_QUOTA,
    main,
)
from researcher.orchestrator import RunConfig
from researcher.observability import read_heartbeat
from researcher.sources import SourceWindow, normalize_extract_result
from tests.test_source_import import _course, _manifest_row


_LLMWIKI = Path(__file__).resolve().parents[2] / "tool-llm-wiki"


class _ExtractAdapter:
    name = "claude"
    calls: list[str] = []
    prompts: list[str] = []
    run_kwargs: list[dict] = []
    scripted_texts: list[str] = []
    quota_on_call: int | None = None

    def default_model_for_role(self, role):
        return None

    def run(self, prompt, **kwargs):
        type(self).prompts.append(prompt)
        type(self).run_kwargs.append(kwargs)
        if type(self).scripted_texts:
            return RunResult(
                True, type(self).scripted_texts.pop(0), 0, stop="done"
            )
        if prompt.startswith("You are building the map of wiki pages"):
            ids = re.findall(r"^(clm_[0-9a-f]{12}) \|", prompt, re.MULTILINE)
            payload = {"pages": [{"slug": "osnovy", "title": "Basics", "claims": ids}]}
            return RunResult(
                True, json.dumps(payload, ensure_ascii=False), 0, stop="done",
                raw={"usage": {"input_tokens": 7}},
            )
        if prompt.startswith("You are writing a domain wiki page"):
            ids = re.findall(r"^- (clm_[0-9a-f]{12}):", prompt, re.MULTILINE)
            filler = " ".join(["word"] * 70)
            body = "Intro paragraph.\n\n" + "\n\n".join(
                f"## Section {index}\n\n{filler} ({claim_id})"
                for index, claim_id in enumerate(ids, 1)
            )
            return RunResult(
                True, json.dumps({"body": body}, ensure_ascii=False), 0, stop="done",
                raw={"usage": {"input_tokens": 8}},
            )
        window_text = prompt.rsplit("WINDOW:\n", 1)[1]
        window = json.loads(window_text.split("\n\nReturn EXACTLY", 1)[0])
        type(self).calls.append(window["document_id"])
        if type(self).quota_on_call == len(type(self).calls):
            return RunResult(False, "usage limit", 1, stop="quota")
        # The needles are the transcript sentences of the shared _course fixture in
        # tests/test_source_import.py.
        needle = "Source" if "Source" in window["text"] else "The evidence"
        local = window["text"].index(needle)
        end = window["text"].find(".", local) + 1
        quote = window["text"][local:end]
        start = window["start_offset"] + local
        payload = {
            "claims": [
                {
                    "text": quote,
                    "confidence": "high",
                    "evidence": [
                        {
                            "window_id": window["window_id"],
                            "quote": quote,
                            "start_offset": start,
                            "end_offset": start + len(quote),
                        }
                    ],
                    "tags": ["course-import"],
                }
            ]
        }
        return RunResult(
            True, json.dumps(payload, ensure_ascii=False), 0, stop="done",
            raw={"usage": {"input_tokens": 10}},
        )


class _VerifyAdapter:
    name = "codex"
    calls = 0
    run_kwargs: list[dict] = []

    def default_model_for_role(self, role):
        return None

    def run(self, prompt, **kwargs):
        type(self).calls += 1
        type(self).run_kwargs.append(kwargs)
        candidates = json.loads(prompt.split("CANDIDATES:\n", 1)[1].split("\nWINDOW:\n", 1)[0])
        verdicts = [
            {"index": row["index"], "accepted": True, "reason": "the quote supports the statement"}
            for row in candidates
        ]
        return RunResult(
            True,
            json.dumps({"verdicts": verdicts}, ensure_ascii=False),
            0,
            stop="done",
            raw={"usage": {"input_tokens": 5}},
        )


@pytest.fixture
def import_env(tmp_path, monkeypatch):
    if str(_LLMWIKI) not in sys.path:
        sys.path.insert(0, str(_LLMWIKI))
    import llmwiki

    course = _course(tmp_path)
    # Cyrillic on purpose: INDEX.md is written by the coursedump tool and its line format
    # ("... - NNN слов", the "ТЕКСТА НЕТ" marker) is the external contract parsed by
    # course_import._index_metadata.
    (course / "INDEX.md").write_text(
        "# synthetic\n\n"
        "- [x] [Intro](text/001%20-%20Intro.mp4.md) (10:00) - 100 слов\n"
        "- [x] [Slides](text/002%20-%20Slides.pdf.md) - 50 слов\n",
        "utf-8",
    )
    _ExtractAdapter.calls = []
    _ExtractAdapter.prompts = []
    _ExtractAdapter.run_kwargs = []
    _ExtractAdapter.scripted_texts = []
    _ExtractAdapter.quota_on_call = None
    _VerifyAdapter.calls = 0
    _VerifyAdapter.run_kwargs = []
    monkeypatch.setattr(
        cli, "ADAPTERS", {"claude": _ExtractAdapter, "codex": _VerifyAdapter}
    )
    return course, tmp_path / "topics", llmwiki


def _approve(topic: Path) -> None:
    state = checkpoint.load(topic)
    path = topic / "work" / "plan.md"
    path.write_text(path.read_text("utf-8") + f"approved: {state['plan_id']}\n", "utf-8")


def _start_waiting(course: Path, base: Path, capfd) -> Path:
    assert main(["import", str(course), "--base", str(base), "--json"]) == EXIT_PLAN_REVIEW
    topic = Path(json.loads(capfd.readouterr().out)["topic_dir"])
    return topic


@pytest.mark.parametrize(
    "approval_kind,error_code",
    [
        ("date", "import_plan_approval_invalid"),
        ("foreign", "import_plan_approval_conflict"),
        ("duplicate", "import_plan_approval_conflict"),
        ("trailing-space", "import_plan_approval_invalid"),
    ],
)
def test_import_approval_requires_one_exact_plan_id(
    import_env, capfd, approval_kind, error_code
):
    course, base, _ = import_env
    topic = _start_waiting(course, base, capfd)
    plan_id = checkpoint.load(topic)["plan_id"]
    plan_path = topic / "work" / "plan.md"
    if approval_kind == "date":
        approval = "approved: 2026-08-22\n"
    elif approval_kind == "foreign":
        foreign = "plan_" + ("0" * 24)
        assert foreign != plan_id
        approval = f"approved: {foreign}\n"
    elif approval_kind == "duplicate":
        approval = f"approved: {plan_id}\napproved: {plan_id}\n"
    else:
        approval = f"approved: {plan_id} \n"
    plan_path.write_text(plan_path.read_text("utf-8") + approval, "utf-8")

    assert main(["resume", str(topic)]) == EXIT_FAIL
    captured = capfd.readouterr()
    assert error_code in captured.err
    assert "Traceback" not in captured.err
    if approval_kind in {"date", "trailing-space"}:
        assert f"expected approved: {plan_id}" in captured.err
    assert checkpoint.load(topic)["phase"] == "import:plan"


def test_cli_import_plan_approve_resume_receipt_and_status(import_env, capfd, monkeypatch):
    course, base, llmwiki = import_env
    before = {path.relative_to(course): path.read_bytes() for path in course.rglob("*") if path.is_file()}
    calls = []
    original_add_claims = llmwiki.add_claims

    def recording_add_claims(*args, **kwargs):
        calls.append((args, kwargs))
        return original_add_claims(*args, **kwargs)

    monkeypatch.setattr(llmwiki, "add_claims", recording_add_claims)
    assert main(["import", str(course), "--base", str(base), "--json"]) == EXIT_PLAN_REVIEW
    first = json.loads(capfd.readouterr().out)
    topic = Path(first["topic_dir"])
    plan_text = (topic / "work" / "plan.md").read_text("utf-8")
    assert "| 001 |" in plan_text and "10:00" in plan_text and "100" in plan_text
    assert "approved:" in plan_text and f"approved: {checkpoint.load(topic)['plan_id']}\n" not in plan_text
    assert main(["status", str(topic)]) == EXIT_OK
    waiting_status = json.loads(capfd.readouterr().out)
    assert waiting_status["import_progress"] == {
        "documents": 2,
        "selected": None,
        "registered": 0,
        "extracted": 0,
        "verified": 0,
        "pages_planned": None,
        "pages": 0,
    }
    assert main(["resume", str(topic), "--supervise", "--json"]) == EXIT_PLAN_REVIEW
    capfd.readouterr()
    assert "supervisor_stop" in (topic / "work" / "events.jsonl").read_text("utf-8")

    _approve(topic)
    assert main(["resume", str(topic), "--json"]) == EXIT_OK
    result = json.loads(capfd.readouterr().out)
    assert result["phase"] == "import:done" and result["status"] == "done"
    receipt = json.loads((topic / "work" / "import-receipt.json").read_text("utf-8"))
    assert receipt["role_calls"] == {
        "extract": 2, "verify": 2, "pages-map": 1, "pages-page": 1,
    }
    assert receipt["usage"]["input_tokens"] == 30
    assert receipt["pages"] == 1 and receipt["wiki_words"] > 0
    heartbeat = read_heartbeat(topic)
    assert heartbeat is not None
    assert heartbeat["phase"] == "import:pages" and heartbeat["role"] == "synth"
    assert heartbeat["call_started"] is None and heartbeat["harness_pid"] is None
    assert set(receipt["phase_seconds"]) == {
        "import:plan", "import:register", "import:extract", "import:verify", "import:pages",
    }
    plan = json.loads((topic / "work" / "pages-plan.json").read_text("utf-8"))
    assert [page["status"] for page in plan] == ["done"]
    assert plan[0]["page"] == "osnovy" and len(plan[0]["claims"]) == 2
    pages = llmwiki.read_pages(topic)
    assert len(pages) == 1 and pages[0]["slug"] == "osnovy"
    assert llmwiki.read_claims(topic, zone="staging") == []
    assert len(llmwiki.read_claims(topic, zone="final")) == 2
    audit = json.loads((topic / "work" / "audit-report.json").read_text("utf-8"))
    assert [row for row in audit if row["level"] == "error"] == []
    assert receipt["windows"] == 2
    assert receipt["content_mode"] == "copied"
    assert receipt["dropped"] == []
    assert calls, "verified claims must pass through llmwiki.add_claims"
    content = sorted((topic / "sources" / "content").iterdir())
    assert [path.name for path in content] == ["001.md", "002.md"]
    assert content[0].read_text("utf-8") == (
        course / "text" / "001 - Intro.mp4.md"
    ).read_text("utf-8")
    sources = llmwiki.read_sources(topic)
    assert {row["content_path"] for row in sources} == {
        "sources/content/001.md", "sources/content/002.md"
    }
    sources_map = json.loads((topic / "work" / "sources-map.json").read_text("utf-8"))
    assert set(sources_map) == {"001", "002"}
    assert all(set(row) == {"source_id", "content_path"} for row in sources_map.values())
    assert cli._sources_map(topic) == {
        row["source_id"]: row["content_path"] for row in sources_map.values()
    }
    after = {path.relative_to(course): path.read_bytes() for path in course.rglob("*") if path.is_file()}
    assert after == before

    assert main(["status", str(topic)]) == EXIT_OK
    status = json.loads(capfd.readouterr().out)
    assert status["import_progress"] == {
        "documents": 2, "selected": 2, "registered": 2,
        "extracted": 2, "verified": 2, "pages_planned": 1, "pages": 1,
    }
    assert status["status"] == "done"
    assert main(["status", "--base", str(base), "--json"]) == EXIT_OK
    base_status = json.loads(capfd.readouterr().out)
    assert base_status["topics"][0]["phase"] == "import:done"


def test_import_resume_skips_durable_extracted_lesson(import_env, capfd):
    course, base, _ = import_env
    _ExtractAdapter.quota_on_call = 2
    assert main([
        "import", str(course), "--base", str(base), "--auto-plan", "--json"
    ]) == EXIT_QUOTA
    topic = base / course.name
    capfd.readouterr()
    assert (topic / "work" / "extracted" / "001.json").is_file()
    assert not (topic / "work" / "extracted" / "002.json").exists()
    first_document = _ExtractAdapter.calls[0]
    _ExtractAdapter.quota_on_call = None
    assert main(["resume", str(topic), "--json"]) == EXIT_OK
    capfd.readouterr()
    assert _ExtractAdapter.calls.count(first_document) == 1
    assert checkpoint.load(topic)["phase"] == "import:done"


def test_import_link_dump_keeps_absolute_paths_as_explicit_opt_in(import_env, capfd):
    course, base, llmwiki = import_env

    assert main([
        "import", str(course), "--base", str(base), "--auto-plan",
        "--link-dump", "--json",
    ]) == EXIT_OK
    topic = base / course.name
    capfd.readouterr()

    assert list((topic / "sources" / "content").iterdir()) == []
    assert all(Path(row["content_path"]).is_absolute() for row in llmwiki.read_sources(topic))
    receipt = json.loads((topic / "work" / "import-receipt.json").read_text("utf-8"))
    assert receipt["content_mode"] == "linked"


def test_import_plan_and_receipt_record_forced_drops(import_env, capfd):
    course, base, _ = import_env
    html_target = "code/application.html.md"
    html_path = course / "text" / html_target
    html_path.parent.mkdir()
    html_path.write_text("<html>application</html>\n", "utf-8")
    manifest = course / "manifest.jsonl"
    manifest.write_text(
        manifest.read_text("utf-8")
        + json.dumps(_manifest_row(html_target, kind="html"))
        + "\n",
        "utf-8",
    )
    index = course / "INDEX.md"
    index.write_text(
        index.read_text("utf-8").replace(
            "- [x] [Slides](text/002%20-%20Slides.pdf.md) - 50 слов",
            "- [x] [Slides](text/002%20-%20Slides.pdf.md) - 2 слов - ТЕКСТА НЕТ",
        ),
        "utf-8",
    )

    topic = _start_waiting(course, base, capfd)
    plan_text = (topic / "work" / "plan.md").read_text("utf-8")
    assert "DROP: empty extraction" in plan_text
    assert "DROP: non-transcript attachment" in plan_text

    _approve(topic)
    assert main(["resume", str(topic), "--json"]) == EXIT_OK
    capfd.readouterr()
    receipt = json.loads((topic / "work" / "import-receipt.json").read_text("utf-8"))
    assert receipt["documents"] == 1
    assert {row["relative_path"] for row in receipt["dropped"]} == {
        "002 - Slides.pdf.md", html_target,
    }
    assert all(row["reason"].startswith("DROP:") for row in receipt["dropped"])


@pytest.mark.parametrize(
    "source_tail,model_tail",
    [("<br>", "<br"), ("<br>", "<"), ("&nbsp;", "&nbsp")],
)
def test_extract_trims_incomplete_markup_tail_before_quote_gate(
    import_env, source_tail, model_tail
):
    _, _, _ = import_env
    prefix = "6: User #1 -> Instance #2"
    text = prefix + source_tail + "\n<!-- End of picture text -->"
    document_id = "doc_" + "0" * 24
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    window_id = "win_" + hashlib.sha256(
        f"{document_id}:0:{len(text)}:{sha256}".encode("ascii")
    ).hexdigest()[:24]
    window = SourceWindow(
        schema_version=1,
        window_id=window_id,
        document_id=document_id,
        relative_path="009.md",
        start_offset=0,
        end_offset=len(text),
        start_line=78,
        end_line=79,
        text=text,
        sha256=sha256,
    )
    claims = [{
        "text": "The user is routed to the second instance.",
        "confidence": "high",
        "evidence": [{"window_id": window_id, "quote": prefix + model_tail}],
        "tags": ["routing"],
    }]

    normalized = normalize_extract_result(claims, window)
    quote = normalized[0]["evidence"][0]["quote"]
    assert quote == prefix
    from llmwiki.ext.evidence import _find_spans, _norm
    assert _find_spans(text.split("\n"), _norm(quote)) == [(1, 1)]


def test_import_source_unreadable_gate_refusal_is_still_exit_77(
    import_env, capfd, monkeypatch
):
    course, base, _ = import_env
    monkeypatch.setattr(
        course_import,
        "_verify_quotes",
        lambda topic, wiki: [{"level": "error", "code": "source-unreadable"}],
    )
    assert main([
        "import", str(course), "--base", str(base), "--auto-plan", "--json"
    ]) == EXIT_GATE_REFUSED
    capfd.readouterr()
    state = checkpoint.load(base / course.name)
    assert state["phase"] == "import:verify"
    assert state["stop_kind"] == "gate_refused"


def _add_claims_with_bad_first_evidence(llmwiki, *, keep_second: bool):
    original_add_claims = llmwiki.add_claims
    injected = False

    def add_claims(topic, zone, claims):
        nonlocal injected
        batch = [
            {**claim, "evidence": [dict(item) for item in claim["evidence"]]}
            for claim in claims
        ]
        if not injected:
            injected = True
            if keep_second:
                batch[0]["evidence"].append(dict(batch[0]["evidence"][0]))
            batch[0]["evidence"][0]["quote"] = "this quote is not in the source"
        return original_add_claims(topic, zone, batch)

    return add_claims


def test_quote_gate_retracts_claim_with_one_bad_of_two_evidence_and_finishes_zero(
    import_env, capfd, monkeypatch
):
    course, base, llmwiki = import_env
    monkeypatch.setattr(
        llmwiki, "add_claims",
        _add_claims_with_bad_first_evidence(llmwiki, keep_second=True),
    )

    assert main([
        "import", str(course), "--base", str(base), "--auto-plan"
    ]) == EXIT_OK
    captured = capfd.readouterr()
    topic = base / course.name
    dropped = [
        row for row in _dropped_rows(topic)
        if row.get("reason") == "quote_not_found_at_gate"
    ]
    assert len(dropped) == 1
    assert dropped[0]["remaining_verbatim_evidence"] == 1
    claim = next(
        row for row in llmwiki.read_claims(topic, zone="staging")
        if row["id"] == dropped[0]["claim_id"]
    )
    # The current append-only llmwiki cannot delete evidence: the fallback takes the whole
    # claim off the gate through a tombstone instead of writing claims.jsonl directly.
    assert claim["status"] == "retracted" and len(claim["evidence"]) == 2
    receipt = json.loads((topic / "work" / "import-receipt.json").read_text("utf-8"))
    assert receipt["quote_gate_repair"] == {
        "healed_quotes": 1, "retracted_claims": 1,
    }
    assert receipt["quote_gate_summary"] == (
        "gate: healed 1 quotes, retracted 1 claims"
    )
    assert receipt["quote_gate_summary"] in captured.err


def test_quote_gate_retracts_claim_with_only_bad_evidence_and_finishes_zero(
    import_env, capfd, monkeypatch
):
    course, base, llmwiki = import_env
    monkeypatch.setattr(
        llmwiki, "add_claims",
        _add_claims_with_bad_first_evidence(llmwiki, keep_second=False),
    )

    assert main([
        "import", str(course), "--base", str(base), "--auto-plan"
    ]) == EXIT_OK
    capfd.readouterr()
    topic = base / course.name
    dropped = [
        row for row in _dropped_rows(topic)
        if row.get("reason") == "quote_not_found_at_gate"
    ]
    assert len(dropped) == 1
    assert dropped[0]["remaining_verbatim_evidence"] == 0
    claim = next(
        row for row in llmwiki.read_claims(topic, zone="staging")
        if row["id"] == dropped[0]["claim_id"]
    )
    assert claim["status"] == "retracted"
    receipt = json.loads((topic / "work" / "import-receipt.json").read_text("utf-8"))
    assert receipt["quote_gate_repair"] == {
        "healed_quotes": 1, "retracted_claims": 1,
    }


def test_import_structural_answer_retries_same_extract_role_then_succeeds(
    import_env, capfd, monkeypatch
):
    course, base, _ = import_env
    _ExtractAdapter.scripted_texts = ["not json"]
    sleeps = []
    monkeypatch.setattr("researcher.orchestrator.time.sleep", sleeps.append)

    assert main([
        "import", str(course), "--base", str(base), "--auto-plan", "--json"
    ]) == EXIT_OK
    topic = base / course.name
    capfd.readouterr()

    receipt = json.loads((topic / "work" / "import-receipt.json").read_text("utf-8"))
    assert receipt["role_calls"]["extract"] == 3
    assert checkpoint.load(topic)["structural_failures"] == 1
    assert len(list((topic / "work" / "bad-answers").glob("import-extract-*.txt"))) == 1
    assert sleeps == [5.0]
    assert "Return EXACTLY one JSON object" in _ExtractAdapter.prompts[1]


def test_import_structural_answer_exhaustion_is_fatal_and_keeps_all_evidence(
    import_env, capfd, monkeypatch
):
    course, base, _ = import_env
    _ExtractAdapter.scripted_texts = ["bad one", "bad two", "bad three"]
    sleeps = []
    monkeypatch.setattr("researcher.orchestrator.time.sleep", sleeps.append)

    assert main([
        "import", str(course), "--base", str(base), "--auto-plan", "--json"
    ]) == EXIT_FAIL
    topic = base / course.name
    captured = capfd.readouterr()

    assert "Traceback" not in captured.err
    assert checkpoint.load(topic)["phase"] == "import:extract"
    evidence = sorted((topic / "work" / "bad-answers").glob("import-extract-*.txt"))
    assert len(evidence) == 3
    assert [path.read_text("utf-8").splitlines()[-1] for path in evidence] == [
        "bad one", "bad two", "bad three"
    ]
    assert sleeps == [5.0, 10.0]
    assert all(
        "Return EXACTLY one JSON object" in prompt
        for prompt in _ExtractAdapter.prompts[1:]
    )


def test_initialize_import_has_one_atomic_creator(import_env, monkeypatch):
    course, base, llmwiki = import_env
    options = course_import.ImportRunOptions(
        dump=course, base=base, slug=course.name,
        extract_adapter="claude", verify_adapter="codex",
        auto_plan=True, link_dump=False,
        extract_config=RunConfig(), verify_config=RunConfig(),
    )
    real_init = llmwiki.init_topic
    calls = 0
    calls_lock = threading.Lock()
    first_entered = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()

    def slow_init(*args, **kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
            current = calls
        first_entered.set()
        if current == 2:
            second_entered.set()
        assert release.wait(2)
        return real_init(*args, **kwargs)

    monkeypatch.setattr(llmwiki, "init_topic", slow_init)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(course_import.initialize_import, llmwiki, options)
        assert first_entered.wait(2)
        second = pool.submit(course_import.initialize_import, llmwiki, options)
        raced = second_entered.wait(0.3)
        release.set()
        results = [future.result() for future in (first, second)]

    assert raced is False
    assert calls == 1
    assert results == [base / course.name, base / course.name]


def test_parallel_bad_answers_keep_distinct_attempts_and_counter(tmp_path):
    topic = tmp_path / "topic"
    barrier = threading.Barrier(2)

    class RacingState(dict):
        def get(self, key, default=None):
            value = super().get(key, default)
            if key == "structural_failures":
                try:
                    barrier.wait(0.2)
                except threading.BrokenBarrierError:
                    pass
            return value

    state = RacingState({
        "phase": "import:extract", "run_mode": "import", "topic": "course",
        "queue": [], "done": [], "sessions": {}, "structural_failures": 0,
    })
    checkpoint.save(topic, state)

    with ThreadPoolExecutor(max_workers=2) as pool:
        diagnostics = list(pool.map(
            lambda text: course_import._bad_answer(
                topic, state, "extract", text, ValueError(text)
            ),
            ["first bad answer", "second bad answer"],
        ))

    evidence = sorted((topic / "work" / "bad-answers").glob("import-extract-*.txt"))
    assert state["structural_failures"] == 2
    assert [path.name for path in evidence] == [
        "import-extract-attempt-1.txt", "import-extract-attempt-2.txt",
    ]
    assert {path.read_text("utf-8").splitlines()[-1] for path in evidence} == {
        "first bad answer", "second bad answer",
    }
    assert len(diagnostics) == 2


def test_verify_replay_reports_duplicate_after_write_checkpoint_gap(
    import_env, capfd, monkeypatch
):
    course, base, llmwiki = import_env
    original = course_import._atomic_json
    tripped = False

    def interrupt_applied(path, value):
        nonlocal tripped
        if (
            not tripped and path.parent.name == "verified"
            and isinstance(value, dict) and value.get("applied") is True
        ):
            tripped = True
            raise OSError("synthetic hard stop after llmwiki batch")
        return original(path, value)

    monkeypatch.setattr(course_import, "_atomic_json", interrupt_applied)
    assert main([
        "import", str(course), "--base", str(base), "--auto-plan", "--json"
    ]) == EXIT_FAIL
    capfd.readouterr()
    topic = base / course.name
    assert len(llmwiki.read_claims(topic, zone="staging")) == 1

    monkeypatch.setattr(course_import, "_atomic_json", original)
    assert main(["resume", str(topic), "--json"]) == EXIT_OK
    capfd.readouterr()
    receipt = json.loads((topic / "work" / "import-receipt.json").read_text("utf-8"))
    assert receipt["duplicates"] == 1
    assert len(llmwiki.read_claims(topic, zone="final")) == 2


def _dropped_rows(topic: Path) -> list[dict]:
    path = topic / "work" / "dropped-claims.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def test_replay_conflict_in_the_same_window_is_fatal_with_evidence_on_disk(
    import_env, capfd, monkeypatch
):
    """A fatal refusal of this class must also leave evidence, not a single word."""
    course, base, llmwiki = import_env
    original = course_import._atomic_json
    tripped = False

    def interrupt_applied(path, value):
        nonlocal tripped
        if (
            not tripped and path.parent.name == "verified"
            and isinstance(value, dict) and value.get("applied") is True
        ):
            tripped = True
            raise OSError("synthetic hard stop after llmwiki batch")
        return original(path, value)

    monkeypatch.setattr(course_import, "_atomic_json", interrupt_applied)
    assert main([
        "import", str(course), "--base", str(base), "--auto-plan"
    ]) == EXIT_FAIL
    capfd.readouterr()
    topic = base / course.name
    monkeypatch.setattr(course_import, "_atomic_json", original)

    # The recorded knowledge diverged from what resume will rewrite, in THE SAME window.
    claims_path = topic / "staging" / "claims.jsonl"
    rows = [json.loads(line) for line in claims_path.read_text("utf-8").splitlines()]
    assert len(rows) == 1
    window_id = rows[0]["meta"]["source_window"]
    rows[0]["confidence"] = "low"
    claims_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), "utf-8"
    )

    assert main(["resume", str(topic)]) == EXIT_FAIL
    captured = capfd.readouterr()
    assert "claim_replay_conflict" in captured.err and window_id in captured.err
    assert "diverged confidence" in captured.err
    dropped = [
        row for row in _dropped_rows(topic)
        if row["reason"] == "claim_replay_conflict"
    ]
    assert len(dropped) == 1
    assert dropped[0]["diverged"] == ["confidence"]
    assert dropped[0]["incoming"]["source_window"] == window_id
    assert dropped[0]["existing"]["source_window"] == window_id


def test_same_fact_in_two_lessons_is_a_duplicate_with_evidence_on_disk(
    import_env, capfd
):
    """Incident 26.08: mt-saascoursemt and bc-deepgocourse died at verification because the
    lecturer repeated the same statement in another lesson (claim id = hash(text)).
    The course must reach the end, and the evidence must land on disk with both windows."""
    course, base, llmwiki = import_env
    repeated = "Source verification must precede the conclusion."
    (course / "text" / "002 - Slides.pdf.md").write_text(
        f"# Slides\n\n{repeated}\n", "utf-8"
    )

    # Without --json: it silences stderr, and the milestone evidence goes exactly there.
    assert main([
        "import", str(course), "--base", str(base), "--auto-plan"
    ]) == EXIT_OK
    captured = capfd.readouterr()
    topic = base / course.name
    receipt = json.loads((topic / "work" / "import-receipt.json").read_text("utf-8"))
    assert (receipt["verified"], receipt["duplicates"]) == (1, 1)

    rows = [
        row for row in _dropped_rows(topic)
        if row["reason"] == "cross_document_duplicate"
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["nnn"] == "002" and row["text"] == repeated
    assert row["zone"] == "staging" and row["merge"] == "merged"
    assert sorted(row["diverged"]) == ["evidence", "meta"]
    assert row["incoming"]["source_window"] != row["existing"]["source_window"]
    assert row["incoming"]["source_ids"] != row["existing"]["source_ids"]
    assert f"verification: 002 claim {row['claim_id']}" in captured.err

    # The fact from the second lesson is not lost: the evidence grew onto the recorded claim.
    final = llmwiki.read_claims(topic, zone="final")
    assert len(final) == 1
    assert sorted(item["source_id"] for item in final[0]["evidence"]) == sorted(
        row["incoming"]["source_ids"] + row["existing"]["source_ids"]
    )


def test_same_fact_twice_in_one_lesson_drops_the_repeat_instead_of_failing(
    import_env, capfd
):
    """The same class inside one lesson (formerly the fatal extract_candidate_conflict):
    the first occurrence stays, the repeat becomes evidence, the course does not fail."""
    course, base, _ = import_env
    repeated = "The evidence window must be bounded."
    long_course = course.parent / "long-course"
    (long_course / "text").mkdir(parents=True)
    body = "# Long lesson\n\n" + "".join(
        f"{repeated}\n\n" + ("word " * 120).strip() + "\n\n" for _ in range(30)
    )
    assert len(body) > 12_000
    (long_course / "text" / "001 - Long.mp4.md").write_text(body, "utf-8")
    (long_course / "manifest.jsonl").write_text(
        json.dumps(_manifest_row("001 - Long.mp4.md"), ensure_ascii=False) + "\n",
        "utf-8",
    )
    (long_course / "source.json").write_text(
        json.dumps({"source": "private input", "title": "Long"}), "utf-8"
    )

    assert main([
        "import", str(long_course), "--base", str(base), "--auto-plan"
    ]) == EXIT_OK
    captured = capfd.readouterr()
    topic = base / long_course.name
    extracted = json.loads(
        (topic / "work" / "extracted" / "001.json").read_text("utf-8")
    )
    assert [claim["text"] for claim in extracted["claims"]] == [repeated]
    rows = [
        row for row in _dropped_rows(topic)
        if row["reason"] == "duplicate_text_in_document"
    ]
    assert rows and all(row["nnn"] == "001" for row in rows)
    assert all(row["kept_window"] != row["dropped_window"] for row in rows)
    assert "statement repeated from window" in captured.err


def test_import_rejects_same_harness_and_changed_dump(import_env, capfd):
    course, base, _ = import_env
    assert main([
        "import", str(course), "--base", str(base), "--adapter", "claude",
        "--verify-adapter", "claude", "--json",
    ]) == EXIT_FAIL
    capfd.readouterr()
    assert not (base / course.name).exists()

    assert main(["import", str(course), "--base", str(base), "--json"]) == EXIT_PLAN_REVIEW
    capfd.readouterr()
    topic = base / course.name
    _approve(topic)
    document = course / "text" / "001 - Intro.mp4.md"
    document.write_text(document.read_text("utf-8") + "modified\n", "utf-8")
    assert main(["resume", str(topic), "--json"]) == EXIT_FAIL
    capfd.readouterr()
    assert checkpoint.load(topic)["phase"] == "import:plan"


def test_import_resume_rejects_same_adapter_pair_before_role_calls(import_env, capfd):
    course, base, _ = import_env
    topic = _start_waiting(course, base, capfd)

    assert main([
        "resume", str(topic), "--adapter", "claude",
        "--verify-adapter", "claude",
    ]) == EXIT_FAIL
    captured = capfd.readouterr()

    assert "verification_adapter_not_independent" in captured.err
    assert "Traceback" not in captured.err
    assert _ExtractAdapter.calls == [] and _VerifyAdapter.calls == 0


def test_import_resume_explicitly_allows_same_adapter_reviewer(
    import_env, capfd, monkeypatch
):
    course, base, _ = import_env
    topic = _start_waiting(course, base, capfd)
    _approve(topic)
    extract_run = _ExtractAdapter.run

    def dual_role_run(self, prompt, **kwargs):
        if prompt.startswith("You are an independent critic."):
            return _VerifyAdapter().run(prompt, **kwargs)
        return extract_run(self, prompt, **kwargs)

    monkeypatch.setattr(_ExtractAdapter, "run", dual_role_run)
    assert main([
        "resume", str(topic), "--adapter", "claude",
        "--verify-adapter", "claude", "--allow-same-adapter-verify", "--json",
    ]) == EXIT_OK
    capfd.readouterr()

    state = checkpoint.load(topic)
    receipt = json.loads(
        (topic / "work" / "import-receipt.json").read_text("utf-8")
    )
    assert state["adapters"] == {"extract": "claude", "verify": "claude"}
    assert receipt["verification_independent"] is False
    assert _VerifyAdapter.calls == 2


def test_verify_phase_has_its_own_same_adapter_guard(import_env, capfd):
    course, base, llmwiki = import_env
    assert main([
        "import", str(course), "--base", str(base), "--auto-plan", "--json"
    ]) == EXIT_OK
    topic = base / course.name
    capfd.readouterr()
    state = checkpoint.load(topic)
    state["phase"] = "import:verify"
    state["adapters"] = {"extract": "codex", "verify": "codex"}
    for path in (topic / "work" / "verified").glob("*.json"):
        path.unlink()
    for path in (topic / "work" / "extracted").glob("*.json"):
        payload = json.loads(path.read_text("utf-8"))
        payload["adapter"] = "codex"
        path.write_text(json.dumps(payload), "utf-8")
    runner = course_import.CourseImportRunner(
        topic,
        wiki=llmwiki,
        extract_adapter=_VerifyAdapter(),
        verify_adapter=_VerifyAdapter(),
        extract_config=RunConfig(retry_backoff=0),
        verify_config=RunConfig(retry_backoff=0),
    )
    _ExtractAdapter.calls = []
    _VerifyAdapter.calls = 0

    with pytest.raises(ValueError, match="verification_adapter_not_independent"):
        runner._verify(state)
    assert _ExtractAdapter.calls == [] and _VerifyAdapter.calls == 0


def test_import_verify_rejects_extracted_artifact_from_verify_adapter(
    import_env, capfd
):
    course, base, _ = import_env
    assert main([
        "import", str(course), "--base", str(base), "--auto-plan", "--json"
    ]) == EXIT_OK
    topic = base / course.name
    capfd.readouterr()
    state = checkpoint.load(topic)
    state["phase"] = "import:verify"
    checkpoint.save(topic, state)
    for path in (topic / "work" / "verified").glob("*.json"):
        path.unlink()
    extracted = topic / "work" / "extracted" / "001.json"
    payload = json.loads(extracted.read_text("utf-8"))
    payload["adapter"] = "codex"
    extracted.write_text(json.dumps(payload), "utf-8")

    assert main(["resume", str(topic)]) == EXIT_FAIL
    captured = capfd.readouterr()
    assert "extracted_artifact_adapter_conflict" in captured.err
    assert "Traceback" not in captured.err


def test_import_cannot_change_extract_adapter_after_durable_artifact(import_env, capfd):
    course, base, _ = import_env
    _ExtractAdapter.quota_on_call = 2
    assert main([
        "import", str(course), "--base", str(base), "--auto-plan", "--json"
    ]) == EXIT_QUOTA
    topic = base / course.name
    capfd.readouterr()
    calls_before = list(_ExtractAdapter.calls)

    assert main([
        "resume", str(topic), "--adapter", "codex",
        "--verify-adapter", "claude",
    ]) == EXIT_FAIL
    captured = capfd.readouterr()

    assert "import_extract_adapter_change_after_extract" in captured.err
    assert "Traceback" not in captured.err
    assert _ExtractAdapter.calls == calls_before


def test_done_import_rechecks_dump_and_rejects_changed_corpus(import_env, capfd):
    course, base, _ = import_env
    assert main([
        "import", str(course), "--base", str(base), "--auto-plan", "--json"
    ]) == EXIT_OK
    topic = base / course.name
    capfd.readouterr()
    document = course / "text" / "001 - Intro.mp4.md"
    document.write_text(document.read_text("utf-8") + "changed after done\n", "utf-8")

    assert main(["resume", str(topic)]) == EXIT_FAIL
    captured = capfd.readouterr()

    assert "import_corpus_changed_after_done" in captured.err
    assert "extend" in captured.err or "rebuild" in captured.err
    assert "Traceback" not in captured.err
    assert checkpoint.load(topic)["phase"] == "import:done"


def test_movement_calls_young_process_starting(monkeypatch):
    old = {"phase": "import:extract", "updated_at": "2000-01-01T00:00:00+00:00"}
    assert cli._movement(
        old, stale_minutes=90, pid=123, process_age_minutes=1.5,
        heartbeat={"role": "collect", "call_started": None},
        activity_age_minutes=120,
    ) == "starting"
    assert cli._movement(
        old, stale_minutes=90, pid=123, process_age_minutes=120,
        heartbeat={"role": "collect", "call_started": None},
        activity_age_minutes=120,
    ) == "STALE"


def test_common_topics_root_defaults_and_explicit_base_wins(tmp_path, capfd):
    if str(_LLMWIKI) not in sys.path:
        sys.path.insert(0, str(_LLMWIKI))
    configured = tmp_path / "configured-topics"
    explicit = tmp_path / "explicit-topics"
    data_root = Path.home() / "tools-data"
    data_root.mkdir()
    (data_root / "config.toml").write_text(
        "schema_version = 1\n\n[paths]\n"
        f'vault = "{tmp_path / "vault"}"\n'
        f'topics_root = "{configured}"\n'
        f'sources_root = "{tmp_path / "sources"}"\n',
        "utf-8",
    )
    assert main(["start", "configured topic", "--no-run", "--json"]) == EXIT_OK
    configured_result = json.loads(capfd.readouterr().out)
    assert Path(configured_result["topic_dir"]).parent == configured
    assert main(["status", "--base", "--json"]) == EXIT_OK
    assert json.loads(capfd.readouterr().out)["base"] == str(configured)

    assert main([
        "start", "explicit topic", "--base", str(explicit), "--no-run", "--json"
    ]) == EXIT_OK
    explicit_result = json.loads(capfd.readouterr().out)
    assert Path(explicit_result["topic_dir"]).parent == explicit


@pytest.mark.parametrize(
    "argv",
    [
        ["import", "dump", "--json"],
        ["start", "topic", "--no-run", "--json"],
        ["status", "--base", "--json"],
    ],
)
def test_bad_common_config_is_base_config_failed_machine_envelope(
    tmp_path, monkeypatch, capfd, argv
):
    data = tmp_path / "data"
    data.mkdir()
    (data / "config.toml").write_text("not valid = [\n", "utf-8")
    monkeypatch.setenv("TOOLS_DATA", str(data))

    assert main(argv) == EXIT_FAIL
    captured = capfd.readouterr()
    payload = json.loads(captured.out)
    assert payload["error"] == {"code": "base_config_failed"}
    assert "base_config_failed" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "argv",
    [
        ["import", "dump"],
        ["start", "topic", "--no-run"],
        ["status", "--base"],
    ],
)
def test_bad_common_config_is_human_exit_1_without_traceback(
    tmp_path, monkeypatch, capfd, argv
):
    data = tmp_path / "data"
    data.mkdir()
    (data / "config.toml").write_text("not valid = [\n", "utf-8")
    monkeypatch.setenv("TOOLS_DATA", str(data))

    assert main(argv) == EXIT_FAIL
    captured = capfd.readouterr()
    assert "base_config_failed" in captured.err
    assert "Traceback" not in captured.err


def test_import_extract_and_verify_have_separate_model_and_effort_flags(
    import_env, capfd
):
    course, base, _ = import_env

    assert main([
        "import", str(course), "--base", str(base), "--auto-plan", "--json",
        "--model", "extract-model", "--effort", "high",
        "--verify-model", "verify-model", "--verify-effort", "low",
    ]) == EXIT_OK
    capfd.readouterr()

    assert {row["model"] for row in _ExtractAdapter.run_kwargs} == {"extract-model"}
    assert {row["effort"] for row in _ExtractAdapter.run_kwargs} == {"high"}
    assert {row["model"] for row in _VerifyAdapter.run_kwargs} == {"verify-model"}
    assert {row["effort"] for row in _VerifyAdapter.run_kwargs} == {"low"}


def test_import_extract_model_does_not_leak_into_verify_adapter(import_env, capfd):
    course, base, _ = import_env

    assert main([
        "import", str(course), "--base", str(base), "--auto-plan", "--json",
        "--model", "claude-extract-only", "--effort", "high",
    ]) == EXIT_OK
    capfd.readouterr()

    assert {row["model"] for row in _ExtractAdapter.run_kwargs} == {
        "claude-extract-only"
    }
    assert all(row["model"] != "claude-extract-only" for row in _VerifyAdapter.run_kwargs)
    assert all(row["effort"] is None for row in _VerifyAdapter.run_kwargs)


@pytest.mark.parametrize("failure", [RuntimeError("adapter crashed"), KeyError("field")])
def test_import_json_catches_unexpected_exception_without_traceback(
    import_env, capfd, monkeypatch, failure
):
    course, base, _ = import_env

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(course_import, "initialize_import", fail)

    assert main(["import", str(course), "--base", str(base), "--json"]) == EXIT_FAIL
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"] == {"code": "import_failed"}
    assert captured.err.strip() == "research import: import_failed"
    assert "Traceback" not in captured.err


def test_import_human_mode_normalizes_key_error(import_env, capfd, monkeypatch):
    course, base, _ = import_env

    def fail(*args, **kwargs):
        raise KeyError("field")

    monkeypatch.setattr(course_import, "initialize_import", fail)

    assert main(["import", str(course), "--base", str(base)]) == EXIT_FAIL
    captured = capfd.readouterr()
    assert "import interrupted" in captured.err
    assert "Traceback" not in captured.err


def test_supervisor_import_progress_uses_durable_artifacts_not_only_claims(
    tmp_path, monkeypatch
):
    topic = tmp_path / "topic"
    checkpoint.save(
        topic,
        {
            "phase": "import:extract",
            "run_mode": "import",
            "topic": "course",
            "done": [],
        },
    )
    calls = 0
    sleeps = []

    def resume_with_extract_progress(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= 2:
            path = topic / "work" / "extracted" / f"{calls:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", "utf-8")
            return EXIT_FAIL
        return EXIT_OK

    monkeypatch.setattr(cli, "_resume_once", resume_with_extract_progress)
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)

    class Args:
        transient_retries = 0
        fatal_retries = 3

    assert cli._supervise_resume(topic, Args(), machine=False) == EXIT_OK
    assert calls == 3
    assert sleeps == [120, 120]
    events = [
        json.loads(line)
        for line in (topic / "work" / "events.jsonl").read_text("utf-8").splitlines()
        if json.loads(line)["event"] == "supervisor_attempt"
    ]
    assert [row["progress_after"][-1] for row in events] == [1, 2, 2]
    assert all(row["progress_after"][:2] == ["import:extract", 0] for row in events)


def test_resume_supervisor_normalizes_key_error(tmp_path, monkeypatch, capfd):
    topic = tmp_path / "topic"
    checkpoint.save(
        topic, {"phase": "collecting", "topic": "t", "queue": [], "done": []}
    )
    monkeypatch.setattr(
        cli, "_supervise_resume", lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("x"))
    )

    assert main(["resume", str(topic), "--supervise", "--json"]) == EXIT_FAIL
    captured = capfd.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "error" and payload["phase"] == "collecting"
    assert "supervisor_failed" in captured.err
    assert "Traceback" not in captured.err


def test_import_prints_a_milestone_per_lesson(import_env, capfd):
    # The scheduler watchdog calls a run hung when the job log stops moving, and extract
    # and verify of a course take hours: every lesson must move the log.
    course, base, _ = import_env
    assert main(["import", str(course), "--base", str(base), "--auto-plan"]) == EXIT_OK
    err = capfd.readouterr().err
    # The course slug in the milestone: in the job log of two workers of one queue, lines
    # without it are indistinguishable.
    assert "extraction synthetic-course: 1/2 (50%)" in err
    assert "extraction synthetic-course: 2/2 (100%)" in err
    assert "verification synthetic-course: 1/2 (50%)" in err
    assert "verification synthetic-course: 2/2 (100%)" in err
    assert "min left" in err


def test_milestone_moves_the_log_but_sends_no_notice(capfd):
    # Canon 25.08: course progress is collected by the status snapshot, the milestone is
    # only a log line.
    class _Notifier:
        def __init__(self):
            self.events = []

        def emit(self, event, text, **kwargs):
            self.events.append((event, kwargs.get("title")))
            return True

    runner = object.__new__(course_import.CourseImportRunner)
    runner.notifier = _Notifier()
    runner.topic = Path("/nowhere/bc-gccourse")
    started = __import__("time").monotonic() - 120
    for done in (1, 2, 3):
        runner._milestone("extraction", done, 10, started=started, processed=done)
    assert capfd.readouterr().err.count("extraction bc-gccourse:") == 3
    assert runner.notifier.events == []
