"""P4-RS-01: bounded local/coursedump sources and runtime machine contract."""

from __future__ import annotations

import errno
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from researcher import checkpoint, cli
from researcher.cli import EXIT_FAIL, EXIT_OK, EXIT_PLAN_REVIEW, main
from researcher.sources import (
    ImportPlan,
    ImportReceipt,
    SourceWindow,
    SourceImportError,
    get_import_tool,
    import_verified_corpus,
    iter_source_windows,
    plan_import,
)
from researcher.sources.imports import MAX_DOCUMENT_BYTES, MAX_WINDOW_CHARS


_LLMWIKI = Path(__file__).resolve().parents[2] / "tool-llm-wiki"


def _manifest_row(
    target: str,
    *,
    rel: str | None = None,
    kind: str = "video",
    skip: str = "",
) -> dict[str, object]:
    return {
        "rel": rel or target.removesuffix(".md"),
        "kind": kind,
        "size": 100,
        "remote": "",
        "skip": skip,
        "target": target,
        "index": 1,
        "vid": "stable-id",
        "title": "Lesson",
    }


def _course(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic-course"
    text = root / "text"
    text.mkdir(parents=True)
    (text / "001 - Intro.mp4.md").write_text(
        "# Intro\n\nSource verification must precede the conclusion.\n", "utf-8"
    )
    (text / "002 - Slides.pdf.md").write_text(
        "# Slides\n\nThe evidence window must be bounded.\n", "utf-8"
    )
    (text / "unrelated.md").write_text("not part of the manifest", "utf-8")
    rows = [
        _manifest_row("001 - Intro.mp4.md"),
        _manifest_row("002 - Slides.pdf.md", kind="pdf"),
        _manifest_row("", rel="cover.png", kind="image", skip="image"),
    ]
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), "utf-8"
    )
    (root / "source.json").write_text(
        json.dumps({"source": "private input", "title": "Synthetic"}), "utf-8"
    )
    return root


def _exact_text_roles(text: str):
    class Extract:
        role = "extract"

        def __call__(self, window):
            local = window["text"].index(text)
            start = window["start_offset"] + local
            return [
                {
                    "text": text,
                    "confidence": "high",
                    "evidence": [
                        {
                            "window_id": window["window_id"],
                            "quote": text,
                            "start_offset": start,
                            "end_offset": start + len(text),
                        }
                    ],
                    "tags": [],
                }
            ]

    class Verify:
        role = "verify"

        def __call__(self, candidate, window):
            return {"accepted": True, "reason": "exact quote"}

    return Extract(), Verify()


def test_local_plan_is_deterministic_bounded_and_requires_exact_approval(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "B.md").write_text("second", "utf-8")
    long_text = "# First\n\n" + "evidence paragraph\n" * 900
    (corpus / "a.md").write_text(long_text, "utf-8")

    first = plan_import(corpus)
    second = plan_import(corpus)

    assert first == second
    assert first.schema_version == 1
    assert first.status == "waiting_human"
    assert [row.relative_path for row in first.documents] == ["a.md", "B.md"]
    assert first.documents[0].window_count >= 2
    assert str(corpus) not in json.dumps(first.to_dict())

    with pytest.raises(SourceImportError, match="plan_not_approved"):
        iter_source_windows(corpus, plan=first, approval="plan_wrong")

    windows = iter_source_windows(corpus, plan=first, approval=first.plan_id)
    assert all(0 < len(window.text) <= MAX_WINDOW_CHARS for window in windows)
    assert [window.relative_path for window in windows] == sorted(
        (window.relative_path for window in windows), key=str.casefold
    )
    assert all(window.sha256 for window in windows)

    (corpus / "a.md").write_text(long_text + "a change\n", "utf-8")
    with pytest.raises(SourceImportError, match="source_plan_conflict"):
        iter_source_windows(corpus, plan=first, approval=first.plan_id)


def test_coursedump_plan_uses_manifest_identity_and_preserves_unrelated_files(tmp_path):
    course = _course(tmp_path)
    before = {
        path.relative_to(course): path.read_bytes()
        for path in course.rglob("*")
        if path.is_file()
    }

    plan = plan_import(course, source_kind="coursedump-manifest")
    windows = iter_source_windows(course, plan=plan, approval=plan.plan_id)

    assert plan.source_kind == "coursedump-manifest"
    assert [row.relative_path for row in plan.documents] == [
        "001 - Intro.mp4.md",
        "002 - Slides.pdf.md",
    ]
    assert {window.relative_path for window in windows} == {
        "001 - Intro.mp4.md",
        "002 - Slides.pdf.md",
    }
    after = {
        path.relative_to(course): path.read_bytes()
        for path in course.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_coursedump_manifest_v1_defaults_are_accepted(tmp_path):
    course = _course(tmp_path)
    rows = []
    for line in (course / "manifest.jsonl").read_text("utf-8").splitlines():
        row = json.loads(line)
        for field in ("index", "vid", "title"):
            row.pop(field)
        rows.append(row)
    (course / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        "utf-8",
    )

    plan = plan_import(course, source_kind="coursedump-manifest")

    assert [row.relative_path for row in plan.documents] == [
        "001 - Intro.mp4.md",
        "002 - Slides.pdf.md",
    ]


def test_coursedump_skips_large_and_non_transcript_documents_with_reasons(tmp_path):
    course = tmp_path / "course"
    text = course / "text"
    text.mkdir(parents=True)
    (text / "lesson.mp4.md").write_text("A useful transcript.\n", "utf-8")
    (text / "application.html.md").write_text("<html>app</html>\n", "utf-8")
    (text / "huge.mp4.md").write_bytes(b"x" * (MAX_DOCUMENT_BYTES + 1))
    rows = [
        _manifest_row("lesson.mp4.md"),
        _manifest_row("application.html.md", kind="html"),
        _manifest_row("huge.mp4.md"),
    ]
    (course / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), "utf-8"
    )

    plan = plan_import(course, source_kind="coursedump-manifest")
    windows = iter_source_windows(course, plan=plan, approval=plan.plan_id)

    assert [row.relative_path for row in plan.documents] == ["lesson.mp4.md"]
    assert [row.to_dict() for row in plan.skipped] == [
        {
            "relative_path": "application.html.md",
            "title": "application",
            "reason": "non_transcript",
            "size_bytes": 17,
        },
        {
            "relative_path": "huge.mp4.md",
            "title": "huge",
            "reason": "document_too_large",
            "size_bytes": MAX_DOCUMENT_BYTES + 1,
        },
    ]
    assert [window.relative_path for window in windows] == ["lesson.mp4.md"]


def test_local_markdown_corpus_skips_one_large_document_instead_of_failing(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "small.md").write_text("bounded\n", "utf-8")
    (corpus / "large.md").write_bytes(b"x" * (MAX_DOCUMENT_BYTES + 1))

    plan = plan_import(corpus, source_kind="local-markdown")

    assert [row.relative_path for row in plan.documents] == ["small.md"]
    assert [(row.relative_path, row.reason) for row in plan.skipped] == [
        ("large.md", "document_too_large")
    ]


def test_coursedump_document_error_names_relative_path(tmp_path):
    course = _course(tmp_path)
    missing = "nested/missing.mp4.md"
    (course / "manifest.jsonl").write_text(
        json.dumps(_manifest_row(missing)) + "\n", "utf-8"
    )

    with pytest.raises(SourceImportError) as caught:
        plan_import(course, source_kind="coursedump-manifest")

    assert caught.value.code == "source_document_missing"
    assert caught.value.relative_path == missing
    assert missing in str(caught.value)


def test_coursedump_accepts_apfs_name_over_255_utf8_bytes(tmp_path):
    """Incident iz-captainbuilders: a 151-character APFS name took 266 UTF-8 bytes.

    The name below deliberately stays in Cyrillic: a pure ASCII name cannot have more UTF-8
    bytes than characters, so the case under test would disappear.
    """
    course = _course(tmp_path)
    target = (
        "074 Почему не стоит встраивать монетизацию на старте, если в продукте "
        "нет издержек. И что делать с монетизацией, когда в продукте есть "
        "издержки.html.md"
    )
    assert len(target) < 255 and len(target.encode("utf-8")) > 255
    path = course / "text" / target
    try:
        path.write_text("# Lesson\n\nA long-name check.\n", "utf-8")
    except OSError as error:
        if error.errno == errno.ENAMETOOLONG:
            pytest.skip("the filesystem does not accept an APFS-compatible name")
        raise
    (course / "manifest.jsonl").write_text(
        json.dumps(_manifest_row(target), ensure_ascii=False) + "\n", "utf-8"
    )

    plan = plan_import(course, source_kind="coursedump-manifest")

    assert [row.relative_path for row in plan.documents] == [target]


def test_coursedump_durable_plan_can_exceed_machine_envelope(tmp_path):
    """iz-captainbuilders: 249 rows produced an honest plan of about 102 KB."""
    course = tmp_path / "large-plan"
    text_dir = course / "text"
    text_dir.mkdir(parents=True)
    rows = []
    for index in range(180):
        target = f"{index:03d}-" + ("long-course-title-" * 10) + ".md"
        (text_dir / target).write_text(f"# Lesson {index}\n\nA check.\n", "utf-8")
        rows.append(_manifest_row(target))
    (course / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), "utf-8"
    )

    plan = plan_import(course, source_kind="coursedump-manifest")
    rendered = json.dumps(plan.to_dict(), ensure_ascii=False).encode("utf-8")

    assert len(plan.documents) == 180
    assert len(rendered) > cli.MAX_MACHINE_OUTPUT_BYTES


@pytest.mark.skipif(
    not (_LLMWIKI / "llmwiki").exists(), reason="no sibling llmwiki for the contract test"
)
def test_synthetic_course_cross_role_import_validates_through_real_llmwiki(tmp_path):
    sys.path.insert(0, str(_LLMWIKI))
    import llmwiki

    course = _course(tmp_path)
    topic = llmwiki.init_topic(tmp_path / "topics", "synthetic course pilot")
    plan = plan_import(course, source_kind="coursedump-manifest")
    calls: list[tuple[str, str]] = []

    class Extract:
        role = "extract"

        def __call__(self, window):
            calls.append((self.role, window["window_id"]))
            needle = "Source" if "Source" in window["text"] else "The evidence"
            local = window["text"].index(needle)
            end = window["text"].find(".", local) + 1
            quote = window["text"][local:end]
            start_offset = window["start_offset"] + local
            return [
                {
                    "text": quote,
                    "confidence": "high",
                    "evidence": [
                        {
                            "window_id": window["window_id"],
                            "quote": quote,
                            "start_offset": start_offset,
                            "end_offset": start_offset + len(quote),
                        }
                    ],
                    "tags": ["source-contract"],
                }
            ]

    class Verify:
        role = "verify"

        def __call__(self, candidate, window):
            calls.append((self.role, window["window_id"]))
            assert candidate["evidence"][0]["quote"] in window["text"]
            return {"accepted": True, "reason": "quote and claim agree"}

    receipt = import_verified_corpus(
        topic,
        course,
        plan=plan,
        approval=plan.plan_id,
        extractor=Extract(),
        verifier=Verify(),
        wiki=llmwiki,
    )

    assert receipt.to_dict() == {
        "schema_version": 1,
        "plan_id": plan.plan_id,
        "status": "imported",
        "documents": 2,
        "windows": 2,
        "candidates": 2,
        "verified": 2,
        "duplicates": 0,
        "role_calls": {"extract": 2, "verify": 2},
        "usage": {},
        "phase_seconds": {},
        "total_seconds": 0.0,
    }
    assert [role for role, _ in calls] == ["extract", "verify", "extract", "verify"]
    assert llmwiki.validate_topic(topic) == []
    assert list((topic / "sources/content").iterdir()) == []
    source_rows = [
        json.loads(line)
        for line in (topic / "sources/sources.jsonl").read_text("utf-8").splitlines()
    ]
    assert len(source_rows) == 2
    assert all(
        "content_path" not in row and "private input" not in json.dumps(row)
        for row in source_rows
    )
    assert all(row["meta"]["relative_path"].endswith(".md") for row in source_rows)
    claim_rows = [
        json.loads(line)
        for line in (topic / "staging/claims.jsonl").read_text("utf-8").splitlines()
    ]
    assert all(row["confidence"] == "medium" for row in claim_rows)
    assert all(row["meta"]["single_source"] is True for row in claim_rows)

    replay = import_verified_corpus(
        topic,
        course,
        plan=plan,
        approval=plan.plan_id,
        extractor=Extract(),
        verifier=Verify(),
        wiki=llmwiki,
    )
    assert replay.verified == 0 and replay.duplicates == 2
    assert llmwiki.validate_topic(topic) == []
    assert len((topic / "sources/sources.jsonl").read_text("utf-8").splitlines()) == 2


def test_mismatched_evidence_and_same_role_are_rejected_before_claim_write(tmp_path):
    course = _course(tmp_path)
    plan = plan_import(course, source_kind="coursedump-manifest")

    class Wiki:
        class DuplicateClaim(Exception):
            pass

        def validate_topic(self, topic):
            return []

        def add_source(self, *args, **kwargs):
            return "src_000000000001"

        def add_claim(self, *args, **kwargs):
            pytest.fail("invalid evidence reached llmwiki")

    class BadExtract:
        role = "extract"

        def __call__(self, window):
            return [
                {
                    "text": "unsupported",
                    "confidence": "high",
                    "evidence": [
                        {
                            "window_id": window["window_id"],
                            "quote": "not in source",
                            "start_offset": window["start_offset"],
                            "end_offset": window["start_offset"] + 13,
                        }
                    ],
                    "tags": [],
                }
            ]

    class Verify:
        role = "verify"

        def __call__(self, candidate, window):
            return {"accepted": True, "reason": "wrong"}

    with pytest.raises(SourceImportError, match="evidence_quote_mismatch"):
        import_verified_corpus(
            tmp_path / "topic",
            course,
            plan=plan,
            approval=plan.plan_id,
            extractor=BadExtract(),
            verifier=Verify(),
            wiki=Wiki(),
        )

    Verify.role = "extract"
    with pytest.raises(SourceImportError, match="verification_role_not_independent"):
        import_verified_corpus(
            tmp_path / "topic",
            course,
            plan=plan,
            approval=plan.plan_id,
            extractor=BadExtract(),
            verifier=Verify(),
            wiki=Wiki(),
        )


@pytest.mark.parametrize(
    "mutation,code",
    [
        (lambda row: {**row, "target": "../escape.md"}, "invalid_source_relative_path"),
        (lambda row: {**row, "size": True}, "invalid_source_manifest"),
        (lambda row: {**row, "debug": "private"}, "invalid_source_manifest"),
        (lambda row: [], "invalid_source_manifest"),
    ],
)
def test_malformed_coursedump_manifest_is_normalized(tmp_path, mutation, code):
    course = _course(tmp_path)
    row = mutation(_manifest_row("001 - Intro.mp4.md"))
    (course / "manifest.jsonl").write_text(json.dumps(row) + "\n", "utf-8")

    with pytest.raises(SourceImportError, match=code):
        plan_import(course, source_kind="coursedump-manifest")


def test_public_plan_parser_rejects_bool_counts_and_malformed_container(tmp_path):
    plan = plan_import(_course(tmp_path), source_kind="coursedump-manifest")
    payload = plan.to_dict()
    payload["documents"][0]["size_bytes"] = True
    with pytest.raises(SourceImportError, match="invalid_import_plan"):
        ImportPlan.from_dict(payload)
    with pytest.raises(SourceImportError, match="invalid_import_plan"):
        ImportPlan.from_dict([])
    payload = plan.to_dict()
    payload["source_kind"] = []
    with pytest.raises(SourceImportError, match="invalid_import_plan"):
        ImportPlan.from_dict(payload)
    with pytest.raises(SourceImportError, match="invalid_source_kind"):
        get_import_tool([])


def test_import_tool_wrappers_expose_versioned_plan_gate(tmp_path):
    course = _course(tmp_path)
    tool = get_import_tool("coursedump-manifest")
    plan = tool.plan(str(course))
    assert tool.contract_version == 1
    assert tool.source_kind == "coursedump-manifest"
    with pytest.raises(SourceImportError, match="plan_not_approved"):
        tool.windows(str(course), plan=plan, approval="")
    assert len(tool.windows(str(course), plan=plan, approval=plan.plan_id)) == 2


def test_public_window_and_receipt_normalize_malformed_fields(tmp_path):
    course = _course(tmp_path)
    plan = plan_import(course, source_kind="coursedump-manifest")
    window = iter_source_windows(course, plan=plan, approval=plan.plan_id)[0]
    malformed = SourceWindow(**{**window.__dict__, "start_offset": True})
    with pytest.raises(SourceImportError, match="invalid_source_window"):
        malformed.to_dict()
    receipt = ImportReceipt(1, [], "imported", 1, 1, 1, 1, 0)
    with pytest.raises(SourceImportError, match="invalid_import_receipt"):
        receipt.to_dict()
    receipt = ImportReceipt(2, plan.plan_id, "imported", 1, 1, 0, 1, 0)
    with pytest.raises(SourceImportError, match="invalid_import_receipt"):
        receipt.to_dict()


def test_local_and_coursedump_import_reject_intermediate_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("private source", "utf-8")

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SourceImportError, match="source_path_not_regular"):
        plan_import(corpus)

    course = _course(tmp_path)
    (course / "text" / "linked").symlink_to(outside, target_is_directory=True)
    row = _manifest_row("linked/secret.md")
    (course / "manifest.jsonl").write_text(json.dumps(row) + "\n", "utf-8")
    with pytest.raises(SourceImportError, match="source_document_missing"):
        plan_import(course, source_kind="coursedump-manifest")


def test_repeated_failed_reads_close_descriptors(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "bad.md").write_bytes(b"\xff")
    before = len(os.listdir("/dev/fd"))
    for _ in range(64):
        with pytest.raises(SourceImportError, match="source_document_not_utf8"):
            plan_import(corpus)
    assert len(os.listdir("/dev/fd")) == before


def test_directory_scan_stops_reading_at_total_byte_bound(tmp_path, monkeypatch):
    from researcher.sources import imports as imports_module

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("a.md", "b.md", "c.md"):
        (corpus / name).write_text("abc", "utf-8")
    real_read = imports_module._read_open_file
    reads = 0

    def counted_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        return real_read(*args, **kwargs)

    monkeypatch.setattr(imports_module, "MAX_TOTAL_BYTES", 5)
    monkeypatch.setattr(imports_module, "_read_open_file", counted_read)

    with pytest.raises(SourceImportError, match="source_corpus_too_large"):
        plan_import(corpus)
    assert reads == 2


def test_directory_iterator_stops_at_budget_without_reading_one_more(
    tmp_path, monkeypatch
):
    from researcher.sources import imports as imports_module

    corpus = tmp_path / "corpus"
    corpus.mkdir()

    class Entry:
        def __init__(self, name):
            self.name = name

    class EndlessDirectory:
        def __init__(self):
            self.reads = 0
            self.closed = False

        def __next__(self):
            self.reads += 1
            if self.reads > 3:
                pytest.fail("directory iterator consumed one entry past its budget")
            return Entry(f"{self.reads}.md")

        def close(self):
            self.closed = True

    entries = EndlessDirectory()
    monkeypatch.setattr(imports_module, "MAX_SCAN_ENTRIES", 3)
    monkeypatch.setattr(imports_module.os, "scandir", lambda directory_fd: entries)

    with pytest.raises(SourceImportError, match="too_many_source_entries"):
        plan_import(corpus)
    assert entries.reads == 3
    assert entries.closed is True


def test_directory_iterator_oserror_is_normalized_and_closed(tmp_path, monkeypatch):
    from researcher.sources import imports as imports_module

    corpus = tmp_path / "corpus"
    corpus.mkdir()

    class BrokenDirectory:
        def __init__(self):
            self.closed = False

        def __next__(self):
            raise OSError("private iterator detail")

        def close(self):
            self.closed = True

    entries = BrokenDirectory()
    monkeypatch.setattr(imports_module.os, "scandir", lambda directory_fd: entries)

    with pytest.raises(SourceImportError, match="source_read_failed"):
        plan_import(corpus)
    assert entries.closed is True


def test_capabilities_exactly_match_runtime_p0_probe(capfd):
    assert main(["capabilities", "--json"]) == EXIT_OK
    captured = capfd.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    assert payload["tool"] == "researcher"
    assert payload["version"] == "0.1.0"
    assert [item["name"] for item in payload["capabilities"]] == [
        "start",
        "import",
        "status",
        "status-base",
        "verify-status",
        "resume",
        "extend",
        "doctor",
    ]
    assert payload["capabilities"][0] == {
        "name": "start",
        "argv": ["research", "start", "<topic>", "--base", "<topics-root>", "--json"],
        "machine_output": {
            "format": "json",
            "schema_version": 1,
            "required_fields": ["schema_version", "status", "topic_dir"],
        },
        "exit_codes": {
            "0": "success",
            "1": "failure",
            "75": "quota",
            "77": "gate_refused",
            "111": "transient",
            "76": "waiting_human",
        },
        "idempotency": "deduplicated",
    }
    for name in ("resume", "extend"):
        capability = next(
            item for item in payload["capabilities"] if item["name"] == name
        )
        assert capability["exit_codes"]["76"] == "waiting_human"
        assert capability["exit_codes"]["77"] == "gate_refused"
    status_base = next(
        item for item in payload["capabilities"] if item["name"] == "status-base"
    )
    assert status_base["argv"][-1] == "--json"
    verify = next(
        item for item in payload["capabilities"] if item["name"] == "verify-status"
    )
    assert verify["argv"][-1] == "--json"
    assert verify["idempotency"] == "idempotent"


def test_source_plan_cli_is_one_bounded_json_and_stops_at_review_gate(tmp_path, capfd):
    course = _course(tmp_path)
    assert (
        main(
            [
                "source-plan",
                str(course),
                "--kind",
                "coursedump-manifest",
                "--json",
            ]
        )
        == EXIT_PLAN_REVIEW
    )
    captured = capfd.readouterr()
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 1
    assert len(captured.out.encode()) < 64 * 1024
    payload = json.loads(captured.out)
    assert payload["status"] == "waiting_human"
    assert str(course) not in captured.out


def test_machine_status_and_start_are_versioned_without_human_stdout(
    tmp_path, monkeypatch, capfd
):
    root = tmp_path / "topic"
    checkpoint.save(
        root,
        {"phase": "planned", "topic": "t", "queue": [], "done": [], "sessions": {}},
    )
    assert main(["status", str(root)]) == EXIT_OK
    status_output = capfd.readouterr()
    status_payload = json.loads(status_output.out)
    assert status_payload["schema_version"] == 1
    assert status_payload["status"] == "running"
    assert status_payload["phase"] == "planned"

    fake_wiki = type("Wiki", (), {})()
    monkeypatch.setattr(cli, "_llmwiki", lambda **kwargs: fake_wiki)
    monkeypatch.setattr(cli, "_start_topic", lambda *args, **kwargs: (root, None))
    monkeypatch.setattr(cli, "_run", lambda *args, **kwargs: EXIT_PLAN_REVIEW)
    assert main(["start", "t", "--base", str(tmp_path), "--json"]) == EXIT_PLAN_REVIEW
    start_output = capfd.readouterr()
    assert len(start_output.out.splitlines()) == 1
    start_payload = json.loads(start_output.out)
    assert start_payload == {
        "schema_version": 1,
        "status": "waiting_human",
        "topic_dir": str(root),
        "phase": "planned",
    }
    assert "DONE" not in start_output.out


def test_machine_doctor_is_one_path_free_json(tmp_path, monkeypatch, capfd):
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/private/bin/{name}")
    module = type("Wiki", (), {"__file__": str(cli.WIKI_REPO / "llmwiki/__init__.py")})
    monkeypatch.setitem(sys.modules, "llmwiki", module)

    assert main(["doctor", "--json"]) == EXIT_OK
    captured = capfd.readouterr()
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 1
    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["schema_version"] == 1
    assert str(tmp_path) not in captured.out
    assert "/private/bin" not in captured.out


def test_machine_start_without_llmwiki_is_one_generic_json(
    tmp_path, monkeypatch, capfd
):
    monkeypatch.setitem(sys.modules, "llmwiki", None)

    assert (
        main(["start", "private topic", "--base", str(tmp_path), "--json"]) == EXIT_FAIL
    )
    captured = capfd.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["error"] == {"code": "llmwiki_unavailable"}
    assert "private topic" not in captured.out + captured.err
    assert str(tmp_path) not in captured.out + captured.err
    assert "Traceback" not in captured.err


def test_machine_resume_does_not_reflect_invalid_source_name(
    tmp_path, monkeypatch, capfd
):
    root = tmp_path / "topic"
    checkpoint.save(
        root,
        {"phase": "planned", "topic": "t", "queue": [], "done": [], "sessions": {}},
    )
    secret = "private-source-name"
    assert main(["resume", str(root), "--sources", secret, "--json"]) == EXIT_FAIL
    captured = capfd.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["status"] == "error"
    assert secret not in captured.out + captured.err
    assert "Traceback" not in captured.err


def test_machine_status_rejects_large_or_non_object_checkpoint(tmp_path, capfd):
    root = tmp_path / "topic"
    path = root / "work/checkpoint.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{" + b"x" * (cli.MAX_MACHINE_STATE_BYTES + 1))
    assert main(["status", str(root)]) == EXIT_FAIL
    captured = capfd.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["error"] == {"code": "invalid_checkpoint"}
    assert "Traceback" not in captured.err

    path.write_text("[]", "utf-8")
    assert main(["status", str(root)]) == EXIT_FAIL
    captured = capfd.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["error"] == {"code": "invalid_checkpoint"}


def test_machine_invalid_exit_is_normalized_to_one_json(tmp_path, monkeypatch, capfd):
    root = tmp_path / "topic"
    checkpoint.save(
        root,
        {"phase": "planned", "topic": "t", "queue": [], "done": [], "sessions": {}},
    )
    monkeypatch.setattr(cli, "_llmwiki", lambda **kwargs: object())
    monkeypatch.setattr(cli, "_start_topic", lambda *args, **kwargs: (root, None))
    monkeypatch.setattr(cli, "_run", lambda *args, **kwargs: 99)

    assert main(["start", "t", "--base", str(tmp_path), "--json"]) == EXIT_FAIL
    captured = capfd.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["error"] == {"code": "invalid_exit_code"}
    assert "Traceback" not in captured.err


def test_machine_gate_refused_keeps_exit_77_and_typed_status(
    tmp_path, monkeypatch, capfd
):
    root = tmp_path / "topic"
    checkpoint.save(
        root,
        {"phase": "synthesizing", "topic": "t", "queue": [], "done": []},
    )
    monkeypatch.setattr(cli, "_run", lambda *args, **kwargs: cli.EXIT_GATE_REFUSED)

    assert main(["resume", str(root), "--json"]) == cli.EXIT_GATE_REFUSED
    captured = capfd.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out) == {
        "phase": "synthesizing",
        "schema_version": 1,
        "status": "gate_refused",
        "topic_dir": str(root),
    }
    assert "invalid_exit_code" not in captured.err


def test_source_plan_cli_normalizes_manifest_error_without_path(tmp_path, capfd):
    course = _course(tmp_path)
    (course / "manifest.jsonl").write_text("[]\n", "utf-8")
    assert (
        main(
            [
                "source-plan",
                str(course),
                "--kind",
                "coursedump-manifest",
                "--json",
            ]
        )
        == EXIT_FAIL
    )
    captured = capfd.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["error"] == {"code": "invalid_source_manifest"}
    assert str(course) not in captured.out + captured.err
    assert "Traceback" not in captured.err


@pytest.mark.skipif(
    not (_LLMWIKI / "llmwiki").exists(), reason="no sibling llmwiki for the contract test"
)
@pytest.mark.parametrize("source_kind", ["local-markdown", "coursedump-manifest"])
def test_approved_snapshot_is_loaded_once_and_exact_bytes_reach_roles(
    tmp_path, monkeypatch, source_kind
):
    sys.path.insert(0, str(_LLMWIKI))
    import llmwiki
    from researcher.sources import imports as imports_module

    if source_kind == "local-markdown":
        source = tmp_path / "local"
        source.mkdir()
        selected = source / "lesson.md"
        selected.write_text("# Stable\n\nApproved local evidence.\n", "utf-8")
        expected_documents = 1
    else:
        source = _course(tmp_path)
        selected = source / "text/001 - Intro.mp4.md"
        expected_documents = 2
    plan = plan_import(source, source_kind=source_kind)
    topic = llmwiki.init_topic(tmp_path / "topics", f"snapshot pilot {source_kind}")
    real_load = imports_module._load_documents
    calls = 0

    def load_once(*args, **kwargs):
        nonlocal calls
        documents = real_load(*args, **kwargs)
        calls += 1
        selected.write_text("CHANGED AFTER FD SNAPSHOT\n", "utf-8")
        return documents

    monkeypatch.setattr(imports_module, "_load_documents", load_once)
    observed: list[str] = []

    class Extract:
        role = "extract"

        def __call__(self, window):
            observed.append(window["text"])
            return []

    class Verify:
        role = "verify"

        def __call__(self, candidate, window):
            pytest.fail("empty extraction reached verifier")

    receipt = import_verified_corpus(
        topic,
        source,
        plan=plan,
        approval=plan.plan_id,
        extractor=Extract(),
        verifier=Verify(),
        wiki=llmwiki,
    )

    assert calls == 1
    assert receipt.documents == expected_documents
    assert observed and all("CHANGED AFTER" not in text for text in observed)
    assert llmwiki.validate_topic(topic) == []


def test_concurrent_in_place_source_change_fails_hash_snapshot_recheck(
    tmp_path, monkeypatch
):
    from researcher.sources import imports as imports_module

    source = tmp_path / "source.md"
    source.write_text("approved content\n", "utf-8")
    plan = plan_import(source)
    real_read = imports_module.os.read
    changed = False

    def racing_read(fd, maximum):
        nonlocal changed
        chunk = real_read(fd, maximum)
        if chunk and not changed:
            changed = True
            source.write_text("changed during the held-fd read\n", "utf-8")
        return chunk

    monkeypatch.setattr(imports_module.os, "read", racing_read)
    with pytest.raises(SourceImportError, match="source_changed_while_read"):
        iter_source_windows(source, plan=plan, approval=plan.plan_id)


@pytest.mark.parametrize(
    "manifest",
    [
        '{"rel":"a","kind":"video","size":1,"size":2,"remote":"",'
        '"skip":"","target":"a.md","index":1,"vid":"v","title":"t"}\n',
        json.dumps(_manifest_row("a.md") | {"size": 2**63}) + "\n",
        json.dumps(_manifest_row("a.md") | {"index": 2**31}) + "\n",
    ],
)
def test_manifest_duplicate_keys_and_unbounded_integers_are_rejected(
    tmp_path, manifest
):
    course = _course(tmp_path)
    (course / "manifest.jsonl").write_text(manifest, "utf-8")
    with pytest.raises(SourceImportError, match="invalid_source_manifest"):
        plan_import(course, source_kind="coursedump-manifest")


@pytest.mark.parametrize("forbidden", ["\u202e", "\u2028", "\u2029"])
def test_unicode_format_and_line_separator_are_rejected_in_paths(tmp_path, forbidden):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / f"bad{forbidden}.md").write_text("ordinary", "utf-8")
    with pytest.raises(SourceImportError):
        plan_import(corpus)


def test_unicode_line_separators_are_normalized_before_control_check(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "body.md").write_text("before\u2028middle\u2029after", "utf-8")

    plan = plan_import(corpus)
    windows = iter_source_windows(corpus, plan=plan, approval=plan.plan_id)

    assert windows[0].text == "before\nmiddle\nafter"


def test_unicode_format_control_is_still_rejected_in_document_text(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "body.md").write_text("before\u202eafter", "utf-8")

    with pytest.raises(SourceImportError, match="source_document_has_control"):
        plan_import(corpus)


def test_validate_plan_recomputes_corpus_hash_on_direct_call(tmp_path):
    from researcher.sources import imports as imports_module

    plan = plan_import(_course(tmp_path), source_kind="coursedump-manifest")
    mutated = replace(plan, corpus_sha256="0" * 64)

    with pytest.raises(SourceImportError, match="invalid_import_plan"):
        imports_module._validate_plan(mutated)


def test_machine_json_suppresses_plan_map_and_setup_stdout(
    tmp_path, monkeypatch, capfd
):
    from researcher import orchestrator

    root = tmp_path / "topic"
    checkpoint.save(
        root,
        {"phase": "planned", "topic": "t", "queue": [], "done": [], "sessions": {}},
    )
    private = "/Users/private/plan-map.md"

    class NoisyAdapter:
        def __init__(self):
            print(private)

    class NoisyOrchestrator:
        def __init__(self, *args, **kwargs):
            print(private)

        def run(self):
            print(private)
            return EXIT_PLAN_REVIEW

    def noisy_wiki(**kwargs):
        print(private)
        return object()

    monkeypatch.setitem(cli.ADAPTERS, "claude", NoisyAdapter)
    monkeypatch.setattr(orchestrator, "Orchestrator", NoisyOrchestrator)
    monkeypatch.setattr(cli, "_llmwiki", noisy_wiki)
    monkeypatch.setattr(
        orchestrator,
        "begin_extend",
        lambda topic_dir: print(private) or ("extend-id", True),
    )

    for argv in (
        ["resume", str(root), "--json"],
        ["extend", str(root), "--json"],
    ):
        assert main(argv) == EXIT_PLAN_REVIEW
        captured = capfd.readouterr()
        assert len(captured.out.splitlines()) == 1
        assert json.loads(captured.out)["status"] == "waiting_human"
        assert private not in captured.out + captured.err


def test_machine_setup_failures_are_generic_for_llmwiki_and_adapter(
    tmp_path, monkeypatch, capfd
):
    private = "/Users/private/model-secret"

    def broken_wiki(**kwargs):
        print(private)
        raise OSError(private)

    monkeypatch.setattr(cli, "_llmwiki", broken_wiki)
    assert main(["start", "private", "--base", str(tmp_path), "--json"]) == EXIT_FAIL
    captured = capfd.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["error"] == {"code": "setup_failed"}
    assert private not in captured.out + captured.err
    assert "Traceback" not in captured.err

    root = tmp_path / "topic"
    checkpoint.save(
        root,
        {"phase": "planned", "topic": "t", "queue": [], "done": [], "sessions": {}},
    )
    monkeypatch.setattr(cli, "_llmwiki", lambda **kwargs: object())

    class BrokenAdapter:
        def __init__(self):
            print(private)
            raise RuntimeError(private)

    monkeypatch.setitem(cli.ADAPTERS, "claude", BrokenAdapter)
    assert main(["resume", str(root), "--json"]) == EXIT_FAIL
    captured = capfd.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["status"] == "error"
    assert private not in captured.out + captured.err
    assert "Traceback" not in captured.err


def test_machine_doctor_and_source_plan_setup_failures_are_one_json(
    tmp_path, monkeypatch, capfd
):
    from researcher import sources as sources_module

    private = "/Users/private/setup-detail"

    def broken(*args, **kwargs):
        print(private)
        raise OSError(private)

    monkeypatch.setattr(cli, "_doctor_report", broken)
    assert main(["doctor", "--json"]) == EXIT_FAIL
    captured = capfd.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["error"] == {"code": "setup_failed"}
    assert private not in captured.out + captured.err

    monkeypatch.setattr(sources_module, "plan_import", broken)
    assert main(["source-plan", str(tmp_path), "--json"]) == EXIT_FAIL
    captured = capfd.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["error"] == {"code": "setup_failed"}
    assert private not in captured.out + captured.err
    assert "Traceback" not in captured.err


def test_machine_checkpoint_duplicate_key_is_one_generic_json(tmp_path, capfd):
    root = tmp_path / "topic"
    path = root / "work/checkpoint.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"phase":"planned","phase":"done"}\n', "utf-8")

    assert main(["status", str(root)]) == EXIT_FAIL
    captured = capfd.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["error"] == {"code": "invalid_checkpoint"}
    assert "Traceback" not in captured.err


@pytest.mark.skipif(
    not (_LLMWIKI / "llmwiki").exists(), reason="no sibling llmwiki for the contract test"
)
@pytest.mark.parametrize("zone", ["staging", "final"])
def test_same_text_from_another_origin_is_a_duplicate_not_a_conflict(tmp_path, zone):
    """A claim id is hash(text), so someone else's row with the same text is NOT a replay divergence.

    A semantic clarification after the 26.08 incident (mt-saascoursemt, bc-deepgocourse):
    this situation used to fail an entire course. Now it is a cross-document duplicate: the
    run continues, our evidence grows onto the already written claim through the socket, and
    an evidence note goes to on_conflict. What stays fatal is a divergence within the SAME
    window - that is held by the neighboring test.
    """
    sys.path.insert(0, str(_LLMWIKI))
    import llmwiki

    source = tmp_path / "source.md"
    source.write_text("Exact imported evidence.\n", "utf-8")
    plan = plan_import(source)
    topic = llmwiki.init_topic(tmp_path / "topics", "evidence binding conflict")
    foreign_source = llmwiki.add_source(
        topic,
        kind="local",
        title="Foreign",
        url="urn:foreign:evidence",
        tool="test",
    )
    text = "Exact imported evidence."
    llmwiki.add_claim(
        topic,
        zone,
        text=text,
        confidence="medium",
        status="verified",
        evidence=[
            {
                "source_id": foreign_source,
                "quote": "different quote",
                "stance": "supports",
            }
        ],
    )

    class Extract:
        role = "extract"

        def __call__(self, window):
            local = window["text"].index(text)
            start = window["start_offset"] + local
            return [
                {
                    "text": text,
                    "confidence": "high",
                    "evidence": [
                        {
                            "window_id": window["window_id"],
                            "quote": text,
                            "start_offset": start,
                            "end_offset": start + len(text),
                        }
                    ],
                    "tags": [],
                }
            ]

    class Verify:
        role = "verify"

        def __call__(self, candidate, window):
            return {"accepted": True, "reason": "exact quote"}

    reports: list[dict] = []
    receipt = import_verified_corpus(
        topic,
        source,
        plan=plan,
        approval=plan.plan_id,
        extractor=Extract(),
        verifier=Verify(),
        wiki=llmwiki,
        on_conflict=reports.append,
    )
    assert (receipt.verified, receipt.duplicates) == (0, 1)
    assert llmwiki.validate_topic(topic) == []
    rows = [
        json.loads(line)
        for line in (topic / zone / "claims.jsonl").read_text("utf-8").splitlines()
    ]
    assert len(rows) == 1
    # One row, the same id, evidence grew append-only: theirs plus ours.
    assert [item["source_id"] for item in rows[0]["evidence"]][0] == foreign_source
    assert len(rows[0]["evidence"]) == 2
    if zone == "final":
        assert not (topic / "staging/claims.jsonl").exists()

    assert len(reports) == 1
    report = reports[0]
    assert report["reason"] == "cross_document_duplicate"
    assert report["zone"] == zone and report["merge"] == "merged"
    assert report["claim_id"] == rows[0]["id"] and report["text"] == text
    assert report["existing"]["source_ids"] == [foreign_source]
    assert report["existing"]["source_window"] is None
    assert report["incoming"]["source_window"].startswith("win_")
    assert "evidence" in report["diverged"] and "meta" in report["diverged"]


@pytest.mark.skipif(
    not (_LLMWIKI / "llmwiki").exists(), reason="no sibling llmwiki for the contract test"
)
def test_same_window_divergence_is_still_a_fatal_replay_conflict(tmp_path):
    """The resume idempotency guard is intact: a divergence in the SAME window is fatal, with an evidence note."""
    sys.path.insert(0, str(_LLMWIKI))
    import llmwiki
    from researcher.sources import imports as imports_module

    text = "Exact imported evidence."
    source = tmp_path / "source.md"
    source.write_text(text + "\n", "utf-8")
    plan = plan_import(source)
    topic = llmwiki.init_topic(tmp_path / "topics", "same window divergence")
    extractor, verifier = _exact_text_roles(text)
    import_verified_corpus(
        topic, source, plan=plan, approval=plan.plan_id,
        extractor=extractor, verifier=verifier, wiki=llmwiki,
    )
    rows = [
        json.loads(line)
        for line in (topic / "staging/claims.jsonl").read_text("utf-8").splitlines()
    ]
    assert len(rows) == 1
    window_id = rows[0]["meta"]["source_window"]

    # The same run, the same window, but the recorded knowledge diverged from what we write.
    real_rows = imports_module._claim_rows

    def mutate(raw: bytes, claim_id: str):
        found = real_rows(raw, claim_id)
        for row in found:
            row["confidence"] = "low"
        return found

    imports_module._claim_rows = mutate
    reports: list[dict] = []
    try:
        with pytest.raises(SourceImportError, match="claim_replay_conflict") as failure:
            import_verified_corpus(
                topic, source, plan=plan, approval=plan.plan_id,
                extractor=extractor, verifier=verifier, wiki=llmwiki,
                on_conflict=reports.append,
            )
    finally:
        imports_module._claim_rows = real_rows
    # A single word in the log is no longer the only diagnostic: the code carries the detail.
    assert failure.value.code == "claim_replay_conflict"
    assert window_id in str(failure.value) and "confidence" in str(failure.value)
    assert len(reports) == 1 and reports[0]["reason"] == "claim_replay_conflict"
    assert reports[0]["diverged"] == ["confidence"]
    assert reports[0]["incoming"]["source_window"] == window_id
    assert reports[0]["existing"]["source_window"] == window_id


@pytest.mark.skipif(
    not (_LLMWIKI / "llmwiki").exists(), reason="no sibling llmwiki for the contract test"
)
@pytest.mark.parametrize("link_level", ["final", "claims"])
def test_final_symlink_cannot_supply_replay_or_hide_missing_staging_write(
    tmp_path, link_level
):
    sys.path.insert(0, str(_LLMWIKI))
    import llmwiki

    text = "Exact imported evidence."
    source = tmp_path / "source.md"
    source.write_text(text + "\n", "utf-8")
    plan = plan_import(source)
    topic = llmwiki.init_topic(tmp_path / "topics", "final symlink replay")
    extractor, verifier = _exact_text_roles(text)
    receipt = import_verified_corpus(
        topic,
        source,
        plan=plan,
        approval=plan.plan_id,
        extractor=extractor,
        verifier=verifier,
        wiki=llmwiki,
    )
    assert receipt.verified == 1
    claim_id = json.loads((topic / "staging/claims.jsonl").read_text("utf-8").strip())[
        "id"
    ]
    llmwiki.promote_claim(topic, claim_id)
    if link_level == "final":
        outside_claims = tmp_path / "outside-final" / "claims.jsonl"
        (topic / "final").rename(outside_claims.parent)
        (topic / "final").symlink_to(outside_claims.parent, target_is_directory=True)
    else:
        outside_claims = tmp_path / "outside-claims.jsonl"
        (topic / "final/claims.jsonl").rename(outside_claims)
        (topic / "final/claims.jsonl").symlink_to(outside_claims)
    assert llmwiki.validate_topic(topic) == []

    before_fds = len(os.listdir("/dev/fd"))
    for _ in range(32):
        with pytest.raises(SourceImportError, match="claim_replay_lookup_failed"):
            import_verified_corpus(
                topic,
                source,
                plan=plan,
                approval=plan.plan_id,
                extractor=extractor,
                verifier=verifier,
                wiki=llmwiki,
            )
    assert len(os.listdir("/dev/fd")) == before_fds
    assert (topic / "staging/claims.jsonl").read_text("utf-8") == ""
    assert len(outside_claims.read_text("utf-8").splitlines()) == 1


@pytest.mark.skipif(
    not (_LLMWIKI / "llmwiki").exists(), reason="no sibling llmwiki for the contract test"
)
def test_final_claim_change_after_read_is_detected_before_replay(tmp_path, monkeypatch):
    sys.path.insert(0, str(_LLMWIKI))
    import llmwiki
    from researcher.sources import imports as imports_module

    text = "Exact imported evidence."
    source = tmp_path / "source.md"
    source.write_text(text + "\n", "utf-8")
    plan = plan_import(source)
    topic = llmwiki.init_topic(tmp_path / "topics", "final mutation replay")
    extractor, verifier = _exact_text_roles(text)
    import_verified_corpus(
        topic,
        source,
        plan=plan,
        approval=plan.plan_id,
        extractor=extractor,
        verifier=verifier,
        wiki=llmwiki,
    )
    claim_id = json.loads((topic / "staging/claims.jsonl").read_text("utf-8").strip())[
        "id"
    ]
    llmwiki.promote_claim(topic, claim_id)
    real_assert = imports_module._assert_claim_zone_stable
    changed = False

    def change_after_snapshot(topic_fd, snapshot):
        nonlocal changed
        if snapshot.zone == "final" and not changed:
            changed = True
            (topic / "final/claims.jsonl").write_text("", "utf-8")
        return real_assert(topic_fd, snapshot)

    monkeypatch.setattr(
        imports_module, "_assert_claim_zone_stable", change_after_snapshot
    )
    before_fds = len(os.listdir("/dev/fd"))
    with pytest.raises(SourceImportError, match="claim_replay_lookup_failed"):
        import_verified_corpus(
            topic,
            source,
            plan=plan,
            approval=plan.plan_id,
            extractor=extractor,
            verifier=verifier,
            wiki=llmwiki,
        )
    assert changed is True
    assert len(os.listdir("/dev/fd")) == before_fds
    assert (topic / "staging/claims.jsonl").read_text("utf-8") == ""
    assert llmwiki.validate_topic(topic) == []


@pytest.mark.skipif(
    not (_LLMWIKI / "llmwiki").exists(), reason="no sibling llmwiki for the contract test"
)
def test_topic_parent_replacement_during_replay_lookup_is_rejected_and_closes_fds(
    tmp_path,
):
    sys.path.insert(0, str(_LLMWIKI))
    import llmwiki

    text = "Exact imported evidence."
    source = tmp_path / "source.md"
    source.write_text(text + "\n", "utf-8")
    plan = plan_import(source)
    topics = tmp_path / "topics"
    topic = llmwiki.init_topic(topics, "parent replacement replay")
    detached_parent = tmp_path / "detached-topics"
    extractor, verifier = _exact_text_roles(text)

    class ReplacingWiki:
        def __init__(self):
            self.replaced = False

        def __getattr__(self, name):
            return getattr(llmwiki, name)

        def make_claim_id(self, slug, claim_text):
            if not self.replaced:
                self.replaced = True
                topics.rename(detached_parent)
                topics.mkdir()
                llmwiki.init_topic(topics, "replacement topic", slug=topic.name)
            return llmwiki.make_claim_id(slug, claim_text)

    before_fds = len(os.listdir("/dev/fd"))
    with pytest.raises(SourceImportError, match="claim_replay_lookup_failed"):
        import_verified_corpus(
            topic,
            source,
            plan=plan,
            approval=plan.plan_id,
            extractor=extractor,
            verifier=verifier,
            wiki=ReplacingWiki(),
        )
    assert len(os.listdir("/dev/fd")) == before_fds
    detached_topic = detached_parent / topic.name
    assert not (detached_topic / "staging/claims.jsonl").exists()
    assert not (topic / "staging/claims.jsonl").exists()
    assert llmwiki.validate_topic(detached_topic) == []
    assert llmwiki.validate_topic(topic) == []


def test_evidence_quote_tolerates_case_and_whitespace_and_keeps_source_text():
    from researcher.sources.imports import SourceWindow, _normalize_evidence

    text = "первая строка.\n\nмы Взяли Всю Часть\nИ вычли. мы взяли всю часть опять"
    window = SourceWindow(
        schema_version=1, window_id="win_1", document_id="doc_1", relative_path="a.md",
        start_offset=100, end_offset=100 + len(text), start_line=1, end_line=4,
        text=text, sha256="0" * 64,
    )
    # The window text stays in Cyrillic on purpose: the production case is a multi-byte ASR
    # transcript, where a model reports character offsets wrongly (see _normalize_evidence).
    # Capitalized first letter + a collapsed line break: accepted, the quote is a substring of
    # the original; the span runs from the first to the last letter or digit, the trailing
    # period is not included.
    got = _normalize_evidence({"window_id": "win_1", "quote": "Мы Взяли Всю Часть И вычли."}, window)
    assert got["quote"] == "мы Взяли Всю Часть\nИ вычли"
    assert text[got["start_offset"] - 100:got["end_offset"] - 100] == got["quote"]
    # A repeat up to case: the start_offset hint picks the nearest one.
    far = _normalize_evidence(
        {"window_id": "win_1", "quote": "мы взяли всю часть", "start_offset": 100 + len(text) - 5},
        window,
    )
    assert far["quote"] == "мы взяли всю часть" and far["start_offset"] > got["start_offset"]
    # ASR transcript punctuation dropped by the model: accepted, the span follows the original.
    punct = _normalize_evidence({"window_id": "win_1", "quote": "мы Взяли Всю Часть И вычли"}, window)
    assert punct["quote"] == "мы Взяли Всю Часть\nИ вычли"
    # A paraphrase is still a refusal.
    with pytest.raises(SourceImportError, match="evidence_quote_mismatch"):
        _normalize_evidence({"window_id": "win_1", "quote": "мы взяли часть"}, window)
    with pytest.raises(SourceImportError, match="evidence_quote_mismatch"):
        _normalize_evidence({"window_id": "win_1", "quote": "   "}, window)


def test_evidence_quote_fuzzy_and_claim_drop_backstop():
    import hashlib

    from researcher.sources.imports import (
        SOURCE_IMPORT_SCHEMA_VERSION, SourceWindow, _normalize_evidence,
        normalize_extract_result,
    )

    text = ("Вот в go используется требуемое выравнивание. Его значение равно размеру "
            "памяти, требующемуся самому большему полю в структуре. Дальше идёт "
            "совсем другой разговор про сборщик мусора и стек.")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    document_id = "doc_" + "0" * 24
    window_id = "win_" + hashlib.sha256(
        f"{document_id}:0:{len(text)}:{sha}".encode("ascii")
    ).hexdigest()[:24]
    window = SourceWindow(
        schema_version=SOURCE_IMPORT_SCHEMA_VERSION, window_id=window_id,
        document_id=document_id, relative_path="a.md",
        start_offset=0, end_offset=len(text), start_line=1, end_line=1,
        text=text, sha256=sha,
    )
    # The transcript stays in Cyrillic on purpose: the fuzzy search is exercised on the real
    # production case, a Russian ASR transcript whose word forms the model "fixes" while
    # copying. Here it "fixed" a word (большему -> большому): the fuzzy search still finds the
    # span, and the evidence carries the window original with "большему".
    got = _normalize_evidence(
        {"window_id": window_id, "quote": "значение равно размеру памяти, требующемуся самому большому полю в структуре"},
        window,
    )
    assert "большему" in got["quote"] and "большому" not in got["quote"]
    assert text[got["start_offset"]:got["end_offset"]] == got["quote"]
    # A filler word dropped from the middle is survived as well.
    got2 = _normalize_evidence(
        {"window_id": window_id, "quote": "Вот в go используется выравнивание. Его значение равно размеру памяти"},
        window,
    )
    assert text[got2["start_offset"]:got2["end_offset"]] == got2["quote"]
    # A short quote (fewer than 5 words) gets no fuzziness.
    with pytest.raises(SourceImportError, match="evidence_quote_mismatch"):
        _normalize_evidence({"window_id": window_id, "quote": "сборщик хлама и стек"}, window)

    # Drop backstop: a claim with an unrecoverable quote goes to on_drop, the rest survive.
    claims = [
        {"text": "a live claim", "confidence": "high", "tags": [],
         "evidence": [{"window_id": window_id, "quote": "требуемое выравнивание"}]},
        {"text": "a paraphrased claim", "confidence": "high", "tags": [],
         "evidence": [{"window_id": window_id, "quote": "a completely invented quote that is nowhere in the window"}]},
    ]
    dropped = []
    kept = normalize_extract_result(claims, window, on_drop=dropped.append)
    assert [c["text"] for c in kept] == ["a live claim"]
    assert [c["text"] for c in dropped] == ["a paraphrased claim"]
    # Without on_drop it is a strict refusal (the register/payload boundary does not loosen).
    with pytest.raises(SourceImportError, match="evidence_quote_mismatch"):
        normalize_extract_result(claims, window)
    # ALL claims fell away - that is a garbage answer, a refusal even with on_drop.
    with pytest.raises(SourceImportError, match="evidence_quote_mismatch"):
        normalize_extract_result([claims[1]], window, on_drop=dropped.append)
