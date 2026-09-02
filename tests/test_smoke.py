"""CLI smoke: doctor runs and mutates nothing; usage errors = exit 1.

The researcher has no --home (the data lives in an external topic directory, decision
.agent/decisions.md), so doctor only inspects the environment: harnesses on PATH plus the
importability of llmwiki. The tests are hermetic: PATH and import are mocked, there is no
network and no live harness. Live harness runs do not belong here - they live in smoke/
(exit 77 = skip).
"""
import datetime
import errno
import fcntl
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest
from conftest import _env_sample_vars

from researcher import checkpoint, cli
from researcher.adapters.base import RunResult
from researcher.cli import EXIT_FAIL, EXIT_OK, main
from researcher.tool_config import ToolConfig


_VALID_PAGE_BODY = (
    "## What is known\n\nA coherent walkthrough.\n\n"
    "## Why it matters\n\nThe page claim explains the meaning of the fact.\n\n"
    "## What it is for\n\nPractical application.\n\n"
    "## Confidence and why\n\nThe source confirms the evidence.\n\n"
    "## Alternatives\n\nConsider another approach."
)


def _fake_env(monkeypatch, *, harness: bool, wiki: bool, wiki_file=None):
    monkeypatch.setattr("researcher.cli.shutil.which",
                        lambda name: f"/usr/local/bin/{name}" if harness else None)
    if wiki:
        mod = types.ModuleType("llmwiki")
        # We set __file__: doctor prints where llmwiki RESOLVES to, not just "present"
        mod.__file__ = str(wiki_file or (cli.WIKI_REPO / "llmwiki" / "__init__.py"))
        mod.FrontmatterError = ValueError
        mod.parse_page = lambda text: (
            {"claims": json.loads(text)["claims"]}, json.loads(text)["body"])
        monkeypatch.setitem(sys.modules, "llmwiki", mod)
    else:
        # None in sys.modules => import llmwiki raises ImportError even if the package is installed
        monkeypatch.setitem(sys.modules, "llmwiki", None)


def test_doctor_ready_and_mutates_nothing(tmp_path, monkeypatch, capsys):
    _fake_env(monkeypatch, harness=True, wiki=True)
    before = set(tmp_path.iterdir())  # conftest already chdir'ed into tmp_path
    assert main(["doctor"]) == EXIT_OK
    assert set(tmp_path.iterdir()) == before  # diagnostics, not mutation
    out = capsys.readouterr().out
    assert "READY" in out and "claude=available" in out
    assert "GITHUB_TOKEN" in out  # presence only, never the value


def test_doctor_prints_llmwiki_resolution(monkeypatch, capsys):
    """llmwiki from ../tool-llm-wiki: doctor prints the resolution PATH, with no warning.
    A bare 'present' masked a real case - a global editable install of an old repo."""
    expected = cli.WIKI_REPO / "llmwiki" / "__init__.py"
    _fake_env(monkeypatch, harness=True, wiki=True, wiki_file=expected)
    assert main(["doctor"]) == EXIT_OK
    out = capsys.readouterr().out
    assert str(expected) in out and "resolved:" in out
    assert "warning:" not in out


def test_doctor_warns_on_foreign_llmwiki(monkeypatch, capsys):
    """llmwiki resolves from a foreign (frozen) repo - the path is visible and a warning is
    printed, but this is NOT a failure: other legitimate resolutions exist."""
    foreign = cli.WIKI_REPO.parent / "llm-wiki" / "llmwiki" / "__init__.py"
    _fake_env(monkeypatch, harness=True, wiki=True, wiki_file=foreign)
    assert main(["doctor"]) == EXIT_OK  # a warning, not a problem
    out = capsys.readouterr().out
    assert str(foreign) in out
    assert "warning:" in out and "resolves outside" in out
    assert "READY" in out and "NOT READY" not in out


def test_doctor_not_ready_without_harness_and_wiki(monkeypatch, capsys):
    _fake_env(monkeypatch, harness=False, wiki=False)
    assert main(["doctor"]) == EXIT_FAIL
    out = capsys.readouterr().out
    assert "NOT READY" in out
    assert "no Claude or Codex harness" in out and "tool-llm-wiki" in out


# --- exit code contract: usage errors = 1, the contract knows no code "2" ---
# Regression guard for the 2026-08-05 migration: the old CLI returned 2 (llm-wiki missing,
# unknown source tool) - the workspace convention is 0/75/111/1 (.agent/decisions.md).

def test_status_without_checkpoint_exits_1(tmp_path, capsys):
    assert main(["status", str(tmp_path)]) == EXIT_FAIL
    assert "checkpoint missing" in capsys.readouterr().err


def test_status_derives_funnel_for_old_done_topic(tmp_path, capsys):
    """An old go-2026 topic lacks the new fields: status still shows its funnel."""
    root = tmp_path / "tema"
    checkpoint.save(root, {"phase": "done", "topic": "t", "queue": [], "done": [],
                           "cycles": 4})
    found = root / "work" / "found"
    found.mkdir()
    (found / "t1.jsonl").write_text("".join('{"text":"x"}\n' for _ in range(5)), "utf-8")
    for zone, count in (("staging", 1), ("final", 3)):
        path = root / zone / "claims.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            "".join(json.dumps({"id": f"{zone}-{i}"}) + "\n" for i in range(count)),
            "utf-8",
        )

    assert main(["status", str(root)]) == EXIT_OK

    shown = json.loads(capsys.readouterr().out)
    assert shown["funnel"] == {
        "found": 5,
        "staging": {"keep": 4, "drop": 1},
        "final": {"keep": 3, "drop": 1},
    }
    assert "queue" not in shown
    assert "rounds" not in shown


def test_status_raw_keeps_large_queue_and_rounds_without_output_cap(tmp_path, capfd):
    root = tmp_path / "tema"
    large = "x" * 1000
    checkpoint.save(
        root,
        {
            "phase": "collecting",
            "topic": "t",
            "queue": [
                {"id": f"task-{index}", "query": f"{index}-{large}"}
                for index in range(90)
            ],
            "done": [],
            "rounds": [{"cycle": 1, "note": large}],
        },
    )

    assert main(["status", str(root)]) == EXIT_OK
    summary = json.loads(capfd.readouterr().out)
    assert "queue" not in summary and "rounds" not in summary

    assert main(["status", str(root), "--raw"]) == EXIT_OK
    raw_output = capfd.readouterr().out
    raw = json.loads(raw_output)
    assert len(raw_output.encode("utf-8")) > cli.MAX_MACHINE_OUTPUT_BYTES
    assert len(raw["queue"]) == 90
    assert raw["rounds"] == [{"cycle": 1, "note": large}]


def test_status_reports_search_map_disk_count_after_post_approval_edit(tmp_path, capfd):
    root = tmp_path / "tema"
    checkpoint.save(
        root,
        {
            "phase": "collecting",
            "topic": "t",
            "queue": [],
            "done": [],
            "search_map_progress": {
                "cluster_ids": ["approved"],
                "completed_ids": [],
            },
        },
    )
    (root / "work" / "search-map.json").write_text(
        json.dumps(
            {
                "clusters": [
                    {"id": "approved", "query": "q1"},
                    {"id": "edited-later", "query": "q2"},
                ]
            }
        ),
        "utf-8",
    )

    assert main(["status", str(root)]) == EXIT_OK
    progress = json.loads(capfd.readouterr().out)["search_map_progress"]
    assert progress["total"] == 1
    assert progress["on_disk"] == 2
    assert progress["disk_mismatch"] is True


def test_status_short_has_current_role_call_claim_gain_pid_event_and_map(
    tmp_path, monkeypatch, capfd
):
    root = tmp_path / "tema"
    checkpoint.save(
        root,
        {
            "phase": "collecting",
            "topic": "the topic",
            "queue": [],
            "done": ["c1"],
            "cycles": 2,
            "max_cycles": 24,
            "rounds": [{"cycle": 1, "new_claims": 6}],
            "search_map_progress": {
                "cluster_ids": ["c1", "c2"],
                "completed_ids": ["c1"],
            },
            "judge": {
                "verdict": "continue",
                "cycle": 2,
                "why": "the price question is still open",
            },
        },
    )
    for relative, count in (
        ("staging/claims.jsonl", 3),
        ("final/claims.jsonl", 4),
        ("sources/sources.jsonl", 5),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" * count, "utf-8")
    events = root / "work" / "events.jsonl"
    events.write_text(
        json.dumps({"event": "old", "at": "2026-08-22T09:00:00+00:00"}) + "\n"
        + json.dumps(
            {"event": "collection_round", "at": "2026-08-22T10:00:00+00:00"}
        ) + "\n"
        + "{partial\n",
        "utf-8",
    )
    heartbeat = {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "phase": "collecting",
        "role": "collect",
        "call_started": (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)
        ).isoformat(timespec="seconds"),
        "pid": 4321,
        "harness_pid": 8765,
        "cycle": 2,
    }
    (root / "work" / "heartbeat.json").write_text(
        json.dumps(heartbeat, ensure_ascii=False), "utf-8",
    )
    monkeypatch.setattr(cli, "_live_lock_pid", lambda topic: 4321)
    monkeypatch.setattr(cli, "_process_age_minutes", lambda pid: 1.0)

    assert main(["status", str(root), "--short"]) == EXIT_OK
    lines = capfd.readouterr().out.splitlines()
    assert len(lines) == 5
    assert lines[0] == (
        "topic: tema | phase: collecting | cycle: 2/24 | staging: 3 | final: 4 | "
        "sources: 5 | judge: continue (2) - the price question is still open | "
        "open questions: - | stop: - | "
        "updated_at: "
        + checkpoint.load(root)["updated_at"]
        + " | state: running (role collect, call 5 min) | pid: 4321"
    )
    assert lines[1] == (
        "now: collecting, role collect, call 5 min, cycle 2/24, "
        "claims +6 in this run"
    )
    assert lines[2] == "last event: collection_round 2026-08-22T10:00:00+00:00"
    assert lines[3] == "map: 1/2 clusters"
    assert lines[4] == "judge: continue (2) - the price question is still open"


def test_status_short_keeps_import_phase_and_human_wait_state(tmp_path, capfd):
    root = tmp_path / "course"
    checkpoint.save(
        root,
        {
            "phase": "import:plan",
            "run_mode": "import",
            "topic": "the course",
            "done": [],
            "documents": [],
            "stop_kind": "waiting_human",
            "stop": {"reason": "import_plan_review"},
        },
    )

    assert main(["status", str(root), "--short"]) == EXIT_OK
    lines = capfd.readouterr().out.splitlines()
    assert "phase: import:plan" in lines[0]
    assert "stop: waiting_human/import_plan_review" in lines[0]
    assert "state: waiting for review" in lines[0]
    assert "role no heartbeat, call unknown" in lines[1]
    assert lines[3] == "map: 0/0 clusters"


def test_short_claim_gain_uses_first_attempt_of_current_supervisor_log(tmp_path):
    root = tmp_path / "tema"
    checkpoint.save(root, {
        "phase": "collecting", "topic": "the topic", "run_id": "run-1",
        "rounds": [{"new_claims": 99}],
    })
    for zone, ids in (("staging", ("a", "b", "c")), ("final", ("d", "e"))):
        path = root / zone / "claims.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps({"id": claim_id}) + "\n" for claim_id in ids),
            "utf-8",
        )
    events = root / "work" / "events.jsonl"
    events.write_text(
        "\n".join(json.dumps(row) for row in (
            {"event": "supervisor_attempt", "run_id": "run-1", "log": "old", "claims_before": 0},
            {"event": "supervisor_attempt", "run_id": "run-1", "log": "current", "claims_before": 3},
            {"event": "supervisor_attempt", "run_id": "run-1", "log": "current", "claims_before": 4},
        )) + "\n",
        "utf-8",
    )

    assert cli._claims_gained_this_run(root, checkpoint.load(root)) == 2


def test_status_base_json_has_no_64k_output_cap(tmp_path, monkeypatch, capfd):
    base = tmp_path / "topics"
    base.mkdir()
    monkeypatch.setattr(cli, "_llmwiki", lambda **kwargs: object())
    monkeypatch.setattr(
        cli,
        "_status_base_rows",
        lambda *args, **kwargs: [
            {"topic": f"topic-{index}", "phase": "done", "detail": "x" * 100}
            for index in range(700)
        ],
    )

    assert main(["status", "--base", str(base), "--json"]) == EXIT_OK
    output = capfd.readouterr().out
    assert len(output.encode("utf-8")) > cli.MAX_MACHINE_OUTPUT_BYTES
    assert len(json.loads(output)["topics"]) == 700


def test_status_base_json_covers_run_states_counts_and_no_checkpoint(
    tmp_path, monkeypatch, capfd
):
    base = tmp_path / "topics"
    base.mkdir()

    def make_topic(name, state=None):
        root = base / name
        root.mkdir()
        (root / "topic.json").write_text(
            json.dumps({"slug": name, "topic": name}), "utf-8"
        )
        if state is not None:
            checkpoint.save(root, {"topic": name, "queue": [], "done": [], **state})
        return root

    moving = make_topic(
        "moving", {"phase": "collecting", "cycles": 2, "max_cycles": 24}
    )
    (moving / "work" / "heartbeat.json").write_text(
        json.dumps({
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "phase": "collecting",
            "role": "collect",
            "call_started": None,
            "pid": 4321,
            "harness_pid": None,
            "cycle": 2,
        }),
        "utf-8",
    )
    waiting = make_topic(
        "waiting",
        {"phase": "planned", "plan_gate": {"status": "waiting"}},
    )
    refused = make_topic(
        "refused",
        {
            "phase": "synthesizing",
            "cycles": 24,
            "stop_kind": "gate_refused",
            "stop_detail": {"max_cycles": 24, "reasons": ["compiler-gap"]},
        },
    )
    make_topic("done", {"phase": "done", "cycles": 4, "max_cycles": 24})
    stale = make_topic("stale", {"phase": "collecting", "cycles": 1, "max_cycles": 24})
    stale_checkpoint = stale / "work" / "checkpoint.json"
    stale_state = json.loads(stale_checkpoint.read_text("utf-8"))
    stale_state["updated_at"] = "2000-01-01T00:00:00+00:00"
    stale_checkpoint.write_text(json.dumps(stale_state), "utf-8")
    make_topic("legacy")

    _fake_env(monkeypatch, harness=True, wiki=True)
    counts = {
        name: {
            "claims_staging": index,
            "claims_final": index + 10,
            "sources": index + 20,
        }
        for index, name in enumerate(
            ("done", "legacy", "moving", "refused", "stale", "waiting")
        )
    }
    sys.modules["llmwiki"].stats = lambda topic: counts[topic.name]
    monkeypatch.setattr(
        cli, "_live_lock_pid", lambda topic: 4321 if topic.name == "moving" else None
    )

    assert main(["status", "--base", str(base), "--json"]) == EXIT_OK
    payload = json.loads(capfd.readouterr().out)
    rows = {row["topic"]: row for row in payload["topics"]}
    assert payload["status"] == "ok" and payload["stale_minutes"] == 90.0
    assert payload["fresh_minutes"] == payload["stale_minutes"]
    assert rows["moving"]["movement"] == "running (role collect)"
    assert rows["moving"]["pid"] == 4321
    assert rows["moving"]["cycles"] == 2 and rows["moving"]["max_cycles"] == 24
    assert rows["waiting"]["movement"] == "waiting for review"
    assert rows["refused"]["movement"] == "gate refused"
    assert rows["refused"]["stop_reason"] == "compiler-gap"
    assert rows["done"]["movement"] == "done"
    assert rows["stale"]["movement"] == "idle"
    assert rows["legacy"]["phase"] == rows["legacy"]["movement"] == "not started"
    assert rows["legacy"]["claims_staging"] == counts["legacy"]["claims_staging"]
    assert moving.is_dir() and waiting.is_dir() and refused.is_dir()


def test_status_base_human_is_one_table(tmp_path, monkeypatch, capsys):
    base = tmp_path / "topics"
    root = base / "legacy"
    root.mkdir(parents=True)
    (root / "topic.json").write_text('{"slug":"legacy","topic":"legacy"}', "utf-8")
    _fake_env(monkeypatch, harness=True, wiki=True)
    sys.modules["llmwiki"].stats = lambda topic: {
        "claims_staging": 7,
        "claims_final": 0,
        "sources": 3,
    }

    assert main(["status", "--base", str(base)]) == EXIT_OK
    shown = capsys.readouterr().out
    assert "topic" in shown and "state" in shown and "legacy" in shown
    assert "not started" in shown and "7" in shown and "3" in shown


def test_status_base_human_shows_llmwiki_setup_diagnostic(tmp_path, monkeypatch, capsys):
    base = tmp_path / "topics"
    base.mkdir()
    monkeypatch.setitem(sys.modules, "llmwiki", None)

    with pytest.raises(SystemExit) as stopped:
        main(["status", "--base", str(base)])

    assert stopped.value.code == EXIT_FAIL
    assert "llmwiki is required" in capsys.readouterr().err


def test_status_base_keeps_other_topics_when_checkpoint_or_stats_is_broken(
    tmp_path, monkeypatch, capfd
):
    base = tmp_path / "topics"
    for name in ("broken-checkpoint", "broken-stats", "healthy"):
        root = base / name
        root.mkdir(parents=True)
        (root / "topic.json").write_text(json.dumps({"topic": name}), "utf-8")
        checkpoint.save(
            root,
            {"phase": "collecting", "topic": name, "queue": [], "done": []},
        )
    (base / "broken-checkpoint" / "work" / "checkpoint.json").write_text(
        "{not json", "utf-8"
    )
    _fake_env(monkeypatch, harness=True, wiki=True)

    def stats(topic):
        if topic.name == "broken-stats":
            raise ValueError("refused")
        return {"claims_staging": 1, "claims_final": 2, "sources": 3}

    sys.modules["llmwiki"].stats = stats
    assert main(["status", "--base", str(base), "--json"]) == EXIT_OK
    rows = {
        row["topic"]: row
        for row in json.loads(capfd.readouterr().out)["topics"]
    }
    assert rows["broken-checkpoint"]["phase"] == "checkpoint unreadable"
    assert rows["broken-checkpoint"]["claims_final"] == 2
    assert rows["broken-stats"]["phase"] == "statistics unavailable"
    assert rows["broken-stats"]["claims_final"] is None
    assert rows["healthy"]["phase"] == "collecting"


def test_status_base_reports_pid_only_while_lock_is_really_held(tmp_path):
    root = tmp_path / "topic"
    lock_path = root / "work" / "resume.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(f"pid={os.getpid()}\n", "utf-8")

    with lock_path.open("r+", encoding="utf-8") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert cli._live_lock_pid(root) == os.getpid()
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
    assert cli._live_lock_pid(root) is None


def test_status_movement_covers_heartbeat_stale_start_stopped_and_legacy():
    now = datetime.datetime.now(datetime.timezone.utc)
    fresh = {"phase": "collecting", "updated_at": now.isoformat()}
    stale = {
        "phase": "collecting",
        "updated_at": (now - datetime.timedelta(hours=1)).isoformat(),
    }
    active = {
        "role": "synth",
        "call_started": (now - datetime.timedelta(minutes=36)).isoformat(),
    }
    very_old_call = {
        "role": "synth",
        "call_started": (now - datetime.timedelta(hours=3, minutes=12)).isoformat(),
    }

    assert cli._movement(
        fresh, stale_minutes=90, pid=123, heartbeat=active,
        activity_age_minutes=36,
    ) == "running (role synth, call 36 min)"
    assert cli._movement(
        stale, stale_minutes=90, pid=123, heartbeat=very_old_call,
        activity_age_minutes=192,
    ) == "STALE (role synth call running for 3h12m)"
    assert cli._movement(
        fresh, stale_minutes=90, pid=None, heartbeat=active,
        activity_age_minutes=1,
    ) == "idle"
    assert cli._movement(
        stale, stale_minutes=90, pid=123, process_age_minutes=2,
        heartbeat=very_old_call, activity_age_minutes=192,
    ) == "starting"
    legacy = cli._movement(
        stale, stale_minutes=90, pid=123, process_age_minutes=200,
        heartbeat=None, activity_age_minutes=192,
    )
    assert legacy == "no heartbeat (legacy run)" and "STALE" not in legacy
    assert cli._movement(
        {"phase": "planned", "plan_gate": {"status": "waiting"}},
        stale_minutes=90,
        pid=None,
    ) == "waiting for review"
    assert cli._movement(
        {"phase": "synthesizing", "stop_kind": "gate_refused"},
        stale_minutes=90,
        pid=None,
    ) == "gate refused"
    assert cli._movement({"phase": "done"}, stale_minutes=90, pid=None) == "done"


def test_status_movement_max_mtime_includes_heartbeat(tmp_path, monkeypatch):
    """Mutation: drop heartbeat from _activity_age_minutes and we get a false STALL."""
    root = tmp_path / "tema"
    checkpoint.save(root, {
        "phase": "synthesizing", "topic": "the topic", "cycles": 7,
        "max_cycles": 24,
    })
    events = root / "work" / "events.jsonl"
    events.write_text("{}\n", "utf-8")
    old = datetime.datetime.now().timestamp() - 4 * 3600
    os.utime(root / "work" / "checkpoint.json", (old, old))
    os.utime(events, (old, old))
    (root / "work" / "heartbeat.json").write_text(
        json.dumps({
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "phase": "synthesizing",
            "role": "synth",
            "call_started": (
                datetime.datetime.now(datetime.timezone.utc)
                - datetime.timedelta(minutes=36)
            ).isoformat(),
            "pid": 4321,
            "harness_pid": 8765,
            "cycle": 7,
        }),
        "utf-8",
    )
    monkeypatch.setattr(cli, "_live_lock_pid", lambda topic: 4321)
    monkeypatch.setattr(cli, "_process_age_minutes", lambda pid: 240)

    row = cli._status_row(
        root, checkpoint.load(root),
        {"claims_staging": 0, "claims_final": 0, "sources": 0},
        stale_minutes=90,
    )

    assert row["movement"].startswith("running (role synth, call 36 min)")


def _verify_topic(tmp_path):
    root = tmp_path / "topic"
    (root / "work").mkdir(parents=True)
    (root / "staging").mkdir()
    (root / "topic.json").write_text('{"slug":"topic","topic":"topic"}', "utf-8")
    source_a = tmp_path / "source-a.txt"
    source_b = tmp_path / "source-b.txt"
    source_a.write_text("quote a", "utf-8")
    source_b.write_text("quote b", "utf-8")
    (root / "work" / "sources-map.json").write_text(
        json.dumps({"src_b": str(source_b), "src_a": str(source_a)}), "utf-8"
    )
    claims = [
        {"id": "clm_good", "status": "candidate"},
        {"id": "clm_bad", "status": "candidate"},
        {"id": "clm_old", "status": "verified"},
    ]
    (root / "staging" / "claims.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in claims), "utf-8"
    )
    return root


def _fake_verify_cli(calls):
    verbs = "update-source verify-quotes set-status validate stats"
    flags = {
        "update-source": "--content-path",
        "verify-quotes": "--zone",
        "set-status": "--batch",
        "validate": "",
        "stats": "",
    }

    def run(args, *, input_text=None):
        calls.append((args, input_text))
        if args == ["--help"]:
            return types.SimpleNamespace(returncode=0, stdout=verbs, stderr="")
        if len(args) == 2 and args[1] == "--help":
            return types.SimpleNamespace(returncode=0, stdout=flags[args[0]], stderr="")
        if args[0] == "stats":
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"sources": 2, "claims_staging": 3, "claims_final": 0}
                ),
                stderr="",
            )
        if args[0] == "update-source":
            return types.SimpleNamespace(returncode=0, stdout="{}", stderr="")
        if args[0] == "verify-quotes":
            return types.SimpleNamespace(
                returncode=1,
                stdout=json.dumps(
                    [
                        {"level": "error", "claim": "clm_bad"},
                        {"level": "warn", "claim": "clm_good"},
                    ]
                ),
                stderr="",
            )
        if args[0] == "set-status":
            assert json.loads(input_text) == ["clm_good"]
            return types.SimpleNamespace(
                returncode=0,
                stdout='[{"id":"clm_good","result":"verified"}]',
                stderr="",
            )
        if args[0] == "validate":
            return types.SimpleNamespace(returncode=0, stdout="OK\n", stderr="")
        raise AssertionError(args)

    return run


def test_verify_status_uses_llmwiki_cli_gate_and_never_promotes(
    tmp_path, monkeypatch, capsys
):
    root = _verify_topic(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "_run_llmwiki_cli", _fake_verify_cli(calls))

    assert main(["verify-status", str(root)]) == EXIT_OK
    shown = capsys.readouterr().out
    assert "updated=2" in shown and "errors=1" in shown and "verified=1" in shown
    actual = [args for args, _ in calls if "--help" not in args]
    assert [args[2] for args in actual if args[0] == "update-source"] == [
        "src_a",
        "src_b",
    ]
    assert any(args[0] == "verify-quotes" and "--zone" in args for args in actual)
    assert any(args[0] == "set-status" and "--batch" in args for args in actual)
    assert any(args[0] == "validate" for args in actual)
    assert not any(args[0] == "promote" for args in actual)


def test_verify_status_dry_run_checks_shape_and_inputs_without_mutation(
    tmp_path, monkeypatch, capsys
):
    root = _verify_topic(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "_run_llmwiki_cli", _fake_verify_cli(calls))

    assert main(["verify-status", str(root), "--dry-run"]) == EXIT_OK
    assert "DRY-RUN" in capsys.readouterr().out
    actual = [args for args, _ in calls if "--help" not in args]
    assert actual == [["stats", str(root)]]


def test_verify_status_json_is_one_machine_object(tmp_path, monkeypatch, capfd):
    root = _verify_topic(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "_run_llmwiki_cli", _fake_verify_cli(calls))

    assert main(["verify-status", str(root), "--json"]) == EXIT_OK
    captured = capfd.readouterr()
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 1
    payload = json.loads(captured.out)
    assert payload["schema_version"] == 1
    assert payload["status"] == "ok"
    assert payload["topic"] == "topic"
    assert payload["verified"] == 1


def test_verify_status_refuses_verb_without_required_flag(tmp_path, monkeypatch, capsys):
    root = _verify_topic(tmp_path)

    def broken(args, *, input_text=None):
        if args == ["--help"]:
            return types.SimpleNamespace(
                returncode=0,
                stdout="update-source verify-quotes set-status validate stats",
                stderr="",
            )
        return types.SimpleNamespace(returncode=0, stdout="wrong flags", stderr="")

    monkeypatch.setattr(cli, "_run_llmwiki_cli", broken)
    assert main(["verify-status", str(root)]) == EXIT_FAIL
    assert "shape_mismatch:update-source" in capsys.readouterr().err


def test_verify_status_never_mutates_knowledge_tree_outside_fake_cli(
    tmp_path, monkeypatch
):
    """Write-path truth: the wrapper only reads, every change belongs to the CLI."""
    root = _verify_topic(tmp_path)
    sources = root / "sources" / "sources.jsonl"
    sources.parent.mkdir()
    sources.write_text('{"id":"src_a","title":"A"}\n', "utf-8")
    final = root / "final" / "claims.jsonl"
    final.parent.mkdir()
    final.write_text('{"id":"clm_final","status":"verified"}\n', "utf-8")

    def snapshot():
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for zone in ("sources", "staging", "final")
            for path in (root / zone).rglob("*")
            if path.is_file()
        }

    calls = []
    fake = _fake_verify_cli(calls)

    def fake_with_own_trace(args, *, input_text=None):
        result = fake(args, input_text=input_text)
        trace = root / "work" / "fake-cli.log"
        with trace.open("a", encoding="utf-8") as stream:
            stream.write(args[0] + "\n")
        return result

    monkeypatch.setattr(cli, "_run_llmwiki_cli", fake_with_own_trace)
    before = snapshot()
    report = cli._verify_status(root, dry_run=False)
    after = snapshot()

    assert report["verified"] == 1
    assert after == before
    assert (root / "work" / "fake-cli.log").read_text("utf-8").splitlines()


def _supervisor_topic(tmp_path):
    root = tmp_path / "topic"
    checkpoint.save(
        root,
        {
            "phase": "collecting",
            "topic": "topic",
            "queue": [],
            "done": [],
            "run_id": "run-supervisor",
        },
    )
    return root


@pytest.mark.parametrize(
    "terminal",
    [cli.EXIT_OK, cli.EXIT_PLAN_REVIEW, cli.EXIT_GATE_REFUSED, cli.EXIT_QUOTA],
)
def test_resume_supervise_returns_terminal_codes_without_retry(
    tmp_path, monkeypatch, terminal
):
    root = _supervisor_topic(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "_run", lambda *args, **kwargs: calls.append(1) or terminal)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: pytest.fail("sleep forbidden"))

    assert main(["resume", str(root), "--supervise"]) == terminal
    assert calls == [1]
    events = [
        json.loads(line)
        for line in (root / "work" / "events.jsonl").read_text("utf-8").splitlines()
    ]
    assert [row["event"] for row in events] == [
        "supervisor_attempt",
        "supervisor_stop",
    ]
    assert events[-1]["exit_code"] == terminal


def test_resume_supervise_tees_full_streams_and_records_bounded_reason(
    tmp_path, monkeypatch, capfd
):
    root = _supervisor_topic(tmp_path)
    adapter_stderr = "adapter-" + "x" * 400

    def transient(*args, adapter_results, **kwargs):
        print("full engine stdout")
        for index in range(1, 5):
            print(f"progress: line {index}", file=sys.stderr)
        adapter_results.append(
            RunResult(
                ok=False,
                text="",
                exit_code=1,
                stop="transient",
                stderr=adapter_stderr,
            )
        )
        return cli.EXIT_TRANSIENT

    monkeypatch.setattr(cli, "_resume_once", transient)

    assert main(
        ["resume", str(root), "--supervise", "--transient-retries", "0"]
    ) == cli.EXIT_FAIL
    captured = capfd.readouterr()
    assert "full engine stdout" in captured.out
    assert "progress: line 1" in captured.err

    logs = list((root / "work" / "logs").glob("supervise-*.log"))
    assert len(logs) == 1
    log = logs[0].read_text("utf-8")
    assert "full engine stdout" in log
    assert all(f"progress: line {index}" in log for index in range(1, 5))

    events = [
        json.loads(line)
        for line in (root / "work" / "events.jsonl").read_text("utf-8").splitlines()
    ]
    attempt = next(row for row in events if row["event"] == "supervisor_attempt")
    assert attempt["reason"] == {
        "stderr_tail": [
            "progress: line 2",
            "progress: line 3",
            "progress: line 4",
        ],
        "adapter": {"stop": "transient", "stderr": adapter_stderr[:300]},
    }
    assert attempt["log"] == f"work/logs/{logs[0].name}"


@pytest.mark.parametrize(
    ("code", "stop"),
    [
        (cli.EXIT_OK, "done"),
        (cli.EXIT_QUOTA, "quota"),
        (cli.EXIT_TRANSIENT, "transient"),
        (cli.EXIT_FAIL, "fatal"),
    ],
)
def test_supervisor_reason_selects_adapter_stop_matching_exit(code, stop):
    results = [
        RunResult(True, "", 0, stop="done", stderr="done stderr"),
        RunResult(False, "", 1, stop=stop, stderr=f"{stop} stderr"),
    ]
    reason = cli._supervisor_reason("engine stderr\n", results, code)
    assert reason["adapter"] == {"stop": stop, "stderr": f"{stop} stderr"}


def test_resume_supervise_json_keeps_engine_streams_in_log_only(
    tmp_path, monkeypatch, capfd
):
    root = _supervisor_topic(tmp_path)

    def done(*args, adapter_results, **kwargs):
        print("engine stdout")
        print("engine stderr", file=sys.stderr)
        adapter_results.append(RunResult(True, "ok", 0, stop="done"))
        return cli.EXIT_OK

    monkeypatch.setattr(cli, "_resume_once", done)

    assert main(["resume", str(root), "--supervise", "--json"]) == EXIT_OK
    captured = capfd.readouterr()
    assert json.loads(captured.out)["schema_version"] == 1
    assert "engine stdout" not in captured.out
    assert "engine stderr" not in captured.err
    log = next((root / "work" / "logs").glob("supervise-*.log")).read_text("utf-8")
    assert "engine stdout" in log and "engine stderr" in log


def test_resume_supervise_observes_real_adapter_result(
    tmp_path, monkeypatch
):
    root = _supervisor_topic(tmp_path)
    _fake_env(monkeypatch, harness=True, wiki=True)

    class TransientAdapter:
        def run(self, *args, **kwargs):
            return RunResult(
                False, "", 1, stop="transient", stderr="adapter transport failed"
            )

    class CaptureOrchestrator:
        def __init__(self, topic_dir, *, adapter, wiki, config, notifier=None):
            self.adapter = adapter

        def run(self):
            assert self.adapter.run("prompt").stop == "transient"
            return cli.EXIT_TRANSIENT

    monkeypatch.setitem(cli.ADAPTERS, "claude", TransientAdapter)
    monkeypatch.setattr("researcher.orchestrator.Orchestrator", CaptureOrchestrator)

    assert main(
        ["resume", str(root), "--supervise", "--transient-retries", "0"]
    ) == cli.EXIT_FAIL
    events = [
        json.loads(line)
        for line in (root / "work" / "events.jsonl").read_text("utf-8").splitlines()
    ]
    attempt = next(row for row in events if row["event"] == "supervisor_attempt")
    assert attempt["reason"]["adapter"] == {
        "stop": "transient",
        "stderr": "adapter transport failed",
    }


def test_resume_supervise_detects_three_identical_zero_delta_attempts(
    tmp_path, monkeypatch, capsys
):
    root = _supervisor_topic(tmp_path)
    sleeps = []
    monkeypatch.setattr(cli, "_run", lambda *args, **kwargs: cli.EXIT_FAIL)
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)

    assert main(["resume", str(root), "--supervise"]) == EXIT_FAIL
    assert sleeps == [120, 120]
    events = [
        json.loads(line)
        for line in (root / "work" / "events.jsonl").read_text("utf-8").splitlines()
    ]
    attempts = [row for row in events if row["event"] == "supervisor_attempt"]
    assert [row["exit_code"] for row in attempts] == [1, 1, 1]
    assert events[-1]["reason"] == "loop"
    assert "LOOP: collecting 0 1 x3" in capsys.readouterr().err


def test_resume_supervise_caps_transient_retries_and_returns_1(
    tmp_path, monkeypatch
):
    root = _supervisor_topic(tmp_path)
    calls = []
    sleeps = []
    monkeypatch.setattr(
        cli,
        "_run",
        lambda *args, **kwargs: calls.append(1) or cli.EXIT_TRANSIENT,
    )
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)

    assert main(
        ["resume", str(root), "--supervise", "--transient-retries", "0"]
    ) == cli.EXIT_FAIL
    assert calls == [1]
    assert sleeps == []
    events = [
        json.loads(line)
        for line in (root / "work" / "events.jsonl").read_text("utf-8").splitlines()
    ]
    assert events[-1]["exit_code"] == cli.EXIT_FAIL
    assert events[-1]["reason"] == "transient_retry_limit"
    assert events[-1]["transient_retries"] == 0


def test_three_transients_use_retry_limit_and_never_loop(tmp_path, monkeypatch):
    root = _supervisor_topic(tmp_path)
    calls = []
    monkeypatch.setattr(
        cli, "_run", lambda *args, **kwargs: calls.append(1) or cli.EXIT_TRANSIENT,
    )
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    assert main([
        "resume", str(root), "--supervise", "--transient-retries", "2",
    ]) == EXIT_FAIL

    events = [json.loads(line) for line in
              (root / "work" / "events.jsonl").read_text("utf-8").splitlines()]
    assert len(calls) == 3
    assert events[-1]["reason"] == "transient_retry_limit"
    assert not any(row.get("reason") == "loop" for row in events)


def test_queue_removal_without_claims_resets_supervisor_loop_token(tmp_path, monkeypatch):
    root = _supervisor_topic(tmp_path)
    state = checkpoint.load(root)
    state["queue"] = [{"id": f"t{index}", "query": "q"} for index in range(3)]
    checkpoint.save(root, state)

    def fatal_with_queue_progress(*args, **kwargs):
        current = checkpoint.load(root)
        current["done"].append(current["queue"].pop(0)["id"])
        checkpoint.save(root, current)
        return EXIT_FAIL

    monkeypatch.setattr(cli, "_run", fatal_with_queue_progress)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    assert main([
        "resume", str(root), "--supervise", "--fatal-retries", "2",
    ]) == EXIT_FAIL

    events = [json.loads(line) for line in
              (root / "work" / "events.jsonl").read_text("utf-8").splitlines()]
    assert events[-1]["reason"] == "fatal_retry_limit"
    assert not any(row.get("reason") == "loop" for row in events)


def test_supervisor_progress_token_covers_all_durable_progress(tmp_path):
    root = _supervisor_topic(tmp_path)
    state = checkpoint.load(root)
    state.update(
        phase="verifying", cycles=4,
        queue=[{"id": "q1"}, {"id": "q2"}], done=["d1"],
    )
    checkpoint.save(root, state)
    (root / "work" / "verdicts.jsonl").write_text("{}\n{}\n", "utf-8")
    for zone, count in (("staging", 3), ("final", 2)):
        path = root / zone / "claims.jsonl"
        path.parent.mkdir()
        path.write_text("{}\n" * count, "utf-8")

    assert cli._supervisor_progress_token(root) == (
        "verifying", 4, 2, 1, 2, 3, 2, 0,
    )


@pytest.mark.parametrize("artifact_dir", ["extracted", "verified"])
def test_supervisor_progress_token_counts_import_artifacts(tmp_path, artifact_dir):
    root = _supervisor_topic(tmp_path)
    state = checkpoint.load(root)
    state["run_mode"] = "import"
    checkpoint.save(root, state)
    directory = root / "work" / artifact_dir
    directory.mkdir()
    (directory / "one.json").write_text("{}", "utf-8")

    assert cli._supervisor_progress_token(root)[-1] == 1


def test_supervise_suppresses_start_and_emits_one_canonical_loop_stop(
    tmp_path, monkeypatch
):
    root = _supervisor_topic(tmp_path)
    hook_envs = []
    notify_flags = []
    monkeypatch.setattr(
        "researcher.tool_config.load_tool_config",
        lambda *args, **kwargs: ToolConfig(Path("config.toml"), True, "hook"),
    )

    def hook_run(argv, **kwargs):
        hook_envs.append(kwargs["env"])
        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    def fatal(*args, **kwargs):
        notify_flags.append(kwargs.get("notify"))
        return EXIT_FAIL

    monkeypatch.setattr("researcher.notifications.subprocess.run", hook_run)
    monkeypatch.setattr(cli, "_run", fatal)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    assert main(["resume", str(root), "--supervise"]) == EXIT_FAIL

    assert notify_flags == [False, False, False]
    assert [env["RESEARCHER_EVENT"] for env in hook_envs] == ["stop"]
    env = hook_envs[0]
    assert env["RESEARCHER_KIND"] == "fail"
    assert env["RESEARCHER_TITLE"] == "stop: loop"
    assert env["RESEARCHER_CLASS"] == "alert"
    assert env["RESEARCHER_ACTION"] == f"resume {root} --supervise"
    lines = env["RESEARCHER_LINES"].splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("Phase collecting, cycle 0:")
    assert lines[1].startswith("Log: ") and "supervise-" in lines[1]
    assert lines[2] == f"Action: resume {root} --supervise"


def test_resume_notifier_suppresses_only_run_started():
    class Notifier:
        def __init__(self):
            self.events = []

        def emit(self, event, text, **notice):
            self.events.append((event, text, notice))
            return True

    base = Notifier()
    notifier = cli._ResumeNotifier(base)

    assert notifier.emit("run_started", "start", kind="info") is False
    assert notifier.emit("phase_changed", "check", kind="progress") is True
    assert notifier.emit("stop", "quota", kind="wait") is True
    assert [event for event, _text, _notice in base.events] == [
        "phase_changed", "stop",
    ]


@pytest.mark.parametrize(
    ("code", "title", "line"),
    [
        (cli.EXIT_QUOTA, "waiting for quota", "Phase collecting, cycle 2 saved in the checkpoint."),
        (
            cli.EXIT_PLAN_REVIEW,
            "waiting for plan review",
            "Action: inspect work/search-map.json and resume",
        ),
    ],
)
def test_wait_notices_follow_quota_and_plan_review_canon(
    tmp_path, code, title, line
):
    root = _supervisor_topic(tmp_path)
    state = checkpoint.load(root)
    state["cycles"] = 2
    checkpoint.save(root, state)

    class Notifier:
        def __init__(self):
            self.calls = []

        def emit(self, event, text, **notice):
            self.calls.append((event, text, notice))

    notifier = Notifier()
    cli._emit_wait_notice(notifier, root, code)

    event, _text, notice = notifier.calls[0]
    assert event == "stop" and notice["kind"] == "wait"
    assert notice["title"] == title and notice["lines"] == [line]


def test_resume_supervise_caps_fatal_retries_even_when_claims_keep_growing(
    tmp_path, monkeypatch
):
    root = _supervisor_topic(tmp_path)
    calls = 0
    sleeps = []

    def fatal_with_progress(*args, **kwargs):
        nonlocal calls
        calls += 1
        path = root / "staging" / "claims.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": f"clm_{calls}"}) + "\n")
        return EXIT_FAIL

    monkeypatch.setattr(cli, "_run", fatal_with_progress)
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)

    assert main(
        ["resume", str(root), "--supervise", "--fatal-retries", "2"]
    ) == EXIT_FAIL
    assert calls == 3 and sleeps == [120, 120]
    events = [
        json.loads(line)
        for line in (root / "work" / "events.jsonl").read_text("utf-8").splitlines()
    ]
    assert events[-1]["reason"] == "fatal_retry_limit"
    assert [
        row["claims_delta"] for row in events if row["event"] == "supervisor_attempt"
    ] == [1, 1, 1]


def test_resume_unknown_source_tool_exits_1(tmp_path, capsys):
    root = tmp_path / "tema"
    checkpoint.save(root, {"phase": "planned", "topic": "t", "queue": [],
                           "done": [], "sessions": {}})
    assert main(["resume", str(root), "--sources", "bogus"]) == EXIT_FAIL
    assert "unknown source tools" in capsys.readouterr().err


def test_start_without_llmwiki_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "llmwiki", None)
    with pytest.raises(SystemExit) as e:
        main(["start", "t", "--base", str(tmp_path)])
    assert e.value.code == EXIT_FAIL
    assert "tool-llm-wiki" in capsys.readouterr().err


def test_resume_with_unknown_phase_exits_1(tmp_path, monkeypatch, capsys):
    """The phase guard raises ValueError - the CLI must return an honest stderr line and 1,
    not a bare traceback: a scheduler decides nothing from a traceback (contract 0/75/111/1)."""
    root = tmp_path / "tema"
    checkpoint.save(root, {"phase": "planned", "topic": "t", "queue": [], "done": []})
    path = root / "work" / "checkpoint.json"
    st = json.loads(path.read_text("utf-8"))
    st["phase"] = "chilling"  # bypassing the checkpoint.save gate - we write the file directly
    path.write_text(json.dumps(st, ensure_ascii=False), "utf-8")
    _fake_env(monkeypatch, harness=True, wiki=True)
    assert main(["resume", str(root)]) == EXIT_FAIL
    err = capsys.readouterr().err
    assert "run interrupted" in err and "unknown checkpoint phase" in err


def test_resume_lock_enotsup_exits_1_with_network_fs_hint(tmp_path, monkeypatch, capsys):
    """A filesystem without flock gives the contractual stderr + 1, not a raw OSError/traceback."""
    root = tmp_path / "tema"
    checkpoint.save(root, {"phase": "done", "topic": "t", "queue": [], "done": []})
    final = root / "final" / "claims.jsonl"
    final.parent.mkdir(parents=True)
    final.write_text('{"id":"clm_x"}\n', "utf-8")
    _fake_env(monkeypatch, harness=True, wiki=True)

    def no_flock(*_args):
        raise OSError(errno.ENOTSUP, "Operation not supported")

    monkeypatch.setattr("researcher.orchestrator.fcntl.flock", no_flock)
    assert main(["resume", str(root)]) == EXIT_FAIL
    err = capsys.readouterr().err
    assert "run interrupted" in err
    assert "filesystem does not support locking" in err
    assert "Operation not supported" in err and "Traceback" not in err


def test_resume_parity_without_web_tool_exits_1(tmp_path, monkeypatch, capsys):
    """The config gate fires in the orchestrator CONSTRUCTOR - the CLI must catch that too
    (codex review fix-3: the constructor sat outside try, and `--parity --sources hn` gave a
    bare traceback instead of the contractual stderr+1)."""
    root = tmp_path / "tema"
    checkpoint.save(root, {"phase": "planned", "topic": "t", "queue": [],
                           "done": [], "sessions": {}})
    _fake_env(monkeypatch, harness=True, wiki=True)
    assert main(["resume", str(root), "--sources", "hn", "--parity"]) == EXIT_FAIL
    err = capsys.readouterr().err
    assert "run interrupted" in err and "parity mode" in err
    assert "Traceback" not in err


# --- start on an existing topic = resume, not a crash (tasks item 6) -------------
# A live rake from the 2026-08-10 ladder: a driver restart after a quota died with a bare
# FileExistsError from init_topic - that is, the normal path of the nightly job was broken.

def _wiki_with_init(monkeypatch, fixed_slug=None):
    """A fake llmwiki that accepts the researcher-computed slug like the real one."""
    mod = types.ModuleType("llmwiki")
    mod.__file__ = str(cli.WIKI_REPO / "llmwiki" / "__init__.py")

    def init_topic(base, topic, *, slug=None):
        root = cli.Path(base) / (fixed_slug or slug)
        manifest = root / "topic.json"
        if manifest.exists():
            raise FileExistsError(f"topic already exists: {manifest}")
        root.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"topic": topic}, ensure_ascii=False), "utf-8")
        return root

    mod.init_topic = init_topic
    monkeypatch.setitem(sys.modules, "llmwiki", mod)
    return mod


def test_start_on_existing_topic_resumes_instead_of_failing(tmp_path, monkeypatch, capsys):
    _wiki_with_init(monkeypatch)
    monkeypatch.setattr("researcher.cli.shutil.which", lambda name: "/usr/local/bin/" + name)
    assert main(["start", "topic of the year", "--base", str(tmp_path), "--no-run"]) == EXIT_OK
    root = capsys.readouterr().out.splitlines()[0]
    st = checkpoint.load(root)
    st["phase"] = "collecting"          # a run already happened and was interrupted
    st["queue"] = [{"id": "t1", "query": "q"}]
    checkpoint.save(root, st)

    ran = {}

    def fake_run(topic_dir, args):
        ran["dir"] = str(topic_dir)
        return 0

    monkeypatch.setattr("researcher.cli._run", fake_run)
    assert main(["start", "topic of the year", "--base", str(tmp_path)]) == 0
    out = capsys.readouterr()
    assert ran["dir"] == root                       # start went into a run of THE SAME topic
    assert out.out.splitlines()[0] == root          # contract: the first stdout line is the path
    assert "continuing from the checkpoint" in out.err and "Traceback" not in out.err
    after = checkpoint.load(root)
    assert after["phase"] == "collecting" and after["queue"]  # the checkpoint was NOT rewritten


def test_start_hash_suffix_separates_topics_that_used_to_collide(tmp_path, monkeypatch, capsys):
    # Cyrillic on purpose: the collision this guards against only happened for Cyrillic
    # topics, whose slug used to degenerate to the few Latin fragments they contained.
    _wiki_with_init(monkeypatch)
    assert main(["start", "тема года", "--base", str(tmp_path), "--no-run"]) == EXIT_OK
    first = Path(capsys.readouterr().out.splitlines()[0])
    assert main(["start", "СОВСЕМ ДРУГАЯ тема того же года", "--base", str(tmp_path),
                 "--no-run"]) == EXIT_OK
    second_output = capsys.readouterr()
    second = Path(second_output.out.splitlines()[0])
    assert first != second
    assert first.name.endswith(cli._topic_folder_slug("тема года")[-9:])
    assert second.name.endswith(cli._topic_folder_slug("СОВСЕМ ДРУГАЯ тема того же года")[-9:])
    assert json.loads(first.joinpath("topic.json").read_text("utf-8"))["topic"] == "тема года"
    assert checkpoint.load(first)["topic"] == "тема года"
    assert "Traceback" not in second_output.err


def test_start_on_two_topics_with_same_text_is_fail_closed(tmp_path, monkeypatch, capsys):
    """Two directories with the SAME topic text (a moved base, a copy, a merged base): the
    engine does not pick the lexicographically first one but refuses with the list of paths
    (codex review fix-2)."""
    _wiki_with_init(monkeypatch)
    for slug in ("tema-a", "tema-b"):
        (tmp_path / slug).mkdir()
        (tmp_path / slug / "topic.json").write_text(
            json.dumps({"topic": "topic of the year"}, ensure_ascii=False), "utf-8")
    assert main(["start", "topic of the year", "--base", str(tmp_path), "--no-run"]) == EXIT_FAIL
    err = capsys.readouterr().err
    assert "multiple topics have the same text" in err and "resume" in err and "Traceback" not in err
    assert str(tmp_path / "tema-a") in err and str(tmp_path / "tema-b") in err


def test_start_skips_non_dict_manifest_without_traceback(tmp_path, monkeypatch, capsys):
    """Valid JSON but not an object in a foreign topic.json - we skip it like a broken
    manifest instead of failing with AttributeError on .get (codex review fix-2)."""
    _wiki_with_init(monkeypatch)
    (tmp_path / "chuzhaya").mkdir()
    (tmp_path / "chuzhaya" / "topic.json").write_text('["not an object"]', "utf-8")
    assert main(["start", "topic of the year", "--base", str(tmp_path), "--no-run"]) == EXIT_OK
    out = capsys.readouterr()
    assert "Traceback" not in out.err
    assert out.out.splitlines()[0] == str(
        tmp_path / cli._topic_folder_slug("topic of the year")
    )


def test_start_on_topic_without_checkpoint_starts_from_plan(tmp_path, monkeypatch, capsys):
    """The topic was created by llmwiki and no run happened in it: with no checkpoint we
    write planned and go into the run instead of failing with FileExistsError."""
    _wiki_with_init(monkeypatch)
    topic_dir = tmp_path / cli._topic_folder_slug("topic of the year")
    topic_dir.mkdir()
    (topic_dir / "topic.json").write_text(
        json.dumps({"topic": "topic of the year"}, ensure_ascii=False), "utf-8")
    assert main(["start", "topic of the year", "--base", str(tmp_path), "--no-run"]) == EXIT_OK
    out = capsys.readouterr()
    assert "without a checkpoint" in out.err
    assert checkpoint.load(topic_dir)["phase"] == "planned"


def test_resume_done_topic_with_empty_final_and_staging_exits_77(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "tema"
    checkpoint.save(root, {"phase": "done", "topic": "t", "queue": [],
                           "done": [], "sessions": {}})
    _fake_env(monkeypatch, harness=True, wiki=True)
    assert main(["resume", str(root)]) == cli.EXIT_GATE_REFUSED
    err = capsys.readouterr().err
    assert "publication gate refused" in err and "nothing to publish" in err
    assert "Traceback" not in err


def test_done_prints_explicit_banner_with_topic_and_counts(tmp_path, monkeypatch, capsys):
    """A successful done must not be confused with a stop: the result is one banner from disk."""
    root = tmp_path / "tema"
    checkpoint.save(root, {"phase": "done", "topic": "t", "queue": [], "done": [],
                           "cycles": 4, "sessions": {}})
    final = root / "final" / "claims.jsonl"
    final.parent.mkdir(parents=True)
    final.write_text("".join(json.dumps({
        "id": f"clm_{index}",
        "evidence": [{"source_id": f"src_{index}"}],
    }) + "\n" for index in range(1, 4)), "utf-8")
    sources = root / "sources" / "sources.jsonl"
    sources.parent.mkdir(parents=True)
    source_rows = [
        {"id": "src_1", "kind": "web", "url": "https://one.example/1"},
        {"id": "src_2", "kind": "paper", "url": "https://two.example/2"},
        {"id": "src_3", "kind": "repo", "url": "https://two.example/3"},
    ]
    sources.write_text("".join(json.dumps(row) + "\n" for row in source_rows), "utf-8")
    wiki = root / "final" / "wiki"
    wiki.mkdir()
    (wiki / "concept.md").write_text(json.dumps({
        "claims": ["clm_1", "clm_2", "clm_3"],
        "body": _VALID_PAGE_BODY,
    }), "utf-8")
    _fake_env(monkeypatch, harness=True, wiki=True)

    assert main(["resume", str(root)]) == EXIT_OK

    out = capsys.readouterr()
    assert "progress: phase=done" in out.err
    assert out.out.strip() == (
        f"DONE: {root} | claims: 3 / sources: 3 / pages: 1 / cycles: 4"
    )


def test_usage_error_exits_1(capsys):
    """A broken flag = 1, not argparse's 2 (enforced by shared/cliargs, docs/conventions.md).
    Subparsers inherit the parser class - we check that on a subcommand."""
    with pytest.raises(SystemExit) as e:
        main(["doctor", "--no-such-flag"])
    assert e.value.code == EXIT_FAIL
    assert "error" in capsys.readouterr().err


@pytest.mark.parametrize(("extra", "expected"), [([], False), (["--auto-plan"], True)])
def test_cli_auto_plan_is_explicit(tmp_path, monkeypatch, extra, expected):
    root = tmp_path / "tema"
    checkpoint.save(root, {"phase": "planned", "topic": "t", "queue": [], "done": []})
    _fake_env(monkeypatch, harness=True, wiki=True)
    got = {}

    class CaptureOrchestrator:
        def __init__(self, topic_dir, *, adapter, wiki, config, notifier=None):
            got["auto_plan"] = config.auto_plan

        def run(self):
            return cli.EXIT_PLAN_REVIEW

    monkeypatch.setattr("researcher.orchestrator.Orchestrator", CaptureOrchestrator)
    assert main(["resume", str(root), *extra]) == cli.EXIT_PLAN_REVIEW
    assert got["auto_plan"] is expected


def test_cli_extend_starts_separate_delta_run_on_done_topic(tmp_path, monkeypatch, capsys):
    root = tmp_path / "tema"
    checkpoint.save(root, {"phase": "done", "topic": "t", "queue": [], "done": [],
                           "run_id": "run_initial", "funnel": {
                               "found": 3, "staging": {"keep": 3, "drop": 0},
                               "final": {"keep": 3, "drop": 0},
                           }})
    _fake_env(monkeypatch, harness=True, wiki=True)
    got = {}

    class CaptureOrchestrator:
        def __init__(self, topic_dir, *, adapter, wiki, config, notifier=None):
            got["topic_dir"] = topic_dir
            got["auto_plan"] = config.auto_plan

        def run(self):
            return cli.EXIT_PLAN_REVIEW

    monkeypatch.setattr("researcher.orchestrator.Orchestrator", CaptureOrchestrator)

    assert main(["extend", str(root)]) == cli.EXIT_PLAN_REVIEW
    state = checkpoint.load(root)
    assert state["phase"] == "planned" and state["run_mode"] == "extend"
    assert state["parent_run_id"] == "run_initial"
    assert state["run_id"] != "run_initial"
    assert got == {"topic_dir": str(root), "auto_plan": False}
    assert f"extension started: run_id={state['run_id']}" in capsys.readouterr().out


def _cli_codex_model_commands(tmp_path, monkeypatch, extra_args):
    """Runs the real CLI parsing but replaces only the phase machine.

    Returns the actual codex commands for both roles: this way the test catches both the loss
    of the "--model was passed" flag and the adapter-specific drop of the claude sonnet default.
    """
    from researcher.adapters.base import ROLE_COLLECT, ROLE_THINK

    root = tmp_path / "tema"
    checkpoint.save(root, {"phase": "planned", "topic": "t", "queue": [], "done": []})
    _fake_env(monkeypatch, harness=True, wiki=True)
    got = {}

    class CaptureOrchestrator:
        def __init__(self, topic_dir, *, adapter, wiki, config, notifier=None):
            for role in (ROLE_THINK, ROLE_COLLECT):
                model, _ = config.for_role(
                    role, harness_default=adapter.default_model_for_role(role))
                got[role] = (model, adapter.build_cmd("p", model=model))

        def run(self):
            return EXIT_OK

    monkeypatch.setattr("researcher.orchestrator.Orchestrator", CaptureOrchestrator)
    argv = ["resume", str(root), "--adapter", "codex", *extra_args]
    assert main(argv) == EXIT_OK
    return got


def test_topic_folder_slug_bounds_prefix_and_hashes_full_text():
    topic = "Very long topic " * 30
    folder = cli._topic_folder_slug(topic)

    prefix, digest = folder.rsplit("-", 1)
    assert len(prefix) <= 120
    assert digest == hashlib.sha256(topic.encode("utf-8")).hexdigest()[:8]
    assert cli._topic_folder_slug(topic + "x") != folder


# Cyrillic on purpose: this is the transliteration contract of _topic_folder_slug.
@pytest.mark.parametrize(("topic", "prefix"), [
    ("тема года", "tema-goda-"),
    ("Ёж и чай", "yozh-i-chay-"),
    ("Щука, юла и мяч", "shchuka-yula-i-myach-"),
])
def test_topic_folder_slug_transliterates_cyrillic_readably(topic, prefix):
    folder = cli._topic_folder_slug(topic)

    assert folder.startswith(prefix)
    assert not folder.startswith("topic-")


def test_status_short_prints_stop_and_first_three_open_questions(tmp_path, capfd):
    root = tmp_path / "tema"
    checkpoint.save(root, {
        "phase": "synthesizing", "topic": "the topic", "cycles": 3,
        "judge": {"verdict": "stuck", "cycle": 3, "why": "the sources are exhausted"},
        "stop": {"reason": "stuck", "why": "the sources are exhausted"},
        "coverage": {"open_questions": ["first", "second", "third", "fourth"]},
    })
    (root / "work" / "coverage.md").write_text(
        "# Coverage\n\n## Open questions\n\n"
        "- first\n- second\n- third\n- fourth\n",
        "utf-8",
    )

    assert main(["status", str(root), "--short"]) == EXIT_OK
    output = capfd.readouterr().out
    assert "judge: stuck (3) - the sources are exhausted" in output
    assert "stop reason: stuck" in output
    assert "open questions: first | second | third" in output
    assert "fourth" not in output


def test_status_base_includes_judge_stop_and_three_coverage_questions(
    tmp_path, monkeypatch, capfd
):
    base = tmp_path / "topics"
    root = base / "tema"
    (root / "work").mkdir(parents=True)
    (root / "topic.json").write_text('{"topic":"the topic"}', "utf-8")
    checkpoint.save(root, {
        "phase": "done", "topic": "the topic", "cycles": 2,
        "judge": {"verdict": "stuck", "cycle": 2, "why": "there is no more data"},
        "stop": {"reason": "stuck", "why": "there is no more data"},
        "coverage": {"open_questions": ["one", "two", "three", "four"]},
    })
    (root / "work" / "coverage.md").write_text(
        "## Open questions\n\n- one\n- two\n- three\n- four\n", "utf-8",
    )
    _fake_env(monkeypatch, harness=True, wiki=True)

    assert main(["status", "--base", str(base), "--json"]) == EXIT_OK
    row = json.loads(capfd.readouterr().out)["topics"][0]
    assert row["judge"] == "stuck (2) - there is no more data"
    assert row["stop_reason"] == "stuck"
    assert row["coverage_questions"] == ["one", "two", "three"]


def test_cli_bare_codex_keeps_owners_hybrid(tmp_path, monkeypatch):
    got = _cli_codex_model_commands(tmp_path, monkeypatch, [])
    think_model, think_cmd = got["think"]
    collect_model, collect_cmd = got["collect"]
    assert think_model == "sonnet" and "--model" not in think_cmd
    assert collect_model == "gpt-5.6-luna"
    assert collect_cmd[collect_cmd.index("--model") + 1] == "gpt-5.6-luna"


def test_cli_explicit_model_uses_one_model_in_all_roles(tmp_path, monkeypatch):
    got = _cli_codex_model_commands(tmp_path, monkeypatch, ["--model", "gpt-5.6-terra"])
    for model, cmd in got.values():
        assert model == "gpt-5.6-terra"
        assert cmd[cmd.index("--model") + 1] == "gpt-5.6-terra"


def test_cli_explicit_model_collect_makes_explicit_hybrid(tmp_path, monkeypatch):
    got = _cli_codex_model_commands(
        tmp_path, monkeypatch,
        ["--model", "gpt-5.6-terra", "--model-collect", "gpt-5.6-luna"])
    assert got["think"][0] == "gpt-5.6-terra"
    assert got["collect"][0] == "gpt-5.6-luna"
    assert got["think"][1][got["think"][1].index("--model") + 1] == "gpt-5.6-terra"
    assert got["collect"][1][got["collect"][1].index("--model") + 1] == "gpt-5.6-luna"


def test_env_sample_creds_scrubbed(tmp_path):
    """The conftest isolator strips the variables of .env.sample: the owner's real
    GITHUB_TOKEN is invisible to the hermetic tests of the github source tool."""
    sample = tmp_path / ".env.sample"
    sample.write_text("# prose header: name=\n# FOO_TOKEN=   # secret\nBAR_KEY=x\n", "utf-8")
    assert _env_sample_vars(sample) == ["FOO_TOKEN", "BAR_KEY"]
    leaked = [n for n in _env_sample_vars() if n in os.environ]
    assert not leaked, f".env.sample credentials are visible to the test: {leaked}"


def test_public_readme_has_release_gate_sections():
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text("utf-8")
    for heading in (
        "## Problem",
        "## What it does",
        "## Architecture",
        "## Quick start",
        "## Demo",
        "## Limitations",
        "## Data and credential boundary",
        "## Tests",
        "## License",
    ):
        assert heading in readme


def test_public_docs_describe_configurable_data_and_credentials():
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text("utf-8")
    assert "RESEARCHER_HOME" in readme and "TOOLS_DATA" in readme
    assert "GITHUB_TOKEN" in readme and "EXA_API_KEY" in readme
    assert "Never commit" in readme and "signed URLs" in readme


def test_public_design_explains_concurrency_and_durable_state():
    repo = Path(__file__).resolve().parents[1]
    design = (repo / "docs" / "DESIGN.md").read_text("utf-8")
    assert "Durable state machine" in design
    assert "Files and concurrency" in design
    assert "Collector work may run concurrently" in design


def test_public_tree_omits_internal_handoff_documents():
    repo = Path(__file__).resolve().parents[1]
    assert (repo / "docs" / "DESIGN.md").is_file()
    assert not (repo / "docs" / "STATUS.md").exists()
    assert not (repo / "docs" / "tasks.md").exists()
    assert not (repo / ".agent").exists()
