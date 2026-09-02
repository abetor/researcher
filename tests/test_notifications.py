import json
import re
import subprocess
from pathlib import Path

import pytest

from researcher import checkpoint
from researcher.notifications import (
    _DEFAULT_TITLES,
    NOTICE_CLASS_BY_KIND,
    EventNotifier,
    HOOK_TIMEOUT_SECONDS,
    NOTICE_LINE_LIMIT,
    NOTICE_LINES_LIMIT,
    NOTICE_SUBJECT_LIMIT,
)
from researcher.tool_config import ToolConfig, ToolConfigError, load_tool_config


def _topic(tmp_path, name="a very long topic name " * 5):
    root = tmp_path / "topic-dir"
    root.mkdir()
    (root / "topic.json").write_text(json.dumps({"topic": name}), "utf-8")
    checkpoint.save(root, {"phase": "collecting", "topic": name, "run_id": "run-hook"})
    return root


def test_tool_config_missing_means_no_hook(tmp_path):
    config = load_tool_config(tmp_path / "missing")

    assert config.exists is False and config.on_event is None


def test_tool_config_rejects_unknown_field_fail_closed(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text('on_event = "true"\ntypo = 1\n', "utf-8")

    with pytest.raises(ToolConfigError, match="unknown fields.*typo"):
        load_tool_config(home)


def test_hook_invocation_receives_required_env_and_one_line_text(tmp_path, monkeypatch):
    root = _topic(tmp_path)
    captured = {}

    def run(argv, **kwargs):
        captured.update(argv=argv, kwargs=kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("researcher.notifications.subprocess.run", run)
    notifier = EventNotifier(root, ToolConfig(tmp_path / "config.toml", True, "deliver now"))

    assert notifier.emit(
        "cycle_done",
        "line 1\nline 2",
        kind="progress",
        title="cycle 2/4",
        lines=["staging 7 (+2 per cycle)", "Rough guide: ~4 min/cycle"],
    ) is True

    assert captured["argv"] == ["/bin/sh", "-c", "deliver now"]
    env = captured["kwargs"]["env"]
    assert env["RESEARCHER_EVENT"] == "cycle_done"
    assert env["RESEARCHER_KIND"] == "progress"
    assert env["RESEARCHER_SUBJECT"] == env["RESEARCHER_TOPIC"]
    assert env["RESEARCHER_TITLE"] == "cycle 2/4"
    assert env["RESEARCHER_LINES"].splitlines() == [
        "staging 7 (+2 per cycle)", "Rough guide: ~4 min/cycle",
    ]
    assert env["RESEARCHER_TEXT"] == "line 1 line 2"
    assert len(env["RESEARCHER_SUBJECT"]) <= NOTICE_SUBJECT_LIMIT
    assert env["RESEARCHER_TOPIC_DIR"] == str(root.resolve())
    assert captured["kwargs"]["timeout"] == HOOK_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    "kind", ["progress", "done", "fail", "stall", "wait", "info"],
)
def test_hook_accepts_every_canonical_kind(tmp_path, monkeypatch, kind):
    root = _topic(tmp_path)
    captured = {}

    def run(argv, **kwargs):
        captured.update(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("researcher.notifications.subprocess.run", run)

    assert EventNotifier(
        root, ToolConfig(tmp_path / "config.toml", True, "deliver")
    ).emit("test", kind, kind=kind, title="check", lines=["line"]) is True

    assert captured["RESEARCHER_KIND"] == kind


def test_subject_uses_whole_first_words_without_tail(tmp_path, monkeypatch):
    name = "first words of the topic must stay whole but this tail must disappear"
    root = _topic(tmp_path, name)
    captured = {}
    monkeypatch.setattr(
        "researcher.notifications.subprocess.run",
        lambda argv, **kwargs: captured.update(kwargs["env"])
        or subprocess.CompletedProcess(argv, 0, "", ""),
    )

    EventNotifier(
        root, ToolConfig(tmp_path / "config.toml", True, "deliver")
    ).emit("test", "text")

    subject = captured["RESEARCHER_SUBJECT"]
    assert subject == "first words of the topic must stay whole"
    assert len(subject) <= NOTICE_SUBJECT_LIMIT
    assert name.startswith(subject) and name[len(subject)] == " "


def test_title_and_lines_are_bounded_and_lines_have_no_embedded_newlines(
    tmp_path, monkeypatch
):
    root = _topic(tmp_path)
    captured = {}
    monkeypatch.setattr(
        "researcher.notifications.subprocess.run",
        lambda argv, **kwargs: captured.update(kwargs["env"])
        or subprocess.CompletedProcess(argv, 0, "", ""),
    )

    EventNotifier(
        root, ToolConfig(tmp_path / "config.toml", True, "deliver")
    ).emit(
        "test", "fallback\nstays a single line", title="t" * 200,
        lines=["a" * 250, "two\nlines\rinside", "third", "extra"],
    )

    lines = captured["RESEARCHER_LINES"].splitlines()
    assert len(lines) == NOTICE_LINES_LIMIT
    assert all(len(line) <= NOTICE_LINE_LIMIT for line in lines)
    assert lines[1] == "two lines inside"
    assert captured["RESEARCHER_TEXT"] == "fallback stays a single line"
    assert len(captured["RESEARCHER_TITLE"]) == 80


def test_hook_rejects_unknown_kind_before_shell(tmp_path, monkeypatch):
    root = _topic(tmp_path)
    monkeypatch.setattr(
        "researcher.notifications.subprocess.run",
        lambda *args, **kwargs: pytest.fail("shell must not run"),
    )

    with pytest.raises(ValueError, match="unknown notice kind"):
        EventNotifier(
            root, ToolConfig(tmp_path / "config.toml", True, "deliver")
        ).emit("test", "text", kind="warning")


def test_hook_shell_command_can_write_delivery_log(tmp_path):
    root = _topic(tmp_path, "short topic")
    log = tmp_path / "hook.log"
    command = f'printf "%s | %s\\n" "$RESEARCHER_EVENT" "$RESEARCHER_TEXT" >> "{log}"'

    assert EventNotifier(
        root, ToolConfig(tmp_path / "config.toml", True, command)
    ).emit("done", "done: 3 claims") is True

    assert log.read_text("utf-8") == "done | done: 3 claims\n"


def test_hook_timeout_records_hook_failed_and_does_not_raise(tmp_path, monkeypatch, capsys):
    root = _topic(tmp_path)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr("researcher.notifications.subprocess.run", timeout)
    notifier = EventNotifier(root, ToolConfig(tmp_path / "config.toml", True, "slow"))

    assert notifier.emit("cycle_done", "cycle") is False

    assert "hook_failed cycle_done" in capsys.readouterr().err
    event = json.loads((root / "work" / "events.jsonl").read_text("utf-8").splitlines()[-1])
    assert event["event"] == "hook_failed" and event["hook_event"] == "cycle_done"
    assert "timeout after 30 s" in event["reason"]


def test_hook_nonzero_records_failure_but_run_can_continue(tmp_path, monkeypatch):
    root = _topic(tmp_path)
    monkeypatch.setattr(
        "researcher.notifications.subprocess.run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 9, "", "bad gateway"),
    )

    assert EventNotifier(
        root, ToolConfig(tmp_path / "config.toml", True, "false")
    ).emit("done", "done") is False

    event = json.loads((root / "work" / "events.jsonl").read_text("utf-8").splitlines()[-1])
    assert event["event"] == "hook_failed" and "exit 9" in event["reason"]


def test_hook_non_utf8_output_is_replaced_and_run_stays_alive(tmp_path, monkeypatch):
    root = _topic(tmp_path)
    monkeypatch.setattr(
        "researcher.notifications.subprocess.run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 9, b"", b"bad\xffgateway"),
    )

    assert EventNotifier(
        root, ToolConfig(tmp_path / "config.toml", True, "bad-bytes")
    ).emit("done", "done") is False

    event = json.loads((root / "work" / "events.jsonl").read_text("utf-8").splitlines()[-1])
    assert event["event"] == "hook_failed"
    assert "bad�gateway" in event["reason"]


def test_hook_and_failure_journal_errors_still_do_not_break_run(
    tmp_path, monkeypatch, capsys
):
    root = _topic(tmp_path)
    monkeypatch.setattr(
        "researcher.notifications.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 7, "", "bad"),
    )
    monkeypatch.setattr(
        "researcher.notifications._hook_failed",
        lambda *args: (_ for _ in ()).throw(OSError("disk refused")),
    )

    assert EventNotifier(
        root, ToolConfig(tmp_path / "config.toml", True, "false")
    ).emit("done", "done") is False

    err = capsys.readouterr().err
    assert "hook_failed done" in err and "hook_failed journal" in err


def _capture(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "researcher.notifications.subprocess.run",
        lambda argv, **kwargs: captured.update(kwargs["env"])
        or subprocess.CompletedProcess(argv, 0, "", ""),
    )
    return captured


def test_hook_env_carries_v2_envelope_for_search_topic(tmp_path, monkeypatch):
    # Canon 25.08 item 7: from=researcher, about=short name, name=full topic,
    # class from kind, action from the "Action: ..." line.
    name = "Senior Go engineer interview meta 2026: formats, what is actually asked"
    root = _topic(tmp_path, name)
    captured = _capture(monkeypatch)

    EventNotifier(
        root, ToolConfig(tmp_path / "config.toml", True, "deliver")
    ).emit(
        "stop", "failed", kind="fail", title="fail",
        lines=["Phase collecting, cycle 2: failure", f"Action: resume {root} --supervise"],
    )

    assert captured["RESEARCHER_FROM"] == "researcher"
    assert captured["RESEARCHER_ABOUT"] == "Senior Go engineer interview meta 2026:"
    assert captured["RESEARCHER_NAME"] == name
    assert captured["RESEARCHER_CLASS"] == "alert"
    assert captured["RESEARCHER_ACTION"] == f"resume {root} --supervise"
    # The legacy contract stays.
    assert captured["RESEARCHER_KIND"] == "fail" and captured["RESEARCHER_SUBJECT"]

    state = checkpoint.load(root)
    state["name"] = "sobes-meta"
    checkpoint.save(root, state)
    EventNotifier(
        root, ToolConfig(tmp_path / "config.toml", True, "deliver")
    ).emit("cycle_done", "cycle", kind="progress", title="cycle 3/24", lines=["staging 7"])
    assert captured["RESEARCHER_ABOUT"] == "sobes-meta"
    assert captured["RESEARCHER_CLASS"] == "event"
    assert captured["RESEARCHER_ACTION"] == ""


@pytest.mark.parametrize("kind,klass", sorted(NOTICE_CLASS_BY_KIND.items()))
def test_hook_class_follows_kind_map(tmp_path, monkeypatch, kind, klass):
    root = _topic(tmp_path, "short topic")
    captured = _capture(monkeypatch)
    EventNotifier(
        root, ToolConfig(tmp_path / "config.toml", True, "deliver")
    ).emit("test", "text", kind=kind, title="t")
    assert captured["RESEARCHER_CLASS"] == klass


def _course_topic(tmp_path, dumps, slug="bc-gccourse"):
    root = tmp_path / "topics" / slug
    root.mkdir(parents=True)
    (root / "topic.json").write_text(json.dumps({"topic": f"Course: {slug}"}), "utf-8")
    checkpoint.save(root, {
        "phase": "import:extract", "run_mode": "import", "topic": f"Course: {slug}",
        "slug": slug, "dump": str(dumps / slug),
    })
    return root


def test_hook_env_for_course_topic_uses_slug_and_catalog_title(tmp_path, monkeypatch):
    dumps = tmp_path / "coursedump" / "out"
    dumps.mkdir(parents=True)
    (dumps.parent / "catalog.tsv").write_text(
        "# course registry\n"
        "slug\ttitle\turl\tnote\n"
        "bc-gccourse\tGo concurrency\thttps://x\t-\n",
        "utf-8",
    )
    root = _course_topic(tmp_path, dumps)
    captured = _capture(monkeypatch)
    config = ToolConfig(tmp_path / "config.toml", True, "deliver")

    # The default catalog is <dump>/../../catalog.tsv taken from the checkpoint.
    EventNotifier(root, config).emit("pages_done", "done", kind="done", title="pages done")
    assert captured["RESEARCHER_FROM"] == "coursedump"
    assert captured["RESEARCHER_ABOUT"] == "bc-gccourse"
    assert captured["RESEARCHER_NAME"] == "Go concurrency"
    assert captured["RESEARCHER_CLASS"] == "event"

    # An explicit catalog wins; the slug is absent from it - name is empty, not a crash.
    other = tmp_path / "other.tsv"
    other.write_text("slug\ttitle\nzzz\tZ\n", "utf-8")
    EventNotifier(root, config, catalog=other).emit("stop", "failed", kind="fail")
    assert captured["RESEARCHER_NAME"] == ""
    assert captured["RESEARCHER_ABOUT"] == "bc-gccourse"

    # No file at all - empty as well.
    EventNotifier(root, config, catalog=tmp_path / "missing.tsv").emit("stop", "failed")
    assert captured["RESEARCHER_NAME"] == ""


def test_notice_titles_have_no_caps():
    # Canon 25.08: titles without caps. A regression guard over the defaults and over every
    # title="..." literal in the engine code (the only title= literals are on notice emits).
    assert all(title == title.lower() for title in _DEFAULT_TITLES.values())
    package = Path(__file__).resolve().parents[1] / "researcher"
    offenders = []
    for path in sorted(package.glob("*.py")):
        for number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
            for match in re.finditer(r'title=f?"([^"]*)"', line):
                title = match.group(1)
                if title != title.lower():
                    offenders.append(f"{path.name}:{number}: {title}")
    assert offenders == []
