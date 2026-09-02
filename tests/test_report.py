"""`research report`: synthesis report over a done topic - sintez-tem moved from ~/prep into the tool (23.08).

Contract: terminal guard (done + free resume.lock), an "already exists" report is never
rewritten, the harness is called once with cwd = the report directory and no network,
exit 76/0/75/111/1, report done / report failed notices through on_event.
"""
import fcntl
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from researcher import checkpoint
from researcher.adapters.base import Capabilities, HarnessAdapter, RunResult
from researcher.cli import main
from researcher.notifications import EventNotifier
from researcher.report import (
    DEFAULT_TEMPLATE, EXIT_FAIL, EXIT_OK, EXIT_QUOTA, EXIT_REPORT_READY, EXIT_TRANSIENT,
    render_prompt, run_report, topic_terminal,
)
from researcher.tool_config import ToolConfig


class WriterHarness(HarnessAdapter):
    """A process-free harness: writes the file named in the prompt (or not) and returns the given stop."""
    name = "writer"

    def __init__(self, *, write=True, stop="done", text="ok", stderr="",
                 before_write=None):
        self.write, self.stop_kind, self.text, self.stderr = write, stop, text, stderr
        self.before_write = before_write
        self.calls = []

    def capabilities(self):
        return Capabilities(False, False, False, False, False)

    def build_cmd(self, prompt, **kw):
        return ["true"]

    def parse_output(self, stdout, exit_code):
        return RunResult(ok=True, text=stdout, exit_code=exit_code)

    def run(self, prompt, *, cwd, timeout=None, retries=0, retry_backoff=5.0,
            on_harness_pid=None, **build_kw):
        self.calls.append({"prompt": prompt, "cwd": cwd, "timeout": timeout, **build_kw})
        if self.before_write is not None:
            self.before_write()
        if self.write:
            out = prompt.split("Output: ", 1)[1].split(" ", 1)[0]
            Path(out).write_text("# report\n\ntext\n", "utf-8")
        ok = self.stop_kind == "done"
        return RunResult(ok=ok, text=self.text, exit_code=0 if ok else 1,
                         stop=self.stop_kind, stderr=self.stderr)


def _topic(tmp_path, phase="done", name="Contractor rates 2026"):
    root = tmp_path / "tema"
    (root / "work").mkdir(parents=True)
    (root / "topic.json").write_text(json.dumps({"topic": name}), "utf-8")
    checkpoint.save(root, {"phase": phase, "topic": name, "queue": [], "done": [],
                           "sessions": {}})
    return root


def _hooked_notifier(tmp_path, root, monkeypatch):
    events = []

    def run(argv, **kwargs):
        env = kwargs["env"]
        events.append({k: env[k] for k in env if k.startswith("RESEARCHER_")})
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("researcher.notifications.subprocess.run", run)
    return EventNotifier(root, ToolConfig(tmp_path / "config.toml", True, "deliver")), events


def _no_pid(_topic_dir):
    return None


def test_render_prompt_replaces_all_placeholders(tmp_path):
    text = render_prompt(DEFAULT_TEMPLATE, topic_dir=tmp_path / "t", out=tmp_path / "o.md",
                         name="Name", topics_root=tmp_path)
    assert "__" not in text.replace("__init__", "")
    assert str(tmp_path / "t") in text and "Topic: Name" in text
    custom = render_prompt("A __NAME__ B __TOPICS_ROOT__ C __OUT__", topic_dir=tmp_path,
                           out=Path("/x/o.md"), name="N", topics_root=Path("/r"))
    assert custom == "A N B /r C /x/o.md"


def test_terminal_requires_done_and_free_lock(tmp_path):
    root = _topic(tmp_path, phase="collecting")
    assert topic_terminal(root) == (False, "phase='collecting'")
    checkpoint.save(root, {**checkpoint.load(root), "phase": "done"})
    assert topic_terminal(root) == (True, "")
    assert topic_terminal(tmp_path / "nope")[0] is False


def test_report_writes_file_notifies_and_returns_ready(tmp_path, monkeypatch):
    root = _topic(tmp_path)
    out = tmp_path / "reports" / "sintez.md"
    out.parent.mkdir()
    harness = WriterHarness()
    notifier, events = _hooked_notifier(tmp_path, root, monkeypatch)

    code = run_report(root, out, adapter=harness, live_pid=_no_pid, notifier=notifier,
                      effort="xhigh", timeout=30.0)

    assert code == EXIT_REPORT_READY and out.read_text("utf-8").startswith("# report")
    call = harness.calls[0]
    assert call["cwd"] == str(out.parent) and call["network"] is False
    assert call["effort"] == "xhigh" and call["timeout"] == 30.0
    assert "Topic: Contractor rates 2026" in call["prompt"]
    assert str(root.parent) in call["prompt"]  # topics_root defaults to the topic's parent
    assert [e["RESEARCHER_EVENT"] for e in events] == ["report_done"]
    assert events[0]["RESEARCHER_KIND"] == "done"
    assert events[0]["RESEARCHER_TITLE"] == "report done"
    assert str(out) in events[0]["RESEARCHER_LINES"]
    logs = list((root / "work" / "logs").glob("report-*.log"))
    assert len(logs) == 1 and "stop=done" in logs[0].read_text("utf-8")


def test_existing_report_is_never_rewritten(tmp_path, capsys):
    root = _topic(tmp_path)
    out = tmp_path / "sintez.md"
    out.write_text("old\n", "utf-8")
    harness = WriterHarness()
    assert run_report(root, out, adapter=harness, live_pid=_no_pid) == EXIT_REPORT_READY
    assert harness.calls == [] and out.read_text("utf-8") == "old\n"
    assert "already exists" in capsys.readouterr().err


def test_not_done_topic_waits_only_with_guard(tmp_path, capsys):
    root = _topic(tmp_path, phase="collecting")
    out = tmp_path / "sintez.md"
    harness = WriterHarness()
    assert run_report(root, out, adapter=harness, live_pid=_no_pid) == EXIT_FAIL
    assert run_report(root, out, adapter=harness, live_pid=_no_pid, guard=True) == EXIT_OK
    assert harness.calls == [] and not out.exists()
    err = capsys.readouterr().err
    assert "--guard" in err and "waiting" in err


def test_stale_lock_text_without_flock_does_not_block_report(tmp_path):
    root = _topic(tmp_path)
    (root / "work" / "resume.lock").write_text("pid=77\n", "utf-8")
    harness = WriterHarness()
    assert run_report(root, tmp_path / "o.md", adapter=harness, live_pid=_no_pid,
                      guard=True) == EXIT_REPORT_READY
    assert len(harness.calls) == 1


def test_empty_held_resume_lock_blocks_report_guard(tmp_path):
    root = _topic(tmp_path)
    lock_path = root / "work" / "resume.lock"
    lock_path.touch()
    harness = WriterHarness()

    with lock_path.open("a+", encoding="utf-8") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run_report(
            root, tmp_path / "o.md", adapter=harness, live_pid=_no_pid, guard=True
        ) == EXIT_OK

    assert harness.calls == []
    assert not (tmp_path / "o.md").exists()


def test_report_holds_shared_topic_lock_through_harness_call(tmp_path):
    root = _topic(tmp_path)
    mutation_lock_acquired = []

    def try_mutation_lock():
        with (root / "work" / "resume.lock").open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                mutation_lock_acquired.append(False)
            else:
                mutation_lock_acquired.append(True)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    harness = WriterHarness(before_write=try_mutation_lock)
    assert run_report(
        root, tmp_path / "o.md", adapter=harness, live_pid=_no_pid
    ) == EXIT_REPORT_READY
    assert mutation_lock_acquired == [False]


def test_partial_output_after_transient_never_becomes_ready(tmp_path):
    root = _topic(tmp_path)
    out = tmp_path / "o.md"
    harness = WriterHarness(stop="transient")

    first = run_report(root, out, adapter=harness, live_pid=_no_pid)
    second = run_report(root, out, adapter=harness, live_pid=_no_pid)

    assert (first, second) == (EXIT_TRANSIENT, EXIT_TRANSIENT)
    assert len(harness.calls) == 2
    assert not out.exists()


def test_two_reporters_cannot_generate_same_output_concurrently(tmp_path):
    root = _topic(tmp_path)
    out = tmp_path / "o.md"
    entered = threading.Event()
    release = threading.Event()

    def wait_in_harness():
        entered.set()
        assert release.wait(2)

    first_harness = WriterHarness(before_write=wait_in_harness)
    second_harness = WriterHarness()
    result = []

    thread = threading.Thread(
        target=lambda: result.append(
            run_report(root, out, adapter=first_harness, live_pid=_no_pid)
        )
    )
    thread.start()
    assert entered.wait(2)
    second = run_report(root, out, adapter=second_harness, live_pid=_no_pid)
    release.set()
    thread.join(2)

    assert not thread.is_alive()
    assert result == [EXIT_REPORT_READY]
    assert second == EXIT_TRANSIENT
    assert second_harness.calls == []


def test_missing_file_after_run_is_failure_with_notice(tmp_path, monkeypatch):
    root = _topic(tmp_path)
    out = tmp_path / "sintez.md"
    harness = WriterHarness(write=False, stderr="something\nthe model refused to write")
    notifier, events = _hooked_notifier(tmp_path, root, monkeypatch)
    assert run_report(root, out, adapter=harness, live_pid=_no_pid,
                      notifier=notifier) == EXIT_FAIL
    assert not out.exists()
    assert [e["RESEARCHER_EVENT"] for e in events] == ["report_failed"]
    assert events[0]["RESEARCHER_KIND"] == "fail"
    lines = events[0]["RESEARCHER_LINES"].splitlines()
    assert lines[0] == "the model refused to write" and lines[2].startswith("Action:")


@pytest.mark.parametrize("stop,code", [("quota", EXIT_QUOTA), ("transient", EXIT_TRANSIENT)])
def test_quota_and_transient_pass_through_without_notice(tmp_path, monkeypatch, stop, code):
    root = _topic(tmp_path)
    harness = WriterHarness(write=False, stop=stop)
    notifier, events = _hooked_notifier(tmp_path, root, monkeypatch)
    assert run_report(root, tmp_path / "o.md", adapter=harness, live_pid=_no_pid,
                      notifier=notifier) == code
    assert events == []


def test_missing_out_dir_is_input_error(tmp_path):
    root = _topic(tmp_path)
    harness = WriterHarness()
    assert run_report(root, tmp_path / "nope" / "o.md", adapter=harness,
                      live_pid=_no_pid) == EXIT_FAIL
    assert harness.calls == []


def test_cli_report_uses_template_file_and_guard(tmp_path, monkeypatch):
    root = _topic(tmp_path, phase="collecting")
    template = tmp_path / "prompt.md"
    template.write_text("Output: __OUT__ for __NAME__", "utf-8")
    monkeypatch.setenv("RESEARCHER_HOME", str(tmp_path / "home"))
    assert main(["report", str(root), "--out", str(tmp_path / "o.md"),
                 "--prompt", str(template), "--guard"]) == EXIT_OK
    assert main(["report", str(root), "--out", str(tmp_path / "o.md"),
                 "--prompt", str(tmp_path / "nope.md")]) == EXIT_FAIL
