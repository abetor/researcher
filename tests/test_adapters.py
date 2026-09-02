import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from researcher.adapters import ADAPTERS, ClaudeAdapter, CodexAdapter, RunResult  # noqa: E402
from researcher.adapters.base import HarnessAdapter, Capabilities  # noqa: E402

_PREFIX = "PYTHONPATH=/repo /usr/bin/python3 -m researcher.sources"


@pytest.fixture
def ad():
    return ClaudeAdapter()


@pytest.fixture
def cx():
    return CodexAdapter()


def test_build_cmd_basics(ad):
    cmd = ad.build_cmd("prompt", model="sonnet")
    assert cmd[:4] == ["claude", "-p", "--output-format", "json"]
    # '--' terminates options (without it allowedTools swallows the prompt - a rig footgun)
    assert cmd[-2:] == ["--", "prompt"]
    assert "--allowedTools" in cmd and "--model" in cmd


def test_claude_physical_tool_set_and_closed_permission_mode(ad):
    """Closing the tool set in claude takes TWO layers, and both must be in the command
    (codex review, block-1).

    Live probes 2026-08-10 (claude 2.1.226, haiku):
    - with `--tools Read,Grep,Glob,Bash` (third wave, already without the nonexistent LS) the
      model's inventory was "Bash, Glob, Grep, Read", and to "do you have WebSearch?" and "do
      you have LS?" it answered "no": web is absent from the set, not denied by permissions.
      With LS in the list the inventory was the same - the name was silently ignored, which
      is why it was removed;
    - without `--strict-mcp-config --mcp-config {"mcpServers":{}}` the owner's MCP tools are
      mixed into the set (34 mcp__claude_ai_* of them) - WITHOUT them the physical set is not
      closed;
    - `--permission-mode dontAsk` denies curl, nslookup and a compound command appended after
      an allowed prefix (permission_denials), while the prefix itself runs.
    """
    cmd = ad.build_cmd("p", tools="Read Grep Bash", allowed_tools="Read Grep Bash(x:*)")
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert cmd[cmd.index("--tools") + 1] == "Read,Grep,Bash"   # comma-separated names (the --help form)
    assert "--strict-mcp-config" in cmd
    assert cmd[cmd.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert cmd[cmd.index("--allowedTools") + 1] == "Read Grep Bash(x:*)"
    # without a physical set the flags are absent entirely (the harness keeps its default set)
    plain = ad.build_cmd("p")
    assert "--tools" not in plain and "--strict-mcp-config" not in plain


def test_claude_default_allowlist_has_no_web(ad):
    """A bare build_cmd without an allowlist must not silently hand out web: the default is
    read only. A latent footgun from the claude review (a fallback with WebSearch/WebFetch
    would break parity silently)."""
    cmd = ad.build_cmd("p")
    assert cmd[cmd.index("--allowedTools") + 1] == "Read Grep Glob"
    assert "WebSearch" not in " ".join(cmd) and "WebFetch" not in " ".join(cmd)


def test_build_cmd_resume_and_schema(ad, tmp_path):
    schema = tmp_path / "s.json"
    schema.write_text('{"type":"object"}')
    cmd = ad.build_cmd("p", resume_session_id="sid-1", schema_path=str(schema))
    assert "--resume" in cmd and "sid-1" in cmd
    # the schema inline (as a string), not as a path
    assert '{"type":"object"}' in cmd
    with pytest.raises(ValueError):
        empty = tmp_path / "e.json"
        empty.write_text("")
        ad.build_cmd("p", schema_path=str(empty))


def test_parse_single_json(ad):
    out = json.dumps({"result": "answer", "session_id": "s1", "is_error": False,
                      "total_cost_usd": 0.5})
    r = ad.parse_output(out, 0)
    assert r.ok and r.text == "answer" and r.session_id == "s1" and r.cost_usd == 0.5


def test_parse_error_and_garbage(ad):
    r = ad.parse_output(json.dumps({"result": "", "is_error": True}), 0)
    assert not r.ok
    r = ad.parse_output("not json at all", 1)
    assert not r.ok and r.text == "not json at all"


def test_classify_quota_even_on_ok(ad):
    r = RunResult(ok=True, text="You have hit your usage limit, resets at 6pm", exit_code=0)
    assert ad.classify(r) == "quota"


@pytest.mark.parametrize("text", [
    '```json\n{"summary":"usage limit of the tool"}\n```',
    'Role answer:\n{"summary":"usage limit of the tool"}',
])
def test_classify_structured_ok_output_with_quota_words_is_done(ad, text):
    assert ad.classify(RunResult(ok=True, text=text, exit_code=0)) == "done"


def test_classify_structured_ok_output_still_reads_quota_diagnostics(ad):
    result = RunResult(
        ok=True,
        text='```json\n{"summary":"answer"}\n```',
        exit_code=0,
        stderr="usage limit reached",
    )
    assert ad.classify(result) == "quota"


def test_classify_transient_vs_fatal(ad):
    assert ad.classify(RunResult(ok=False, text="", exit_code=1,
                                 stderr="api error 529 overloaded")) == "transient"
    assert ad.classify(RunResult(ok=False, text="", exit_code=1,
                                 stderr="no such flag")) == "fatal"
    assert ad.classify(RunResult(ok=True, text="a normal answer", exit_code=0)) == "done"


def test_classify_live_codex_capacity_error_is_transient(ad, cx):
    """Live task_complete/codex_error_info from the 2026-08-11 smoke test are cured by a retry."""
    message = "Selected model is at capacity. Please try a different model."
    assert ad.classify(RunResult(ok=False, text=message, exit_code=1)) == "transient"
    assert ad.classify(RunResult(
        ok=False, text="", exit_code=1,
        raw={"codex_error_info": "server_overloaded"},
    )) == "transient"
    live = json.dumps({
        "type": "error",
        "message": message,
        "codex_error_info": "server_overloaded",
    })
    parsed = cx.parse_output(live, 1)
    assert parsed.raw["codex_error_info"] == "server_overloaded"
    assert cx.classify(parsed) == "transient"


# --- the quota / transient boundary (production ladder bug 2026-08-10) --------
#
# A burst of parallel codex calls produced an ordinary 429, and the old _QUOTA swallowed it
# as a real quota -> exit 75 -> the ladder driver stopped the whole matrix, even though
# waiting a few seconds would have fixed it. Both sides of the boundary are pinned below.
# The wordings come from the harness docs (docs/phase1/L §quota, docs/phase1/M §5): we have
# no live samples and cannot get them cheaply - one cannot hit a quota on purpose.


@pytest.mark.parametrize("msg", [
    "usage limit reached",                              # a form common to both harnesses
    "You've hit your weekly limit - resets Mon 12:00am",   # claude, weekly window
    "You've hit your usage limit. Try again in 4 hours.",  # codex; "try again" does not win
    "5-hour limit reached, resets at 6pm",              # claude, session window
    "You have hit your limit",                          # claude, short form
    "you are out of credits",                           # codex/API, credits exhausted
    "monthly quota exhausted",
    "429: usage limit reached",                         # MIXED - decided in favor of quota
])
def test_classify_real_quota(ad, msg):
    """The subscription budget is exhausted: only waiting for the window helps -> stop with
    a checkpoint (75).

    The mixed "429: usage limit reached" is deliberately classified as QUOTA: budget wording
    is more specific than the generic 429 code, and the cost of an error is asymmetric - an
    extra wait is cheaper than hammering a real quota.
    """
    assert ad.classify(RunResult(ok=False, text=msg, exit_code=1)) == "quota"
    assert ad.classify(RunResult(ok=False, text="", exit_code=1, stderr=msg)) == "quota"


@pytest.mark.parametrize("msg", [
    "429",                                              # the bare burst code
    "stream error: unexpected status 429 Too Many Requests",  # the codex stream form
    "too many requests, please try again later",
    "Rate limit reached for gpt-5.4",                   # the generic throttling wording
    "rate limit exceeded",
    "you are being rate-limited",
])
def test_classify_burst_rate_limit_is_transient(ad, msg):
    """Burst throttling is NOT quota: it is cured by a retry within seconds (exit 111).

    This is exactly the class that stopped the ladder on 2026-08-10 by landing in quota. The
    reverse mutation (putting 429/rate limit back into _QUOTA) must fail precisely these
    parameters.
    """
    assert ad.classify(RunResult(ok=False, text=msg, exit_code=1)) == "transient"
    assert ad.classify(RunResult(ok=False, text="", exit_code=1, stderr=msg)) == "transient"


def test_classify_throttle_on_ok_run_stays_done(ad):
    """A 429 on a SUCCESSFUL call is not a stop: the harness already did its own retry
    (claude retries transients itself, docs/phase1/L §"Automatic retries"). On top of that a
    searcher's text holds material collected from the web: a page about rate limits must not
    fail the collection."""
    r = RunResult(ok=True, text="the source explains 429 Too Many Requests and rate limit",
                  exit_code=0)
    assert ad.classify(r) == "done"


# --- the collector capability profile (tasks item 2) -------------------------


def test_claude_profile_native_web_and_scoped_bash(ad):
    p = ad.collect_profile(source_cmd_prefix=_PREFIX)
    assert p.web == "native" and p.network is True
    # a static set: native web + reading + EXACTLY the source CLI prefix (a scoped rule;
    # live check 2026-08-10: with it touch/curl are denied while the prefix itself runs)
    assert p.allowed_tools == f"WebSearch WebFetch Read Grep Glob Bash({_PREFIX}:*)"
    assert "Task" not in p.allowed_tools and "Agent" not in p.allowed_tools
    # the physical set holds the same tools as plain names (the scoped rule lives in the allowlist only)
    assert p.tools == "WebSearch WebFetch Read Grep Glob Bash"
    # without source tools Bash is not handed out at all - neither in the set nor in the allowlist
    assert ad.collect_profile().allowed_tools == "WebSearch WebFetch Read Grep Glob"
    assert ad.collect_profile().tools == "WebSearch WebFetch Read Grep Glob"


def test_claude_profile_parity_drops_native_web(ad):
    p = ad.collect_profile(source_cmd_prefix=_PREFIX, parity=True)
    assert p.web == "tool"
    assert p.allowed_tools == f"Read Grep Glob Bash({_PREFIX}:*)"
    # web is absent from the PHYSICAL set too: in parity it is not attached, not merely "denied"
    assert p.tools == "Read Grep Glob Bash"
    assert "WebSearch" not in p.tools and "WebFetch" not in p.tools


def test_codex_profile_has_no_web_and_no_allowlist(cx):
    for parity in (False, True):
        p = cx.collect_profile(source_cmd_prefix=_PREFIX, parity=parity)
        assert p.web == "tool"          # codex has no native web - the profile knows it
        assert p.allowed_tools is None  # no per-tool allowlist, the sandbox is the boundary
        assert p.tools is None          # and it has no physical tool set either
        assert p.network is True        # but source tools are attached - the sandbox needs network


def test_profile_network_is_computed_not_always_on(cx, ad):
    """Network in the profile is computed, not a constant True (codex review, fix-1).

    A blind codex (`--sources ""`, no native web) must not get an open sandbox: it has
    nothing to browse with anyway, and network access there is arbitrary, not limited to our
    tools.
    """
    assert cx.collect_profile().network is False                       # neither web nor source tools
    assert cx.collect_profile(source_cmd_prefix=_PREFIX).network is True
    assert ad.collect_profile().network is True                        # native web means network is needed
    assert ad.collect_profile(parity=True).network is False            # web removed, no tools


def test_unknown_harness_profile_degrades_honestly():
    """A new adapter with no declared capabilities is NOT assumed to have web by default."""
    p = ShAdapter("true").collect_profile(source_cmd_prefix=_PREFIX)
    assert p.web == "tool" and p.allowed_tools is None and p.tools is None


def test_codex_network_opens_sandbox(cx):
    """workspace-write comes WITHOUT network by default: source tools fail with a DNS error
    (live check on codex 0.147, 2026-08-10). The network is opened only where it is asked for."""
    assert "sandbox_workspace_write.network_access=true" not in cx.build_cmd("p")
    cmd = cx.build_cmd("p", network=True)
    assert "sandbox_workspace_write.network_access=true" in cmd
    # and on the resume subcommand too (it has its own set of flags)
    assert "sandbox_workspace_write.network_access=true" in cx.build_cmd(
        "p", resume_session_id="t-1", network=True)


def test_codex_closed_network_is_pinned_explicitly(cx):
    """A closed network is an explicit `-c sandbox_workspace_write.network_access=false`
    rather than the codex default: that default lives in someone else's ~/.codex/config.toml
    and may change with a CLI version, while the prompt of the thinking phases carries
    someone else's text (course transcripts). Live check 0.150.1, 2026-08-28."""
    cmd = cx.build_cmd("p")
    assert "sandbox_workspace_write.network_access=false" in cmd
    assert "sandbox_workspace_write.network_access=true" not in cmd
    cmd = cx.build_cmd("p", network=True)
    assert "sandbox_workspace_write.network_access=false" not in cmd
    assert "sandbox_workspace_write.network_access=false" in cx.build_cmd(
        "p", resume_session_id="t-1")


def test_codex_empty_tools_means_read_only_sandbox(cx):
    """tools="" (course_import/pages: the agent may do nothing but answer) means a read-only
    sandbox in codex - the only physical switch against writing; None (the collector, a run's
    synthesis) stays workspace-write as before. Live check 0.150.1, 2026-08-28: read-only with
    a schema answers per the schema, `echo > probe.txt` -> "Operation not permitted"; resume
    accepts `-c sandbox_mode=read-only`."""
    cmd = cx.build_cmd("p", tools="")
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert "workspace-write" not in cmd
    cmd = cx.build_cmd("p")
    assert cmd[cmd.index("--sandbox") + 1] == "workspace-write"
    cmd = cx.build_cmd("p", resume_session_id="t-1", tools="")
    assert "sandbox_mode=read-only" in cmd and "--sandbox" not in cmd
    # in read-only the network is always closed: asking for it together with tools="" is a
    # contradiction, fail-closed
    with pytest.raises(ValueError):
        cx.build_cmd("p", tools="", network=True)


def test_claude_ignores_network_flag(ad):
    """In claude the network is a function of the tool set, there is no separate flag: the command does not change."""
    assert ad.build_cmd("p", network=True) == ad.build_cmd("p", network=False)


def test_claude_effort_is_a_real_flag(ad):
    """In claude effort is a SEPARATE flag `--effort <level>` (in codex it is a config
    override). The form was verified live 2026-08-10 on claude 2.1.226: `--help` declares
    `--effort <level> (low, medium, high, xhigh, max)`, and a live call with `--effort low`
    worked. Without the parameter the flag is absent from the command entirely - we do not
    override the harness default (tasks item 5: effort used not to reach the adapter at all)."""
    cmd = ad.build_cmd("p", effort="high")
    assert cmd[cmd.index("--effort") + 1] == "high"
    assert "--effort" not in ad.build_cmd("p")


# --- codex (the mechanics are a copy of the ones verified in rig, see adapters/codex.py) ---


def test_adapters_registry():
    assert ADAPTERS == {"claude": ClaudeAdapter, "codex": CodexAdapter}


def test_codex_build_cmd_basics(cx):
    cmd = cx.build_cmd("prompt", model="gpt-5-codex")
    # headless = the exec subcommand (not -p); the prompt is the last positional
    assert cmd[:2] == ["codex", "exec"] and cmd[-1] == "prompt"
    assert "--json" in cmd
    # approval is a config override (-c), not an exec flag; sandbox is a separate flag
    assert "approval_policy=never" in cmd and "--sandbox" in cmd and "workspace-write" in cmd
    assert "--model" in cmd and "gpt-5-codex" in cmd
    # effort is a config override, not --effort
    cmd = cx.build_cmd("p", effort="low")
    assert "model_reasoning_effort=low" in cmd and "--effort" not in cmd


def test_codex_role_defaults_are_the_owners_hybrid(cx, ad):
    """OWNER'S DECISION 2026-08-10: on codex, collection runs on the lightweight model and
    the thinking phases on the account default. This is an ENGINE DEFAULT (adapter knowledge),
    not a driver flag."""
    from researcher.adapters.base import ROLE_COLLECT, ROLE_THINK
    assert cx.default_model_for_role(ROLE_COLLECT) == "gpt-5.6-luna"
    assert cx.default_model_for_role(ROLE_THINK) is None      # the account default means no --model
    assert ad.default_model_for_role(ROLE_COLLECT) is None    # role defaults were left alone in claude
    assert ad.default_model_for_role(ROLE_THINK) is None
    # and the role default really does reach the harness command
    cmd = cx.build_cmd("p", model=cx.default_model_for_role(ROLE_COLLECT))
    assert "--model" in cmd and "gpt-5.6-luna" in cmd
    assert "--model" not in cx.build_cmd("p", model=cx.default_model_for_role(ROLE_THINK))


def test_codex_claude_family_model_dropped(cx):
    # a claude name (including the RunConfig default "sonnet") gives a 400 on a ChatGPT
    # subscription; omitting --model means the account default (the rig model_for_harness behavior)
    for m in ("sonnet", "opus", "haiku", "claude-sonnet-4-6", None):
        assert "--model" not in cx.build_cmd("p", model=m)
    assert "--model" in cx.build_cmd("p", model="gpt-5-codex")


def test_codex_resume_subcommand(cx):
    # resume is a SUBCOMMAND exec resume <id>, not a --resume flag; resume has no --sandbox
    # (live check 0.137) - the mode goes through the sandbox_mode config override
    cmd = cx.build_cmd("p", resume_session_id="tid-1")
    assert cmd[:4] == ["codex", "exec", "resume", "tid-1"]
    assert "--resume" not in cmd and "--sandbox" not in cmd
    assert "sandbox_mode=workspace-write" in cmd


def test_codex_schema_by_path_and_resume_conflict(cx, tmp_path):
    schema = tmp_path / "s.json"
    schema.write_text('{"type":"object"}')
    cmd = cx.build_cmd("p", schema_path=str(schema))
    # the schema by PATH (--output-schema), not inline as in claude
    assert "--output-schema" in cmd and str(schema) in cmd
    assert '{"type":"object"}' not in cmd
    # schema+resume are incompatible in codex (#14343) - fail-closed, not a silent drop
    with pytest.raises(ValueError):
        cx.build_cmd("p", resume_session_id="tid-1", schema_path=str(schema))
    empty = tmp_path / "e.json"
    empty.write_text("")
    with pytest.raises(ValueError):
        cx.build_cmd("p", schema_path=str(empty))


def test_codex_parse_jsonl(cx):
    lines = [
        {"type": "thread.started", "thread_id": "t-1"},
        {"type": "item.completed", "item": {"item_type": "reasoning", "text": "thinking"}},
        {"type": "item.completed", "item": {"item_type": "agent_message", "text": "ans"}},
        {"type": "item.completed", "item": {"item_type": "agent_message",
                                            "content": [{"type": "output_text", "text": "wer"}]}},
        {"type": "turn.completed", "usage": {"input_tokens": 5}},
    ]
    r = cx.parse_output("\n".join(json.dumps(l) for l in lines), 0)
    # agent_message parts are joined (a content list -> text), reasoning does not reach text
    assert r.ok and r.text == "ans\nwer" and r.session_id == "t-1"
    assert r.cost_usd is None and r.raw["usage"] == {"input_tokens": 5}


def test_codex_parse_error_event_and_garbage(cx):
    # an error event in the JSONL: not-ok even on exit 0, quota visible to the classifier through raw
    r = cx.parse_output(json.dumps({"type": "error", "message": "usage limit reached"}), 0)
    assert not r.ok
    assert cx.classify(r) == "quota"
    r = cx.parse_output(json.dumps({"type": "error", "message": "invalid_request"}), 1)
    assert not r.ok and cx.classify(r) == "fatal"
    # non-JSONL output (a crash before the format) - an honest not-ok with the raw text
    r = cx.parse_output("not jsonl at all", 1)
    assert not r.ok and r.text == "not jsonl at all"


def _codex_lines(*events) -> str:
    return "\n".join(json.dumps(e) for e in events)


# The event shapes are SYNTHETIC: we have no live sample of a quota arriving as an error
# item (generating one would mean burning the owner's quota). We use the documented
# item.completed/item.type=error shape that codex 0.147 really sends (observed live on hook
# warnings) plus quota text observed from the harnesses ("usage limit reached").


def test_codex_item_level_quota_is_not_swallowed(cx):
    """A quota that arrives as an ITEM with exit 0 must reach the classifier (codex review, fix-2).

    Before the fix: err was set only on a TOP-level error event, and an item error never made
    it into the blob -> ok=True, stop=done. During the collection phase that reads as "the
    task completed empty" instead of exit 75, and the run keeps going, burning the queue for
    nothing (checked by an in-process claude review).
    """
    out = _codex_lines(
        {"type": "item.completed",
         "item": {"type": "error", "message": "usage limit reached, resets at 6pm"}},
        {"type": "item.completed", "item": {"item_type": "agent_message", "text": "partial"}})
    r = cx.parse_output(out, 0)
    assert r.raw["item_errors"]            # the error text reached raw
    assert cx.classify(r) == "quota"       # and the classifier saw it
    assert r.text == "partial"             # the partial answer is not lost in the process


def test_codex_benign_item_warning_stays_done(cx):
    """A benign warning arriving as an item does NOT make the run fatal: codex 0.147 sends two
    of those per call for the owner (a global --dangerously-bypass-hook-trust). The cure "any
    item error means failure" would be worse than the disease - it would fail every run."""
    out = _codex_lines(
        {"type": "item.completed",
         "item": {"type": "error", "message": "hook is not trusted, skipping"}},
        {"type": "item.completed", "item": {"item_type": "agent_message", "text": "answer"}})
    r = cx.parse_output(out, 0)
    assert r.ok and cx.classify(r) == "done" and r.text == "answer"


def test_cli_adapter_flag():
    from researcher import cli
    # an unknown adapter is cut off by argparse choices before any orchestrator import,
    # and it is a usage error -> exit 1 (shared/cliargs), not argparse's stock 2
    with pytest.raises(SystemExit) as e:
        cli.main(["start", "t", "--base", "b", "--adapter", "bogus"])
    assert e.value.code == cli.EXIT_FAIL


class ShAdapter(HarnessAdapter):
    """A fake harness on /bin/sh - the M contract: one-shot, stdout, a distinguishable failure."""
    name = "sh"

    def __init__(self, script):
        self.script = script

    def capabilities(self):
        return Capabilities(False, False, False, False, False)

    def build_cmd(self, prompt, **kw):
        return ["/bin/sh", "-c", self.script]

    def parse_output(self, stdout, exit_code):
        return RunResult(ok=exit_code == 0, text=stdout.strip(), exit_code=exit_code)


class _FlakyAdapter(ShAdapter):
    """A harness that fails with a transient the first `fail` times, then answers."""

    def __init__(self, fail, script_ok="echo ok", script_bad="echo '429' >&2; exit 1"):
        super().__init__(script_ok)
        self.fail, self.spawns = fail, 0

    def build_cmd(self, prompt, **kw):
        self.spawns += 1
        return ["/bin/sh", "-c",
                "echo '429 Too Many Requests' >&2; exit 1" if self.spawns <= self.fail
                else "echo ok"]


@pytest.fixture
def no_sleep(monkeypatch):
    """The retry really sleeps - in tests we capture the DELAYS instead of waiting them out."""
    slept: list[float] = []
    monkeypatch.setattr("researcher.adapters.base.time.sleep", slept.append)
    return slept


def test_transient_is_retried_with_backoff(tmp_path, no_sleep):
    """tasks item 6: there was NO transient retry at all - a single 429 burst left the run
    unfinished. Now the call is repeated a bounded number of times with a growing pause."""
    ad = _FlakyAdapter(fail=2)
    res = ad.run("x", cwd=str(tmp_path), retries=2, retry_backoff=5.0)
    assert res.stop == "done" and res.text == "ok"
    assert ad.spawns == 3          # the first attempt plus two retries
    assert no_sleep == [5.0, 10.0]  # the backoff doubles


def test_retry_budget_is_bounded(tmp_path, no_sleep):
    """Bounded means it has a ceiling: hammering the harness forever is not allowed."""
    ad = _FlakyAdapter(fail=99)
    res = ad.run("x", cwd=str(tmp_path), retries=2, retry_backoff=1.0)
    assert res.stop == "transient" and not res.ok
    assert ad.spawns == 3 and len(no_sleep) == 2


def test_quota_and_fatal_are_not_retried(tmp_path, no_sleep):
    """We retry ONLY what a retry can cure. A quota is cured by waiting for the window (the
    scheduler's territory, exit 75) - hammering only prolongs it; fatal is not cured by
    repetition at all."""
    for script, stop in (("echo 'usage limit reached' >&2; exit 1", "quota"),
                         ("exit 7", "fatal"),
                         ("echo ok", "done")):
        ad = ShAdapter(script)
        assert ad.run("x", cwd=str(tmp_path), retries=5).stop == stop
    assert no_sleep == []


def test_run_default_is_still_one_shot(tmp_path, no_sleep):
    """The base.run default with no parameters is as it was: one spawn, no pauses."""
    ad = _FlakyAdapter(fail=99)
    assert ad.run("x", cwd=str(tmp_path)).stop == "transient"
    assert ad.spawns == 1 and no_sleep == []


def test_run_oneshot_contract(tmp_path, monkeypatch):
    assert ShAdapter("echo hello").run("x", cwd=str(tmp_path)).text == "hello"
    # "usage limit" is a real quota (75); "rate limit exceeded" stood here until 2026-08-10
    # and yielded quota - that was the bug: burst throttling is transient now (111).
    r = ShAdapter("echo 'usage limit reached' >&2; exit 1").run("x", cwd=str(tmp_path))
    assert r.stop == "quota" and not r.ok
    r = ShAdapter("echo 'rate limit exceeded' >&2; exit 1").run("x", cwd=str(tmp_path))
    assert r.stop == "transient" and not r.ok
    r = ShAdapter("exit 7").run("x", cwd=str(tmp_path))
    assert r.stop == "fatal" and r.exit_code == 7
    r = ShAdapter("sleep 5").run("x", cwd=str(tmp_path), timeout=0.2)
    assert r.stop == "transient" and r.stderr == "wall-timeout"

    # stdin is never handed to a harness anywhere (README, "how to call it"): base.run spawns
    # with DEVNULL, otherwise codex exec reads the inherited input to EOF and hangs (a live
    # footgun, 2026-08-05). The assert is double: what we spawned with (deterministically
    # catches a revert of DEVNULL) and what came out - a process reading input sees EOF
    # immediately rather than a timeout.
    seen: dict = {}
    real_popen = subprocess.Popen
    monkeypatch.setattr(
        "researcher.adapters.base.subprocess.Popen",
        lambda cmd, **kw: (seen.update(kw), real_popen(cmd, **kw))[1],
    )
    r = ShAdapter("cat").run("x", cwd=str(tmp_path), timeout=5)
    assert seen["stdin"] is subprocess.DEVNULL
    assert r.stop == "done" and r.text == ""


def test_run_reports_real_harness_pid_and_clears_it(tmp_path):
    observed = []
    result = ShAdapter("echo ok").run(
        "x", cwd=str(tmp_path), on_harness_pid=observed.append,
    )
    assert result.stop == "done"
    assert len(observed) == 2
    assert isinstance(observed[0], int) and observed[0] > 0
    assert observed[1] is None


def test_codex_recovered_stream_error_keeps_completed_answer(cx):
    """A stream break that codex survived on its own (an error event, but the turn ran through
    to agent_message with exit 0) is NOT a failed call. Before 23.08 such an answer was
    rejected -> transient -> the role rerun: a synthesis spun for a whole day (18 sessions of
    40 minutes each) without accepting a single answer."""
    out = _codex_lines(
        {"type": "error", "message": "stream disconnected before completion: idle timeout; retrying"},
        {"type": "item.completed", "item": {"item_type": "agent_message", "text": '{"final": []}'}},
        {"type": "turn.completed", "usage": {"output_tokens": 5}})
    r = cx.parse_output(out, 0)
    assert r.ok and r.text == '{"final": []}'
    assert r.raw["error"]                      # the error text is not lost
    assert cx.classify(r) == "done"
    # a quota inside a survived error is still visible to the classifier (classify rule 1)
    out_q = _codex_lines(
        {"type": "error", "message": "usage limit reached"},
        {"type": "item.completed", "item": {"item_type": "agent_message", "text": "x"}})
    assert cx.classify(cx.parse_output(out_q, 0)) == "quota"
    # with no answer or with a nonzero exit it is not-ok, as before
    assert not cx.parse_output(_codex_lines({"type": "error", "message": "timed out"}), 0).ok
    assert not cx.parse_output(out, 1).ok


def test_codex_long_stream_provider_only_for_chatgpt_login(cx, tmp_path, monkeypatch):
    """The provider clone with a long idle timeout: only when auth_mode=chatgpt, disabled by
    RESEARCHER_CODEX_PROVIDER=builtin; without auth.json or on an api key the command is unchanged."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.delenv("RESEARCHER_CODEX_PROVIDER", raising=False)
    assert "model_provider=openai-long" not in cx.build_cmd("p")          # no auth.json
    (tmp_path / "auth.json").write_text(json.dumps({"auth_mode": "apikey"}), "utf-8")
    assert "model_provider=openai-long" not in cx.build_cmd("p")
    (tmp_path / "auth.json").write_text(json.dumps({"auth_mode": "chatgpt"}), "utf-8")
    cmd = cx.build_cmd("p")
    assert "model_provider=openai-long" in cmd
    assert "model_providers.openai-long.stream_idle_timeout_ms=1800000" in cmd
    assert "model_providers.openai-long.requires_openai_auth=true" in cmd
    assert "model_providers.openai-long.base_url=https://chatgpt.com/backend-api/codex" in cmd
    assert cmd.index("model_provider=openai-long") < cmd.index("p")      # the prompt comes last
    monkeypatch.setenv("RESEARCHER_CODEX_PROVIDER", "builtin")
    assert "model_provider=openai-long" not in cx.build_cmd("p")


def test_quota_words_in_long_ok_answer_are_not_quota(cx, ad):
    """A topic about limits must not stop a run with a false quota (live case 23.08: the
    planner on 'usage limits GitHub Actions' -> exit 75). Structured or long ok output is a
    role's answer; diagnostics are checked separately."""
    long_plan = json.dumps({"plan_md": "GitHub Actions usage limit: 2000 minutes quota " * 20,
                            "clusters": []})
    out = _codex_lines({"type": "item.completed",
                        "item": {"item_type": "agent_message", "text": long_plan}})
    r = cx.parse_output(out, 0)
    assert r.ok and cx.classify(r) == "done"
    assert ad.classify(RunResult(
        ok=True, text='{"answer":"usage limit quota"}', exit_code=0,
    )) == "done"
    # a short harness phrase with exit 0 is still a quota
    short = _codex_lines({"type": "item.completed",
                          "item": {"item_type": "agent_message", "text": "You have hit your usage limit."}})
    assert cx.classify(cx.parse_output(short, 0)) == "quota"
    assert ad.classify(RunResult(
        ok=True, text="plain usage limit " + "x" * 1000, exit_code=0,
    )) == "done"
    prefix = "plain usage limit "
    assert ad.classify(RunResult(
        ok=True, text=prefix + "x" * (399 - len(prefix)), exit_code=0,
    )) == "quota"
    assert ad.classify(RunResult(
        ok=True, text=prefix + "x" * (400 - len(prefix)), exit_code=0,
    )) == "done"
    # not-ok: quota is looked for in the whole text, as before
    r = cx.parse_output(out, 1)
    assert cx.classify(r) == "quota"
    # a quota in stderr/diag is visible even with a long ok answer
    r = cx.parse_output(out, 0)
    r.stderr = "usage limit reached"
    assert cx.classify(r) == "quota"
