"""OpenAI Codex CLI adapter with live-verified flag behavior.

`codex exec [resume <id>] --json -c approval_policy=never --sandbox workspace-write|read-only
-c sandbox_workspace_write.network_access=true|false [--model M] [-c model_reasoning_effort=E] [-c model_instructions_file=PATH]
[--output-schema PATH] "prompt"` -> stdout is a JSONL event stream
(thread.started/thread_id, item.completed/agent_message, turn.completed/usage).

Verified constraints:
- headless execution is the `exec` subcommand; resume is `exec resume <id>`;
- resume rejects --sandbox, so sandbox mode is a configuration override;
- approval policy and reasoning effort are configuration overrides rather than flags;
- --output-schema accepts a file path, unlike Claude's inline schema;
- output schema and resume are incompatible;
- ChatGPT subscription mode rejects explicit Claude model names;
- Codex reports token usage rather than monetary cost, so usage remains in raw.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .base import ROLE_COLLECT, Capabilities, HarnessAdapter, RunResult

# claude-family aliases: full names (claude-*) and the short forms of the claude CLI.
_CLAUDE_ALIASES = {"sonnet", "opus", "haiku"}

# OWNER'S DECISION 2026-08-10: a hybrid, and the default of the ENGINE itself on codex
# rather than a driver flag - crawling the web is done by the light model, while synthesis
# into the knowledge corpus goes to the strong one. Collection runs on the lightweight
# model, thinking phases on the account default (sol for the owner): ROLE_THINK deliberately
# has no name of its own here - hardcoding a specific sol would break someone else's account
# with a different default, and a claude-family name coming from RunConfig is dropped by the
# adapter anyway. Overridden by the --model / --model-collect flags.
_ROLE_DEFAULT_MODEL = {ROLE_COLLECT: "gpt-5.6-luna"}

# Long stream. Thinking roles (synthesis over 800+ claims, a critic over 900 claims) stay
# silent in reasoning for more than 5 minutes; the built-in codex provider tears such a
# stream down on stream_idle_timeout_ms=300000, reconnects, and the server starts generating
# from scratch. Observed 22-23.08 (rollout files in ~/.codex/sessions): 6-7 reconnects per
# call at roughly 335 s intervals, a 5-minute answer stretched to 40-48, and the adapter
# rejected it on top of that (see parse_output). The built-in `openai` provider cannot be
# overridden (codex 0.148, live: "Built-in providers cannot be overridden"), hence a clone
# of the provider with a long idle timeout and generous retries - only for a ChatGPT login
# (auth_mode=chatgpt in $CODEX_HOME/auth.json; with an api key the clone would need env_key,
# which we do not do). To disable: RESEARCHER_CODEX_PROVIDER=builtin.
LONG_STREAM_PROVIDER = "openai-long"
LONG_STREAM_BASE_URL = "https://chatgpt.com/backend-api/codex"
STREAM_IDLE_TIMEOUT_MS = 1_800_000
STREAM_MAX_RETRIES = 20
REQUEST_MAX_RETRIES = 8


def _codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def long_stream_overrides() -> list[str]:
    """Return provider-clone overrides, or [] when unavailable or disabled."""
    if os.environ.get("RESEARCHER_CODEX_PROVIDER", "").lower() == "builtin":
        return []
    try:
        auth = json.loads((_codex_home() / "auth.json").read_text("utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(auth, dict) or auth.get("auth_mode") != "chatgpt":
        return []
    pid = LONG_STREAM_PROVIDER
    pairs = [
        (f"model_providers.{pid}.name", "OpenAI (long stream)"),
        (f"model_providers.{pid}.base_url", LONG_STREAM_BASE_URL),
        (f"model_providers.{pid}.wire_api", "responses"),
        (f"model_providers.{pid}.requires_openai_auth", "true"),
        (f"model_providers.{pid}.stream_idle_timeout_ms", str(STREAM_IDLE_TIMEOUT_MS)),
        (f"model_providers.{pid}.stream_max_retries", str(STREAM_MAX_RETRIES)),
        (f"model_providers.{pid}.request_max_retries", str(REQUEST_MAX_RETRIES)),
        ("model_provider", pid),
    ]
    out: list[str] = []
    for key, value in pairs:
        out += ["-c", f"{key}={value}"]
    return out


def _claude_family(model: str) -> bool:
    m = model.lower()
    return m.startswith("claude") or m in _CLAUDE_ALIASES


class CodexAdapter(HarnessAdapter):
    name = "codex"

    def __init__(self, binary: str = "codex"):
        self.binary = binary

    def capabilities(self) -> Capabilities:
        # subagents: TOML agents in .codex/agents/; mcp: client plus codex itself as a
        # server (M doc, codex page). native_web=False - codex has NO web tools at all (web
        # goes only through our source tool); tool_allowlist=False - there is no per-tool
        # allowlist, the sandbox is the boundary.
        return Capabilities(json_events=True, schema_output=True, native_resume=True,
                            subagents=True, mcp=True, native_web=False, tool_allowlist=False)

    def default_model_for_role(self, role: str):
        """Use the lightweight collection model and the account default for thinking."""
        return _ROLE_DEFAULT_MODEL.get(role)

    def build_cmd(self, prompt, *, model=None, effort=None, resume_session_id=None,
                  schema_path=None, system_prompt_path=None, allowed_tools=None,
                  tools=None, network=False):
        # allowed_tools and tools are IGNORED (as in rig): codex has neither a per-tool
        # allowlist nor a flag for the physical tool set - what is available is a function of
        # the sandbox mode. The profile knows this (both fields None) and does not invent a
        # string with nowhere to go.
        if resume_session_id and schema_path:
            # `codex exec resume` does NOT accept --output-schema (codex#14343, closed not
            # planned). Fail-closed, like an empty schema in the claude adapter: a constraint
            # must never be dropped silently. The caller picks: resume without a schema
            # (answer validated after the fact - the orchestrator runs _extract_json anyway)
            # OR a fresh session with a schema.
            raise ValueError("codex: schema and resume are incompatible (codex#14343) - pick one")
        # An explicitly empty tool set (tools="" - how course_import/pages shut tools off in
        # claude) means a read-only sandbox in codex: the only physical switch that can deny
        # writes to an agent whose prompt carries someone else's text (course transcripts).
        # In read-only the network is always closed, so network=True with tools="" is a
        # contradiction that must not be dropped silently (a collector with source tools
        # would end up without network access).
        read_only = tools == ""
        if read_only and network:
            raise ValueError("codex: tools='' (read-only sandbox) is incompatible with network=True")
        mode = "read-only" if read_only else "workspace-write"
        cmd = [self.binary, "exec"]
        if resume_session_id:
            cmd += ["resume", resume_session_id]  # resume is a subcommand (structurally != claude --resume)
            # `exec resume` does NOT accept --sandbox ("unexpected argument", verified live on
            # 0.137; the rig reference appends --sandbox after resume too - on 0.137 that
            # fails). The same mode goes through a config override; the form was verified by a
            # live smoke test 2026-08-05, read-only on 2026-08-28
            # (codex 0.150.1: `exec resume ... -c sandbox_mode=read-only`).
            sandbox = ["-c", f"sandbox_mode={mode}"]
        else:
            sandbox = ["--sandbox", mode]
        # Autonomous headless run: approval never + sandbox (the 'auto' mapping from rig).
        # approval_policy is a config override (-c), NOT an exec flag (verified on live codex
        # in rig).
        cmd += ["--json", "-c", "approval_policy=never", *sandbox]
        # Under workspace-write the network is CLOSED by default, and that breaks the
        # collector: source tools fail with a DNS error. Verified live on codex 0.147
        # (2026-08-10): without opening it, `... researcher.sources hn search` ->
        # "urlopen error [Errno 8] nodename nor servname provided"; with it the same call
        # returns JSON. We open the network ONLY for the collection phase - thinking phases
        # do not need it (the sandbox boundary is our R2b on codex). A closed network is
        # written EXPLICITLY rather than relying on the codex default: that default lives in
        # someone else's ~/.codex/config.toml and in the next CLI version, while the prompt of
        # the thinking phases carries someone else's text. Verified live on codex 0.150.1
        # (2026-08-28): with `=false`, curl from the sandbox reports "Could not resolve host",
        # and a write in read-only reports "Operation not permitted".
        cmd += ["-c", f"sandbox_workspace_write.network_access={'true' if network else 'false'}"]
        # This flag is absent from the rig reference (rig always works inside git repos): a
        # researcher topic directory is NOT a git repo, and without the flag codex refuses
        # with "Not inside a trusted directory". Verified live on codex 0.137
        # (smoke test 2026-08-05).
        cmd += ["--skip-git-repo-check"]
        cmd += long_stream_overrides()
        # A claude-family name (including the RunConfig default "sonnet") gives codex a 400
        # invalid_request on a ChatGPT subscription; omitting --model means the account
        # default. The fix belongs to the model_for_harness class of fixes in rig, where it
        # sits a layer above the adapter; researcher has no such layer, so the mapping lives
        # here (VISION §7: roles and model names are the adapter's business). An explicit
        # codex name is passed through unchanged.
        if model and not _claude_family(model):
            cmd += ["--model", model]
        if effort:
            cmd += ["-c", f"model_reasoning_effort={effort}"]  # config override, NOT --effort
        if system_prompt_path:
            cmd += ["-c", f"model_instructions_file={system_prompt_path}"]
        if schema_path:
            schema = Path(schema_path).read_text("utf-8").strip()
            if not schema:
                raise ValueError(f"empty schema {schema_path} - refusing an unconstrained call")
            cmd += ["--output-schema", schema_path]  # codex takes a PATH to the schema (claude takes it inline)
        cmd.append(prompt)
        return cmd

    def parse_output(self, stdout, exit_code) -> RunResult:
        """Parse the JSONL stream rather than Claude's single JSON object.

        Join text blocks from agent_message content, map thread_id to session_id, and
        retain usage in raw. Malformed lines are skipped; a stream with no valid JSON
        becomes an explicit unsuccessful result with its raw output.
        """
        text_parts: list[str] = []
        item_errors: list[str] = []
        usage = err = session_id = codex_error_info = None
        parsed_any = False
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(ev, dict):
                continue
            parsed_any = True
            t = ev.get("type")
            if t == "thread.started":
                session_id = ev.get("thread_id")  # the codex equivalent of session_id (for resume)
                continue
            item = ev.get("item") if isinstance(ev.get("item"), dict) else ev
            if ev.get("codex_error_info") or item.get("codex_error_info"):
                codex_error_info = ev.get("codex_error_info") or item.get("codex_error_info")
            itype = item.get("item_type") or item.get("type")
            if t == "item.completed" and itype == "agent_message":
                txt = item.get("text")
                if txt is None:
                    txt = item.get("content")
                if isinstance(txt, list):
                    txt = "".join(b.get("text", "") for b in txt if isinstance(b, dict))
                if txt:
                    text_parts.append(str(txt))
            elif t == "item.completed" and itype == "error":
                # An error ITEM (not a fatal top-level error event). On its own it does not
                # fail the run: codex 0.147 sends benign warnings here (with
                # --dangerously-bypass-hook-trust in the owner's config, two such items per
                # call), and turning every run fatal would be worse than the disease. But the
                # text must reach base.classify: a real quota arriving as an item would
                # otherwise quietly become "done" (checked by an in-process reviewer
                # 2026-08-10).
                msg = item.get("message") or item.get("text") or item.get("error")
                if msg:
                    item_errors.append(str(msg))
            elif t == "turn.completed":
                usage = ev.get("usage")
            elif t == "error" or ev.get("is_error"):
                err = str(ev.get("message") or ev.get("error") or "codex error")
        if not parsed_any:
            # non-JSONL (a crash before the format) - hand it back as is; classify will sort
            # it out from stderr
            return RunResult(ok=False, text=stdout or "", exit_code=exit_code)
        # item_errors do NOT take part in ok (see above). An error event with exit 0 AND an
        # agent_message that did arrive is a survived stream break (codex reconnected and
        # finished the turn), not a failure: before 23.08 such an answer was rejected outright
        # -> transient -> the role rerun, and a 40-minute synthesis spun for a whole day
        # without a single accepted answer. The error text stays in raw - the classifier sees
        # a quota even on an ok result.
        ok = exit_code == 0 and (err is None or bool(text_parts))
        # err and item_errors go into raw: base.classify includes both keys in the blob, so a
        # quota (whether it arrived as an event or as an item) is visible to the classifier
        # without smearing classification across adapters (the AGENTS rule: the classifier
        # lives only in adapters/base).
        return RunResult(ok=ok, text="\n".join(text_parts), exit_code=exit_code,
                         session_id=session_id, cost_usd=None,
                         raw={"usage": usage, "error": err or "",
                              "item_errors": "\n".join(item_errors),
                              "codex_error_info": codex_error_info or ""})
