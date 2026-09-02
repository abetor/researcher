"""OpenAI Codex CLI adapter with live-verified flag behavior.

``codex exec [resume <id>] --json`` emits a JSONL stream containing thread, item, and
usage events. The shared runner supplies ``stdin=DEVNULL`` to prevent inherited-input
hangs. ``--skip-git-repo-check`` permits non-repository working directories. Resume is
the ``exec resume`` subcommand and requires sandbox mode as a config override because it
does not accept ``--sandbox``. Approval policy and reasoning effort are also config
overrides. Output schemas are file paths and cannot be combined with resume. Claude
model aliases are omitted in subscription mode, and usage is recorded as tokens rather
than a monetary cost.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import Capabilities, HarnessAdapter, RunResult

# claude-family aliases: full names (claude-*) and the short forms of the claude CLI.
_CLAUDE_ALIASES = {"sonnet", "opus", "haiku"}


def _claude_family(model: str) -> bool:
    m = model.lower()
    return m.startswith("claude") or m in _CLAUDE_ALIASES


class CodexAdapter(HarnessAdapter):
    name = "codex"

    def __init__(self, binary: str = "codex"):
        self.binary = binary

    def capabilities(self) -> Capabilities:
        # subagents: TOML agents in .codex/agents/; mcp: client plus codex itself as a server.
        return Capabilities(json_events=True, schema_output=True, native_resume=True,
                            subagents=True, mcp=True)

    def build_cmd(self, prompt, *, model=None, effort=None, resume_session_id=None,
                  schema_path=None, system_prompt_path=None, allowed_tools=None):
        # allowed_tools is IGNORED: codex has no per-tool allowlist, what is available is a
        # function of the sandbox mode.
        if resume_session_id and schema_path:
            # Fail-closed, like an empty schema in claude: a constraint must never be
            # dropped silently. The caller picks: resume without a schema (validation after
            # the fact) or a fresh session with a schema.
            raise ValueError("codex: schema and resume are incompatible (codex#14343) - pick one")
        cmd = [self.binary, "exec"]
        if resume_session_id:
            cmd += ["resume", resume_session_id]  # a subcommand, not a --resume flag
            # `exec resume` does not accept --sandbox - the mode goes through a config override.
            sandbox = ["-c", "sandbox_mode=workspace-write"]
        else:
            sandbox = ["--sandbox", "workspace-write"]
        # Autonomous headless run: approval never + workspace-write.
        cmd += ["--json", "-c", "approval_policy=never", *sandbox]
        # The working directory (--home) is not a git repo; without the flag codex refuses with
        # "Not inside a trusted directory".
        cmd += ["--skip-git-repo-check"]
        # A claude-family name gives codex a 400 invalid_request on a ChatGPT subscription;
        # omitting --model means the account default. An explicit codex name is passed through.
        if model and not _claude_family(model):
            cmd += ["--model", model]
        if effort:
            cmd += ["-c", f"model_reasoning_effort={effort}"]  # a config override, not --effort
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
        """Parse JSONL events into text, session id, usage, and structured errors.

        Agent-message block text is joined rather than represented as a Python list.
        Malformed lines are skipped; a stream containing no valid JSON produces an
        explicit unsuccessful result with the raw output preserved.
        """
        text_parts: list[str] = []
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
            elif t == "turn.completed":
                usage = ev.get("usage")
            elif t == "error" or ev.get("is_error"):
                err = str(ev.get("message") or ev.get("error") or "codex error")
        if not parsed_any:
            # non-JSONL (a crash before the format) - hand it back as is; classify sorts it out from stderr
            return RunResult(ok=False, text=stdout or "", exit_code=exit_code)
        ok = exit_code == 0 and err is None
        # err and codex_error_info reach base.classify through raw: classification is not
        # smeared across adapters.
        return RunResult(ok=ok, text="\n".join(text_parts), exit_code=exit_code,
                         session_id=session_id, cost_usd=None,
                         raw={"usage": usage, "error": err or "",
                              "codex_error_info": codex_error_info or ""})
