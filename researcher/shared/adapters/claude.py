"""Claude Code CLI adapter with live-verified flag behavior.

``claude -p --output-format json [...] -- "prompt"`` emits one JSON object containing
``result``, ``session_id``, ``total_cost_usd``, ``is_error``, and ``subtype``.

The JSON schema is passed inline rather than by path. ``--`` before the prompt is
required because ``--allowedTools`` otherwise consumes the next positional argument.
Resume uses ``--resume <session_id>``. Subscription usage should still be monitored
because upstream issue 43333 documents unexpected API-style billing behavior.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .base import Capabilities, HarnessAdapter, RunResult

# Default whitelist: web and reading are available, writing into the project is not (a
# physical read-only guarantee, not a promise). Pass your own allowed_tools for an agent
# that needs to write.
READ_ONLY_TOOLS = "WebSearch WebFetch Read Grep Glob LS"


class ClaudeAdapter(HarnessAdapter):
    name = "claude"

    def __init__(self, binary: str = "claude"):
        self.binary = binary

    def capabilities(self) -> Capabilities:
        return Capabilities(json_events=True, schema_output=True, native_resume=True,
                            subagents=True, mcp=True)

    def build_cmd(self, prompt, *, model=None, effort=None, resume_session_id=None,
                  schema_path=None, system_prompt_path=None, allowed_tools=None):
        cmd = [self.binary, "-p", "--output-format", "json",
               "--permission-mode", "default"]
        if model:
            cmd += ["--model", model]
        # Under mode=default a tool OUTSIDE the whitelist will not run in non-interactive
        # -p mode (there is nobody to ask) - the whitelist is the boundary of what the call
        # can do.
        cmd += ["--allowedTools", allowed_tools or READ_ONLY_TOOLS]
        if effort:
            cmd += ["--effort", effort]
        if system_prompt_path:
            # our own system prompt as a separate file, WITHOUT clobbering the project CLAUDE.md
            cmd += ["--append-system-prompt-file", system_prompt_path]
        if resume_session_id:
            cmd += ["--resume", resume_session_id]
        if schema_path:
            schema = Path(schema_path).read_text("utf-8").strip()
            if not schema:
                raise ValueError(f"empty schema {schema_path} - refusing an unconstrained call")
            cmd += ["--json-schema", schema]  # claude wants the schema inline (codex takes a path)
        cmd += ["--", prompt]
        return cmd

    def parse_output(self, stdout, exit_code) -> RunResult:
        try:
            data = json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            # non-json (a crash before the format was produced) - hand it back as is
            return RunResult(ok=exit_code == 0 and bool(stdout.strip()),
                             text=stdout, exit_code=exit_code)
        ok = exit_code == 0 and not data.get("is_error", False)
        return RunResult(ok=ok, text=data.get("result") or "", exit_code=exit_code,
                         session_id=data.get("session_id"),
                         cost_usd=data.get("total_cost_usd"), raw=data)

    def quota_patterns(self) -> re.Pattern:
        # subtype in the json answer when subscription/API limits are hit (observed forms)
        return re.compile(r"error_max_turns|resets at|upgrade to|claude usage", re.I)
