"""Claude Code CLI adapter with live-verified flag behavior.

`claude -p --output-format json [...] -- "prompt"` -> stdout single-json:
{result, session_id, total_cost_usd, is_error, subtype, ...}.

Verified constraints:
- the schema is passed inline with --json-schema rather than as a path;
- '--' before the prompt is mandatory because --allowedTools consumes following values;
- resume uses --resume <session_id>;
- subscription execution has had silent API-billing reports, so usage is recorded under
  work/ and must be compared with the expected account mode.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .base import Capabilities, HarnessAdapter, RunResult

# Fallback tool set when the caller supplied neither a profile nor an explicit list:
# READ ONLY. Web is deliberately absent - "quieter by default": a bare build_cmd must not
# silently hand out WebSearch/WebFetch (in parity mode that would be a leak, and only the
# smoke test would catch it). Names are only the ones that exist in the harness (LS
# removed 2026-08-10: it is gone in 2.1.226).
READ_ONLY_TOOLS = "Read Grep Glob"

# Empty MCP configuration + --strict-mcp-config: the physical --tools set narrows only the
# BUILT-IN tools, while the user's MCP servers (Gmail/Calendar/Drive for the owner) stay
# connected. Live probe 2026-08-10: without these flags the collector's tool inventory held
# 34 mcp__claude_ai_* tools; with them it is exactly "Bash, Glob, Grep, Read".
_NO_MCP = '{"mcpServers":{}}'


class ClaudeAdapter(HarnessAdapter):
    name = "claude"

    def __init__(self, binary: str = "claude"):
        self.binary = binary

    def capabilities(self) -> Capabilities:
        return Capabilities(json_events=True, schema_output=True, native_resume=True,
                            subagents=True, mcp=True, native_web=True, tool_allowlist=True,
                            static_tool_set=True)

    def build_cmd(self, prompt, *, model=None, effort=None, resume_session_id=None,
                  schema_path=None, system_prompt_path=None, allowed_tools=None,
                  tools=None, network=False):
        # network is ignored deliberately: in claude, network access is a function of the
        # TOOL SET (WebSearch/WebFetch, an allowed Bash command); there is no separate
        # network sandbox. The parameter exists in the port contract for harnesses that do
        # have a sandbox (codex).
        #
        # permission-mode dontAsk is the closed mode: only what is explicitly allowed runs,
        # everything else is denied (in default mode an unlisted call is merely "asked
        # about", which headless turns into a de facto denial - but the contract is weaker).
        # Live probe 2026-08-10 on dontAsk: curl, nslookup and the compound command
        # "<allowed prefix> ; touch pwned.txt" were denied (permission_denials, no file
        # created) while the prefix itself ran. REMAINING GAP: local read-only commands
        # (ls -la) are auto-approved by the harness even in dontAsk - see docs/STATUS.md.
        cmd = [self.binary, "-p", "--output-format", "json",
               "--permission-mode", "dontAsk"]
        if model:
            cmd += ["--model", model]
        allowed = allowed_tools or READ_ONLY_TOOLS
        cmd += ["--allowedTools", allowed]
        if tools is not None:
            # PHYSICAL set: what is not here does not exist for the model (unlike an
            # allowlist). The form was verified live 2026-08-10: names separated by COMMAS
            # (as in --help); WebSearch left out of the list disappears from the model
            # entirely ("that tool is not in my set") rather than being denied by
            # permissions. A nonexistent name is silently ignored by the harness (checked
            # with LS in 2.1.226) - so a typo in the set shows up nowhere except a live
            # inventory probe; that is why the sets are built from existing names only.
            cmd += ["--tools", ",".join(tools.split()),
                    "--strict-mcp-config", "--mcp-config", _NO_MCP]
        if effort:
            cmd += ["--effort", effort]
        if system_prompt_path:
            cmd += ["--append-system-prompt-file", system_prompt_path]
        if resume_session_id:
            cmd += ["--resume", resume_session_id]
        if schema_path:
            schema = Path(schema_path).read_text("utf-8").strip()
            if not schema:
                raise ValueError(f"empty schema {schema_path} - refusing an unconstrained call")
            cmd += ["--json-schema", schema]
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
