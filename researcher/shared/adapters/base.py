"""Harness port for running LLM work through interchangeable subscription CLIs.

This repository owns its vendored copy. CLI flags and output patterns reflect live
verification and should not change without another live check.

The minimum contract is a headless one-shot process receiving a prompt and cwd, a
textual final answer on stdout, and an observable failure. Capability flags describe
everything else so the wrapper can emulate missing behavior without pretending it is
native. This module owns done/quota/transient/fatal classification because no machine
quota endpoint exists; classification relies on process output.
"""
from __future__ import annotations

import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

# Patterns shared by the whole CLI harness class; an adapter adds its own via quota_patterns().
#
# CLASS BOUNDARY (production bug in tool-researcher 2026-08-10). Before it, 429 and
# "rate limit" lived in _QUOTA next to real quota exhaustion: a burst of our own
# parallelism (2 runs x pool of 3 = up to 6 simultaneous codex calls) produced an ordinary
# 429, the classifier said "quota", the CLI returned exit 75, and the external driver
# dropped everything else from the queue - even though waiting a few seconds would have
# fixed it. Classes must be split by WHAT CURES THEM, not by what they are called.
#
# _QUOTA - the SUBSCRIPTION budget is exhausted: the only cure is waiting for the reset
# window (hours to days). The marker is wording about the BUDGET itself, not about request
# rate. Observed forms: claude - "Claude usage limit reached ... resets at", "You've hit
# your weekly limit", "5-hour limit reached"; codex - "You've hit your usage limit",
# "out of credits". `(?<!rate[ -])\blimit (reached|exceeded)` is exactly that thin
# boundary: "usage limit exceeded" is quota, while "rate limit exceeded" (the classic
# throttling wording of OpenAI-compatible APIs) is not.
_QUOTA = re.compile(r"usage limit|weekly limit|session limit|5-hour|out of credits|quota"
                    r"|(?:hit|reached) your [^.\n]{0,24}limit"
                    r"|(?<!rate[ -])\blimit (?:reached|exceeded)", re.I)
# _TRANSIENT - cured by a RETRY within seconds to minutes. Burst throttling
# (429 / "too many requests" / "rate limit") belongs here too: per the docs of both
# harnesses this is a class separate from quota, and they retry it themselves with backoff
# (claude - CLAUDE_CODE_MAX_RETRIES, default 10; codex - its own attempts on a stream
# error), so it reaches us only after their attempts are exhausted and the right reaction
# is to wait some more rather than stop with a checkpoint.
# A mixed message ("429: usage limit reached") is resolved by the ORDER of checks in
# classify: quota is checked first and wins. That is deliberate - hammering a real quota
# is worse than waiting one extra time, and budget wording is more specific than the
# generic 429 code.
_TRANSIENT = re.compile(r"\b(?:429|500|502|503|529)\b|too many requests|rate.?limit"
                        r"|at capacity|try a different model|server[_ -]?overloaded|\boverloaded\b|timed?.?out|connection|"
                        r"temporarily|try again|ECONNRESET|EAI_AGAIN", re.I)

# Mapping from a stop reason to CLI exit codes (the shared workspace contract, see cli.py).
STOP_TO_EXIT = {"done": 0, "quota": 75, "transient": 111, "fatal": 1}


@dataclass
class Capabilities:
    """Describe native harness features; the wrapper emulates the rest."""
    json_events: bool
    schema_output: bool
    native_resume: bool
    subagents: bool
    mcp: bool


@dataclass
class RunResult:
    ok: bool
    text: str
    exit_code: int
    session_id: Optional[str] = None
    stop: str = "done"                 # done | quota | transient | fatal
    cost_usd: Optional[float] = None
    raw: dict = field(default_factory=dict)
    stderr: str = ""


class HarnessAdapter(ABC):
    """Own per-CLI command construction and parsing while sharing ``run()``."""
    name: str = "harness"

    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abstractmethod
    def build_cmd(self, prompt: str, *, model: Optional[str] = None,
                  effort: Optional[str] = None, resume_session_id: Optional[str] = None,
                  schema_path: Optional[str] = None, system_prompt_path: Optional[str] = None,
                  allowed_tools: Optional[str] = None) -> list[str]: ...

    @abstractmethod
    def parse_output(self, stdout: str, exit_code: int) -> RunResult: ...

    def quota_patterns(self) -> Optional[re.Pattern]:
        """Return per-CLI quota indicators in addition to shared patterns."""
        return None

    def classify(self, result: RunResult) -> str:
        """Classify a result as done, quota, transient, or fatal in contract order.

        Quota checks come first even on exit zero because a harness may report a usage
        limit in otherwise successful output. A clean successful process then wins over
        generic throttling text: the harness may already have retried, and collected
        material can legitimately discuss rate limits. Transient throttling and network
        errors apply only to failed calls. Structured ``raw`` fields preserve the
        provider-specific error signals used by this port.
        """
        blob = f"{result.text}\n{result.stderr}\n{result.raw.get('subtype', '')}\n" \
               f"{result.raw.get('api_error_status', '')}\n{result.raw.get('error', '')}\n" \
               f"{result.raw.get('codex_error_info', '')}"
        extra = self.quota_patterns()
        if _QUOTA.search(blob) or (extra and extra.search(blob)):
            return "quota"
        if result.ok:
            return "done"
        if _TRANSIENT.search(blob):
            return "transient"
        return "fatal"

    def run(self, prompt: str, *, cwd: str, timeout: Optional[float] = None,
            **build_kw) -> RunResult:
        """Run one harness call; this is the only place that spawns a process."""
        cmd = self.build_cmd(prompt, **build_kw)
        try:
            # stdin=DEVNULL - a production fix: with inherited stdin, codex exec reads it
            # to EOF ("Reading additional input from stdin...") and hangs.
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                                  timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") if isinstance(e.stdout, str) else ""
            return RunResult(ok=False, text=out, exit_code=-1,
                             stderr="wall-timeout", stop="transient")
        res = self.parse_output(proc.stdout, proc.returncode)
        res.stderr = proc.stderr or ""
        res.stop = self.classify(res)
        if res.stop != "done":
            res.ok = False
        return res
