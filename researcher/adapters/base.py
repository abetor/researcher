"""Harness port for LLM work through interchangeable subscription-backed agent CLIs.

The minimum harness contract is a headless one-shot process with a prompt and working
directory, a textual final response on stdout, and a distinguishable failure. Every
other feature is a capability flag. The shared wrapper owns behavior unavailable from
either harness: done/quota/transient/fatal classification from output patterns, quota
window handling by the scheduler, and logical checkpoints in the topic directory.
Harness sessions are mutually incompatible, so moving a job between harnesses depends
on researcher-owned state rather than a vendor session.

The adapter implementation descends from a separately owned CLI adapter design. It
keeps live-verified CLI flag behavior, removes coding-agent orchestration concerns,
and adds stop classification suitable for multi-day research runs.
"""
from __future__ import annotations

import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

# Patterns shared by the whole CLI harness class; an adapter adds its own via quota_patterns().
#
# CLASS BOUNDARY (production ladder bug 2026-08-10, see docs/STATUS.md). Before it, 429
# and "rate limit" lived in _QUOTA next to real quota exhaustion: a burst of parallel
# codex calls (2 runs x pool of 3) produced an ordinary 429, the classifier said "quota",
# the orchestrator returned exit 75 and the driver stopped the ENTIRE matrix - even though
# waiting a few seconds would have fixed it.
#
# _QUOTA - the SUBSCRIPTION budget is exhausted: the only cure is waiting for the reset
# window (hours to days). The marker is wording about the BUDGET itself, not about request
# rate. Observed forms: claude - "Claude usage limit reached ... resets at", "You've hit
# your weekly limit", "5-hour limit reached" (docs/phase1/L §"Behavior when hitting a
# rate limit / quota"); codex - "You've hit your usage limit", "out of credits"
# (docs/phase1/M §5). `(?<!rate[ -])\blimit (reached|exceeded)` is exactly that thin
# boundary: "usage limit exceeded" is quota, while "rate limit exceeded" (the classic
# throttling wording of OpenAI-compatible APIs) is not.
_QUOTA = re.compile(r"usage limit|weekly limit|session limit|5-hour|out of credits|quota"
                    r"|(?:hit|reached) your [^.\n]{0,24}limit"
                    r"|(?<!rate[ -])\blimit (?:reached|exceeded)", re.I)
# _TRANSIENT - cured by a RETRY within seconds to minutes. Burst throttling
# (429 / "too many requests" / "rate limit") moved here: per the docs of both harnesses
# this is a class separate from quota, and they retry it themselves with backoff (claude -
# CLAUDE_CODE_MAX_RETRIES, default 10; codex - its own attempts on a stream error), so it
# reaches us only after their attempts are exhausted.
# A mixed message ("429: usage limit reached") is resolved by the ORDER of checks in
# classify: quota is checked first and wins. That is deliberate - hammering a real quota
# is worse than waiting one extra time, and budget wording is more specific than the
# generic 429 code.
_TRANSIENT = re.compile(r"\b(?:429|500|502|503|529)\b|too many requests|rate.?limit"
                        r"|at capacity|server[_ -]?overloaded|\boverloaded\b|timed?.?out|connection|"
                        r"temporarily|try again|ECONNRESET|EAI_AGAIN", re.I)

# Engine roles live here, on the port BOUNDARY: they are known both to the orchestrator
# (which phase acts as which role) and to the adapter (which model id backs a role on its
# harness - VISION §7: "cheap/worker/synth roles + mapping in the adapter"). The names are
# ours, not the harnesses'. The orchestrator re-exports them so older imports keep working.
ROLE_THINK = "think"      # planning, adversarial critique, synthesis - where judgment is needed
ROLE_COLLECT = "collect"  # collecting sources and facts - mechanical work


@dataclass
class Capabilities:
    """Describe native harness features; the wrapper emulates only declared gaps."""
    json_events: bool
    schema_output: bool
    native_resume: bool
    subagents: bool
    mcp: bool
    # Below is what the collector profile is assembled from (tasks item 2). The defaults
    # are conservative: an unknown harness counts as "no web, no allowlist" - an honest
    # downgrade rather than a silent assumption that web exists (VISION §4, the
    # "as is -> as intended" diff).
    native_web: bool = False       # native WebSearch/WebFetch in the harness
    tool_allowlist: bool = False   # per-tool allowlist at the permission level (claude --allowedTools)
    # Physically narrowing the tool set is a DIFFERENT mechanism from a permission
    # allowlist: an allowlist only auto-approves calls, while an unlisted tool is still
    # attached (and visible to the model). In claude that is --tools (live probe
    # 2026-08-10: with it WebSearch disappears from the set and the model answers that
    # there is no such tool); codex has no such mechanism at all.
    static_tool_set: bool = False


@dataclass
class CollectProfile:
    """Describe how a collector can access external information on this harness.

    The harness-independent contract permits source tools and read access but forbids
    sub-agents. A harness with a per-tool allowlist receives a static allowed_tools
    value; a harness with a sandbox relies on that sandbox and leaves allowed_tools
    as None rather than claiming a capability it lacks.

    The tool set is static within a run. Every collector receives the same value,
    derived from parity mode and configured source tools rather than an individual
    task, because changing it per call breaks prefix caching.

    Two distinct layers are used when available. tools is the physical set exposed to
    the model, while allowed_tools identifies exposed tools that execute without a
    permission prompt. Codex provides neither layer, so its sandbox holds the boundary.
    """
    web: str                       # "native" - native WebSearch/WebFetch; "tool" - only our own web tool
    tools: Optional[str]           # physical tool set; None - the harness cannot do this
    allowed_tools: Optional[str]   # for a harness with an allowlist; None - no mechanism, the sandbox holds the boundary
    network: bool                  # whether the collector needs network access (the sandbox must open it)


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
    """Own per-CLI command construction and output parsing; share run execution."""
    name: str = "harness"

    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abstractmethod
    def build_cmd(self, prompt: str, *, model: Optional[str] = None,
                  effort: Optional[str] = None, resume_session_id: Optional[str] = None,
                  schema_path: Optional[str] = None, system_prompt_path: Optional[str] = None,
                  allowed_tools: Optional[str] = None, tools: Optional[str] = None,
                  network: bool = False) -> list[str]: ...

    @abstractmethod
    def parse_output(self, stdout: str, exit_code: int) -> RunResult: ...

    def collect_profile(self, *, source_cmd_prefix: Optional[str] = None,
                        parity: bool = False) -> CollectProfile:
        """Build the collector profile from capabilities.

        In parity mode native WebSearch and WebFetch are physically absent, so every
        harness uses the same source CLI and produces comparable corpora. The source
        command prefix also becomes a scoped Bash allowlist rule rather than granting
        arbitrary shell commands. Network access follows the effective profile: a
        collector with neither native web nor source tools does not need network access,
        and opening the Codex sandbox in that state would grant unjustified access.
        """
        caps = self.capabilities()
        native = caps.native_web and not parity
        names = ["WebSearch", "WebFetch"] if native else []
        # Only names that exist in the harness: a nonexistent one (LS in claude 2.1.226)
        # is silently ignored today, but a future version may turn it into an error or
        # give it different semantics. Directory listing is covered by Glob.
        names += ["Read", "Grep", "Glob"]
        if source_cmd_prefix:
            names.append("Bash")
        allowed = None
        if caps.tool_allowlist:
            allowed = " ".join(f"Bash({source_cmd_prefix}:*)" if n == "Bash" else n
                               for n in names)
        return CollectProfile(
            web="native" if native else "tool",
            tools=" ".join(names) if caps.static_tool_set else None,
            allowed_tools=allowed,
            network=native or source_cmd_prefix is not None)

    def default_model_for_role(self, role: str) -> Optional[str]:
        """Return this harness's default model for a role, or None for no preference.

        Concrete model IDs belong only in adapters because the engine operates on
        roles. An explicit role setting in RunConfig overrides this value. The base
        adapter knows no model names and therefore preserves legacy harness behavior.
        """
        return None

    def quota_patterns(self) -> Optional[re.Pattern]:
        """Return additional per-CLI quota patterns on top of shared patterns."""
        return None

    def classify(self, result: RunResult) -> str:
        """Classify a result as done, quota, transient, or fatal in contract order.

        Quota is checked first, including short unstructured text from an otherwise
        successful process, because a harness can exit 0 with a usage-limit message.
        Quota also wins a mixed message such as a 429 with explicit usage exhaustion.
        A remaining successful result is done before transient patterns are considered:
        a recovered 429 or source text discussing rate limits must not invalidate a
        successful run. Burst throttling and network failures are transient only for a
        failed call. Raw subtype, API status, fatal event errors, nonfatal item errors,
        and structured Codex error information are all part of the port contract.
        """
        diag = f"{result.stderr}\n{result.raw.get('subtype', '')}\n" \
               f"{result.raw.get('api_error_status', '')}\n{result.raw.get('error', '')}\n" \
               f"{result.raw.get('item_errors', '')}\n{result.raw.get('codex_error_info', '')}"
        text = str(result.text or "")
        blob = f"{text}\n{diag}"
        # On an ok result the model text resembles a harness message only when it is
        # short and does not look like a structured role answer. JSON inside a fence or
        # with a prose prefix may contain a subject-matter "usage limit"; quota is then
        # looked for in stderr/raw only. On a failed call the text stays diagnostic
        # regardless of its form.
        short_plain_ok = (
            result.ok and len(text) < 400 and "{" not in text and "```" not in text
        )
        quota_blob = blob if not result.ok or short_plain_ok else diag
        extra = self.quota_patterns()
        if _QUOTA.search(quota_blob) or (extra and extra.search(quota_blob)):
            return "quota"
        if result.ok:
            return "done"
        if _TRANSIENT.search(blob):
            return "transient"
        return "fatal"

    def run(self, prompt: str, *, cwd: str, timeout: Optional[float] = None,
            retries: int = 0, retry_backoff: float = 5.0,
            on_harness_pid: Optional[Callable[[Optional[int]], None]] = None,
            **build_kw) -> RunResult:
        """Invoke a harness with a bounded retry policy for transient failures.

        Only transient failures are retried: burst 429 responses, 5xx responses,
        network errors, and wall timeouts. Quota requires waiting for a reset window,
        while fatal failures are not repaired by repetition. Classification remains in
        classify rather than being redefined here.

        The default of zero retries preserves one-shot behavior. The caller chooses the
        policy through RunConfig. Execution belongs here because this is the only process
        spawn point; retrying an entire collection phase would repeat successful tasks.
        """
        delay = retry_backoff
        while True:
            res = self._run_once(
                prompt, cwd=cwd, timeout=timeout,
                on_harness_pid=on_harness_pid, **build_kw,
            )
            if res.stop != "transient" or retries <= 0:
                return res
            retries -= 1
            time.sleep(delay)
            delay *= 2

    def _run_once(self, prompt: str, *, cwd: str, timeout: Optional[float] = None,
                  on_harness_pid: Optional[Callable[[Optional[int]], None]] = None,
                  **build_kw) -> RunResult:
        """Invoke the harness once. The only place where a process is spawned."""
        cmd = self.build_cmd(prompt, **build_kw)
        proc = None
        try:
            # stdin=DEVNULL - rig mechanics ("never hang on input"): with inherited stdin
            # codex exec reads it to EOF ("Reading additional input from stdin...") and hangs.
            proc = subprocess.Popen(
                cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, stdin=subprocess.DEVNULL,
            )
            if on_harness_pid is not None:
                on_harness_pid(proc.pid)
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            assert proc is not None
            proc.kill()
            stdout, _ = proc.communicate()
            res = RunResult(ok=False, text=stdout or "", exit_code=-1,
                            stderr="wall-timeout", stop="transient")
            return res
        finally:
            if on_harness_pid is not None:
                on_harness_pid(None)
        res = self.parse_output(stdout, proc.returncode)
        res.stderr = stderr or ""
        res.stop = self.classify(res)
        if res.stop != "done":
            res.ok = False
        return res
