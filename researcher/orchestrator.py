"""Run orchestrator built on the harness port and durable checkpoint.

Phases are plan, collect, verify, synthesize, and done. Every phase resumes from
work/checkpoint.json rather than a harness session. Quota or interruption leaves a
valid checkpoint, and resume continues from the same durable point.

Harness failure classification exists only in adapters/base.classify; this module maps
RunResult.stop to exit codes. Knowledge is written only through the injected llmwiki
contract, while collection candidates stay under work/found/. Only collection runs in
parallel, with a pool of at most three and no nested sub-agents; synthesis is one call.
Executable stop gates combine max_cycles with saturation after rounds without new
claims. Only the main thread mutates llmwiki sequentially; worker threads run harness
subprocesses and return results, preventing races in sources.jsonl.

Process exits are 0 for done, 76 for plan review, 75 for quota, 77 for a refused gate,
111 for transient failure, and 1 for fatal failure. Separate review and gate codes avoid
confusing a normal pause or refusal with success or breakage.
"""
from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from . import checkpoint
from .adapters.base import ROLE_COLLECT, ROLE_THINK, HarnessAdapter, RunResult
from .observability import (
    RoleHeartbeat,
    completed_funnel_from_disk,
    empty_funnel,
    search_map_progress,
    topic_counts,
)

EXIT_DONE = 0
EXIT_PLAN_REVIEW = 76
EXIT_QUOTA = 75
EXIT_GATE_REFUSED = 77
EXIT_TRANSIENT = 111
EXIT_FATAL = 1
_EXIT = {"done": EXIT_DONE, "plan_review": EXIT_PLAN_REVIEW, "quota": EXIT_QUOTA,
         "gate_refused": EXIT_GATE_REFUSED,
         "transient": EXIT_TRANSIENT, "fatal": EXIT_FATAL}

_JUDGE_VERDICT_LABELS = {
    "continue": "continue",
    "enough": "enough",
    "stuck": "stuck",
}

_STOP_REASON_LABELS = {
    "max_cycles": "cycle cap reached",
    "saturated": "search saturated",
    "enough": "the judge decided there is enough data",
    "stuck": "the judge considers the search stuck",
}


def _round_tasks_with_map_priority(
    queue: list, pool: int, pending_map_ids: set[str]
) -> list:
    """Round tasks: the approved map is executed in full before any generated gaps.

    Gaps keep their priority between the already traversed map and the later saturation
    tasks. But while the queue still holds at least one map cluster, the collection round
    covers every such cluster. They run in slices no wider than `pool`, so the physical
    parallelism limit does not change. A partial round is already durable in the
    checkpoint: a quota stop or an interruption continues the rest of the map instead of
    restarting it.

    This policy is stronger than the earlier reservation of a single slot: at pool=1 that
    slot did not move the map at all, and at pool>=2 it lost to a stream of 3-10 new gaps
    per cycle. A full first pass guarantees that every approved cluster runs before the
    first gap cycle.
    """
    map_tasks = [task for task in queue if task.get("id") in pending_map_ids]
    return map_tasks if map_tasks else queue[:pool]


class GateRefused(ValueError):
    """Nothing can be published because both final and staging are empty."""

_MAX_POOL = 3
_CONF = {"high", "medium", "low"}
_STANCE = {"supports", "contradicts", "mentions"}
_KIND = {"web", "youtube", "telegram", "paper", "book", "repo", "dataset", "local", "other"}
_SUBSTRATE_SECTIONS = (
    "## What is known",
    "## Why it matters",
    "## What it is for",
    "## Confidence and why",
    "## Alternatives",
)
_MIN_PAGE_DOMAINS = 2
_MIN_PAGE_KINDS = 2
_MIN_TOPIC_DOMAINS = 2
_MIN_TOPIC_KINDS = 3
_SEARCH_MAP_REL = Path("work/search-map.json")
_SEARCH_MAP_VERSION = 1
_MAP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RUN_MODE_EXTEND = "extend"
_STRUCTURAL_PROTOCOL_REMINDER = (
    "Return EXACTLY one JSON object: no plan, no explanations, no multiple objects."
)

# The collector tool set is no longer an engine constant: it is built from the adapter
# capability profile (adapters/base.CollectProfile) - claude has a static set plus an
# allowlist, codex has neither (its boundary is the sandbox). Task/Agent are never included:
# nested fan-out is forbidden. Planner, critic and synthesizer work on already collected
# material, so they need no web access.
THINK_TOOLS = "Read Grep Glob"

# Engine roles for model/effort routing. The names are OURS, not the harness's: the engine
# knows "who thinks and who collects", while the model-id behind a role is a matter of run
# config and of the adapter (VISION §7: "roles cheap/worker/synth plus a mapping inside the
# adapter"). The constants themselves live in adapters/base (the port boundary, known to
# both sides); this is only a re-export.

# Who answered in a phase - for diagnosing a structural parse failure (codex review block-2).
# The key is the name of the raw-answer file under work/bad-answers/, the value is the role
# named in the error text.
_WHO = {
    "plan": "planner", "collect": "collector", "verify": "critic",
    "judge": "judge", "synth": "synthesizer",
}
_HEARTBEAT_ROLE = {
    "plan": "plan", "collect": "collect", "verify": "critic", "judge": "judge",
    "synth": "synth",
}
_VERDICT_BATCH = 150
_JUDGE_CLAIMS = 60
_JUDGE_COVERAGE_WORDS = 1500
_JUDGE_GAPS = 3
_JUDGE_CLUSTERS = 2
_JUDGE_OPEN_GAPS = 20
_JUDGE_EXHAUSTED_GAPS = 10
_JUDGE_QUESTIONS = 5
_JUDGE_PROMPT_BYTES = 40_000


def _structured_answer_policy(
    prompt: str,
    *,
    call,
    bad_answer,
    retries: int,
    retry_backoff: float,
    schema: Optional[dict[str, type]] = None,
    validator=None,
    role_label: str,
    first_result: Optional[RunResult] = None,
) -> tuple[RunResult, Optional[dict], Optional[str]]:
    """Shared bounded retry policy for any role returning one structured object."""

    retries_left = retries
    delay = retry_backoff
    result = first_result
    retrying = False
    while True:
        if result is None:
            call_prompt = prompt + (
                "\n\n" + _STRUCTURAL_PROTOCOL_REMINDER if retrying else ""
            )
            result = call(call_prompt)
        if result.stop != "done":
            return result, None, None
        try:
            data = _extract_json(result.text)
            if schema is not None:
                data = _role_payload(data, role_label, schema)
            if validator is not None:
                validator(data)
            return result, data, None
        except (ValueError, json.JSONDecodeError) as error:
            diagnostic = bad_answer(result.text, error)
            if retries_left <= 0:
                return result, None, diagnostic
            retries_left -= 1
            time.sleep(delay)
            delay *= 2
            retrying = True
            result = None


def _sources_block(names: list[str], profile, prefix: str) -> str:
    """The "how to reach the world" block of the collector prompt - BUILT FROM THE PROFILE,
    not from an assumption that "the web is there".

    profile.web='native' means the harness has WebSearch/WebFetch, they are the default and
    source tools only add to them; 'tool' means there is no native web (codex) or it was
    switched off by parity mode: all web access goes through our web source tool, and if it
    was not wired in we say so to the collector honestly instead of pretending it has web
    access. The prompt does not lie about the ban on other Bash commands either: on a harness
    with an allowlist they physically will not run, on a harness with a sandbox the boundary
    rests on the collector itself - and that is what the text says. The case "no native web
    AND no web tool" gets its own header branch: hedging with "if it is in the list below"
    would be dishonest.
    """
    from . import sources  # local import: the source layer is stdlib-only
    # On claude native web stays the default and our tool is the fallback (owner decision §9.1).
    fallback = ("; the web source tool below is a fallback if native search finds nothing"
                if "web" in names else "")
    if profile.web == "native":
        head = (f"Search the web and fetch pages with the NATIVE WebSearch/WebFetch tools "
                f"(this is the default{fallback}).\n")
    elif "web" in names:
        head = ("You have NO native web search (a harness without web, or acceptance parity "
                "mode): reach the web ONLY through the web source tool listed below.\n")
    else:
        # Neither native web nor a web tool: there is NO general web access AT ALL. Say it
        # directly instead of hedging with "if it is in the list below" - the collector must
        # not go looking for a path that does not exist.
        head = ("You have NO general web access AT ALL: no native search and no web source "
                "tool.\n")
    if not names:
        return head + ("No external source tools are wired in - work with what you have.\n"
                       if profile.web == "tool" else "")
    enforced = ("everything else is forbidden by harness permissions and simply will not run"
                if profile.allowed_tools else
                "the harness does not block them, so the boundary here rests on you")
    return (
        f"{head}"
        "Available source tools:\n"
        f"{sources.describe(names)}\n"
        "Call them with a Bash command using STRICTLY this prefix (run no other Bash command -\n"
        f"{enforced}):\n"
        f'  {prefix} <tool> search "<query>"\n'
        f'  {prefix} <tool> fetch "<url>"\n'
        "search prints JSON [{url,title,snippet}], fetch prints markdown. In a claim put the "
        "source url into evidence and the name of the tool you used into sources[].tool. Skip "
        "an irrelevant source, do not force it in.\n"
    )


@dataclass
class RunConfig:
    """Run parameters controlling quality gates and execution cost."""
    model: Optional[str] = "sonnet"
    # The CLI must tell the common default apart from an explicit --model: an explicit value
    # means the measurement scenario "this model in every role" and outranks the adapter's
    # per-role default. A direct library call sets this flag itself.
    model_explicit: bool = False
    # Engine roles (VISION §5b): thinking phases (plan/critic/synthesis) get the strong
    # model, collection gets the cheap one. The role is an ENGINE abstraction; the mapping
    # from role to model-id lives outside (flag/config) and inside the adapter (VISION §7:
    # no model-id is hardcoded here). None = the role is not split out and collection runs
    # on the same model as the thinking phases (the old behavior).
    model_collect: Optional[str] = None
    # Reasoning effort: before this wave it was not passed to the adapter AT ALL - the
    # harness config default was picked up silently. It is split by the same roles: where a
    # harness cannot switch models, effort is the only remaining control left for a role
    # (owner answer, VISION §9.3).
    effort: Optional[str] = None
    effort_collect: Optional[str] = None
    max_cycles: int = 3               # cap on collection rounds (hard gate)
    saturation_rounds: int = 2        # K rounds in a row without a new claim -> enough
    pool_size: int = 3                # parallel collectors; clamped to <= 3
    timeout: Optional[float] = None   # wall timeout of one harness call, seconds
    # Bounded retry of a transient failure and of a structural role failure inside one run:
    # before these waves there was no retry AT ALL - a single 429 spike left the run short of
    # the finish until an external restart. Only transient failures are retried (quota is
    # handled by waiting for the window, which is the scheduler's job); the mechanics live in
    # base.run, the policy lives here. Two retries at 5s/10s is a cap, not a "keep trying
    # until it works".
    retries: int = 2
    retry_backoff: float = 5.0
    # Source tools for collectors: empty = only the harness native web, no Bash.
    # The library default is conservative (empty); the CLI turns them on (--sources).
    sources: list = field(default_factory=list)
    # Acceptance parity mode (VISION §9.1): native web is switched off even where it exists,
    # and both sides are REQUIRED to search only with our source tools - otherwise the claude
    # and codex corpora are not comparable. Only claude enforces it physically (the tool set);
    # on codex it is enforced by the prompt.
    parity: bool = False
    # An interactive run stops after plan by default. Nightly jobs must enable auto mode
    # explicitly (`--auto-plan`), otherwise it is exit 76 with the map awaiting approval.
    auto_plan: bool = False

    def pool(self) -> int:
        return max(1, min(_MAX_POOL, self.pool_size))

    def for_role(self, role: str, *,
                 harness_default: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
        """Map an engine role to the model and effort passed to the harness.

        COLLECT covers mechanical evidence gathering and may use a cheaper model. THINK
        covers planning, adversarial review, and synthesis, where judgment quality has
        greater impact.

        Model precedence is an explicit role flag, an explicitly supplied common model,
        the adapter's role default, then the implicit common default. Without a role flag
        or adapter default, the role inherits the common value for backward-compatible
        behavior. An unknown role raises ``ValueError`` instead of silently selecting an
        expensive thinking model after a typo.
        """
        if role == ROLE_COLLECT:
            model = self.model_collect
            if model is None and self.model_explicit:
                model = self.model
            if model is None:
                model = harness_default
            if model is None:
                model = self.model
            return model, self.effort_collect or self.effort
        if role == ROLE_THINK:
            model = self.model if self.model_explicit else None
            if model is None:
                model = harness_default
            if model is None:
                model = self.model
            return model, self.effort
        raise ValueError(f"unknown engine role: {role!r} "
                         f"(only {ROLE_THINK!r} and {ROLE_COLLECT!r} exist)")


def _now_id() -> str:
    return "run_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def _run_work_rel(state: dict) -> Path:
    """Work area of the current run; initial keeps the historical paths without migration."""
    if state.get("run_mode") == _RUN_MODE_EXTEND and isinstance(state.get("run_id"), str):
        return Path("work") / "runs" / state["run_id"]
    return Path("work")


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".",
            suffix=".tmp", delete=False) as fh:
        fh.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        tmp = Path(fh.name)
    os.replace(tmp, path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".",
            suffix=".tmp", delete=False) as fh:
        fh.write(text)
        tmp = Path(fh.name)
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    _write_text_atomic(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _begin_extend_locked(topic_dir: str | Path) -> tuple[str, bool]:
    """Start a new delta run of a completed topic, or resume an extend already started.

    Returns `(run_id, created)`. An ordinary unfinished run is not taken over: it has to be
    continued with `resume`. The checkpoint history of the previous done run is kept both in
    the `run_history` array and as a separate immutable snapshot next to its work artifacts.
    """
    root = Path(topic_dir)
    state = checkpoint.load(root)
    if state is None:
        raise ValueError(f"no checkpoint - there is nothing to extend: {root}")
    if state.get("phase") != "done":
        if state.get("run_mode") == _RUN_MODE_EXTEND and isinstance(state.get("run_id"), str):
            return state["run_id"], False
        raise ValueError(
            f"extend only runs on a done topic; the current phase={state.get('phase')!r}. "
            "Continue the current run with resume")
    current_work = root / _run_work_rel(state)
    pending = current_work / "synthesis-pending.json"
    if pending.exists():
        raise ValueError(
            f"the done topic contains pending {pending.relative_to(root)}; run resume first "
            "to close the crash window, then extend")
    previous_run_id = state.get("run_id") if isinstance(state.get("run_id"), str) \
        else "legacy_initial"
    history = list(state.get("run_history")) if isinstance(state.get("run_history"), list) else []
    if not any(isinstance(item, dict) and item.get("run_id") == previous_run_id
               for item in history):
        previous_funnel = state.get("funnel")
        if not isinstance(previous_funnel, dict):
            previous_funnel = completed_funnel_from_disk(root)
        previous = {
            "run_id": previous_run_id,
            "run_mode": state.get("run_mode", "initial"),
            "phase": "done",
            "work_dir": str(_run_work_rel(state)),
            "funnel": previous_funnel,
            "rounds": state.get("rounds", []),
            "stop": state.get("stop"),
            "breadth": state.get("breadth", {}),
            "completed_at": state.get("updated_at"),
        }
        history.append(previous)
        snapshot = root / "work" / "runs" / previous_run_id / "checkpoint-final.json"
        _write_json_atomic(snapshot, previous)
    run_id = _now_id()
    work_rel = Path("work") / "runs" / run_id
    (root / work_rel).mkdir(parents=True, exist_ok=False)
    next_state = {
        "phase": "planned",
        "topic": state["topic"],
        "queue": [],
        "done": [],
        "sessions": {},
        "run_id": run_id,
        "run_mode": _RUN_MODE_EXTEND,
        "parent_run_id": previous_run_id,
        "run_history": history,
        "funnel": empty_funnel(),
        "rounds": [],
        "gaps": [],
        "gap_history_version": 1,
        "breadth": {"topic": {}, "pages": {}},
        "cycles": 0,
        "rounds_without_claim": 0,
        "structural_failures": 0,
    }
    checkpoint.save(root, next_state)
    return run_id, True


def begin_extend(topic_dir: str | Path) -> tuple[str, bool]:
    """Serialize extend initialization with the same local lock used by resume."""
    root = Path(topic_dir)
    path = root / "work" / "resume.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise ValueError(
                f"the topic is already running in another process: lock {path}. "
                "Extend cannot start while a resume is in progress") from e
        except OSError as e:
            raise ValueError(
                f"filesystem does not support locking; is the topic on a network filesystem? "
                f"lock {path}: {e}") from e
        try:
            return _begin_extend_locked(root)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _is_plan_noise(value: Any) -> bool:
    """Narrow signatures of the collector preamble seen in the wild (11 real bad answers).

    The collector sometimes prints a housekeeping object BEFORE the answer. Exactly two
    shapes were observed live, both single-key: `{"plan": [{"step": "...", ...}, ...]}` - a
    work plan (once) and `{"status": "searching"}` - a progress ping (three times). Only
    those are discarded: any other object still counts as a standalone one, and the "take
    the last one" heuristic is forbidden - it silently picks a draft instead of the answer.
    """
    if not isinstance(value, dict) or len(value) != 1:
        return False
    if set(value) == {"status"}:
        return isinstance(value["status"], str) and bool(value["status"].strip())
    if set(value) != {"plan"}:
        return False
    steps = value["plan"]
    return bool(steps) and isinstance(steps, list) and all(
        isinstance(step, dict)
        and isinstance(step.get("step"), str)
        and bool(step["step"].strip())
        for step in steps
    )


def _truncation_error(text: str, start: int, err: ValueError) -> Optional[ValueError]:
    """Tells a TRUNCATED answer apart from a stray '{' inside prose.

    The live shape (3 cases out of 11 in one night): the harness returned an answer without
    finishing the JSON, so parsing from the first '{' fails exactly at the END of the text.
    The scanner then starts collecting inner fragments of that same truncated object
    (sources/claims elements) and reports "TWO standalone objects" - a diagnosis that leads
    away from the cause. Here the cause is named for what it is; the cure is the same role
    retry.
    """
    m = re.search(r"char (\d+)", str(err))
    if not m:
        return None
    pos = int(m.group(1))
    if pos < len(text.rstrip()) - 2:
        return None  # a failure in the middle of the text is ordinary junk, not truncation
    return ValueError(
        f"the harness answer is TRUNCATED: parsing JSON from position {start} fails at the "
        f"end of the text ({len(text.rstrip())} characters) - the model did not finish the "
        f"object. Cause: {err}")


def _extract_json(text: str) -> dict:
    """Extract a JSON OBJECT from the harness answer text (strips ``` fences and prose).

    The policy (codex review fix-1) is exactly one object:
    - if the whole answer parses as JSON and is an object, that is the answer (normal case);
    - otherwise scan the text for COMPLETE objects: exactly one means we take it, and an
      incomplete tail after it is allowed and discarded (the known shape of a live bug - a
      glued-on duplicate `],"summary_md":"..."}` that silently turned 52 claims into nothing);
    - a narrowly recognizable housekeeping plan object is discarded before counting;
    - any other SECOND complete object is fail-closed: which of them is the answer is
      unknown, and the "take the last one" heuristic is forbidden;
    - valid JSON that is not an object (array, scalar) fails here rather than as an
      AttributeError on `.get` somewhere further down the phase.

    Why not the old wide slice "first '{' .. last '}'": a stray '{' in the prose BEFORE the
    answer broke parsing of the valid JSON that followed. The scan looks for the object
    where it actually is.
    """
    t = text.strip()
    if t.startswith("```"):
        t = t[3:]
        if t[:4].lower() == "json":
            t = t[4:]
        t = t.rsplit("```", 1)[0]
    try:
        whole = json.loads(t)
    except ValueError:
        pass
    else:
        if not isinstance(whole, dict):
            raise ValueError(f"the harness answer is valid JSON but not an object "
                             f"({type(whole).__name__}); the phase protocol requires an object")
        return whole
    dec = json.JSONDecoder()
    found = None
    i = t.find("{")
    first = True
    while i != -1:
        try:
            value, end = dec.raw_decode(t, i)
        except ValueError as err:
            if first:
                # The first '{' is the most likely start of the real answer. If parsing
                # from it fails at the end of the text, the answer is truncated and the
                # inner fragments below are pieces of it, not a "second object".
                truncated = _truncation_error(t, i, err)
                if truncated is not None:
                    raise truncated
            first = False
            i = t.find("{", i + 1)  # not an object's '{' (prose, junk) - keep looking
            continue
        first = False
        if _is_plan_noise(value):
            i = t.find("{", end)
            continue
        if found is not None:
            raise ValueError("the harness answer contains TWO standalone JSON objects - "
                             "which one is the answer is unknown (the protocol requires "
                             "exactly one)")
        found, i = value, t.find("{", end)
    if found is None:
        raise ValueError("the harness answer contains no JSON object")
    return found


def _norm_conf(v: Any, default: str = "low") -> str:
    return v if v in _CONF else default


def _claim_text_key(text: Any) -> str:
    """Exact dedup layer of the wiki socket: lowercase plus whitespace collapsing."""
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _claim_verdict_key(text: Any) -> str:
    """Durable critic key built on the same normalization as the exact dedup."""
    return hashlib.sha256(_claim_text_key(text).encode("utf-8")).hexdigest()


def _one_sentence(value: Any) -> str:
    text = " ".join(str(value or "").split())
    match = re.match(r".*?[.!?](?:\s|$)", text)
    return (match.group(0).strip() if match else text)[:300]


def _model_line(value: Any, limit: Optional[int] = None) -> str:
    """Model text for one Markdown line: collapse controls, strip heading prefix."""
    text = re.sub(r"(?m)^[ \t]*#+[ \t]*", "", str(value or ""))
    text = " ".join(text.split())
    return _truncate_text(text, limit) if limit is not None else text


def _as_list(v: Any) -> list:
    """A list field from the model answer. Anything else (string, number, null) yields an
    empty list rather than a crash: structural validity of the ANSWER is checked while
    parsing, the shape of a single field is checked here."""
    return v if isinstance(v, list) else []


def _as_dicts(v: Any) -> list[dict]:
    """The same for a list of objects: non-object elements are dropped (else .get fails)."""
    return [x for x in _as_list(v) if isinstance(x, dict)]


def _model_gaps(v: Any) -> list[dict]:
    """Normalize model gaps, dropping fields reserved for the compiler.

    The model only supplies the content of a task. Origin, gate, scope and measured
    thresholds are decisions made by the code and must never arrive from a harness JSON
    answer. The allowlist here is deliberately narrower than the current `_queue_gaps`
    schema, so new compiler housekeeping fields cannot become trusted by accident.
    """
    return [
        {key: gap[key] for key in ("query", "reason", "priority") if key in gap}
        for gap in _as_dicts(v)
    ]


def _without_system_section(body: str, title: str) -> str:
    """Remove every previous instance of a reserved H2 section, leaving neighbors alone."""
    heading = re.escape(title)
    return re.sub(
        rf"(?ms)^##[ \t]+{heading}[ \t]*(?:\n|\Z).*?(?=^##[ \t]+|\Z)",
        "",
        body,
    ).rstrip()


def _append_system_section(body: str, section: str) -> str:
    clean = body.rstrip()
    prefix = clean + "\n\n" if clean else ""
    return prefix + section.rstrip() + "\n"


def _truncate_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip()


def _truncate_words(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    words = list(re.finditer(r"\S+", text))
    return text if len(words) <= limit else text[:words[limit - 1].end()].rstrip()


def _demote_markdown_headings(value: Any) -> str:
    """Prevent model markdown from breaking apart a reserved system section."""
    return re.sub(
        r"(?m)^\s*#{1,6}\s+(.+?)\s*$",
        lambda match: f"**{match.group(1).strip()}**",
        str(value or ""),
    )


def _role_payload(data: dict, role: str, schema: dict[str, type]) -> dict:
    """Check the mandatory top-level fields of a particular role's answer.

    A valid JSON object does not yet mean a valid phase answer: `{}` from the collector used
    to mark a task as completed, and `{}` from the critic meant "drop everything". Such
    answers now take the same fail-closed path as syntactic junk, bad-answers/ included.
    """
    errors = []
    for key, expected in schema.items():
        if key not in data:
            errors.append(f"missing required field {key!r}")
        elif not isinstance(data[key], expected):
            errors.append(
                f"field {key!r}: expected {expected.__name__}, got {type(data[key]).__name__}")
    if errors:
        raise ValueError(f"the {role!r} answer does not match the schema: " + "; ".join(errors))
    return data


def _validate_collector_sources(data: dict) -> None:
    """Check the mandatory publication date in the collector answer.

    `published_at` is an ISO-8601 date/time or null. Null without a non-empty reason is
    forbidden: otherwise a missing date is again indistinguishable from a forgotten field.
    The usual reason is `not found`, but a specific honest diagnosis is kept as written.
    The check lives in the code, not only in the prompt.
    """
    for index, source in enumerate(data["sources"]):
        if not isinstance(source, dict):
            raise ValueError(
                f"sources[{index}]: expected object, got {type(source).__name__}")
        if "published_at" not in source:
            raise ValueError(f"sources[{index}]: missing required field 'published_at'")
        published_at = source["published_at"]
        reason = source.get("published_at_reason")
        if published_at is None:
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"sources[{index}]: published_at=null requires a non-empty "
                    "published_at_reason")
            continue
        if not isinstance(published_at, str) or not published_at.strip():
            raise ValueError(
                f"sources[{index}]: published_at must be an ISO-8601 string or null")
        value = published_at.strip()
        try:
            if len(value) == 10:
                datetime.date.fromisoformat(value)
            else:
                datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(
                f"sources[{index}]: published_at={published_at!r} is not ISO-8601") from e


class Orchestrator:
    """Run one topic against an injected llmwiki implementation.

    The wiki socket supplies ``add_source``, ``add_claim``, ``DuplicateClaim``,
    ``promote_claim``, ``add_wiki_page``, and ``update_wiki_page``. Synthesis no longer
    inserts final claims directly; it promotes verified staging claims instead. One
    harness adapter serves every phase in the current implementation.
    """

    def __init__(self, topic_dir: str | Path, *, adapter: HarnessAdapter, wiki,
                 config: Optional[RunConfig] = None, notifier=None):
        self.dir = Path(topic_dir)
        self.adapter = adapter
        self.wiki = wiki
        self.cfg = config or RunConfig()
        self.notifier = notifier
        self._notice_started_at = time.monotonic()
        self._notice_cycle_count = 0
        initial_state = checkpoint.load(self.dir) or {}
        self._run_mode = initial_state.get("run_mode", "initial")
        self._is_extend = self._run_mode == _RUN_MODE_EXTEND
        self._run_id = initial_state.get("run_id")
        self._work_rel = _run_work_rel(initial_state)
        self._work_dir = self.dir / self._work_rel
        self._found_dir = self._work_dir / "found"
        self._heartbeat = RoleHeartbeat(self.dir)
        # The collector profile is computed ONCE per run: every collector must get the same
        # tool set (prefix cache), and the prompt must be built from the profile rather than
        # from an assumption that "the web is there".
        from . import sources as sources_pkg  # local import: the source layer is stdlib-only
        self._sources = list(self.cfg.sources or [])
        prefix = sources_pkg.cmd_prefix()
        self._profile = self.adapter.collect_profile(
            source_cmd_prefix=prefix if self._sources else None, parity=self.cfg.parity)
        if self.cfg.parity and "web" not in self._sources:
            # Parity without our web tool is a contradiction: native web is off and there is
            # no replacement. The gate is executable, not a wish expressed in the prompt.
            raise ValueError("parity mode requires the web source tool (--sources ...,web): "
                             "native web access is disabled in this mode")
        self._sources_block = _sources_block(self._sources, self._profile, prefix)

    # --- housekeeping ----------------------------------------------------

    def _state(self) -> dict:
        st = checkpoint.load(self.dir)
        if st is None:
            raise FileNotFoundError(
                f"no checkpoint in {self.dir} - this topic has not been started (run start)")
        return st

    def _call(self, prompt: str, *, role: str, tools: Optional[str],
              allowed_tools: Optional[str], heartbeat_role: str,
              network: bool = False) -> RunResult:
        """One harness call. The stop class is already set by base.classify; we leave it be.

        role is the engine role (ROLE_THINK/ROLE_COLLECT): it selects the model and effort
        (cfg.for_role), and the adapter IS ASKED for its own default for that role
        (default_model_for_role) - codex has one by owner decision, and a role flag overrides
        it. An adapter that cannot switch model/effort ignores them honestly - on codex, for
        example, claude model names fall back to the account default.
        tools is the PHYSICAL tool set (--tools on claude), allowed_tools is the permission
        allowlist. A harness adapter without such a mechanism simply ignores them.
        network says whether network access is needed (it opens the sandbox where there is
        one: codex workspace-write blocks the network by default). Thinking phases need none.
        """
        model, effort = self.cfg.for_role(
            role, harness_default=self.adapter.default_model_for_role(role))
        state = self._state()
        active_round = state.get("collecting_round")
        cycle = active_round.get("cycle") if isinstance(active_round, dict) else None
        if not isinstance(cycle, int):
            cycle = state.get("cycles") if isinstance(state.get("cycles"), int) else None
        token = self._heartbeat.start(
            phase=str(state.get("phase")), role=heartbeat_role, cycle=cycle,
        )
        try:
            return self.adapter.run(
                prompt, cwd=str(self.dir), timeout=self.cfg.timeout,
                retries=self.cfg.retries, retry_backoff=self.cfg.retry_backoff,
                model=model, effort=effort, tools=tools,
                allowed_tools=allowed_tools, network=network,
                on_harness_pid=lambda pid: self._heartbeat.harness_pid(token, pid),
            )
        finally:
            self._heartbeat.finish(token)

    def _bad_answer(self, st: dict, kind: str, text: str, err: Exception, *,
                    task: Optional[str] = None) -> str:
        """A structural parse failure of a harness answer: keep the raw text, build a diagnosis.

        Why NOT `except -> empty value` (codex review block-2): an empty dict is
        indistinguishable from an honest "found nothing" and silently destroys work - that is
        how the collector marked a task done without data, and how the critic rejected ALL
        candidates and pushed the checkpoint into the next phase. A structural error must
        leave the task in the queue and the phase in the checkpoint, and put the cause on disk.

        Why the raw answer goes to work/bad-answers/: the evidence for the tail-duplicate bug
        in a synthesizer answer had to be dug out of ~/.codex/sessions rollouts because the
        engine kept NOT A SINGLE line. Now every attempt gets its own suffix and does not
        overwrite the previous one; its number comes from a durable checkpoint counter.
        Returns the text for the ValueError (the CLI prints it to stderr and exits 1).
        """
        d = self._work_dir / "bad-answers"
        d.mkdir(parents=True, exist_ok=True)
        attempt = st.get("structural_failures", 0)
        attempt = attempt if isinstance(attempt, int) and attempt >= 0 else 0
        attempt += 1
        stem = f"{kind}-{task}" if task else kind
        path = d / f"{stem}-attempt-{attempt}.txt"
        while path.exists():
            attempt += 1
            path = d / f"{stem}-attempt-{attempt}.txt"
        st["structural_failures"] = attempt
        checkpoint.save(self.dir, st)
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        path.write_text(f"# {stamp} {kind}{' ' + task if task else ''}: {err}\n{text}", "utf-8")
        return (f"the '{_WHO.get(kind, kind)}' answer"
                f"{f' (task {task})' if task else ''} does not parse structurally: {err}. "
                f"The raw answer is saved to {path.relative_to(self.dir)} "
                f"({len(text)} characters). Such an answer is NOT replaced by an empty value "
                "- the work stayed where it was (checkpoint in the current phase, unfinished "
                "tasks in the queue), and resume will repeat exactly this role")

    def _structured_answer(
            self, st: dict, prompt: str, *, kind: str, role: str,
            tools: Optional[str], allowed_tools: Optional[str],
            task: Optional[str] = None, schema: Optional[dict[str, type]] = None,
            validator=None, network: bool = False,
            first_result: Optional[RunResult] = None,
    ) -> tuple[RunResult, Optional[dict], Optional[str]]:
        """Call one role until it returns a valid object or the structural retry cap is hit.

        Neither the strict parser nor the role schema is weakened: every rejection
        immediately writes the raw text and bumps the counter in the checkpoint. The same
        role is repeated with the same model, effort and tools; only an explicit protocol
        reminder is added. The cap/backoff policy is the same as for transient failures:
        RunConfig.retries/retry_backoff.
        """
        return _structured_answer_policy(
            prompt,
            call=lambda call_prompt: self._call(
                call_prompt,
                role=role,
                tools=tools,
                allowed_tools=allowed_tools,
                heartbeat_role=_HEARTBEAT_ROLE[kind],
                network=network,
            ),
            bad_answer=lambda text, error: self._bad_answer(
                st, kind, text, error, task=task
            ),
            retries=self.cfg.retries,
            retry_backoff=self.cfg.retry_backoff,
            schema=schema,
            validator=validator,
            role_label=_WHO.get(kind, kind),
            first_result=first_result,
        )

    def _found_files(self) -> list[Path]:
        if not self._found_dir.exists():
            return []
        return sorted(self._found_dir.glob("*.jsonl"))

    def _read_candidates(self, *, exclude_tasks: Optional[set[str]] = None) -> list[dict]:
        rows: list[dict] = []
        for f in self._found_files():
            if exclude_tasks and f.stem in exclude_tasks:
                continue
            for line in f.read_text("utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def _progress(self, phase: str, *, cycle: Optional[int] = None,
                  task: Optional[str] = None, task_status: Optional[str] = None) -> None:
        """Write one live progress line to stderr, preserving stdout for the CLI contract."""
        counts = topic_counts(self.dir)
        parts = [f"phase={phase}"]
        if cycle is not None:
            parts.append(f"cycle={cycle} of {self.cfg.max_cycles}")
        if task is not None:
            suffix = f" ({task_status})" if task_status else ""
            parts.append(f"task={task}{suffix}")
        if counts["map_total"]:
            parts.append(f"map={counts['map_completed']} of {counts['map_total']}")
        parts.extend([
            f"claims found={counts['found']} staging={counts['staging']} "
            f"final={counts['final']}",
            f"sources={counts['sources']}",
            f"pages={counts['pages']}",
        ])
        print("progress: " + " | ".join(parts), file=sys.stderr, flush=True)

    @staticmethod
    def _observability(st: dict) -> tuple[dict, list]:
        """Add the fields to both new and old checkpoints without a separate migration."""
        funnel = st.setdefault("funnel", empty_funnel())
        rounds = st.setdefault("rounds", [])
        return funnel, rounds

    @staticmethod
    def _breadth(st: dict) -> dict:
        """Add the durable breadth snapshot to both new and old checkpoints."""
        return st.setdefault("breadth", {"topic": {}, "pages": {}})

    def _breadth_signature(self) -> dict:
        breadth = self._topic_breadth()
        return {"domains": breadth["domains"], "kinds": breadth["kinds"]}

    def _event(self, event: str, *, phase: str, **payload) -> None:
        """Append-only observability log kept next to the checkpoint."""
        path = self.dir / "work" / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            "phase": phase,
            "run_id": self._run_id,
            **payload,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _notify(self, event: str, text: str, **notice) -> None:
        if self.notifier is not None:
            self.notifier.emit(event, text, **notice)

    def _cycle_minutes(self) -> int:
        """Average wall time of the cycles this process has completed."""
        self._notice_cycle_count += 1
        elapsed = max(0.0, time.monotonic() - self._notice_started_at)
        return max(1, round(elapsed / self._notice_cycle_count / 60))

    def _verdicts_path(self) -> Path:
        return self._work_dir / "verdicts.jsonl"

    def _load_verdict_rows(self) -> list[dict]:
        path = self._verdicts_path()
        if not path.exists():
            return []
        rows = []
        for line_no, line in enumerate(path.read_text("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as e:
                raise ValueError(
                    f"corrupt {path.relative_to(self.dir)}:{line_no}: {e}") from e
            if not isinstance(row, dict) \
                    or not isinstance(row.get("key"), str) \
                    or row.get("verdict") not in {"keep", "drop"}:
                raise ValueError(
                    f"corrupt {path.relative_to(self.dir)}:{line_no}: needs verdict keep|drop "
                    "and a string key")
            rows.append(row)
        return rows

    def _verdict_index(self) -> dict[str, dict]:
        """The last row for a key wins; the writer itself never creates duplicates."""
        return {row["key"]: row for row in self._load_verdict_rows()}

    def _append_verdict_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        current = self._load_verdict_rows()
        by_key = {row["key"]: row for row in current}
        order = [row["key"] for row in current]
        for row in rows:
            if row["key"] not in by_key:
                order.append(row["key"])
            by_key[row["key"]] = row
        _write_jsonl_atomic(self._verdicts_path(), [by_key[key] for key in order])

    def _seed_legacy_verdicts(self, st: dict, cands: list[dict]) -> None:
        """First read of a legacy topic: what is already in staging is not re-criticized."""
        path = self._verdicts_path()
        if path.exists():
            return
        staged = {
            _claim_text_key(claim.get("text"))
            for claim in self._read_zone_claims("staging")
        }
        now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        seeded = []
        seen = set()
        for cand in cands:
            normalized = _claim_text_key(cand.get("text"))
            key = _claim_verdict_key(cand.get("text"))
            if normalized not in staged or key in seen:
                continue
            seen.add(key)
            seeded.append({
                "key": key,
                "verdict": "keep",
                "confidence": _norm_conf(cand.get("confidence")),
                "cycle": st.get("cycles", 0),
                "run_id": st.get("run_id"),
                "at": now,
                "seeded": True,
            })
        # Even an empty file is a durable marker that the legacy seed has completed.
        _write_jsonl_atomic(path, seeded)

    def _reconcile_kept_candidates(
            self, st: dict, cands: list[dict], verdicts: dict[str, dict]) -> None:
        """The verdict is written before staging; resume closes the crash window idempotently.

        The same fact backed by a new source is appended to the stored claim through the
        socket's `merge_claim_evidence` (llm-wiki). Each piece of evidence is merged
        separately: one broken FK does not roll back the valid entries of the batch. A socket
        without that API (an older checkout) keeps the previous fallback: a durable
        `evidence_merged_pending` event, and we never write around the socket. A merge failure
        does not fail the run and is recorded in full as `evidence_merge_failed`.
        """
        merge = getattr(self.wiki, "merge_claim_evidence", None)
        existing: dict[str, dict] = {}
        for zone in ("staging", "final"):
            for claim in self._read_zone_claims(zone):
                normalized = _claim_text_key(claim.get("text"))
                row = existing.setdefault(normalized, {
                    "id": claim.get("id"), "zones": [], "source_ids": set(),
                })
                row["zones"].append(zone)
                row["source_ids"].update(self._candidate_source_ids(claim))
        pending = set()
        events_path = self.dir / "work" / "events.jsonl"
        try:
            event_lines = events_path.read_text("utf-8").splitlines()
        except OSError:
            event_lines = []
        for line in event_lines:
            try:
                event = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(event, dict) and event.get("event") == "evidence_merged_pending" \
                    and isinstance(event.get("id"), str) \
                    and isinstance(event.get("source_id"), str):
                pending.add((event["id"], event["source_id"]))
        seen = set()
        for cand in cands:
            key = _claim_verdict_key(cand.get("text"))
            if key in seen:
                continue
            seen.add(key)
            verdict = verdicts.get(key)
            if not isinstance(verdict, dict) or verdict.get("verdict") != "keep":
                continue
            stored = existing.get(_claim_text_key(cand.get("text")))
            if stored is not None:
                claim_id = stored.get("id")
                fresh: list[dict] = []
                for evidence in _as_dicts(cand.get("evidence")):
                    source_id = evidence.get("source_id")
                    marker = (claim_id, source_id)
                    if not isinstance(claim_id, str) or not isinstance(source_id, str) \
                            or source_id in stored["source_ids"] or marker in pending \
                            or any(row.get("source_id") == source_id for row in fresh):
                        continue
                    fresh.append(evidence)
                if not fresh:
                    continue
                if merge is None:
                    for evidence in fresh:
                        self._event(
                            "evidence_merged_pending", phase="verifying", id=claim_id,
                            source_id=evidence["source_id"], zones=stored["zones"],
                            evidence=evidence,
                        )
                        pending.add((claim_id, evidence["source_id"]))
                    continue
                # final is authoritative: after a promote the same id in both zones means a
                # crash-window promote, so the final row is the one to grow.
                zone = "final" if "final" in stored["zones"] else "staging"
                for evidence in fresh:
                    source_id = evidence["source_id"]
                    try:
                        merge(self.dir, zone, claim_id, [evidence])
                    except Exception as exc:  # noqa: BLE001 - socket failure must not fail the cycle
                        self._event(
                            "evidence_merge_failed", phase="verifying", id=claim_id,
                            zone=zone, source_ids=[source_id], evidence=evidence,
                            error=str(exc),
                        )
                        continue
                    stored["source_ids"].add(source_id)
                    self._event(
                        "evidence_merged", phase="verifying", id=claim_id, zone=zone,
                        source_ids=[source_id],
                    )
                continue
            conf = _norm_conf(verdict.get("confidence"), cand.get("confidence"))
            conf, single_source = _single_source_confidence(conf, cand.get("evidence"))
            self._add_claim(
                "staging", "verified", cand["text"], conf, cand["evidence"],
                st.get("run_id"), meta={"single_source": single_source},
            )

    @staticmethod
    def _merge_candidate_evidence(cands: list[dict]) -> list[dict]:
        """One text is criticized together with every source_id found for it."""
        merged: dict[str, dict] = {}
        source_ids: dict[str, set[str]] = {}
        for cand in cands:
            key = _claim_verdict_key(cand.get("text"))
            if key not in merged:
                merged[key] = {**cand, "evidence": []}
                source_ids[key] = set()
            for evidence in _as_dicts(cand.get("evidence")):
                source_id = evidence.get("source_id")
                if not isinstance(source_id, str) or source_id in source_ids[key]:
                    continue
                source_ids[key].add(source_id)
                merged[key]["evidence"].append(evidence)
        return list(merged.values())

    @staticmethod
    def _candidate_source_ids(cand: dict) -> list[str]:
        return sorted({
            evidence["source_id"] for evidence in _as_dicts(cand.get("evidence"))
            if isinstance(evidence.get("source_id"), str)
        })

    @staticmethod
    def _gap_source(source: Any) -> str:
        if source == "judge":
            return "judge"
        if source in {"synth", "synthesis", "synthesizer"}:
            return "synth"
        return "legacy"

    def _ensure_gap_history(self, st: dict) -> list[dict]:
        """Migrate the old open-gaps snapshot into full history without a manual migration."""
        if st.get("gap_history_version") == 1:
            gaps = st.setdefault("gaps", [])
            return gaps if isinstance(gaps, list) else []
        merged: dict[str, dict] = {}
        raw_rows: list[dict] = []
        events_path = self.dir / "work" / "events.jsonl"
        try:
            event_lines = events_path.read_text("utf-8").splitlines()
        except OSError:
            event_lines = []
        for line in event_lines:
            try:
                event = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(event, dict) and event.get("event") == "gaps_queued":
                raw_rows.extend(_as_dicts(event.get("gaps")))
        raw_rows.extend(_as_dicts(st.get("gaps")))
        done_ids = {item for item in st.get("done", []) if isinstance(item, str)}
        for raw in raw_rows:
            query = str(raw.get("base_query") or raw.get("query") or "").strip()
            if not query:
                continue
            norm = _claim_text_key(query)
            task_id = str(raw.get("id") or (
                "gap_" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:10]))
            row = merged.setdefault(norm, {
                "id": "gap_" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:10],
                "query": query,
                "reason": str(raw.get("reason") or "legacy gap"),
                "priority": raw.get("priority") if raw.get("priority") in {
                    "high", "medium", "low"} else "medium",
                "source": self._gap_source(raw.get("source")),
                "status": "done" if task_id in done_ids else "open",
                "attempts": 0,
                "task_ids": [],
                "completed_task_ids": [],
                "claims_brought": 0,
                "empty_attempts": 0,
            })
            if task_id not in row["task_ids"]:
                row["task_ids"].append(task_id)
                row["attempts"] += 1
                found = self._found_dir / f"{task_id}.jsonl"
                claims = sum(1 for line in found.read_text("utf-8").splitlines()
                             if line.strip()) if found.is_file() else 0
                row["claims_brought"] += claims
                if task_id in done_ids and claims == 0:
                    row["empty_attempts"] += 1
                if task_id in done_ids:
                    row["completed_task_ids"].append(task_id)
            if raw.get("status") == "exhausted":
                row["status"] = "exhausted"
        st["gaps"] = list(merged.values())
        st["gap_history_version"] = 1
        checkpoint.save(self.dir, st)
        return st["gaps"]

    def _complete_gap_task(self, st: dict, task: dict, new_claims: int) -> None:
        gap_id = task.get("gap_id")
        if not isinstance(gap_id, str):
            return
        history = self._ensure_gap_history(st)
        row = next((gap for gap in history if gap.get("id") == gap_id), None)
        if row is None:
            return
        task_id = task.get("id")
        completed = row.setdefault("completed_task_ids", [])
        if not isinstance(task_id, str) or task_id in completed:
            return
        completed.append(task_id)
        row["claims_brought"] = int(row.get("claims_brought", 0)) + new_claims
        if new_claims:
            row["status"] = "done"
            row["empty_attempts"] = 0
        else:
            row["empty_attempts"] = int(row.get("empty_attempts", 0)) + 1
            row["status"] = (
                "exhausted" if row["empty_attempts"] >= 2 else "done"
            )

    def _queue_gaps(self, st: dict, raw_gaps: list, *, source: str) -> bool:
        """Add directions to the durable history and put new attempts at the head."""
        priority_order = {"high": 0, "medium": 1, "low": 2}
        history = self._ensure_gap_history(st)
        by_query = {_claim_text_key(gap.get("query")): gap for gap in history}
        queued = []
        queued_ids = {
            str(task.get("id")) for task in _as_dicts(st.get("queue"))
            if isinstance(task.get("id"), str)
        }
        queued_queries = {
            _claim_text_key(task.get("query")) for task in _as_dicts(st.get("queue"))
            if str(task.get("query") or "").strip()
        }
        incoming_queries = set()
        for raw in _as_dicts(raw_gaps):
            query = str(raw.get("query") or "").strip()
            reason = str(raw.get("reason") or "").strip()
            if not query or not reason:
                continue
            priority = raw.get("priority") if raw.get("priority") in priority_order else "medium"
            normalized = _claim_text_key(query)
            if normalized in incoming_queries:
                continue
            incoming_queries.add(normalized)
            gap = by_query.get(normalized)
            if gap is not None and gap.get("status") == "exhausted":
                if source == "judge":
                    self._event(
                        "gap_rejected_exhausted", phase="verifying", query=query,
                        cycle=st.get("cycles", 0),
                    )
                continue
            if gap is None:
                gap_id = "gap_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]
                gap = {
                    "id": gap_id, "query": query, "reason": reason,
                    "priority": priority, "source": self._gap_source(source),
                    "status": "done", "attempts": 0, "task_ids": [],
                    "completed_task_ids": [], "claims_brought": 0,
                    "empty_attempts": 0,
                }
                history.append(gap)
                by_query[normalized] = gap
            if gap.get("status") == "open":
                task_id = gap.get("task_ids", [])[-1]
            else:
                attempt = int(gap.get("attempts", 0)) + 1
                task_id = gap["id"] if attempt == 1 else f"{gap['id']}_a{attempt}"
                gap.update(
                    reason=reason, priority=priority, source=self._gap_source(source),
                    status="open", attempts=attempt,
                )
                gap.setdefault("task_ids", []).append(task_id)
            if task_id in queued_ids or normalized in queued_queries:
                continue
            queued_ids.add(task_id)
            queued_queries.add(normalized)
            queued.append({"id": task_id, "query": query, "gap_id": gap["id"]})
        if not queued:
            checkpoint.save(self.dir, st)
            return False
        if self._collection_stop_reason(st) is not None:
            checkpoint.save(self.dir, st)
            return False
        # A continuation really was accepted: the old terminal diagnosis no longer describes
        # the checkpoint and must not mislead an external watchdog.
        st.pop("stop_kind", None)
        st.pop("stop_detail", None)
        queued = sorted(queued, key=lambda task: (
            priority_order[next(gap for gap in history if gap["id"] == task["gap_id"])[
                "priority"]], task["id"]))
        gap_ids = {task["id"] for task in queued}
        new_ids = {task["id"] for task in queued}
        new_queries = {_claim_text_key(task["query"]) for task in queued}
        st["queue"] = queued + [
            task for task in _as_dicts(st.get("queue"))
            if task.get("id") not in new_ids
            and _claim_text_key(task.get("query")) not in new_queries
        ]
        st["phase"] = "collecting"
        checkpoint.save(self.dir, st)
        self._event("gaps_queued", phase=source, gaps=[
            next(gap for gap in history if gap["id"] == task["gap_id"])
            for task in queued
        ], queue=list(st["queue"]))
        return True

    def _record_synth_gaps(self, st: dict, raw_gaps: list) -> None:
        """Synthesis reports its limitations but never starts a new collection round."""
        history = self._ensure_gap_history(st)
        by_query = {_claim_text_key(gap.get("query")): gap for gap in history}
        recorded = []
        for raw in _as_dicts(raw_gaps):
            query = str(raw.get("query") or "").strip()
            reason = str(raw.get("reason") or "").strip()
            if not query or not reason:
                continue
            normalized = _claim_text_key(query)
            gap = by_query.get(normalized)
            if gap is None:
                gap = {
                    "id": "gap_" + hashlib.sha256(
                        normalized.encode("utf-8")).hexdigest()[:10],
                    "query": query,
                    "reason": reason,
                    "priority": raw.get("priority")
                    if raw.get("priority") in {"high", "medium", "low"} else "medium",
                    "attempts": 0,
                    "task_ids": [],
                    "completed_task_ids": [],
                    "claims_brought": 0,
                    "empty_attempts": 0,
                }
                history.append(gap)
                by_query[normalized] = gap
            gap.update(source="synth", status="unqueued", reason=reason)
            recorded.append(gap)
        checkpoint.save(self.dir, st)
        if recorded:
            self._event("gaps_unqueued", phase="synthesizing", gaps=recorded)

    def _collection_stop_reason(self, st: dict) -> Optional[str]:
        """A safety net above the judge: two empty accepted rounds, or the cycle cap."""
        if st.get("cycles", 0) >= self.cfg.max_cycles:
            return "max_cycles"
        if st.get("rounds_without_claim", 0) >= self.cfg.saturation_rounds:
            return "saturated"
        return None

    def _coverage_path(self) -> Path:
        return self._work_dir / "coverage.md"

    def _coverage_text(self) -> str:
        try:
            text = self._coverage_path().read_text("utf-8")
        except OSError:
            return ""
        return _truncate_words(text, _JUDGE_COVERAGE_WORDS)

    def _accepted_claims_for_cycle(
            self, cands: list[dict], verdicts: dict[str, dict], cycle: int) -> list[dict]:
        priority = {"high": 0, "medium": 1, "low": 2}
        rows = []
        seen = set()
        for position, cand in enumerate(cands):
            key = _claim_verdict_key(cand.get("text"))
            verdict = verdicts.get(key)
            if key in seen or not isinstance(verdict, dict) \
                    or verdict.get("verdict") != "keep" \
                    or verdict.get("seeded") is True \
                    or verdict.get("cycle") != cycle:
                continue
            seen.add(key)
            rows.append({
                "text": cand.get("text"),
                "confidence": _norm_conf(verdict.get("confidence"), cand.get("confidence")),
                "_position": position,
            })
        return sorted(rows, key=lambda row: (
            priority[row["confidence"]], row["_position"]))

    def _record_accepted_round(self, st: dict, accepted: int) -> None:
        cycle = st.get("cycles", 0)
        rounds = st.setdefault("rounds", [])
        row = next((item for item in reversed(rounds)
                    if isinstance(item, dict) and item.get("cycle") == cycle), None)
        if row is not None and "accepted_claims" not in row:
            row["accepted_claims"] = accepted
        if accepted:
            st["rounds_without_claim"] = 0
        elif row is not None and row.get("accepted_claims") == 0:
            previous = rounds[:-1] if rounds and rounds[-1] is row else [
                item for item in rounds if item is not row]
            streak = 1
            for item in reversed(previous):
                if not isinstance(item, dict) or "accepted_claims" not in item:
                    break
                if item.get("accepted_claims") != 0:
                    break
                streak += 1
            st["rounds_without_claim"] = streak
        checkpoint.save(self.dir, st)

    def _judge_map(self, st: dict, cands: list[dict], verdicts: dict[str, dict]) -> list[dict]:
        try:
            search_map = json.loads(self._search_map_path().read_text("utf-8"))
        except (OSError, TypeError, ValueError):
            return []
        clusters = _as_dicts(search_map.get("clusters")) \
            if isinstance(search_map, dict) else []
        done = {item for item in st.get("done", []) if isinstance(item, str)}
        accepted_by_task: dict[str, int] = {}
        seen = set()
        for cand in cands:
            key = _claim_verdict_key(cand.get("text"))
            if key in seen or verdicts.get(key, {}).get("verdict") != "keep":
                continue
            seen.add(key)
            task = cand.get("task")
            if isinstance(task, str):
                accepted_by_task[task] = accepted_by_task.get(task, 0) + 1
        return [{
            "id": str(cluster.get("id")),
            "query": str(cluster.get("query") or ""),
            "status": "done" if cluster.get("id") in done else "pending",
            "accepted_claims": accepted_by_task.get(str(cluster.get("id")), 0),
        } for cluster in clusters]

    def _judge_context(
            self, st: dict, cands: list[dict], verdicts: dict[str, dict],
            accepted: list[dict]) -> dict:
        history = self._ensure_gap_history(st)
        breadth = self._topic_breadth()
        claims = [_truncate_text(row["text"], 240) for row in accepted[:_JUDGE_CLAIMS]]
        more = max(0, len(accepted) - len(claims))
        indexed = list(enumerate(_as_dicts(history)))
        priority = {"high": 0, "medium": 1, "low": 2}
        open_rows = sorted(
            ((index, gap) for index, gap in indexed if gap.get("status") == "open"),
            key=lambda item: (priority.get(item[1].get("priority"), 1), -item[0]),
        )
        exhausted_rows = [
            (index, gap) for index, gap in reversed(indexed)
            if gap.get("status") == "exhausted"
        ]
        closed = [
            gap for _, gap in indexed
            if gap.get("status") not in {"open", "exhausted"}
        ]
        search_map = self._judge_map(st, cands, verdicts)[:40]
        return {
            "topic": _truncate_text(st["topic"], 500),
            "search_map": [{
                **row,
                "id": _truncate_text(row.get("id"), 80),
                "query": _truncate_text(row.get("query"), 160),
            } for row in search_map],
            "breadth": {
                "sources": breadth["sources"],
                "domains": breadth["distinct_domains"],
                "source_types": breadth["distinct_kinds"],
            },
            "gaps": {
                "open": [{
                    "query": _truncate_text(gap.get("query"), 240),
                    "source": gap.get("source"),
                    "priority": gap.get("priority", "medium"),
                    "claims_brought": gap.get("claims_brought", 0),
                } for _, gap in open_rows[:_JUDGE_OPEN_GAPS]],
                "exhausted": [{
                    "query": _truncate_text(gap.get("query"), 160),
                } for _, gap in exhausted_rows[:_JUDGE_EXHAUSTED_GAPS]],
                "rest": (
                    f"{len(closed)} closed, they brought "
                    f"{sum(int(gap.get('claims_brought', 0)) for gap in closed)} claims"
                ),
            },
            "new_accepted_claims": claims,
            "new_accepted_note": f"and {more} more" if more else "",
            "coverage_md": _truncate_text(self._coverage_text(), 8000),
            "counters": {
                "cycle": st.get("cycles", 0),
                "max_cycles": self.cfg.max_cycles,
                "rounds_without_new_claims": st.get("rounds_without_claim", 0),
                "staging": len(self._read_zone_claims("staging")),
            },
        }

    @staticmethod
    def _validate_judge(data: dict) -> None:
        if data.get("verdict") not in {"continue", "enough", "stuck"}:
            raise ValueError("judge: verdict must be continue|enough|stuck")
        data["why"] = _model_line(data.get("why"), 600)
        data["coverage_md"] = _truncate_words(
            data.get("coverage_md"), _JUDGE_COVERAGE_WORDS,
        )
        data["gaps"] = _as_dicts(data.get("gaps"))[:_JUDGE_GAPS]
        data["new_clusters"] = _as_dicts(data.get("new_clusters"))[:_JUDGE_CLUSTERS]
        data["open_questions"] = [
            _model_line(question, 200)
            for question in _as_list(data.get("open_questions"))
            if isinstance(question, str) and _model_line(question)
        ][:_JUDGE_QUESTIONS]

    def _judge(
            self, st: dict, cands: list[dict], verdicts: dict[str, dict],
            accepted: list[dict]) -> dict:
        cycle = st.get("cycles", 0)
        existing = st.get("judge")
        if isinstance(existing, dict) and existing.get("cycle") == cycle:
            return existing
        context = self._judge_context(st, cands, verdicts, accepted)
        res, data, bad = self._structured_answer(
            st, _prompt_judge(context), kind="judge", role=ROLE_THINK,
            tools=THINK_TOOLS, allowed_tools=THINK_TOOLS,
            schema={"verdict": str},
            validator=self._validate_judge,
        )
        if res.stop != "done":
            return {"_stop": res.stop}
        if bad is not None:
            raise ValueError(bad)
        assert data is not None
        coverage = _demote_markdown_headings(data.get("coverage_md")).rstrip() + "\n"
        _write_text_atomic(self._coverage_path(), coverage)
        open_questions = list(data.get("open_questions") or [])
        st["coverage"] = {"open_questions": open_questions}
        judge = {
            "verdict": data["verdict"],
            "why": data["why"].strip(),
            "coverage_md": coverage,
            "open_questions": open_questions,
            "gaps": _model_gaps(data.get("gaps")) if data["verdict"] == "continue" else [],
            "new_clusters": [
                {"query": str(row.get("query") or "").strip()}
                for row in _as_dicts(data.get("new_clusters"))
                if str(row.get("query") or "").strip()
            ] if data["verdict"] == "continue" else [],
            "cycle": cycle,
            "accepted_claims": len(accepted),
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        st["judge"] = judge
        checkpoint.save(self.dir, st)
        self._event(
            "judge_verdict", phase="verifying", verdict=judge["verdict"],
            why=judge["why"], gaps_n=len(judge["gaps"]),
            clusters_n=len(judge["new_clusters"]), cycle=cycle,
        )
        return judge

    def _extend_search_map(self, st: dict, clusters: list[dict]) -> list[dict]:
        if not clusters:
            return []
        path = self._search_map_path()
        try:
            search_map = json.loads(path.read_text("utf-8"))
        except (OSError, TypeError, ValueError) as e:
            self._event(
                "search_map_extend_failed", phase="verifying",
                cycle=st.get("cycles", 0), reason=str(e),
            )
            return []
        if not isinstance(search_map, dict) or not isinstance(search_map.get("clusters"), list):
            self._event(
                "search_map_extend_failed", phase="verifying",
                cycle=st.get("cycles", 0), reason="invalid structure",
            )
            return []
        existing_queries = {
            _claim_text_key(row.get("query")) for row in _as_dicts(search_map["clusters"])
        }
        existing_ids = {
            row.get("id") for row in _as_dicts(search_map["clusters"])
            if isinstance(row.get("id"), str)
        }
        added = []
        cycle = st.get("cycles", 0)
        for row in clusters[:_JUDGE_CLUSTERS]:
            query = str(row.get("query") or "").strip()
            normalized = _claim_text_key(query)
            if not query or normalized in existing_queries:
                continue
            stem = "j" + str(cycle) + "_" + hashlib.sha256(
                normalized.encode("utf-8")).hexdigest()[:8]
            cluster_id = stem
            suffix = 2
            while cluster_id in existing_ids:
                cluster_id = f"{stem}_{suffix}"
                suffix += 1
            cluster = {"id": cluster_id, "query": query}
            search_map["clusters"].append(cluster)
            existing_queries.add(normalized)
            existing_ids.add(cluster_id)
            added.append(cluster)
        if not added:
            return []
        _write_json_atomic(path, search_map)
        progress = search_map_progress(self.dir, st)
        ids = list(progress.get("cluster_ids", []))
        ids.extend(cluster["id"] for cluster in added)
        progress["cluster_ids"] = list(dict.fromkeys(ids))
        st["search_map_progress"] = progress
        st["queue"] = added + list(st.get("queue", []))
        checkpoint.save(self.dir, st)
        self._event(
            "search_map_extended", phase="verifying", cycle=cycle, clusters=added,
        )
        return added

    def _transition_to_synthesis(self, st: dict, reason: str, why: str) -> None:
        st["stop"] = {
            "reason": reason,
            "why": why,
            "cycles": st.get("cycles", 0),
            "rounds_without_claim": st.get("rounds_without_claim", 0),
        }
        st["phase"] = "synthesizing"
        checkpoint.save(self.dir, st)
        self._event(
            "phase_changed", phase="verifying", to="synthesizing",
            reason=reason, why=why,
        )
        reason_label = _STOP_REASON_LABELS.get(reason, reason)
        why_line = _one_sentence(why)[:120] or "no further explanation"
        self._notify(
            "phase_changed",
            f"moving to synthesis: {reason_label} - {why_line}",
            kind="progress",
            title="synth",
            lines=[f"Reason: {reason_label} - {why_line}"],
        )

    def _apply_judge(self, st: dict, judge: dict) -> None:
        stop_reason = self._collection_stop_reason(st)
        if stop_reason is not None:
            if judge.get("verdict") == "continue":
                self._event(
                    "judge_overruled", phase="verifying", cycle=st.get("cycles", 0),
                    verdict="continue", reason=stop_reason,
                )
            self._transition_to_synthesis(st, stop_reason, str(judge.get("why") or ""))
            return
        verdict = judge.get("verdict")
        if verdict in {"enough", "stuck"}:
            self._transition_to_synthesis(st, verdict, str(judge.get("why") or ""))
            return
        added = self._extend_search_map(st, _as_dicts(judge.get("new_clusters")))
        queued = self._queue_gaps(st, _as_dicts(judge.get("gaps")), source="judge")
        # A duplicate direction is not a new insertion, but a gap/map task already in the
        # queue still counts as an accepted continue. The queue holds only unfinished tasks;
        # completed ones are removed by _phase_collect.
        has_open_task = bool(_as_dicts(st.get("queue")))
        if not added and not queued and not has_open_task:
            self._transition_to_synthesis(
                st, "enough", "the judge gave no new directions",
            )
            return
        st["phase"] = "collecting"
        checkpoint.save(self.dir, st)

    # --- search map approval ---------------------------------------------

    def _search_map_path(self) -> Path:
        return self._work_dir / "search-map.json"

    @staticmethod
    def _search_map_from_plan(topic: str, data: dict) -> dict:
        """Convert the planner answer into the one editable form of the map.

        The current prompt asks for `clusters`; `tasks` is accepted as the compatible shape
        of the older planner and is normalized immediately. In both cases one cluster equals
        one queue task: deleting, adding or editing a row in the map means literally that.
        """
        raw_clusters = data.get("clusters")
        if not isinstance(raw_clusters, list):
            raw_clusters = data.get("tasks")
        clusters = []
        for index, raw in enumerate(_as_dicts(raw_clusters)):
            query = str(raw.get("query") or "").strip()
            if not query:
                continue
            cluster_id = str(raw.get("id") or f"c{index + 1}").strip()
            clusters.append({"id": cluster_id, "query": query})
        return {
            "version": _SEARCH_MAP_VERSION,
            "topic": topic,
            "instructions": ("Edit, delete or add items in clusters. "
                             "After resume this list becomes the collection queue."),
            "clusters": clusters,
        }

    def _save_search_map(self, search_map: dict) -> None:
        """Atomically save the human-editable map before the waiting checkpoint."""
        _write_json_atomic(self._search_map_path(), search_map)

    def _queue_from_search_map(self, topic: str) -> list[dict]:
        """Validate the approved map and build the queue from exactly its clusters."""
        path = self._search_map_path()
        try:
            data = json.loads(path.read_text("utf-8"))
        except FileNotFoundError as e:
            raise ValueError(
                f"the search map is awaiting approval but the file is gone: {path}") from e
        except (OSError, ValueError) as e:
            raise ValueError(f"the search map does not read as JSON: {path}: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(f"search map {path}: a JSON object is required")
        if data.get("version") != _SEARCH_MAP_VERSION:
            raise ValueError(
                f"search map {path}: version must be {_SEARCH_MAP_VERSION}")
        if data.get("topic") != topic:
            raise ValueError(
                f"search map {path}: topic does not match the checkpoint topic")
        raw_clusters = data.get("clusters")
        if not isinstance(raw_clusters, list):
            raise ValueError(f"search map {path}: clusters must be a list")
        queue = []
        seen_ids = set()
        for index, raw in enumerate(raw_clusters):
            if not isinstance(raw, dict):
                raise ValueError(f"search map {path}: clusters[{index}] must be an object")
            cluster_id = raw.get("id")
            query = raw.get("query")
            if not isinstance(cluster_id, str) or not _MAP_ID.fullmatch(cluster_id):
                raise ValueError(
                    f"search map {path}: clusters[{index}].id must match "
                    "[A-Za-z0-9][A-Za-z0-9_.-]*")
            if cluster_id in seen_ids:
                raise ValueError(f"search map {path}: duplicate id {cluster_id!r}")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    f"search map {path}: clusters[{index}].query must be a non-empty string")
            seen_ids.add(cluster_id)
            queue.append({"id": cluster_id, "query": query.strip()})
        if not queue:
            raise ValueError(f"search map {path}: no cluster is left after the edits")
        return queue

    def _approve_search_map(self, st: dict) -> None:
        """Resume/auto-plan turns the current map into a queue without another LLM call."""
        queue = self._queue_from_search_map(st["topic"])
        gate = st.get("plan_gate") if isinstance(st.get("plan_gate"), dict) else {}
        st.update(phase="collecting", queue=queue, done=[], cycles=0,
                  max_cycles=self.cfg.max_cycles,
                  rounds_without_claim=0, run_id=st.get("run_id") or _now_id(),
                  funnel=empty_funnel(), rounds=[], gaps=[],
                  gap_history_version=1,
                  breadth={"topic": {}, "pages": {}},
                  search_map_progress={
                      "cluster_ids": [task["id"] for task in queue],
                      "completed_ids": [],
                      "pending_ids": [task["id"] for task in queue],
                      "completed": 0,
                      "total": len(queue),
                  },
                  plan_gate={**gate, "status": "approved",
                             "approved_at": datetime.datetime.now(
                                 datetime.timezone.utc).isoformat(timespec="seconds")})
        checkpoint.save(self.dir, st)
        self._event("search_map_approved", phase="planned", path=str(self._work_rel / "search-map.json"),
                    queue=list(queue))

    # --- main loop -------------------------------------------------------

    def run(self) -> int:
        """Run from the current checkpoint until completion or a durable stop.

        The entire topic run holds an advisory flock. Without it, concurrent resumes
        could publish different synthesis decisions and leave a completed checkpoint
        alongside another process's pending state. The return value is this module's
        process exit code.
        """
        with self._topic_lock():
            return self._run_locked()

    def _run_locked(self) -> int:
        started = self._state()
        self._notice_started_at = time.monotonic()
        self._notice_cycle_count = 0
        self._notify(
            "run_started",
            f"start: phase {started.get('phase')}, cycle {started.get('cycles', 0)}",
            kind="info",
            title="start",
            lines=[
                f"Phase {started.get('phase')}, cycle {started.get('cycles', 0)}."
            ],
        )
        while True:
            phase = self._state()["phase"]
            if phase == "done":
                self._finish_done_pending()
                self._gate_published()  # fail-closed: done without a result is not success
                self._progress("done")
                state = self._state()
                notices = state.setdefault("terminal_notices", {})
                if not notices.get("done"):
                    counts = topic_counts(self.dir)
                    breadth = self._topic_breadth()
                    self._notify(
                        "done",
                        f"done: {counts['final']} claims, {counts['pages']} pages, "
                        f"{counts['cycles']} cycles, {self.dir}",
                        kind="done",
                        title="done",
                        lines=[
                            f"{counts['final']} claims, {counts['pages']} pages, "
                            f"{counts['cycles']} cycles. Coverage: domains "
                            f"{breadth['distinct_domains']}, source types "
                            f"{breadth['distinct_kinds']} of 3.",
                            f"Folder: {self.dir}",
                        ],
                    )
                    notices["done"] = True
                    checkpoint.save(self.dir, state)
                return EXIT_DONE
            handler = {"planned": self._phase_plan, "collecting": self._phase_collect,
                       "verifying": self._phase_verify, "synthesizing": self._phase_synth}.get(phase)
            if handler is None:
                raise ValueError(f"unknown checkpoint phase: {phase!r}")
            self._progress(phase)
            stop = handler()
            if stop != "done":
                return _EXIT.get(stop, EXIT_FATAL)
            # the phase moved `phase` in the checkpoint itself - the next lap continues

    @contextmanager
    def _topic_lock(self):
        """Non-blocking topic lock: a second process gets an honest refusal, not a silent wait.

        The contract is a local filesystem with working advisory flock; cross-host NFS/SMB
        synchronization is not supported. A locking error on an unsupported filesystem becomes
        a ValueError, so the CLI prints a diagnostic to stderr and exits 1 instead of dumping
        a bare traceback.

        The file is not removed after unlock: unlinking an active lock file would let a third
        process open a fresh inode and bypass the first process's lock.
        """
        path = self.dir / "work" / "resume.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                lock.seek(0)
                holder = lock.read().strip()
                detail = f" ({holder})" if holder else ""
                raise ValueError(
                    f"the topic is already running in another process{detail}: lock {path}. "
                    "Concurrent resume is forbidden; retry once the first one finishes") from e
            except OSError as e:
                raise ValueError(
                    f"filesystem does not support locking; is the topic on a network filesystem? lock {path}: "
                    f"{e}") from e
            lock.seek(0)
            lock.truncate()
            lock.write(f"pid={os.getpid()}\n")
            lock.flush()
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    # --- plan phase ------------------------------------------------------

    def _phase_plan(self) -> str:
        st = self._state()
        gate = st.get("plan_gate")
        if isinstance(gate, dict) and gate.get("status") == "waiting":
            self._approve_search_map(st)
            return "done"
        existing_pages = self._existing_pages() if self._is_extend else None
        res, data, bad = self._structured_answer(
            st, _prompt_plan(st["topic"], existing_pages=existing_pages),
            kind="plan", role=ROLE_THINK, tools=THINK_TOOLS,
            allowed_tools=THINK_TOOLS)
        if res.stop != "done":
            return res.stop  # the checkpoint stayed in planned - resume will replan
        if bad is not None:
            # The lead produced no plan within the bounded attempts: without a map there is
            # nothing to continue with.
            raise ValueError(bad)
        assert data is not None
        self._work_dir.mkdir(parents=True, exist_ok=True)
        (self._work_dir / "plan.md").write_text(str(data.get("plan_md", "")), "utf-8")
        search_map = self._search_map_from_plan(st["topic"], data)
        self._save_search_map(search_map)
        st.update(queue=[], run_id=st.get("run_id") or _now_id(), plan_gate={
            "status": "waiting",
            "path": str(self._work_rel / "search-map.json"),
            "saved_at": datetime.datetime.now(
                datetime.timezone.utc).isoformat(timespec="seconds"),
        }, max_cycles=self.cfg.max_cycles)
        if res.session_id:
            st.setdefault("sessions", {})[self.adapter.name] = res.session_id
        checkpoint.save(self.dir, st)
        if self.cfg.auto_plan:
            self._approve_search_map(st)
            return "done"
        print(f"the map is awaiting approval: {self._search_map_path()}; "
              "if it looks right run resume, otherwise edit the file", flush=True)
        self._event("search_map_waiting", phase="planned",
                    path=str(self._work_rel / "search-map.json"))
        return "plan_review"

    # --- collect phase ---------------------------------------------------

    def _run_collector(self, task: dict, topic: str) -> RunResult:
        """Runs on a pool thread. ONLY spawns the harness; no socket or disk mutations."""
        return self._call(_prompt_collect(topic, task["query"], self._sources_block),
                          role=ROLE_COLLECT, tools=self._profile.tools,
                          allowed_tools=self._profile.allowed_tools,
                          heartbeat_role="collect",
                          network=self._profile.network)

    def _phase_collect(self) -> str:
        st = self._state()
        topic = st["topic"]
        for k, v in (("queue", []), ("done", []), ("cycles", 0), ("rounds_without_claim", 0)):
            st.setdefault(k, v)  # tolerate resuming from a checkpoint without counters
        st["max_cycles"] = self.cfg.max_cycles
        funnel, rounds = self._observability(st)
        st["search_map_progress"] = search_map_progress(self.dir, st)
        self._found_dir.mkdir(parents=True, exist_ok=True)
        pool = self.cfg.pool()

        while (st.get("collecting_round") is not None
               or (st["queue"]
                   and st["cycles"] < self.cfg.max_cycles
                   and st["rounds_without_claim"] < self.cfg.saturation_rounds)):
            active = st.get("collecting_round")
            if active is None:
                pending_map_ids = set(
                    st["search_map_progress"].get("pending_ids", [])
                )
                round_tasks = _round_tasks_with_map_priority(
                    st["queue"], pool, pending_map_ids
                )
                active = {
                    "cycle": st["cycles"] + 1,
                    "tasks": [task["id"] for task in round_tasks],
                    "completed": [],
                    "new_claims": 0,
                }
                st["collecting_round"] = active
                # The unfinished round exists BEFORE the batch starts: a stop or a hard kill
                # cannot turn a partial result into a supposedly completed cycle.
                checkpoint.save(self.dir, st)
                batch = round_tasks[:pool]
            else:
                task_by_id = {task["id"]: task for task in st["queue"]}
                remaining = [task_id for task_id in active["tasks"]
                             if task_id not in active["completed"]]
                if not remaining:
                    self._finish_collection_round(st, funnel, rounds)
                    break
                missing = [task_id for task_id in remaining if task_id not in task_by_id]
                if missing:
                    raise ValueError(
                        "the unfinished round refers to tasks outside the queue: "
                        + ", ".join(missing))
                batch = [task_by_id[task_id] for task_id in remaining[:pool]]
            cycle = active["cycle"]
            self._progress("collecting", cycle=cycle)
            for task in batch:
                self._progress("collecting", cycle=cycle, task=task["id"],
                               task_status="started")
            # threads only spawn the harness; add_source and the found write happen below,
            # on the main thread
            with ThreadPoolExecutor(max_workers=pool) as ex:
                futs = {ex.submit(self._run_collector, t, topic): t for t in batch}
                results = {futs[f]["id"]: f.result() for f in futs}
            worst = "done"
            bad: list[str] = []
            for task in batch:
                prompt = _prompt_collect(topic, task["query"], self._sources_block)
                res, data, diagnostic = self._structured_answer(
                    st, prompt, kind="collect", role=ROLE_COLLECT,
                    tools=self._profile.tools, allowed_tools=self._profile.allowed_tools,
                    task=task["id"],
                    schema={"sources": list, "claims": list},
                    validator=_validate_collector_sources,
                    network=self._profile.network,
                    first_result=results[task["id"]])
                if res.stop != "done":
                    worst = _worse(worst, res.stop)
                    continue
                if diagnostic is not None:
                    # The cap is spent: the task stays in the queue, as it did before auto-retry.
                    bad.append(diagnostic)
                    continue
                assert data is not None
                # The file of an unfinished task may have been written atomically just before
                # a hard kill while the checkpoint was not. Such a file is not a baseline for
                # itself: repeating the task replaces it and counts findings honestly again,
                # relative to completed tasks and earlier rounds.
                unfinished = set(active["tasks"]) - set(active["completed"])
                seen = {c["text"] for c in self._read_candidates(exclude_tasks=unfinished)
                        if c.get("text")}
                gained = self._ingest_collector(task, data, seen)
                active["new_claims"] += gained
                self._complete_gap_task(st, task, gained)
                st["queue"] = [q for q in st["queue"] if q["id"] != task["id"]]
                st["done"].append(task["id"])
                active["completed"].append(task["id"])
                st["search_map_progress"] = search_map_progress(self.dir, st)
                funnel["found"] = len(self._read_candidates())
                self._breadth(st)["topic"] = self._topic_breadth()
                # A task's success is durable separately from the whole round. If a sibling
                # task stops, resume does not rerun the completed one and does not spend a
                # whole cycle either.
                checkpoint.save(self.dir, st)
                self._progress("collecting", cycle=cycle, task=task["id"],
                               task_status="completed")
            funnel["found"] = len(self._read_candidates())
            if bad and worst == "done":
                raise ValueError("; ".join(bad))
            if worst != "done":
                # Quota/transient outranks a structural failure: those are cured by waiting
                # and retrying, and the failed task stayed in the queue anyway (the raw text
                # of its answer is in work/bad-answers/) and will be repeated by the same
                # resume.
                return worst  # the checkpoint is valid: unfinished tasks stayed in the queue
            if len(active["completed"]) == len(active["tasks"]):
                self._finish_collection_round(st, funnel, rounds)
                break
        st["phase"] = "verifying"
        checkpoint.save(self.dir, st)
        self._notify(
            "phase_changed",
            f"moving to verification: collection of cycle {st.get('cycles', 0)} is finished",
            kind="progress",
            title="verify",
            lines=[
                f"Collection of cycle {st.get('cycles', 0)} is finished; "
                "verifying the new claims."
            ],
        )
        return "done"

    def _finish_collection_round(self, st: dict, funnel: dict, rounds: list) -> None:
        """Spend the gates only after the whole durable collection batch has finished."""
        active = st["collecting_round"]
        st["cycles"] = active["cycle"]
        round_row = {
            "cycle": st["cycles"],
            "tasks": list(active["tasks"]),
            "new_claims": active["new_claims"],
            "found": funnel["found"],
        }
        rounds.append(round_row)
        del st["collecting_round"]
        checkpoint.save(self.dir, st)
        self._event("collection_round", phase="collecting", round=round_row,
                    funnel=funnel)

    def _ingest_collector(self, task: dict, data: dict, seen: set[str]) -> int:
        """Store an ALREADY PARSED collector answer: add_source into the socket, candidates
        into work/found/. Returns the number of NEW (previously unseen) claims, for the
        saturation gate.

        Parsing the answer is deliberately NOT part of this: a structural failure is handled
        by the phase (the task stays in the queue) rather than by substituting an empty dict
        inside the loader."""
        from . import sources as sources_pkg

        url2id: dict[str, str] = {}
        for s in _as_dicts(data.get("sources")):
            url = (s.get("url") or "").strip()
            title = (s.get("title") or url or "source").strip()
            tool = str(s.get("tool") or "").strip()
            # A known source tool is the machine source of truth about the kind. The model's
            # self-report `kind=web` produced 39/39 web on one live corpus, GitHub and HN
            # included; breadth by kind is impossible on such data. For native or unknown
            # tools we keep the compatible fallback to a valid self-report, because the
            # harness does not expose call traces.
            kind = sources_pkg.kind_for_tool(tool)
            if kind is None:
                kind = s.get("kind") if s.get("kind") in _KIND else "web"
            published_at = s.get("published_at")
            published_reason = s.get("published_at_reason") if published_at is None else None
            sid = self.wiki.add_source(self.dir, kind=kind, title=title,
                                       url=url or None, tool=tool or None,
                                       meta={"published_at": published_at,
                                             "published_at_reason": published_reason})
            if url:
                url2id[url] = sid
        rows: list[dict] = []
        for c in _as_dicts(data.get("claims")):
            text = (c.get("text") or "").strip()
            if not text:
                continue
            evidence = []
            for e in _as_dicts(c.get("evidence")):
                sid = url2id.get((e.get("url") or "").strip())
                quote = (e.get("quote") or "").strip()
                if not sid or not quote:
                    continue  # without a real source and a quote there is no evidence
                stance = e.get("stance") if e.get("stance") in _STANCE else "supports"
                evidence.append({"source_id": sid, "quote": quote, "stance": stance})
            if not evidence:
                continue  # a claim without evidence is meaningless (and the schema rejects it)
            confidence, single_source = _single_source_confidence(
                _norm_conf(c.get("confidence")), evidence)
            rows.append({"text": text, "confidence": confidence,
                         "evidence": evidence, "task": task["id"],
                         "meta": {"single_source": single_source}})
        if self._is_extend:
            # A duplicate of a final claim must not reach the critic or the synthesizer
            # again. The same goes for facts from the old staging and for ones this delta has
            # already found.
            blocked = {
                _claim_text_key(claim.get("text"))
                for zone in ("final", "staging")
                for claim in self._read_zone_claims(zone)
            }
            blocked.update(_claim_text_key(text) for text in seen)
            unique = []
            for row in rows:
                key = _claim_text_key(row["text"])
                if key in blocked:
                    continue
                blocked.add(key)
                unique.append(row)
            rows = unique
        new = 0
        path = self._found_dir / f"{task['id']}.jsonl"
        # One task owns one found file. The atomic replace makes a repeat after a hard kill
        # idempotent: the window between writing found and the checkpoint adds no extra rows.
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent,
                prefix=path.name + ".", suffix=".tmp", delete=False) as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                if r["text"] not in seen:
                    seen.add(r["text"])
                    new += 1
            tmp = Path(fh.name)
        os.replace(tmp, path)
        return new

    # --- verify phase ----------------------------------------------------

    def _phase_verify(self) -> str:
        st = self._state()
        cands = self._merge_candidate_evidence(self._read_candidates())
        funnel, _ = self._observability(st)
        self._seed_legacy_verdicts(st, cands)
        verdicts = self._verdict_index()
        unique_unverified = []
        seen = set()
        for cand in cands:
            key = _claim_verdict_key(cand.get("text"))
            verdict = verdicts.get(key)
            checked_sources = set(verdict.get("source_ids", [])) \
                if isinstance(verdict, dict) else set()
            current_sources = set(self._candidate_source_ids(cand))
            already_checked = isinstance(verdict, dict) and (
                verdict.get("verdict") == "keep"
                or current_sources.issubset(checked_sources)
            )
            if already_checked or key in seen:
                continue
            seen.add(key)
            unique_unverified.append(cand)
        for offset in range(0, len(unique_unverified), _VERDICT_BATCH):
            batch = unique_unverified[offset:offset + _VERDICT_BATCH]
            res, data, bad = self._structured_answer(
                st, _prompt_verify(st["topic"], batch), kind="verify", role=ROLE_THINK,
                tools=THINK_TOOLS, allowed_tools=THINK_TOOLS,
                schema={"verdicts": list})
            if res.stop != "done":
                return res.stop  # stay in verifying; resume will check again
            if bad is not None:
                raise ValueError(bad)
            assert data is not None
            keep = _verdicts(data, len(batch))
            now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            rows = []
            for index, cand in enumerate(batch):
                accepted, confidence = keep.get(index, (False, None))
                rows.append({
                    "key": _claim_verdict_key(cand.get("text")),
                    "verdict": "keep" if accepted else "drop",
                    "confidence": _norm_conf(confidence, cand.get("confidence")),
                    "source_ids": self._candidate_source_ids(cand),
                    "cycle": st.get("cycles", 0),
                    "run_id": st.get("run_id"),
                    "at": now,
                })
            # The batch is durable before the next critic call: quota/resume will not redo it.
            self._append_verdict_rows(rows)
            verdicts.update({row["key"]: row for row in rows})
            self._reconcile_kept_candidates(st, batch, verdicts)
        verdicts = self._verdict_index()
        self._reconcile_kept_candidates(st, cands, verdicts)
        unique_keys = {_claim_verdict_key(cand.get("text")) for cand in cands}
        kept = sum(1 for key in unique_keys if verdicts.get(key, {}).get("verdict") == "keep")
        funnel["found"] = len(cands)
        funnel["staging"] = {"keep": kept, "drop": max(0, len(unique_keys) - kept)}
        cycle = st.get("cycles", 0)
        accepted = self._accepted_claims_for_cycle(cands, verdicts, cycle)
        self._record_accepted_round(st, len(accepted))
        st = self._state()
        judge = self._judge(st, cands, verdicts, accepted)
        if judge.get("_stop"):
            return str(judge["_stop"])
        if not judge.get("cycle_notified"):
            staging = len(self._read_zone_claims("staging"))
            sources = topic_counts(self.dir)["sources"]
            verdict = _JUDGE_VERDICT_LABELS.get(
                str(judge.get("verdict")), "stuck"
            )
            why = _one_sentence(judge.get("why"))[:120] or "no explanation"
            minutes = self._cycle_minutes()
            self._notify(
                "cycle_done",
                f"cycle {cycle}/{self.cfg.max_cycles}: +{len(accepted)} claims "
                f"(staging {staging}), judge: {verdict} - {why}",
                kind="progress",
                title=f"cycle {cycle}/{self.cfg.max_cycles}",
                lines=[
                    f"staging {staging} (+{len(accepted)} this cycle), sources "
                    f"{sources}. Judge: {verdict} - {why}",
                    f"Rough pace: ~{minutes} min/cycle",
                ],
            )
            judge["cycle_notified"] = True
            st["judge"] = judge
            checkpoint.save(self.dir, st)
        self._apply_judge(st, judge)
        self._event("verification_funnel", phase="verifying", funnel=funnel)
        return "done"

    # --- synthesize phase ------------------------------------------------

    def _phase_synth(self) -> str:
        """Synthesis: one call -> the synthesizer decision -> publication. Between the call
        and the publication stands a PENDING FILE - the whole decision, saved atomically
        BEFORE the first mutation.

        Publication means (a) promoting the selected staged claims into final through the
        socket (a row move, NOT add_claim(final); the verified status does not change during
        the move - verified in final is legal under the llm-wiki contract W2); (b) concept
        pages under final/wiki/ via add_wiki_page, falling back to update_wiki_page on
        FileExistsError (W1); (c) work/synthesis.md - the raw summary. Claims the synthesizer
        did NOT select STAY in staging as they are - we do not retract them (rejection is a
        decision for a human or for the critic, not for synthesis).

        Why a pending file (codex review block-1): publication is NOT an atomic operation
        (claims are promoted one by one, pages are written afterwards), while the
        synthesizer's decision lived only in process memory. An interruption in the middle
        lost the index -> claim_id mapping, the pages and the summary: with a partial final,
        resume saw "there are claims" and moved the phase to done with unfinished pages, and
        with an EMPTY staging (every claim already promoted) it did not even call the
        synthesizer. Now the decision survives an interruption and resume FINISHES it instead
        of guessing from the leftovers in the zones.
        """
        st = self._state()
        pending = self._load_pending()
        loaded_pending = pending is not None
        breadth_snapshotted = False
        if pending is None:
            staged = self._read_zone_claims("staging")
            if self._is_extend:
                staged = [claim for claim in staged if claim.get("run_id") == st.get("run_id")]
            if staged:
                res, data, bad = self._structured_answer(
                    st, _prompt_synth(
                        st["topic"], staged,
                        existing_pages=self._existing_pages() if self._is_extend else None),
                    kind="synth", role=ROLE_THINK,
                    tools=THINK_TOOLS, allowed_tools=THINK_TOOLS,
                    schema={"final": list, "pages": list, "summary_md": str,
                            "gaps": list})
                if res.stop != "done":
                    return res.stop  # stop BEFORE mutations: staging is intact, resume redoes it
                if bad is not None:
                    raise ValueError(bad)
                assert data is not None
                candidate = _pending_synthesis(data, staged)
                breadth = self._breadth(st)
                breadth["topic"] = self._topic_breadth()
                breadth["pages"] = self._pending_page_breadth(candidate)
                checkpoint.save(self.dir, st)
                self._event("breadth_snapshot", phase="synthesizing", breadth=breadth)
                breadth_snapshotted = True
                gaps = _model_gaps(data.get("gaps"))
                self._record_synth_gaps(st, gaps)
                pending = candidate
                self._save_pending(pending)  # atomically, BEFORE the first promote
        if pending is not None:
            pending = self._ensure_pending_input_count(pending)
        had_pending = pending is not None
        if had_pending:
            funnel_missing = not isinstance(st.get("funnel"), dict)
            if funnel_missing:
                # A legacy pending predates the whole funnel. By the synthesizing phase
                # verify has already finished, so found and the critic's result can be
                # honestly reconstructed from work/found plus staging plus a partially
                # filled final.
                st["funnel"] = completed_funnel_from_disk(self.dir)
            funnel, _ = self._observability(st)
            if not breadth_snapshotted:
                breadth = self._breadth(st)
                breadth["topic"] = self._topic_breadth()
                breadth["pages"] = self._pending_page_breadth(pending)
                checkpoint.save(self.dir, st)
                self._event("breadth_snapshot", phase="synthesizing", breadth=breadth)
            if loaded_pending:
                available_claim_ids = {
                    claim["id"]
                    for zone in ("staging", "final")
                    for claim in self._read_zone_claims(zone)
                }
                lost = sorted(
                    item["id"] for item in _as_dicts(pending.get("claims"))
                    if isinstance(item.get("id"), str)
                    and item["id"] not in available_claim_ids
                )
                if lost:
                    raise ValueError(
                        f"claims from the synthesizer decision are missing from both zones: "
                        f"{', '.join(lost)} (pending: "
                        f"{self._pending_path().relative_to(self.dir)}). We do not finish a "
                        "publication piecemeal: sort out the topic corpus and run resume again")
            input_count = pending.get("input_count")
            if isinstance(input_count, int):
                final = {"keep": len(_as_dicts(pending.get("claims"))),
                         "drop": max(0, input_count - len(_as_dicts(pending.get("claims"))))}
                if funnel_missing or funnel.get("final") != final:
                    funnel["final"] = final
                    checkpoint.save(self.dir, st)
                    self._event("synthesis_funnel", phase="synthesizing", funnel=funnel)
            self._apply_pending(pending)     # idempotent: reconciliation plus ALL pages
        # The gate runs BEFORE moving to done: if there is nothing to publish, the phase stays
        # synthesizing and pending stays both as evidence and as the recovery plan until the
        # gate confirms a result. Marking done and only then failing on run()'s exit gate
        # would be a dead end.
        self._gate_published(expect_pending=had_pending)
        if had_pending:
            # A separate durable finalization step: done is written while pending still
            # holds a snapshot of the publication that was ACTUALLY applied. If the process
            # dies before save(done), synthesizing idempotently applies the same decision
            # again; if it dies after save(done), the done branch compares the contents of
            # final and the pages and removes pending safely.
            pending = {**pending, "applied": self._publication_snapshot()}
            self._save_pending(pending)
        st["phase"] = "done"
        checkpoint.save(self.dir, st)
        if had_pending:
            self._remove_pending()
        return "done"

    # --- pending synthesis: the decision survives an interruption (codex review block-1) ---

    def _pending_path(self) -> Path:
        return self._work_dir / "synthesis-pending.json"

    def _load_pending(self) -> Optional[dict]:
        path = self._pending_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text("utf-8"))
        except ValueError as e:
            # A corrupt pending (an interruption BEFORE os.replace leaves no such file, but
            # disks differ) is fail-closed: the synthesizer decision must not be thrown away
            # silently.
            raise ValueError(f"corrupt pending synthesis {path}: {e} - delete the file if you "
                             "are ready to lose the synthesizer decision, then run resume again")
        if not isinstance(data, dict):
            raise ValueError(
                f"corrupt pending synthesis {path}: expected a JSON object, got "
                f"{type(data).__name__} - the file exists and is not read as an absent decision")
        try:
            return _role_payload(
                data, "pending-synth", {"claims": list, "pages": list, "summary_md": str})
        except ValueError as e:
            raise ValueError(
                f"corrupt pending synthesis {path}: {e} - the decision is not applied as an "
                "empty one; fix the file, or delete it if you are ready to lose the "
                "synthesizer decision") from e

    def _save_pending(self, pending: dict) -> None:
        """Atomic write (tmp plus os.replace, like the checkpoint): no half-written file."""
        path = self._pending_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent,
                prefix=path.name + ".", suffix=".tmp", delete=False) as fh:
            fh.write(json.dumps(pending, ensure_ascii=False, indent=2) + "\n")
            tmp = Path(fh.name)
        os.replace(tmp, path)

    def _ensure_pending_input_count(self, pending: dict) -> dict:
        """Migrate a pending file up to the input_count field added with the final funnel.

        An older decision may have been applied partially: staging holds the unselected and
        not yet promoted claims, final holds the selected ones already promoted. Their sum
        reconstructs the size of the synthesizer's original input without calling the model
        again.
        """
        input_count = pending.get("input_count")
        if isinstance(input_count, int) and not isinstance(input_count, bool) \
                and input_count >= 0:
            return pending
        if "input_count" in pending:
            raise ValueError(
                f"corrupt pending synthesis {self._pending_path()}: input_count must be a "
                "non-negative integer")
        staging_ids = {c["id"] for c in self._read_zone_claims("staging")}
        final_ids = {c["id"] for c in self._read_zone_claims("final")}
        selected_ids = {item.get("id") for item in _as_dicts(pending.get("claims"))
                        if isinstance(item.get("id"), str)}
        already_promoted = (selected_ids & final_ids) - staging_ids
        migrated = {
            **pending,
            "input_count": len(staging_ids) + len(already_promoted),
        }
        self._save_pending(migrated)
        return migrated

    def _remove_pending(self) -> None:
        """Remove a finalized pending. The separate seam exists for the crash regression."""
        self._pending_path().unlink()

    def _publication_snapshot(self) -> dict:
        """Snapshot of the applied publication, for a safe done + pending finalization.

        We compare contents, not just the presence of files: the selected claims, the summary
        and every concept page must stay exactly what they were when the gate passed. A new
        or vanished page changes the snapshot too and prevents blindly deleting the evidence.
        """
        paths = [self.dir / "final" / "claims.jsonl", self._work_dir / "synthesis.md"]
        wiki_dir = self.dir / "final" / "wiki"
        if wiki_dir.exists():
            paths.extend(sorted(wiki_dir.glob("*.md")))
        files = []
        for path in paths:
            if not path.is_file():
                files.append({"path": str(path.relative_to(self.dir)), "sha256": None})
                continue
            files.append({
                "path": str(path.relative_to(self.dir)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        return {"files": files}

    def _finish_done_pending(self) -> None:
        """Close the crash window after save(done) without weakening the done + pending rule.

        A pending without a durable snapshot, or with contents that do not match, is NOT
        considered applied: that is the old contradictory done + pending state, and it needs
        a manual look.
        """
        pending = self._load_pending()
        if pending is None:
            return
        applied = pending.get("applied")
        if not isinstance(applied, dict) or not isinstance(applied.get("files"), list):
            raise ValueError(
                f"the checkpoint cannot be done while pending "
                f"{self._pending_path().relative_to(self.dir)} is not confirmed as applied. "
                "Resume does not delete such a decision blindly")
        actual = self._publication_snapshot()
        if applied != actual:
            raise ValueError(
                f"the done checkpoint contains pending "
                f"{self._pending_path().relative_to(self.dir)}, but the contents of final and "
                "the pages do not match the snapshot of the applied decision. Pending is kept "
                "for inspection")
        self._remove_pending()

    def _apply_pending(self, pending: dict) -> None:
        """Apply the saved synthesizer decision. Idempotently and in full:

        - the summary is always written (rewritten with the same text);
        - every claim is reconciled BY ZONE: in staging means promote (the socket itself
          drops the duplicate if it is already in final); only in final means already
          published, so skip it; nowhere means a fail-closed error rather than a silent loss;
        - ALL pages are written every time (add, then update on FileExistsError), including
          when staging is already empty: that is exactly the hole the review pointed at - an
          "interruption after the last promote but before writing the pages" used to resume
          as a success without pages.
        """
        self._work_dir.mkdir(parents=True, exist_ok=True)
        st = self._state()
        _write_text_atomic(
            self._work_dir / "synthesis.md",
            self._synthesis_with_coverage(pending, st),
        )
        staging_claims = {c["id"]: c for c in self._read_zone_claims("staging")}
        final_claims = {c["id"]: c for c in self._read_zone_claims("final")}
        in_staging = set(staging_claims)
        in_final = set(final_claims)
        promoted: dict[int, str] = {}
        lost: list[str] = []
        for item in _as_dicts(pending.get("claims")):
            cid, idx = item.get("id"), item.get("index")
            if not isinstance(cid, str):
                continue
            if cid in in_staging:
                staged = staging_claims[cid]
                enforced_confidence, _ = _single_source_confidence(
                    staged.get("confidence"), staged.get("evidence"))
                if enforced_confidence != staged.get("confidence"):
                    raise ValueError(
                        f"legacy staging claim {cid} has a single source but "
                        "confidence=high. The socket cannot change confidence during a "
                        "promote; re-verify the claim before publishing")
                self.wiki.promote_claim(self.dir, cid)  # verified staging -> final, by moving
            elif cid in in_final:
                final_claim = final_claims[cid]
                enforced_confidence, _ = _single_source_confidence(
                    final_claim.get("confidence"), final_claim.get("evidence"))
                if enforced_confidence != final_claim.get("confidence"):
                    raise ValueError(
                        f"legacy final claim {cid} has a single source but "
                        "confidence=high. The socket cannot change the confidence of an "
                        "already published row; re-verify the claim before done")
            elif cid not in in_final:
                lost.append(cid)
                continue
            if isinstance(idx, int):
                promoted[idx] = cid
        if lost:
            raise ValueError(
                f"claims from the synthesizer decision are missing from both zones: "
                f"{', '.join(sorted(lost))} "
                f"(pending: {self._pending_path().relative_to(self.dir)}). We do not finish a "
                "publication piecemeal: sort out the topic corpus and run resume again")
        self._write_pages(_as_list(pending.get("pages")), promoted)
        self._ensure_all_pages_coverage(st)
        if not st.get("coverage_note"):
            st["coverage_note"] = {
                "reason": (st.get("stop") or {}).get("reason"),
                "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            }
            checkpoint.save(self.dir, st)
            self._event(
                "coverage_note", phase="synthesizing",
                reason=(st.get("stop") or {}).get("reason"),
                breadth=self._topic_breadth(),
            )

    def _coverage_section(
            self, st: dict, *, title: Optional[str] = None,
            claim_ids: Optional[list[str]] = None) -> str:
        stop = st.get("stop") if isinstance(st.get("stop"), dict) else {}
        reason = str(stop.get("reason") or "legacy")
        why = _model_line(
            stop.get("why") or (st.get("judge") or {}).get("why") or "not stated"
        ) or "not stated"
        topic_breadth = self._topic_breadth()
        unmet = []
        if topic_breadth["distinct_domains"] < _MIN_TOPIC_DOMAINS:
            unmet.append(
                f"topic: {topic_breadth['distinct_domains']} of {_MIN_TOPIC_DOMAINS} domains")
        if topic_breadth["distinct_kinds"] < _MIN_TOPIC_KINDS:
            unmet.append(
                f"topic: {topic_breadth['distinct_kinds']} of {_MIN_TOPIC_KINDS} "
                "source types")
        if title is not None and claim_ids is not None:
            _, page_breadth = self._page_claims_and_breadth(
                claim_ids, zones=("final",))
            if page_breadth["distinct_domains"] < _MIN_PAGE_DOMAINS:
                unmet.append(
                    f"page {title}: {page_breadth['distinct_domains']} of "
                    f"{_MIN_PAGE_DOMAINS} domains")
            if page_breadth["distinct_kinds"] < _MIN_PAGE_KINDS:
                unmet.append(
                    f"page {title}: {page_breadth['distinct_kinds']} of "
                    f"{_MIN_PAGE_KINDS} source types")
        domains = list(topic_breadth["domains"])
        shown_domains = domains[:15]
        domains_text = ", ".join(shown_domains)
        if len(domains) > len(shown_domains):
            domains_text += f", and {len(domains) - len(shown_domains)} more"
        raw_judge_questions = _as_list((st.get("coverage") or {}).get("open_questions")) \
            if isinstance(st.get("coverage"), dict) else []
        judge_questions = [
            question for question in (_model_line(value) for value in raw_judge_questions)
            if question
        ]
        synth_questions = [
            gap for gap in _as_dicts(st.get("gaps"))
            if gap.get("source") == "synth" and gap.get("status") == "unqueued"
        ]
        coverage = _demote_markdown_headings(self._coverage_text()).strip()
        lines = [
            "## Coverage", "", f"Stop reason: {reason}.", f"Why: {why}", "",
            ("Sources: " + str(topic_breadth["sources"]) + "; domains: "
             + str(topic_breadth["distinct_domains"]) + " ("
             + domains_text + "); types: "
             + str(topic_breadth["distinct_kinds"]) + " ("
             + ", ".join(topic_breadth["kinds"]) + ")."),
            "", "Breadth thresholds not met:",
        ]
        lines.extend("- " + item for item in unmet)
        if not unmet:
            lines.append("- none")
        lines.extend(["", "**Open questions from the judge**", ""])
        if judge_questions:
            lines.extend("- " + question for question in judge_questions)
        else:
            lines.append("the judge named no open questions")
        lines.extend(["", "**Open questions (synthesis)**", ""])
        if synth_questions:
            lines.extend(
                "- " + _model_line(gap.get("query")) + ": "
                + _model_line(gap.get("reason"))
                for gap in synth_questions
            )
        else:
            lines.append("- none")
        if coverage:
            lines.extend(["", "**Judge notes**", "", coverage])
        return "\n".join(lines).rstrip() + "\n"

    def _synthesis_with_coverage(self, pending: dict, st: dict) -> str:
        body = _without_system_section(str(pending.get("summary_md", "")), "Coverage")
        skipped = []
        index_to_id = {
            row.get("index"): row.get("id") for row in _as_dicts(pending.get("claims"))
            if isinstance(row.get("index"), int) and isinstance(row.get("id"), str)
        }
        for page in _as_dicts(pending.get("pages")):
            title = str(page.get("title") or "<untitled>")
            ids = list(dict.fromkeys(
                index_to_id[index] for index in _as_list(page.get("claim_indexes"))
                if isinstance(index, int) and index in index_to_id
            ))
            claims, breadth = self._page_claims_and_breadth(ids)
            if len(claims) < 3:
                skipped.append(f"- {title}: {len(claims)} fact(s), at least 3 are required")
            elif not self._has_knowledge_substrate(str(page.get("body_md") or ""), claims):
                skipped.append(f"- {title}: the required coherent substrate is missing")
        if skipped:
            body = _append_system_section(
                _without_system_section(body, "Unpublished pages"),
                "## Unpublished pages\n\n" + "\n".join(skipped),
            )
        return _append_system_section(body, self._coverage_section(st))

    def _ensure_all_pages_coverage(self, st: dict) -> None:
        for page in self._existing_pages():
            body = _append_system_section(
                _without_system_section(page["body_md"], "Coverage"),
                self._coverage_section(
                    st, title=page["title"], claim_ids=page["claims"],
                ),
            )
            if body == page["body_md"]:
                continue
            self.wiki.update_wiki_page(
                self.dir, title=page["title"], claims=page["claims"], body=body,
            )

    # --- publication gate ------------------------------------------------

    def _gate_published(self, *, expect_pending: bool = False) -> None:
        """Fail-closed publication gate: "the run finished" must mean "there is a result".

        A live case: the synthesizer returned an answer, the parser choked on a duplicated
        tail, `except -> data = {}` swallowed the error silently - and the run reported
        exit 0 with phase=done while work/synthesis.md was 0 bytes and final/ was empty. The
        collected work (67 claims, 21 sources) went nowhere and the scheduler was told
        "success". The truth here is THE DISK (final/claims.jsonl), not the phase's own
        report.

        Why a ValueError rather than an exit code: the CLI already turns an engine ValueError
        into a stderr line plus exit 1 (the 0/75/77/111/1 contract, the same path as the
        phase guard and the config gates) - the scheduler needs a non-zero code AND a clear
        reason, not a bare traceback. An empty corpus counts as a failure deliberately:
        "we found nothing" is a result for a human, but not a success for an autonomous
        nightly job.
        """
        pending = self._pending_path()
        if pending.exists() != expect_pending:
            if pending.exists():
                raise ValueError(
                    f"the checkpoint cannot be done while an unapplied synthesizer decision "
                    f"exists: {pending.relative_to(self.dir)}. Resume must first finish the "
                    "pending decision and remove it after the publication gate")
            raise ValueError(
                "the pending synthesis vanished before the publication gate: the decision "
                "cannot be considered applied atomically")
        if self._read_zone_claims("final"):
            wiki_dir = self.dir / "final" / "wiki"
            pages = sorted(wiki_dir.glob("*.md")) if wiki_dir.is_dir() else []
            invalid_pages = []
            for page in pages:
                issue = self._published_page_issue(page)
                if issue is not None:
                    invalid_pages.append((page.name, issue))
            if invalid_pages:
                state = self._state()
                if state.get("phase") == "synthesizing":
                    self._queue_gaps(state, [{
                        "query": f"{state['topic']}: fix concept page {name}",
                        "reason": f"compiler gap on existing page {name}: {issue}",
                        "priority": "high",
                    } for name, issue in invalid_pages], source="publication")
                raise ValueError(
                    "the publication is incomplete: compiler gaps on existing concept pages: "
                    + "; ".join(f"{name}: {issue}" for name, issue in invalid_pages)
                    + ". Repeat synthesis after closing the gap")
            return
        synth = self._work_dir / "synthesis.md"
        size = synth.stat().st_size if synth.exists() else None
        if not self._read_zone_claims("staging"):
            state = self._state()
            state["stop_kind"] = "gate_refused"
            state["stop_detail"] = {
                "reasons": ["nothing to publish: final and staging are empty"],
                "cycles": state.get("cycles"),
                "max_cycles": self.cfg.max_cycles,
            }
            checkpoint.save(self.dir, state)
            self._event(
                "gate", phase=str(state.get("phase")),
                reason="nothing to publish: final and staging are empty",
            )
            raise GateRefused(
                "nothing to publish: final and staging are empty; inspect work/found and verdicts")
        raise ValueError(
            "the publication is empty: final/claims.jsonl contains no claims at all "
            f"(staging: {len(self._read_zone_claims('staging'))} claims, "
            f"{(self._work_rel / 'synthesis.md')}: "
            f"{'no file' if size is None else str(size) + ' bytes'}). "
            "A run without a result does not count as success (fail-closed publication gate): "
            "look at work/found/ and staging/, then repeat synthesis with resume")

    def _published_page_issue(self, path: Path) -> Optional[str]:
        """Check an existing page with the same content/breadth gate as the write path."""
        parse_error = getattr(self.wiki, "FrontmatterError", ValueError)
        try:
            page, body = self.wiki.parse_page(path.read_text("utf-8"))
        except (OSError, UnicodeError, parse_error) as e:
            return f"does not parse as a wiki page ({e})"
        if not isinstance(page, dict) or not isinstance(body, str):
            return "parse_page returned an invalid structure"
        raw_claim_ids = page.get("claims")
        if not isinstance(raw_claim_ids, list):
            return "the frontmatter has no claims list"
        claim_ids = list(dict.fromkeys(
            claim_id for claim_id in raw_claim_ids if isinstance(claim_id, str)))
        claims, breadth = self._page_claims_and_breadth(claim_ids, zones=("final",))
        if len(claims) != len(claim_ids) or len(claim_ids) != len(raw_claim_ids):
            return "the page claim references do not match the final claims"
        if not breadth["fact_gate"]:
            return f"only {breadth['facts']} fact(s), at least 3 are required"
        if not self._has_knowledge_substrate(body, claims):
            return "no coherent substrate with claim/source references"
        return None

    def _write_pages(self, pages: list, promoted: dict[int, str]) -> None:
        """Write the synthesis concept pages into final/wiki/. A page may reference ONLY
        promoted claims (already in final): claim_indexes are filtered by promoted, otherwise
        a page FK would point at a staging claim and the socket would reject the write.
        Idempotent: add_wiki_page, and on FileExistsError (a page for the same concept already
        exists - a resume) update_wiki_page instead (a rewrite)."""
        for p in _as_dicts(pages):
            title = str(p.get("title") or "").strip()
            idxs = p.get("claim_indexes") or []
            delta_claim_ids = list(dict.fromkeys(  # dedup, preserving order
                promoted[i] for i in idxs if isinstance(i, int) and i in promoted))
            if not title or not delta_claim_ids:
                continue  # without a title or a final claim the page is invalid (wiki schema)
            claim_ids = self._merged_page_claim_ids(title, delta_claim_ids)
            claims, breadth = self._page_claims_and_breadth(claim_ids)
            if not breadth["fact_gate"]:
                continue
            body = str(p.get("body_md") or "")
            if not self._has_knowledge_substrate(body, claims):
                continue
            body = self._with_controversy(body, claims)
            body = self._with_source_freshness(body, claim_ids)
            body = _append_system_section(
                _without_system_section(body, "Coverage"),
                self._coverage_section(
                    self._state(), title=title, claim_ids=claim_ids,
                ),
            )
            try:
                self.wiki.add_wiki_page(self.dir, title=title, claims=claim_ids, body=body)
            except FileExistsError:
                self.wiki.update_wiki_page(self.dir, title=title, claims=claim_ids, body=body)

    def _existing_pages(self) -> list[dict]:
        """Read the existing pages for the extend planner and the delta synthesis."""
        wiki_dir = self.dir / "final" / "wiki"
        if not wiki_dir.is_dir():
            return []
        parse_error = getattr(self.wiki, "FrontmatterError", ValueError)
        pages = []
        for path in sorted(wiki_dir.glob("*.md")):
            try:
                page, body = self.wiki.parse_page(path.read_text("utf-8"))
            except (OSError, UnicodeError, parse_error) as e:
                raise ValueError(
                    f"existing page {path.name} does not parse for the delta merge: {e}") from e
            if not isinstance(page, dict) or not isinstance(body, str):
                raise ValueError(
                    f"existing page {path.name}: parse_page returned an invalid structure")
            title = page.get("title")
            claims = page.get("claims")
            if not isinstance(title, str) or not isinstance(claims, list):
                raise ValueError(
                    f"existing page {path.name}: title and a claims list are required")
            pages.append({"title": title, "claims": claims, "body_md": body,
                          "path": str(path.relative_to(self.dir))})
        return pages

    def _merged_page_claim_ids(self, title: str, delta_claim_ids: list[str]) -> list[str]:
        """Delta merge of a page: the old references are kept and the new ones appended."""
        if not self._is_extend:
            return list(dict.fromkeys(delta_claim_ids))
        existing = {
            _claim_text_key(page["title"]): page
            for page in self._existing_pages()
        }.get(_claim_text_key(title))
        old = existing["claims"] if existing is not None else []
        return list(dict.fromkeys([
            claim_id for claim_id in [*old, *delta_claim_ids]
            if isinstance(claim_id, str)
        ]))

    def _source_rows(self) -> list[dict]:
        path = self.dir / "sources" / "sources.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text("utf-8").splitlines()
                if line.strip()]

    def _page_claims_and_breadth(
            self, claim_ids: list[str], *,
            zones: tuple[str, ...] = ("staging", "final")) -> tuple[list[dict], dict]:
        claims_by_id = {
            claim["id"]: claim
            for zone in zones
            for claim in self._read_zone_claims(zone)
        }
        claims = [claims_by_id[claim_id] for claim_id in claim_ids if claim_id in claims_by_id]
        source_ids = {
            evidence.get("source_id")
            for claim in claims
            for evidence in _as_dicts(claim.get("evidence"))
            if isinstance(evidence.get("source_id"), str)
        }
        breadth = self._source_breadth(source_ids)
        breadth["facts"] = len(claims)
        breadth["fact_gate"] = len(claims) >= 3
        breadth["domain_gate"] = breadth["distinct_domains"] >= _MIN_PAGE_DOMAINS
        breadth["kind_gate"] = breadth["distinct_kinds"] >= _MIN_PAGE_KINDS
        breadth["breadth_gate"] = breadth["domain_gate"] and breadth["kind_gate"]
        return claims, breadth

    def _source_breadth(self, source_ids: Optional[set[str]] = None) -> dict:
        domains = set()
        kinds = set()
        count = 0
        for source in self._source_rows():
            if source_ids is not None and source.get("id") not in source_ids:
                continue
            count += 1
            kind = source.get("kind")
            if isinstance(kind, str) and kind:
                kinds.add(kind)
            host = urlparse(str(source.get("url") or "")).hostname
            if host:
                domains.add(host.removeprefix("www.").lower())
        return {
            "sources": count,
            "distinct_domains": len(domains),
            "domains": sorted(domains),
            "distinct_kinds": len(kinds),
            "kinds": sorted(kinds),
        }

    def _topic_breadth(self) -> dict:
        """Topic-level breadth gate: the corpus is wider by type than a single concept page."""
        breadth = self._source_breadth()
        breadth["domain_gate"] = breadth["distinct_domains"] >= _MIN_TOPIC_DOMAINS
        breadth["kind_gate"] = breadth["distinct_kinds"] >= _MIN_TOPIC_KINDS
        breadth["breadth_gate"] = breadth["domain_gate"] and breadth["kind_gate"]
        return breadth

    def _pending_page_breadth(self, pending: dict) -> dict:
        index_to_id = {
            item.get("index"): item.get("id")
            for item in _as_dicts(pending.get("claims"))
            if isinstance(item.get("index"), int) and isinstance(item.get("id"), str)
        }
        metrics = {}
        for position, page in enumerate(_as_dicts(pending.get("pages"))):
            title = str(page.get("title") or "").strip()
            indexes = page.get("claim_indexes") or []
            delta_claim_ids = list(dict.fromkeys(
                index_to_id[index] for index in indexes
                if isinstance(index, int) and index in index_to_id))
            claim_ids = self._merged_page_claim_ids(title, delta_claim_ids)
            claims, breadth = self._page_claims_and_breadth(claim_ids)
            if self._is_extend:
                breadth["delta_facts"] = len(delta_claim_ids)
                breadth["delta_gate"] = bool(delta_claim_ids)
            breadth["substrate_gate"] = self._has_knowledge_substrate(
                str(page.get("body_md") or ""), claims)
            metrics[title or f"<untitled #{position + 1}>"] = breadth
        return metrics

    @staticmethod
    def _has_knowledge_substrate(body: str, claims: list[dict]) -> bool:
        """A minimal structural gate on content, with no attempt at perfect NLP.

        A heading on its own does not count as content. All five mandatory sections must have
        at least one line of text; `Why` explicitly refers to a claim/fact and `Confidence`
        to evidence/a source. The presence of the page claims themselves is checked by the
        caller. The word lists stay bilingual so that Russian-language corpora written
        before the English prompts keep passing the same gate.
        """
        headings = list(re.finditer(r"(?m)^##[ \t]+([^\n]+?)[ \t]*$", body))
        sections: dict[str, str] = {}
        for index, match in enumerate(headings):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
            title = "## " + match.group(1).strip()
            content = body[match.end():end]
            substantive = "\n".join(
                line.strip() for line in content.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ).strip()
            sections[title] = substantive
        if not claims or any(not sections.get(title) for title in _SUBSTRATE_SECTIONS):
            return False
        why = sections["## Why it matters"]
        confidence = sections["## Confidence and why"]
        refers_to_claim = re.search(r"\b(?:клейм\w*|факт\w*|claim\w*)\b", why, re.I)
        refers_to_source = re.search(
            r"\b(?:источник\w*|source\w*|evidence|доказатель\w*|цитат\w*)\b",
            confidence,
            re.I,
        )
        return bool(refers_to_claim and refers_to_source)

    @staticmethod
    def _with_controversy(body: str, claims: list[dict]) -> str:
        """Do not smooth a contradiction away: sides are decided per piece of evidence.

        One claim can have supporting and contradicting sources at the same time. It then has
        to appear on both sides rather than be classified wholesale as opposing.
        """
        opposing = [claim for claim in claims if any(
            evidence.get("stance") == "contradicts"
            for evidence in _as_dicts(claim.get("evidence")))]
        supporting = [claim for claim in claims if any(
            evidence.get("stance") == "supports"
            for evidence in _as_dicts(claim.get("evidence")))]
        clean = _without_system_section(body, "Contested")
        if not opposing or not supporting:
            return clean
        lines = ["## Contested", "", "### Supporting side", ""]
        lines.extend(f"- {claim['text']}" for claim in supporting)
        lines.extend(["", "### Contradicting side", ""])
        lines.extend(f"- {claim['text']}" for claim in opposing)
        return _append_system_section(clean, "\n".join(lines))

    def _with_source_freshness(self, body: str, claim_ids: list[str]) -> str:
        """Add a machine-checkable freshness list of the page's sources.

        Sources and final claims are read from disk; the page itself is still written only
        through the llm-wiki socket. If the injected fake keeps no source registry on disk,
        the section stays empty.
        """
        clean = _without_system_section(body, "Sources and freshness")
        claims = {c["id"]: c for c in self._read_zone_claims("final")}
        source_ids = {
            evidence.get("source_id")
            for claim_id in claim_ids
            for evidence in _as_dicts(claims.get(claim_id, {}).get("evidence"))
            if isinstance(evidence.get("source_id"), str)
        }
        sources = self._source_rows()
        rows = []
        for source in sources:
            if source.get("id") not in source_ids:
                continue
            meta = source.get("meta") if isinstance(source.get("meta"), dict) else {}
            published_at = meta.get("published_at")
            publication = published_at or meta.get("published_at_reason") or "not found"
            title = str(source.get("title") or source.get("url") or source["id"])
            url = source.get("url")
            label = f"[{title}]({url})" if url else title
            rows.append(f"- {label} - publication date: {publication}")
        if not rows:
            return clean
        return _append_system_section(
            clean, "## Sources and freshness\n\n" + "\n".join(rows)
        )

    # --- wiki socket -----------------------------------------------------

    def _add_claim(self, zone: str, status: str, text: str, conf: str,
                   evidence: list[dict], run_id: Optional[str],
                   meta: Optional[dict] = None) -> None:
        try:
            self.wiki.add_claim(self.dir, zone, text=text, confidence=conf, status=status,
                                evidence=evidence, run_id=run_id, meta=meta)
        except self.wiki.DuplicateClaim:
            pass  # the same fact is already in the zone (needed for idempotent resume)

    def _read_zone_claims(self, zone: str) -> list[dict]:
        path = self.dir / zone / "claims.jsonl"
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


# --- helpers (module level, no self) -------------------------------------

def _worse(a: str, b: str) -> str:
    """Stop priority within a batch: quota beats transient beats fatal beats done."""
    order = {"quota": 3, "transient": 2, "fatal": 1, "done": 0}
    return a if order.get(a, 0) >= order.get(b, 0) else b


def _single_source_confidence(confidence: Any, evidence: Any) -> tuple[str, bool]:
    """One confidence enforcement shared by collect, verify and the pre-promote guard."""
    normalized = _norm_conf(confidence)
    distinct_sources = {
        row.get("source_id") for row in _as_dicts(evidence)
        if isinstance(row.get("source_id"), str)
    }
    single_source = len(distinct_sources) == 1
    if single_source and normalized == "high":
        normalized = "medium"
    return normalized, single_source


def _pending_synthesis(data: dict, staged: list[dict]) -> dict:
    """The synthesizer decision in a form that survives an interruption (codex review block-1).

    The key part is that the index -> claim_id mapping is resolved HERE, before the first
    mutation: after the first promote the staging list is already different, and nothing is
    left to reconstruct what index 3 meant in that answer. From then on publication works
    with ids only, never with positions.
    """
    picked = sorted({i for i in (f.get("index") for f in _as_dicts(data.get("final")))
                     if isinstance(i, int) and 0 <= i < len(staged)})
    return {"saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "input_count": len(staged),
            "claims": [{"index": i, "id": staged[i]["id"]} for i in picked],
            "pages": _as_dicts(data.get("pages")),
            "summary_md": str(data.get("summary_md", ""))}


def _verdicts(data: dict, n: int) -> dict[int, tuple[bool, Any]]:
    """Critic verdicts from an ALREADY parsed answer. Parsing happens in the phase: a
    structural failure there stops the phase instead of turning into "no candidate
    survived" here."""
    out: dict[int, tuple[bool, Any]] = {}
    for v in _as_dicts(data.get("verdicts")):
        i = v.get("index")
        if isinstance(i, int) and 0 <= i < n:
            out[i] = (bool(v.get("keep")), v.get("confidence"))
    return out


# --- prompts (the role header is used both by the model and to route the test fake) ------

def _prompt_plan(topic: str, *, existing_pages: Optional[list[dict]] = None) -> str:
    extend = ""
    if existing_pages is not None:
        lite = [{"title": page["title"], "claims": len(page["claims"]),
                 "body_md": page["body_md"]} for page in existing_pages]
        extend = (
            "This is an extension of an already completed topic. The existing pages are given "
            "below. Plan only the gaps, updates and new angles on top of them; do not redo "
            "what is already covered.\n"
            f"Existing pages:\n{json.dumps(lite, ensure_ascii=False)}\n"
        )
    return (
        f"Role: deep research planner.\n"
        f"Topic: \"{topic}\".\n"
        f"{extend}"
        "Build a search map - clusters of concrete sub-questions/queries that cover the topic "
        "broadly (different angles, source types, points of view). Scale the effort to the "
        "breadth of the topic: narrow means 3-4 clusters, broad means 6-10. One cluster "
        "becomes one collection task; do not duplicate queries. After your answer a human "
        "will read and edit this map.\n"
        "Return STRICTLY JSON with no explanations:\n"
        '{"plan_md": "<plan in markdown>", "clusters": '
        '[{"id": "c1", "query": "<cluster query>"}]}'
    )


def _prompt_collect(topic: str, query: str, sources_block: str = "") -> str:
    return (
        f"Role: source collector (one query, one pass).\n"
        f"Topic: \"{topic}\". Query: \"{query}\".\n"
        "Find 2-4 relevant sources with the available search/fetch tools and extract atomic "
        "candidate facts with exact quotes. Do the work YOURSELF: do NOT start sub-agents, do "
        "NOT use the Task/Agent tools.\n"
        f"{sources_block}"
        "For EVERY source find the publication date. published_at is an ISO-8601 date/time; "
        "if no date is found, pass null and published_at_reason='not found'.\n"
        "Return STRICTLY JSON with no explanations:\n"
        '{"sources": [{"kind": "web", "title": "...", "url": "...", "tool": "web", '
        '"published_at": "2026-08-11", "published_at_reason": null}], '
        '"claims": [{"text": "<fact>", "confidence": "high|medium|low", '
        '"evidence": [{"url": "<source url>", "quote": "<exact quote>", '
        '"stance": "supports|contradicts|mentions"}]}]}\n'
        "Every claim must cite at least one of the listed sources with a quote."
    )


def _prompt_verify(topic: str, cands: list[dict]) -> str:
    lite = [{"index": i, "text": c["text"], "confidence": c["confidence"],
             "evidence": [{"source_id": e.get("source_id"), "quote": e["quote"],
                           "stance": e["stance"]} for e in c["evidence"]]}
            for i, c in enumerate(cands)]
    return (
        f"Role: adversarial critic. Topic: \"{topic}\".\n"
        "Below are candidate facts with their evidence (JSON). Try to REFUTE each one: does "
        "the quote really support the fact? is there a stretch or a substitution? Keep only "
        "the survivors, and lower confidence when in doubt. You judge only this new batch and "
        "do NOT decide whether the search should continue.\n"
        "Return STRICTLY JSON:\n"
        '{"verdicts": [{"index": <index from the input>, "keep": true, '
        '"confidence": "high|medium|low"}]}\n'
        f"Candidates:\n{json.dumps(lite, ensure_ascii=False)}"
    )


def _render_judge_prompt(context: dict) -> str:
    return (
        "Role: judge of the research round. Decide on the merits whether there are new useful "
        "search directions, whether the answer is already sufficient, or whether the search "
        "is stuck. Do not ask for an exhausted gap to be retried. The input below is a "
        "strictly bounded summary; you do not have the full corpus.\n"
        "continue is allowed only with concrete new gaps (at most 3) or an extension of the "
        "map (at most 2 clusters). enough means the answer is already sufficient. stuck means "
        "nothing essential was found and what is available should be published honestly. "
        "Update coverage_md as coverage notes: what is closed, what is not, and where to "
        f"look; at most {_JUDGE_COVERAGE_WORDS} words. In open_questions list up to "
        f"{_JUDGE_QUESTIONS} concrete open questions, each at most 200 characters.\n"
        "Return STRICTLY JSON with no explanations:\n"
        '{"verdict":"continue|enough|stuck","why":"...",'
        '"coverage_md":"...","gaps":[{"query":"...","reason":"...",'
        '"priority":"high|medium|low"}],"new_clusters":[{"query":"..."}],'
        '"open_questions":["..."]}\n'
        "Bounded input:\n" + json.dumps(context, ensure_ascii=False)
    )


def _prompt_judge(context: dict) -> str:
    """Serialize the judge context under one UTF-8 byte budget.

    Nominal per-section limits are character/row based, so Cyrillic can make the same
    context roughly twice as large in bytes. Only an oversized prompt is reduced, in
    semantic priority order: exhausted gaps, open gaps, new claims, previous coverage,
    then the search map.
    """
    prompt = _render_judge_prompt(context)
    if len(prompt.encode("utf-8")) < _JUDGE_PROMPT_BYTES:
        return prompt
    # Context is intentionally JSON-shaped. The roundtrip gives us a private deep copy:
    # prompt budgeting must not mutate the durable state or the context used by tests/status.
    bounded = json.loads(json.dumps(context, ensure_ascii=False))

    def fitted() -> Optional[str]:
        candidate = _render_judge_prompt(bounded)
        return candidate if len(candidate.encode("utf-8")) < _JUDGE_PROMPT_BYTES else None

    gaps = bounded.get("gaps") if isinstance(bounded.get("gaps"), dict) else {}
    gaps["exhausted"] = []
    candidate = fitted()
    if candidate is not None:
        return candidate

    gaps["open"] = _as_dicts(gaps.get("open"))[:10]
    candidate = fitted()
    if candidate is not None:
        return candidate

    claims = [row for row in _as_list(bounded.get("new_accepted_claims"))
              if isinstance(row, str)]
    if len(claims) > 30:
        bounded["new_accepted_claims"] = claims[:30]
        budget_note = f"byte budget: showing 30 of the {len(claims)} passed in"
        previous_note = _model_line(bounded.get("new_accepted_note"))
        bounded["new_accepted_note"] = "; ".join(
            part for part in (previous_note, budget_note) if part
        )
    candidate = fitted()
    if candidate is not None:
        return candidate

    bounded["coverage_md"] = _truncate_text(bounded.get("coverage_md"), 4000)
    candidate = fitted()
    if candidate is not None:
        return candidate

    bounded["search_map"] = _as_dicts(bounded.get("search_map"))[:20]
    candidate = fitted()
    if candidate is not None:
        return candidate

    # Defensive tail for malformed legacy state with unexpectedly large map fields. At this
    # point all specified reductions have happened; keep trimming the last-priority section.
    while bounded["search_map"]:
        bounded["search_map"].pop()
        candidate = fitted()
        if candidate is not None:
            return candidate
    # Bounded producers above make this unreachable for valid state, but fail closed on the
    # byte contract rather than hand an oversized prompt to the harness.
    raise ValueError("the judge context does not fit into 40000 bytes after bounded trimming")


def _prompt_synth(topic: str, claims: list[dict], *,
                  existing_pages: Optional[list[dict]] = None) -> str:
    lite = [{"index": i, "text": c["text"], "confidence": c["confidence"],
             "single_source": bool((c.get("meta") or {}).get("single_source")),
             "evidence": [{"source_id": e.get("source_id"), "stance": e.get("stance")}
                          for e in _as_dicts(c.get("evidence"))]}
            for i, c in enumerate(claims)]
    extend = ""
    if existing_pages is not None:
        extend = (
            "This is a DELTA MERGE of a completed topic. The input facts below are only the "
            "new delta. The existing pages are given as separate JSON. Return only new pages "
            "and those existing pages that at least one delta index really touches; do not "
            "rewrite unrelated pages. For a page you update, keep its title, give the full "
            "updated body_md, and list in claim_indexes only the indexes of the NEW delta "
            "that belong to it: the engine keeps the old claim references and adds the new "
            "ones itself.\n"
            f"Existing pages:\n{json.dumps(existing_pages, ensure_ascii=False)}\n"
        )
    return (
        f"Role: synthesizer (one pass). Topic: \"{topic}\".\n"
        f"{extend}"
        "Below are the verified facts (JSON). Do the following:\n"
        "1) Select the facts for the final set, removing only duplicates and subsumed ones; "
        "where contradicting positions are both proven, keep both and do not smooth them "
        "away. Return the indexes (final).\n"
        "2) Group the final facts into concept pages. A page is created only with >=3 facts. "
        "A variety of domains and source types is desirable, but falling short of it will be "
        "stated honestly in the coverage section and does not block the page. body_md is a "
        "coherent unit of knowledge, not a list of claims, and contains EXACTLY these "
        "sections: ## What is known, ## Why it matters, ## What it is for, "
        "## Confidence and why, ## Alternatives. "
        "Fill every section with text beyond its heading; in `Why` explicitly tie the "
        "conclusion to the claims/facts of the page, and in `Confidence` to its "
        "evidence/sources. Do not hide contested facts: the engine adds both sides under "
        "## Contested.\n"
        "3) Write a short overall synthesis of the topic (summary_md).\n"
        "4) If there is not enough data for a coherent result, or an important conclusion "
        "rests on a single source, return gaps with concrete follow-up queries; otherwise "
        "gaps=[].\n"
        "Return STRICTLY JSON with no explanations:\n"
        '{"final": [{"index": <index from the input>}], '
        '"pages": [{"title": "<concept>", "claim_indexes": [<indexes from the input>], '
        '"body_md": "<page markdown>"}], '
        '"summary_md": "<overall markdown synthesis>", '
        '"gaps": [{"query": "<what to collect>", "reason": "<why>", '
        '"priority": "high|medium|low"}]}\n'
        f"Facts:\n{json.dumps(lite, ensure_ascii=False)}"
    )
