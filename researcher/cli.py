"""Researcher CLI: start, extend, status, resume, import, report, and doctor.

Start runs the orchestrator unless --no-run is selected, while resume continues from
durable topic state. Exit codes are 0 for success, 76 for plan review, 75 for quota
wait, 77 for a refused publication gate, 111 for a transient retry, and 1 for
failure. Usage and configuration errors also use 1 rather than argparse's 2.
"""

import datetime
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path

from . import checkpoint
from .adapters import ADAPTERS
from .observability import read_heartbeat, search_map_progress, status_payload, topic_counts
from .shared.cliargs import ArgumentParser
from .shared.config import CommonConfigError, load_common_config

EXIT_OK = 0  # success (done)
EXIT_PLAN_REVIEW = 76  # a planned pause: the search map is waiting for approval
EXIT_QUOTA = 75  # EX_TEMPFAIL: quota/subscription window - not an error, wait for the window
EXIT_GATE_REFUSED = 77  # final and staging are empty: there is nothing to publish
EXIT_TRANSIENT = 111  # a transient failure (network/5xx) - safe to retry
EXIT_FAIL = 1  # a real failure (including config/usage errors)

MACHINE_SCHEMA_VERSION = 1
TOOL_VERSION = "0.1.0"
MAX_MACHINE_OUTPUT_BYTES = 64 * 1024
MAX_MACHINE_STATE_BYTES = 1024 * 1024
DEFAULT_STATUS_STALE_MINUTES = 90.0
SUPERVISOR_TRANSIENT_WAIT_SECONDS = 60
SUPERVISOR_FATAL_WAIT_SECONDS = 120
DEFAULT_SUPERVISOR_TRANSIENT_RETRIES = 12
DEFAULT_SUPERVISOR_FATAL_RETRIES = 6

# The expected source of llmwiki is the sibling repo ../tool-llm-wiki (.agent/decisions.md).
WIKI_REPO = Path(__file__).resolve().parents[2] / "tool-llm-wiki"


class _DiscardText:
    """Bounded-by-construction sink for human progress in machine mode."""

    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None


class _TeeText:
    """Write one stream to a log and the original CLI stream without owning either."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, value: str) -> int:
        for stream in self._streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


class _ObservedAdapter:
    """Transparent adapter that records the current supervisor attempt result."""

    def __init__(self, adapter, results: list) -> None:
        self._adapter = adapter
        self._results = results

    def __getattr__(self, name):
        return getattr(self._adapter, name)

    def run(self, *args, **kwargs):
        result = self._adapter.run(*args, **kwargs)
        self._results.append(result)
        return result


class _SupervisorAttemptNotifier:
    """Forward useful attempt events while the supervisor owns lifecycle events."""

    def __init__(self, notifier):
        self.notifier = notifier

    def emit(self, event: str, text: str, **notice) -> bool:
        if event in {"run_started", "stop"}:
            return False
        return self.notifier.emit(event, text, **notice)


class _ResumeNotifier:
    """Suppress duplicate start events during resume or extend, but pass terminal events."""

    def __init__(self, notifier):
        self.notifier = notifier

    def emit(self, event: str, text: str, **notice) -> bool:
        if event == "run_started":
            return False
        return self.notifier.emit(event, text, **notice)


class _MachineSetupError(RuntimeError):
    pass


def _notice_state_line(topic_dir: str | Path, reason: str) -> str:
    try:
        state = checkpoint.load(topic_dir) or {}
    except (OSError, TypeError, ValueError):
        state = {}
    phase = state.get("phase") or "unknown"
    cycle = state.get("cycles", 0)
    maximum = state.get("max_cycles")
    cycle_text = f"{cycle}/{maximum}" if isinstance(maximum, int) else str(cycle)
    one_line_reason = " ".join(str(reason).split())
    return f"Phase {phase}, cycle {cycle_text}: {one_line_reason}"


def _emit_wait_notice(notifier, topic_dir: str | Path, code: int) -> None:
    try:
        state = checkpoint.load(topic_dir) or {}
    except (OSError, TypeError, ValueError):
        state = {}
    phase = state.get("phase") or "unknown"
    cycle = state.get("cycles", 0)
    if code == EXIT_QUOTA:
        notifier.emit(
            "stop",
            f"waiting for quota: phase {phase}, cycle {cycle} saved",
            kind="wait",
            title="waiting for quota",
            lines=[f"Phase {phase}, cycle {cycle} saved in the checkpoint."],
        )
    elif code == EXIT_PLAN_REVIEW:
        notifier.emit(
            "stop",
            "waiting for map review: inspect work/search-map.json and resume",
            kind="wait",
            title="waiting for plan review",
            lines=["Action: inspect work/search-map.json and resume"],
        )


def _emit_fail_notice(
    notifier,
    topic_dir: str | Path,
    reason: str,
    *,
    title: str = "fail",
    log_path: str | Path | None = None,
    action: bool = True,
    event: str = "stop",
) -> None:
    lines = [_notice_state_line(topic_dir, reason)]
    if log_path is not None:
        lines.append(f"Log: {Path(log_path).resolve()}")
    if action:
        lines.append(f"Action: resume {Path(topic_dir)} --supervise")
    notifier.emit(
        event,
        f"{title}: {' '.join(str(reason).split())}",
        kind="fail",
        title=title,
        lines=lines,
    )


@contextmanager
def _machine_silence():
    sink = _DiscardText()
    with redirect_stdout(sink), redirect_stderr(sink):
        yield


def _llmwiki(*, machine: bool = False):
    try:
        import llmwiki

        return llmwiki
    except ImportError:
        message = (
            "research: llmwiki_unavailable"
            if machine
            else "llmwiki is required: pip install -e ../tool-llm-wiki "
            "(or add it to PYTHONPATH)"
        )
        print(message, file=sys.stderr)
        if machine:
            raise _MachineSetupError("llmwiki_unavailable") from None
        raise SystemExit(EXIT_FAIL)
    except Exception:
        if not machine:
            raise
        print("research: llmwiki_unavailable", file=sys.stderr)
        raise _MachineSetupError("llmwiki_unavailable") from None


def _run(
    topic_dir, args, *, machine: bool = False,
    adapter_results: list | None = None, notify: bool = True,
) -> int:
    # Import before try: the type is needed in except, and inside try it would arrive too
    # late. Laziness is not lost - this is the run branch, it loads the orchestrator anyway.
    from .orchestrator import GateRefused

    # Under supervise the machine diagnostics are already redirected into the per-run log,
    # so we do not drop them; the outer stdout still stays a single JSON object.
    diagnostic_context = (
        _machine_silence() if machine and adapter_results is None else nullcontext()
    )
    notifier = None
    try:
        with diagnostic_context:
            from . import sources as sources_pkg
            from .notifications import EventNotifier
            from .orchestrator import Orchestrator, RunConfig
            from .tool_config import load_tool_config

            base_notifier = EventNotifier(topic_dir, load_tool_config())
            if not notify:
                notifier = _SupervisorAttemptNotifier(base_notifier)
            elif getattr(args, "cmd", None) == "start":
                notifier = base_notifier
            else:
                notifier = _ResumeNotifier(base_notifier)

            names = [s.strip() for s in (args.sources or "").split(",") if s.strip()]
            unknown = [n for n in names if n not in sources_pkg.tool_names()]
            if unknown:
                if machine:
                    raise ValueError("invalid_sources")
                print(
                    f"unknown source tools: {', '.join(unknown)}; available: "
                    f"{', '.join(sources_pkg.tool_names())}",
                    file=sys.stderr,
                )
                return EXIT_FAIL
            # argparse stores None only when --model was NOT passed. The effective shared
            # default is still sonnet, but role priority needs to know it was explicit.
            cfg = RunConfig(
                model=args.model if args.model is not None else "sonnet",
                model_explicit=args.model is not None,
                model_collect=args.model_collect,
                effort=args.effort,
                effort_collect=args.effort_collect,
                max_cycles=args.max_cycles,
                pool_size=args.pool,
                timeout=args.timeout,
                sources=names,
                parity=args.parity,
                auto_plan=args.auto_plan,
            )
            # The constructor is INSIDE try: some config gates (for example "parity without
            # the web source tool") run in it, and a bare traceback is as useless to the
            # scheduler as one from run().
            adapter = ADAPTERS[args.adapter or "claude"]()
            if adapter_results is not None:
                adapter = _ObservedAdapter(adapter, adapter_results)
            orch = Orchestrator(
                topic_dir,
                adapter=adapter,
                wiki=_llmwiki(machine=machine),
                config=cfg,
                notifier=notifier,
            )
            code = orch.run()
    except GateRefused as e:
        # A publication refusal is NOT a failure: collection worked, the quality gates were
        # not cleared before the ceiling ran out. It gets its own code because a resume with
        # THE SAME parameters gives exactly the same result; on the night of 19-20.08 an
        # external watchdog read the refusal as a crash and spun 183 idle restarts on five
        # topics.
        print(
            "research: gate_refused" if machine else f"publication gate refused: {e}",
            file=sys.stderr,
        )
        if notifier is not None:
            _emit_fail_notice(
                notifier, topic_dir, str(e),
                event="gate", action=False,
            )
        return EXIT_GATE_REFUSED
    except ValueError as e:
        # A ValueError from the engine (a config gate, the checkpoint phase guard, the
        # publication gate of item 7, an llmwiki validation refusal) is a data/config failure,
        # not a harness failure: the scheduler needs code 1 and a line, not a traceback.
        print(
            "research: run_failed" if machine else f"run interrupted: {e}",
            file=sys.stderr,
        )
        if notifier is not None:
            _emit_fail_notice(notifier, topic_dir, str(e))
        return EXIT_FAIL
    except Exception:
        if notifier is not None:
            _emit_fail_notice(notifier, topic_dir, "unexpected error")
        if not machine:
            raise
        print("research: run_failed", file=sys.stderr)
        return EXIT_FAIL
    if notifier is not None and code in {EXIT_QUOTA, EXIT_PLAN_REVIEW}:
        _emit_wait_notice(notifier, topic_dir, code)
    elif notifier is not None and code == EXIT_FAIL:
        _emit_fail_notice(notifier, topic_dir, "fatal run stop")
    if code == EXIT_OK and not machine:
        counts = topic_counts(topic_dir)
        print(
            f"DONE: {Path(topic_dir)} | claims: {counts['claims']} / "
            f"sources: {counts['sources']} / pages: {counts['pages']} / "
            f"cycles: {counts['cycles']}"
        )
    elif code != EXIT_PLAN_REVIEW and not machine:
        print(
            f"run stopped, exit={code} "
            "(0 done / 76 plan-review / 75 quota / 77 gate-refused / 111 transient / 1 fatal)"
        )
    return code


def _existing_topics(base, topic: str) -> list[Path]:
    """ALL topics of the base carrying the same text (tasks item 6, codex review fix-2).

    We scan manifests instead of computing the slug: slugify is an internal llmwiki function
    (its __init__ does not export it), and the slug of a Cyrillic topic degenerates into the
    few Latin fragments it contains ("...agentnogo dzhob-sercha 2026" -> "2026"), so one
    directory is easily shared by DIFFERENT topics. Comparing the topic text rules that out:
    we continue only our own.

    We return a LIST, not the first match: the same text in two directories (a moved base, a
    manual copy, a merged base) is an ambiguity, and silently resuming the lexicographically
    first corpus is not allowed. A non-object manifest (valid JSON but a list or a scalar) is
    skipped like a broken one: `.get` on it would raise AttributeError as a bare traceback.
    """
    base = Path(base)
    if not base.is_dir():
        return []
    hits = []
    for manifest in sorted(base.glob("*/topic.json")):
        try:
            data = json.loads(manifest.read_text("utf-8"))
        except (OSError, ValueError):
            continue  # a foreign/broken manifest - not our topic, silently skipped
        if isinstance(data, dict) and data.get("topic") == topic:
            hits.append(manifest.parent)
    return hits


def _topic_folder_slug(topic: str) -> str:
    transliteration = str.maketrans({
        "\u0430": "a", "\u0431": "b", "\u0432": "v", "\u0433": "g",
        "\u0434": "d", "\u0435": "e", "\u0451": "yo", "\u0436": "zh",
        "\u0437": "z", "\u0438": "i", "\u0439": "y", "\u043a": "k",
        "\u043b": "l", "\u043c": "m", "\u043d": "n", "\u043e": "o",
        "\u043f": "p", "\u0440": "r", "\u0441": "s", "\u0442": "t",
        "\u0443": "u", "\u0444": "f", "\u0445": "kh", "\u0446": "ts",
        "\u0447": "ch", "\u0448": "sh", "\u0449": "shch", "\u044a": "",
        "\u044b": "y", "\u044c": "", "\u044d": "e", "\u044e": "yu",
        "\u044f": "ya",
    })
    slug = re.sub(
        r"[^a-z0-9]+", "-", topic.casefold().translate(transliteration),
    ).strip("-")
    slug = slug[:120].rstrip("-") or "topic"
    digest = hashlib.sha256(topic.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


def _start_topic(lw, args, *, machine: bool = False):
    """The topic directory for start: continue an existing one or create a new one (tasks item 6).

    `start` on an existing topic RESUMES instead of failing: this is the normal path of the
    nightly job and of the driver, both of which restart after a quota or a disconnect (a live
    rake from the 08-10 ladder - an unconditional init_topic threw FileExistsError as a bare
    traceback).
    Returns (topic_dir, None) or (None, refusal reason).
    """
    hits = _existing_topics(args.base, args.topic)
    if len(hits) > 1:
        # Ambiguity is fail-closed: which corpus to continue is a human decision.
        paths = "\n  ".join(str(p) for p in hits)
        return None, (
            "multiple topics have the same text; choose one explicitly:\n  "
            f"{paths}\nrun: research resume <topic_dir>"
        )
    if hits:
        root = hits[0]
        st = checkpoint.load(root)
        if st is None:
            # The topic was created (by llmwiki or an earlier start) but no run began in it.
            checkpoint.save(
                root,
                {
                    "phase": "planned",
                    "topic": args.topic,
                    "queue": [],
                    "done": [],
                    "sessions": {},
                    "structural_failures": 0,
                },
            )
            if not machine:
                print(
                    f"topic already exists without a checkpoint; starting from plan: {root}",
                    file=sys.stderr,
                )
        else:
            if not machine:
                print(
                    f"topic already started (phase={st.get('phase')}); continuing from "
                    f"the checkpoint instead of recreating it: {root}",
                    file=sys.stderr,
                )
        return root, None
    try:
        root = lw.init_topic(args.base, args.topic, slug=_topic_folder_slug(args.topic))
    except FileExistsError as e:
        # The slug matched but the topic text differs: silently continuing a foreign topic
        # is not allowed.
        return None, (
            f"another topic already uses the same slug ({e}); rephrase the topic or "
            f"resume the existing one explicitly: research resume <topic_dir>"
        )
    checkpoint.save(
        root,
        {
            "phase": "planned",
            "topic": args.topic,
            "queue": [],
            "done": [],
            "sessions": {},
            "structural_failures": 0,
        },
    )
    return root, None


def _add_run_opts(sp, *, adapter_default="claude"):
    sp.add_argument("--json", action="store_true", help="emit one machine JSON object v1")
    sp.add_argument(
        "--name", default=None,
        help="short topic name for events and status, stored in the checkpoint",
    )
    sp.add_argument(
        "--adapter",
        default=adapter_default,
        choices=sorted(ADAPTERS),
        help="harness adapter (default: claude)",
    )
    sp.add_argument(
        "--model",
        default=None,
        help="model for the whole run (default: sonnet). Codex omits Claude model "
        "names and uses the account default. An explicit value applies to every role "
        "unless --model-collect overrides collection",
    )
    sp.add_argument(
        "--model-collect",
        default=None,
        help="collection model. Precedence: --model-collect, --model, adapter role "
        "default (gpt-5.6-luna for Codex), then the general default",
    )
    sp.add_argument(
        "--effort",
        default=None,
        help="reasoning effort for thinking phases, passed through unchanged because "
        "supported levels and invalid-value behavior are harness-specific",
    )
    sp.add_argument(
        "--effort-collect",
        default=None,
        help="reasoning effort for collection; defaults to --effort",
    )
    sp.add_argument("--max-cycles", type=int, default=3, help="maximum collection rounds")
    sp.add_argument(
        "--pool", type=int, default=3, help="parallel collectors, capped at 3"
    )
    sp.add_argument(
        "--timeout", type=float, default=None, help="harness wall timeout in seconds"
    )
    sp.add_argument(
        "--sources",
        default="hn,arxiv,github,web",
        help="source tools available to collectors through Bash; empty means native "
        "harness web only, which Codex does not provide",
    )
    sp.add_argument(
        "--parity",
        action="store_true",
        help="parity mode: disable native harness web and require the configured source "
        "tools. Claude enforces the physical tool set; Codex relies on the prompt",
    )
    sp.add_argument(
        "--auto-plan",
        action="store_true",
        help="continue after saving the search map. Without this flag, start returns "
        "exit 76 for review and resume accepts the current map",
    )


def _resolve_base(value: str | None) -> str:
    """Explicit path wins; an omitted/empty --base reads common paths.topics_root."""
    if isinstance(value, str) and value:
        return value
    common = load_common_config()
    if not common.exists:
        raise ValueError(
            f"--base was omitted and shared configuration was not found: {common.path}"
        )
    if common.paths is None:
        raise ValueError("shared configuration has no paths.topics_root")
    return str(common.paths.topics_root)


def _import_run_configs(args):
    from .orchestrator import RunConfig

    return (
        RunConfig(
            model=args.model if args.model is not None else "sonnet",
            model_explicit=args.model is not None,
            effort=args.effort,
            timeout=args.timeout,
        ),
        RunConfig(
            model=args.verify_model if args.verify_model is not None else "sonnet",
            model_explicit=args.verify_model is not None,
            effort=args.verify_effort,
            timeout=args.timeout,
        ),
    )


def _pages_run_config(args):
    """Build page-synthesis configuration, falling back to extraction settings."""
    from .orchestrator import RunConfig

    pages_model = getattr(args, "pages_model", None)
    pages_effort = getattr(args, "pages_effort", None)
    model = pages_model if pages_model is not None else args.model
    return RunConfig(
        model=model if model is not None else "sonnet",
        model_explicit=model is not None,
        effort=pages_effort if pages_effort is not None else args.effort,
        timeout=args.timeout,
    )


def _import_notifier(topic_dir, catalog=None):
    from .notifications import EventNotifier
    from .tool_config import load_tool_config

    return EventNotifier(topic_dir, load_tool_config(), catalog=catalog)


def _store_topic_name(topic_dir, name: str | None) -> None:
    """Store --name in the checkpoint for event subjects and status items."""
    if not name or not name.strip():
        return
    state = checkpoint.load(topic_dir)
    if state is None or state.get("name") == name.strip():
        return
    state["name"] = name.strip()
    checkpoint.save(topic_dir, state)


def _add_import_opts(sp, *, adapter_default=None) -> None:
    sp.add_argument("--json", action="store_true", help="emit one machine JSON object v1")
    sp.add_argument(
        "--adapter", default=adapter_default, choices=sorted(ADAPTERS),
        help="extraction harness (default for a new topic: claude)",
    )
    sp.add_argument(
        "--verify-adapter", choices=sorted(ADAPTERS), default=None,
        help="independent verification harness (default: the other harness)",
    )
    sp.add_argument(
        "--allow-same-adapter-verify", action="store_true",
        help="explicitly allow verification by the extraction harness",
    )
    sp.add_argument("--model", default=None, help="model for thinking roles")
    sp.add_argument("--effort", default=None, help="reasoning effort for thinking roles")
    sp.add_argument("--verify-model", default=None, help="independent verification model")
    sp.add_argument(
        "--verify-effort", default=None,
        help="reasoning effort for independent verification",
    )
    sp.add_argument("--pages-model", default=None, help="page synthesis model (default: --model)")
    sp.add_argument(
        "--pages-effort", default=None,
        help="page synthesis effort (default: --effort)",
    )
    sp.add_argument("--timeout", type=float, default=None, help="harness wall timeout in seconds")
    sp.add_argument(
        "--catalog", default=None,
        help="coursedump catalog.tsv for course names; default: <dumps>/../catalog.tsv",
    )


def _wiki_resolution(mod) -> tuple[str, str | None]:
    """Return the llmwiki path and warn when it is not the sibling checkout.

    Importability does not prove identity: a global editable installation can point
    at an older checkout. The resolved module path makes this visible. Other package
    installations and independent checkouts are valid, so this is only a warning.
    """
    path = getattr(mod, "__file__", None)
    if not path:
        return (
            "path unknown because the module has no __file__",
            "cannot determine which llmwiki installation was loaded; check the installation",
        )
    resolved = Path(path).resolve()
    expected = WIKI_REPO.resolve()
    if expected in resolved.parents:
        return str(resolved), None
    return str(resolved), (
        f"llmwiki resolves outside {expected}; the run will use code from {resolved}. "
        f"A global editable installation may point at an old checkout; use "
        f"PYTHONPATH={expected} or reinstall the intended package"
    )


def _doctor_report() -> tuple[dict[str, object], list[str], list[str], str | None]:
    """Return a path-free machine report plus human-only details."""

    problems: list[str] = []
    warnings: list[str] = []
    harnesses = {name: shutil.which(name) is not None for name in ("claude", "codex")}
    if not any(harnesses.values()):
        problems.append(
            "no Claude or Codex harness found; a live run is unavailable"
        )
    where = None
    wiki_ok = False
    try:
        import llmwiki

        wiki_ok = True
        where, warn = _wiki_resolution(llmwiki)
        if warn:
            warnings.append(warn)
    except ImportError:
        problems.append(
            "llmwiki is not importable: pip install -e ../tool-llm-wiki "
            "or add it to PYTHONPATH"
        )
    except Exception:
        problems.append("llmwiki failed the resolution check")
    common_status = "missing"
    try:
        common = load_common_config()
        if common.exists:
            common_status = "ok"
    except CommonConfigError as error:
        common_status = "error"
        problems.append(f"shared configuration is invalid: {error}")
    report = {
        "schema_version": MACHINE_SCHEMA_VERSION,
        "status": "ok" if not problems else "error",
        "checks": [
            {
                "name": "harness",
                "status": "ok" if any(harnesses.values()) else "error",
                "available": [name for name in ("claude", "codex") if harnesses[name]],
            },
            {
                "name": "llmwiki",
                "status": "ok" if wiki_ok else "error",
            },
            {
                "name": "common-config",
                "status": common_status,
            },
        ],
    }
    return report, problems, warnings, where


def doctor(*, machine: bool = False) -> int:
    """Diagnose the environment without creating or modifying files.

    Researcher has no --home because topics are external directories selected by
    --base or a topic argument. A live run requires at least one harness on PATH and
    an importable llmwiki package. Credential checks reveal only boolean presence and
    never print values.
    """
    if machine:
        try:
            with _machine_silence():
                report, problems, warnings, where = _doctor_report()
        except Exception:
            print("research doctor: setup_failed", file=sys.stderr)
            _emit_machine(
                {
                    "schema_version": MACHINE_SCHEMA_VERSION,
                    "status": "error",
                    "checks": [],
                    "error": {"code": "setup_failed"},
                },
                operation="doctor",
            )
            return EXIT_FAIL
        emitted = _emit_machine(report, operation="doctor")
        if problems:
            print("research doctor: unavailable", file=sys.stderr)
            return EXIT_FAIL
        return emitted
    report, problems, warnings, where = _doctor_report()
    harnesses = {name: shutil.which(name) is not None for name in ("claude", "codex")}
    wiki = "available" if report["checks"][1]["status"] == "ok" else "MISSING"
    print("researcher doctor: " + ("READY" if not problems else "NOT READY"))
    print(
        "  harnesses: "
        + ", ".join(n + "=" + ("available" if ok else "MISSING") for n, ok in harnesses.items())
    )
    print(
        "  llmwiki (import llmwiki): "
        + wiki
        + (", resolved: " + where if where else "")
    )
    config_check = next(row for row in report["checks"] if row["name"] == "common-config")
    print("  shared config.toml: " + str(config_check["status"]))
    print(
        "  credentials (presence only): GITHUB_TOKEN="
        + ("present" if os.environ.get("GITHUB_TOKEN") else "absent")
    )
    for w in warnings:
        print("  warning: " + w)
    for pr in problems:
        print("  problem: " + pr)
    return EXIT_OK if not problems else EXIT_FAIL


def _capability(
    name: str,
    argv: list[str],
    required_fields: list[str],
    exits: dict[str, str],
    idempotency: str,
) -> dict[str, object]:
    return {
        "name": name,
        "argv": argv,
        "machine_output": {
            "format": "json",
            "schema_version": MACHINE_SCHEMA_VERSION,
            "required_fields": required_fields,
        },
        "exit_codes": exits,
        "idempotency": idempotency,
    }


def _capabilities() -> dict[str, object]:
    local_exits = {"0": "success", "1": "failure"}
    remote_exits = {
        **local_exits,
        "75": "quota",
        "77": "gate_refused",
        "111": "transient",
        "76": "waiting_human",
    }
    return {
        "schema_version": MACHINE_SCHEMA_VERSION,
        "tool": "researcher",
        "version": TOOL_VERSION,
        "capabilities": [
            _capability(
                "start",
                ["research", "start", "<topic>", "--base", "<topics-root>", "--json"],
                ["schema_version", "status", "topic_dir"],
                remote_exits,
                "deduplicated",
            ),
            _capability(
                "import",
                ["research", "import", "<dump>", "--base", "<topics-root>", "--json"],
                ["schema_version", "status", "topic_dir", "phase"],
                remote_exits,
                "idempotent",
            ),
            _capability(
                "status",
                ["research", "status", "<topic-dir>"],
                ["schema_version", "status", "phase"],
                local_exits,
                "read-only",
            ),
            _capability(
                "status-base",
                ["research", "status", "--base", "<topics-root>", "--json"],
                ["schema_version", "status", "topics"],
                local_exits,
                "read-only",
            ),
            _capability(
                "verify-status",
                ["research", "verify-status", "<topic-dir>", "--json"],
                ["schema_version", "status", "topic", "verified"],
                local_exits,
                "idempotent",
            ),
            _capability(
                "resume",
                ["research", "resume", "<topic-dir>", "--json"],
                ["schema_version", "status", "topic_dir"],
                remote_exits,
                "idempotent",
            ),
            _capability(
                "extend",
                ["research", "extend", "<topic-dir>", "--json"],
                ["schema_version", "status", "topic_dir"],
                remote_exits,
                "deduplicated",
            ),
            _capability(
                "doctor",
                ["research", "doctor", "--json"],
                ["schema_version", "status", "checks"],
                local_exits,
                "read-only",
            ),
        ],
    }


def _emit_machine(
    payload: dict[str, object],
    *,
    operation: str,
    max_bytes: int | None = MAX_MACHINE_OUTPUT_BYTES,
) -> int:
    failed = False
    try:
        rendered = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeError):
        rendered = b'{"error":{"code":"serialization_error"},"schema_version":1,"status":"error"}\n'
        failed = True
        print(f"research {operation}: serialization_error", file=sys.stderr)
    if max_bytes is not None and len(rendered) > max_bytes:
        rendered = b'{"error":{"code":"output_too_large"},"schema_version":1,"status":"error"}\n'
        print(f"research {operation}: output_too_large", file=sys.stderr)
        sys.stdout.buffer.write(rendered)
        return EXIT_FAIL
    sys.stdout.buffer.write(rendered)
    return EXIT_FAIL if failed else EXIT_OK


def _phase_status(phase: object, code: int) -> str:
    if type(code) is not int or code not in {
        EXIT_OK,
        EXIT_PLAN_REVIEW,
        EXIT_QUOTA,
        EXIT_GATE_REFUSED,
        EXIT_TRANSIENT,
        EXIT_FAIL,
    }:
        raise ValueError("invalid_exit_code")
    if code == EXIT_PLAN_REVIEW:
        return "waiting_human"
    if code == EXIT_QUOTA:
        return "quota"
    if code == EXIT_GATE_REFUSED:
        return "gate_refused"
    if code == EXIT_TRANSIENT:
        return "transient"
    if code == EXIT_FAIL:
        return "error"
    return "done" if phase in {"done", "import:done"} else "running"


def _run_payload(topic_dir: str | Path, code: int) -> dict[str, object]:
    state = _load_machine_checkpoint(topic_dir) or {}
    phase = state.get("phase")
    if phase not in checkpoint.PHASES:
        raise ValueError("invalid_checkpoint")
    return {
        "schema_version": MACHINE_SCHEMA_VERSION,
        "status": _phase_status(phase, code),
        "topic_dir": str(Path(topic_dir)),
        "phase": phase if type(phase) is str else None,
    }


def _machine_error(code: str) -> dict[str, object]:
    return {
        "schema_version": MACHINE_SCHEMA_VERSION,
        "status": "error",
        "topic_dir": None,
        "phase": None,
        "error": {"code": code},
    }


def _emit_run_result(topic_dir: str | Path, code: int, *, operation: str) -> int:
    try:
        payload = _run_payload(topic_dir, code)
    except ValueError as error:
        error_code = (
            "invalid_exit_code"
            if str(error) == "invalid_exit_code"
            else "invalid_checkpoint"
        )
        print(f"research {operation}: {error_code}", file=sys.stderr)
        _emit_machine(_machine_error(error_code), operation=operation)
        return EXIT_FAIL
    except (OSError, TypeError, UnicodeError, RecursionError):
        print(f"research {operation}: invalid_checkpoint", file=sys.stderr)
        _emit_machine(_machine_error("invalid_checkpoint"), operation=operation)
        return EXIT_FAIL
    emitted = _emit_machine(payload, operation=operation)
    return code if emitted == EXIT_OK else EXIT_FAIL


def _load_machine_checkpoint(topic_dir: str | Path) -> dict | None:
    try:
        path = Path(topic_dir) / "work" / "checkpoint.json"
        if not path.is_file():
            return None
        with path.open("rb") as stream:
            raw = stream.read(MAX_MACHINE_STATE_BYTES + 1)
        if len(raw) > MAX_MACHINE_STATE_BYTES:
            raise ValueError("state_too_large")
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_machine_object_no_duplicates,
        )
    except (OSError, UnicodeError, ValueError, RecursionError):
        raise ValueError("invalid_checkpoint") from None
    if type(value) is not dict:
        raise ValueError("invalid_checkpoint")
    return value


def _resume_lock_status(topic_dir: str | Path) -> tuple[bool, int | None]:
    """A held flock and a best-effort live PID are two different signals.

    Empty or corrupted text inside a held lock does not make it free. The PID serves
    observability only; busy/free is decided by the kernel through flock.
    """
    path = Path(topic_dir) / "work" / "resume.lock"
    try:
        with path.open("r", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                match = re.search(r"(?:^|\s)pid=(\d+)(?:\s|$)", lock.read())
                if match is None:
                    return True, None
                pid = int(match.group(1))
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return True, None
                except PermissionError:
                    pass
                return True, pid
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                return False, None
    except OSError:
        return False, None


def _live_lock_pid(topic_dir: str | Path) -> int | None:
    """The PID of a held resume.lock when it is recorded and the process is alive, else None."""
    return _resume_lock_status(topic_dir)[1]


@contextmanager
def _exclusive_topic_lock(topic_dir: str | Path):
    """The atomic mutator lock of a topic; yields False when another process already holds it."""
    path = Path(topic_dir) / "work" / "resume.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        try:
            yield True
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _timestamp_age_minutes(timestamp: object) -> float | None:
    if not isinstance(timestamp, str):
        return None
    try:
        moment = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.timezone.utc)
    now = datetime.datetime.now(datetime.timezone.utc)
    return max(0.0, (now - moment.astimezone(datetime.timezone.utc)).total_seconds() / 60)


def _activity_age_minutes(topic_dir: str | Path) -> float | None:
    """The age of the freshest observable record, including the role heartbeat."""
    work = Path(topic_dir) / "work"
    mtimes = []
    for name in ("checkpoint.json", "events.jsonl", "heartbeat.json"):
        try:
            mtimes.append((work / name).stat().st_mtime)
        except OSError:
            continue
    if not mtimes:
        return None
    now = datetime.datetime.now(datetime.timezone.utc).timestamp()
    return max(0.0, (now - max(mtimes)) / 60)


def _duration(minutes: float | None) -> str:
    if minutes is None:
        return "unknown"
    total = max(0, int(minutes))
    if total < 60:
        return f"{total} min"
    hours, rest = divmod(total, 60)
    if hours < 24:
        return f"{hours}h{rest:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


def _movement(
    state: dict | None,
    *,
    stale_minutes: float,
    pid: int | None = None,
    process_age_minutes: float | None = None,
    heartbeat: dict | None = None,
    activity_age_minutes: float | None = None,
) -> str:
    if state is None:
        return "not started"
    if state.get("phase") in {"done", "import:done"}:
        return "done"
    age = activity_age_minutes
    if age is None:
        age = _timestamp_age_minutes(state.get("updated_at"))
    if pid is None:
        if state.get("stop_kind") == "gate_refused":
            return "gate refused"
        gate = state.get("plan_gate")
        if (
            state.get("stop_kind") == "waiting_human"
            or isinstance(gate, dict) and gate.get("status") == "waiting"
        ):
            return "waiting for review"
        return "idle"
    if heartbeat is None:
        legacy = "no heartbeat (legacy run)"
        if age is not None and age <= stale_minutes:
            return f"running ({legacy})"
        if process_age_minutes is not None and process_age_minutes <= stale_minutes:
            return f"starting ({legacy})"
        return legacy
    role = str(heartbeat.get("role") or "?")
    call_started = heartbeat.get("call_started")
    call_age = _timestamp_age_minutes(call_started)
    if age is not None and age <= stale_minutes:
        if call_started is not None:
            return f"running (role {role}, call {_duration(call_age)})"
        return f"running (role {role})"
    if process_age_minutes is not None and process_age_minutes <= stale_minutes:
        return "starting"
    if call_started is not None:
        return f"STALE (role {role} call running for {_duration(call_age)})"
    return "STALE"


def _process_age_minutes(pid: int) -> float | None:
    """Best-effort process age from portable ps etime (DD-HH:MM:SS)."""
    try:
        result = subprocess.run(
            ["ps", "-o", "etime=", "-p", str(pid)], capture_output=True,
            text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    rendered = result.stdout.strip()
    try:
        day_part, clock = rendered.split("-", 1) if "-" in rendered else ("0", rendered)
        parts = [int(part) for part in clock.split(":")]
        if len(parts) == 2:
            hours, minutes, seconds = 0, parts[0], parts[1]
        elif len(parts) == 3:
            hours, minutes, seconds = parts
        else:
            return None
        return (int(day_part) * 86400 + hours * 3600 + minutes * 60 + seconds) / 60
    except (TypeError, ValueError):
        return None


def _stop_reason(state: dict) -> object:
    stop = state.get("stop")
    if isinstance(stop, dict) and stop.get("reason") is not None:
        return stop["reason"]
    detail = state.get("stop_detail")
    reasons = detail.get("reasons") if isinstance(detail, dict) else None
    if isinstance(reasons, list):
        return "; ".join(str(item) for item in reasons)
    return None


def _judge_fields(state: dict | None) -> tuple[object, object, object]:
    judge = state.get("judge") if isinstance(state, dict) else None
    if not isinstance(judge, dict):
        return None, None, None
    return judge.get("verdict"), judge.get("cycle"), judge.get("why")


def _judge_line(state: dict | None) -> str:
    verdict, cycle, why = _judge_fields(state)
    if verdict is None:
        return "judge: -"
    return f"judge: {verdict} ({cycle}) - {why or '-'}"


def _coverage_questions(
    topic_dir: str | Path, state: dict | None = None,
) -> list[str]:
    if state is None:
        try:
            state = checkpoint.load(topic_dir) or {}
        except (OSError, TypeError, ValueError):
            state = {}
    coverage = state.get("coverage")
    if not isinstance(coverage, dict):
        return []
    return [
        question.strip() for question in coverage.get("open_questions", [])
        if isinstance(question, str) and question.strip()
    ][:3]


def _status_row(
    topic_dir: str | Path,
    state: dict | None,
    stats: dict,
    *,
    stale_minutes: float,
    phase_override: str | None = None,
) -> dict:
    stop_detail = state.get("stop_detail") if isinstance(state, dict) else None
    max_cycles = state.get("max_cycles") if isinstance(state, dict) else None
    if max_cycles is None and isinstance(stop_detail, dict):
        max_cycles = stop_detail.get("max_cycles")
    pid = _live_lock_pid(topic_dir)
    heartbeat = read_heartbeat(topic_dir)
    if pid is not None and heartbeat is not None and heartbeat.get("pid") != pid:
        heartbeat = None
    judge_verdict, judge_cycle, judge_why = _judge_fields(state)
    coverage_questions = _coverage_questions(topic_dir, state)
    return {
        "topic": Path(topic_dir).name,
        "phase": phase_override or (
            state.get("phase") if state is not None else "not started"
        ),
        "cycles": state.get("cycles") if state is not None else None,
        "max_cycles": max_cycles,
        "claims_staging": stats.get("claims_staging"),
        "claims_final": stats.get("claims_final"),
        "sources": stats.get("sources"),
        "stop_kind": state.get("stop_kind") if state is not None else None,
        "stop_reason": _stop_reason(state) if state is not None else None,
        "judge_verdict": judge_verdict,
        "judge_cycle": judge_cycle,
        "judge_why": judge_why,
        "judge": _judge_line(state).removeprefix("judge: "),
        "coverage_questions": coverage_questions,
        "questions": " | ".join(coverage_questions) or "-",
        "updated_at": state.get("updated_at") if state is not None else None,
        "movement": (
            "idle" if phase_override else
            _movement(
                state, stale_minutes=stale_minutes, pid=pid,
                process_age_minutes=_process_age_minutes(pid) if pid else None,
                heartbeat=heartbeat,
                activity_age_minutes=_activity_age_minutes(topic_dir),
            )
        ),
        "pid": pid,
        "heartbeat": heartbeat,
    }


def _status_base_rows(base: str | Path, *, stale_minutes: float, wiki) -> list[dict]:
    root = Path(base)
    if not root.is_dir():
        raise ValueError("base_not_found")
    rows = []
    for topic_dir in sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "topic.json").is_file()
    ):
        phase_override = None
        try:
            state = _load_machine_checkpoint(topic_dir)
        except ValueError:
            state = None
            phase_override = "checkpoint unreadable"
        try:
            stats = wiki.stats(topic_dir)
            if not isinstance(stats, dict):
                raise ValueError("invalid_stats")
        except Exception:
            stats = {}
            if phase_override is None:
                phase_override = "statistics unavailable"
        rows.append(
            _status_row(
                topic_dir, state, stats,
                stale_minutes=stale_minutes, phase_override=phase_override,
            )
        )
    return rows


_SEARCH_STAGES = {
    "planned": (1, "cycles"), "collecting": (1, "cycles"), "verifying": (1, "cycles"),
    "synthesizing": (2, "synth"), "pages": (3, "pages"),
}


def _run_started_at(state: dict) -> object:
    """The run start taken from run_id `run_YYYYMMDDTHHMMSS...`, otherwise updated_at."""
    run_id = state.get("run_id")
    if isinstance(run_id, str):
        match = re.fullmatch(r"run_(\d{8}T\d{6})\d*", run_id)
        if match:
            try:
                moment = datetime.datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
            except ValueError:
                moment = None
            if moment is not None:
                return moment.replace(tzinfo=datetime.timezone.utc).isoformat(
                    timespec="seconds"
                )
    return state.get("updated_at")


def _status_snapshot(base: str | Path, rows: list[dict]) -> dict:
    """The status fragment for `gateway snapshot` (canon v2 item 6), flow researcher.

    Schema (stable, covered by tests/test_snapshot.py):
      {"flow": "researcher",
       "workers": [{"item": short name (checkpoint name | first 40 characters of topic),
                    "name": full topic, "pid": int, "job": null,
                    "stage": {"i": 1..3, "n": 3, "name": "cycles|synth|pages"}|null,
                    "done": cycles|null, "total": max_cycles|null,
                    "started_at": iso|null, "eta_seconds": null, "eta_total_seconds": null,
                    "state": "running",
                    "note": "judge: <verdict> - <why<=100>"|null,
                    "metrics": ["claims N", "sources N", "map done/total"]}],
       "queue": []}
    workers lists topics with a live pid (movement is not "stalled"); import topics are not
    included: their snapshot comes from `course-queue --dry-run --json` (flow coursedump).
    """
    from .notifications import short_name
    from .observability import search_map_progress

    workers = []
    for row in rows:
        if row.get("pid") is None or row.get("movement") == "idle":
            continue
        topic_dir = Path(base) / row["topic"]
        try:
            state = checkpoint.load(topic_dir) or {}
        except (OSError, TypeError, ValueError):
            state = {}
        if state.get("run_mode") == "import":
            continue
        topic = " ".join(str(state.get("topic") or row["topic"]).split())
        name = state.get("name")
        stage = _SEARCH_STAGES.get(state.get("phase"))
        verdict, _cycle, why = _judge_fields(state)
        note = None
        if verdict is not None:
            why_text = " ".join(str(why or "").split())[:100]
            note = f"judge: {verdict}" + (f" - {why_text}" if why_text else "")
        metrics = []
        claims = sum(
            value for value in (row.get("claims_staging"), row.get("claims_final"))
            if type(value) is int
        )
        metrics.append(f"claims {claims}")
        if type(row.get("sources")) is int:
            metrics.append(f"sources {row['sources']}")
        try:
            progress = search_map_progress(topic_dir, state)
        except (OSError, TypeError, ValueError):
            progress = {}
        if progress.get("total"):
            metrics.append(f"map {progress['completed']}/{progress['total']}")
        workers.append({
            "item": short_name(name) if isinstance(name, str) and name.strip()
            else short_name(topic),
            "name": topic,
            "pid": row["pid"],
            "job": None,
            "stage": {"i": stage[0], "n": 3, "name": stage[1]} if stage else None,
            "done": row.get("cycles") if type(row.get("cycles")) is int else None,
            "total": row.get("max_cycles") if type(row.get("max_cycles")) is int else None,
            "started_at": _run_started_at(state),
            "eta_seconds": None,
            "eta_total_seconds": None,
            "state": "running",
            "note": note,
            "metrics": metrics,
        })
    return {"flow": "researcher", "workers": workers, "queue": []}


_STATUS_HEADINGS = (
    ("topic", "topic"),
    ("phase", "phase"),
    ("cycle", "cycle"),
    ("claims_staging", "staging"),
    ("claims_final", "final"),
    ("sources", "sources"),
    ("judge", "judge"),
    ("questions", "open questions"),
    ("stop", "stop"),
    ("updated_at", "updated_at"),
    ("movement", "state"),
    ("pid", "pid"),
)


def _render_status_row(row: dict) -> dict:
    cycle = "-" if row["cycles"] is None else str(row["cycles"])
    maximum = "?" if row["max_cycles"] is None else str(row["max_cycles"])
    return {
        **row,
        "topic": row["topic"] if len(row["topic"]) <= 36 else row["topic"][:33] + "...",
        "cycle": f"{cycle}/{maximum}",
        "stop": "/".join(
            str(item)
            for item in (row["stop_kind"], row["stop_reason"])
            if item is not None
        ) or "-",
        "updated_at": row["updated_at"] or "-",
        "pid": "-" if row["pid"] is None else str(row["pid"]),
    }


def _print_base_table(rows: list[dict]) -> None:
    rendered = [_render_status_row(row) for row in rows]
    widths = {
        key: max(len(title), *(len(str(row.get(key, "-"))) for row in rendered))
        for key, title in _STATUS_HEADINGS
    }
    print("  ".join(title.ljust(widths[key]) for key, title in _STATUS_HEADINGS))
    print("  ".join("-" * widths[key] for key, _ in _STATUS_HEADINGS))
    for row in rendered:
        print(
            "  ".join(
                str(row.get(key, "-")).ljust(widths[key])
                for key, _ in _STATUS_HEADINGS
            )
        )


def _latest_topic_event(topic_dir: str | Path) -> tuple[str, str]:
    rows = _topic_events(topic_dir)
    if not rows:
        return "-", "-"
    row = rows[-1]
    return str(row.get("event") or "-"), str(row.get("at") or "-")


def _topic_events(topic_dir: str | Path) -> list[dict]:
    path = Path(topic_dir) / "work" / "events.jsonl"
    try:
        lines = path.read_text("utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _claims_gained_this_run(topic_dir: str | Path, state: dict) -> int:
    """The gain of the supervise session; without one, the sum of durable collection rounds."""
    attempts = [
        row for row in _topic_events(topic_dir)
        if row.get("event") == "supervisor_attempt"
        and row.get("run_id") == state.get("run_id")
        and isinstance(row.get("log"), str)
    ]
    if attempts:
        current_log = attempts[-1]["log"]
        first = next(row for row in attempts if row["log"] == current_log)
        before = first.get("claims_before")
        if isinstance(before, int):
            return max(0, _knowledge_claim_count(topic_dir) - before)
    gained = sum(
        row.get("new_claims", 0)
        for row in state.get("rounds", [])
        if isinstance(row, dict) and isinstance(row.get("new_claims"), int)
    )
    active = state.get("collecting_round")
    if isinstance(active, dict) and isinstance(active.get("new_claims"), int):
        gained += active["new_claims"]
    return max(0, gained)


def _print_short_status(
    topic_dir: str | Path, state: dict, payload: dict, *, stale_minutes: float
) -> None:
    counts = topic_counts(topic_dir)
    row = _status_row(
        topic_dir,
        state,
        {
            "claims_staging": counts["staging"],
            "claims_final": counts["final"],
            "sources": counts["sources"],
        },
        stale_minutes=stale_minutes,
    )
    rendered = _render_status_row(row)
    print(" | ".join(
        f"{title}: {rendered.get(key, '-')}" for key, title in _STATUS_HEADINGS
    ))
    heartbeat = row.get("heartbeat")
    role = "no heartbeat"
    call = "unknown"
    cycle = state.get("cycles") if isinstance(state.get("cycles"), int) else 0
    if isinstance(heartbeat, dict):
        role = str(heartbeat.get("role") or "?")
        started = heartbeat.get("call_started")
        call = _duration(_timestamp_age_minutes(started)) if started is not None else "none"
        if isinstance(heartbeat.get("cycle"), int):
            cycle = heartbeat["cycle"]
    maximum = row.get("max_cycles")
    maximum = maximum if isinstance(maximum, int) else "?"
    print(
        f"now: {state.get('phase')}, role {role}, call {call}, "
        f"cycle {cycle}/{maximum}, claims +{_claims_gained_this_run(topic_dir, state)} "
        "in this run"
    )
    event, at = _latest_topic_event(topic_dir)
    print(f"last event: {event} {at}")
    progress = payload.get("search_map_progress")
    if not isinstance(progress, dict):
        progress = search_map_progress(topic_dir, state)
    completed = progress.get("completed")
    total = progress.get("total")
    completed = completed if isinstance(completed, int) else 0
    total = total if isinstance(total, int) else 0
    print(f"map: {completed}/{total} clusters")
    print(_judge_line(state))
    stop = _stop_reason(state)
    if stop is not None:
        print(f"stop reason: {stop}")
    questions = _coverage_questions(topic_dir)
    if questions:
        print("open questions: " + " | ".join(questions))


def _run_llmwiki_cli(args: list[str], *, input_text: str | None = None):
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(WIKI_REPO) + (os.pathsep + existing if existing else "")
    return subprocess.run(
        [sys.executable, "-m", "llmwiki", *args],
        input=input_text,
        capture_output=True,
        text=True,
        env=env,
    )


def _require_verify_cli_contract() -> None:
    required = {
        "update-source": ("--content-path",),
        "verify-quotes": ("--zone",),
        "set-status": ("--batch",),
        "validate": (),
        "stats": (),
    }
    top = _run_llmwiki_cli(["--help"])
    if top.returncode != EXIT_OK:
        raise ValueError("llmwiki_help_failed")
    top_help = top.stdout + top.stderr
    for verb, flags in required.items():
        if verb not in top_help:
            raise ValueError(f"llmwiki_cli_missing:{verb}")
        sub = _run_llmwiki_cli([verb, "--help"])
        help_text = sub.stdout + sub.stderr
        if sub.returncode != EXIT_OK or any(flag not in help_text for flag in flags):
            raise ValueError(f"llmwiki_cli_shape_mismatch:{verb}")


def _json_cli_result(result, *, operation: str, allow_exit_one: bool = False):
    allowed = {EXIT_OK, EXIT_FAIL} if allow_exit_one else {EXIT_OK}
    if result.returncode not in allowed:
        raise ValueError(f"{operation}:exit={result.returncode}")
    try:
        return json.loads(result.stdout)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{operation}:invalid_json") from error


def _read_staging_claims(topic_dir: Path) -> list[dict]:
    path = topic_dir / "staging" / "claims.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise ValueError(f"staging_invalid_jsonl:{line_number}") from error
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError(f"staging_invalid_claim:{line_number}")
        rows.append(row)
    return rows


def _sources_map(topic_dir: Path) -> dict[str, str]:
    path = topic_dir / "work" / "sources-map.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("sources_map_invalid") from error
    if not isinstance(value, dict):
        raise ValueError("sources_map_invalid")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, str):
            source_id, content_path = key, item
        elif (
            isinstance(key, str)
            and isinstance(item, dict)
            and set(item) == {"source_id", "content_path"}
            and isinstance(item["source_id"], str)
            and isinstance(item["content_path"], str)
        ):
            source_id, content_path = item["source_id"], item["content_path"]
        else:
            raise ValueError("sources_map_invalid")
        if source_id in normalized:
            raise ValueError("sources_map_invalid")
        normalized[source_id] = content_path
    for source_id, content_path in normalized.items():
        probe = Path(content_path)
        if not probe.is_absolute():
            probe = topic_dir / probe
        if not probe.is_file():
            raise ValueError(f"sources_map_missing_path:{source_id}")
    return normalized


def _verify_status(topic_dir: str | Path, *, dry_run: bool) -> dict:
    root = Path(topic_dir)
    if not (root / "topic.json").is_file():
        raise ValueError("topic_not_found")
    _require_verify_cli_contract()
    source_paths = _sources_map(root)
    claims = _read_staging_claims(root)
    candidates = [row["id"] for row in claims if row.get("status") == "candidate"]
    before = _json_cli_result(
        _run_llmwiki_cli(["stats", str(root)]), operation="stats_before"
    )
    report = {
        "topic": root.name,
        "dry_run": dry_run,
        "sources_mapped": len(source_paths),
        "sources_updated": 0,
        "claims_staging": len(claims),
        "candidates": len(candidates),
        "quote_errors": 0,
        "quote_warnings": 0,
        "eligible": 0,
        "verified": 0,
        "already_verified": 0,
        "stats_before": before,
        "stats_after": before,
        "validated": False,
    }
    if dry_run:
        return report

    for source_id, content_path in sorted(source_paths.items()):
        updated = _run_llmwiki_cli(
            ["update-source", str(root), source_id, "--content-path", content_path]
        )
        if updated.returncode != EXIT_OK:
            raise ValueError(f"update_source_failed:{source_id}")
        report["sources_updated"] += 1

    findings = _json_cli_result(
        _run_llmwiki_cli(["verify-quotes", str(root), "--zone", "staging"]),
        operation="verify_quotes",
        allow_exit_one=True,
    )
    if not isinstance(findings, list):
        raise ValueError("verify_quotes:invalid_shape")
    bad = {
        finding.get("claim")
        for finding in findings
        if isinstance(finding, dict) and finding.get("level") == "error"
    }
    report["quote_errors"] = sum(
        1
        for finding in findings
        if isinstance(finding, dict) and finding.get("level") == "error"
    )
    report["quote_warnings"] = sum(
        1
        for finding in findings
        if isinstance(finding, dict) and finding.get("level") == "warn"
    )
    eligible = [claim_id for claim_id in candidates if claim_id not in bad]
    report["eligible"] = len(eligible)
    if eligible:
        results = _json_cli_result(
            _run_llmwiki_cli(
                ["set-status", str(root), "--batch"],
                input_text=json.dumps(eligible),
            ),
            operation="set_status",
        )
        if not isinstance(results, list):
            raise ValueError("set_status:invalid_shape")
        result_by_id = {
            item.get("id"): item.get("result")
            for item in results
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if set(result_by_id) != set(eligible) or any(
            result not in {"verified", "already-verified"}
            for result in result_by_id.values()
        ):
            raise ValueError("set_status:incomplete_result")
        report["verified"] = sum(
            1 for item in results if isinstance(item, dict) and item.get("result") == "verified"
        )
        report["already_verified"] = sum(
            1
            for item in results
            if isinstance(item, dict) and item.get("result") == "already-verified"
        )
    validated = _run_llmwiki_cli(["validate", str(root)])
    if validated.returncode != EXIT_OK:
        raise ValueError("validate_failed")
    report["validated"] = True
    report["stats_after"] = _json_cli_result(
        _run_llmwiki_cli(["stats", str(root)]), operation="stats_after"
    )
    return report


def _print_verify_report(report: dict) -> None:
    mode = "DRY-RUN" if report["dry_run"] else "APPLIED"
    print(f"verify-status {report['topic']}: {mode}")
    print(
        f"  sources-map={report['sources_mapped']} updated={report['sources_updated']} | "
        f"staging={report['claims_staging']} candidate={report['candidates']}"
    )
    print(
        f"  quote errors={report['quote_errors']} warn={report['quote_warnings']} | "
        f"eligible={report['eligible']} verified={report['verified']} "
        f"already={report['already_verified']}"
    )
    print(f"  validate={'OK' if report['validated'] else 'not run'}")


def _knowledge_claim_count(topic_dir: str | Path) -> int:
    ids = set()
    for zone in ("staging", "final"):
        path = Path(topic_dir) / zone / "claims.jsonl"
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{zone}_invalid_jsonl:{line_number}") from error
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"{zone}_invalid_claim:{line_number}")
            ids.add(row["id"])
    return len(ids)


def _supervisor_progress_token(topic_dir: str | Path) -> tuple[object, ...]:
    state = checkpoint.load(topic_dir) or {}
    root = Path(topic_dir)
    work = root / "work"
    if state.get("run_mode") == "extend" and isinstance(state.get("run_id"), str):
        work = work / "runs" / state["run_id"]

    def nonempty_lines(path: Path) -> int:
        try:
            return sum(1 for line in path.read_text("utf-8").splitlines() if line.strip())
        except OSError:
            return 0

    import_artifacts = 0
    if state.get("run_mode") == "import":
        for name in ("extracted", "verified"):
            directory = root / "work" / name
            if directory.is_dir():
                import_artifacts += len(list(directory.glob("*.json")))
        wiki_dir = root / "final" / "wiki"
        if wiki_dir.is_dir():
            import_artifacts += len(list(wiki_dir.glob("*.md")))
        if (root / "work" / "pages-plan.json").is_file():
            import_artifacts += 1
    return (
        state.get("phase"),
        state.get("cycles", 0),
        len(state.get("queue")) if isinstance(state.get("queue"), list) else 0,
        len(state.get("done")) if isinstance(state.get("done"), list) else 0,
        nonempty_lines(work / "verdicts.jsonl"),
        nonempty_lines(root / "staging" / "claims.jsonl"),
        nonempty_lines(root / "final" / "claims.jsonl"),
        import_artifacts,
    )


def _supervisor_reason_text(reason: dict) -> str:
    tail = reason.get("stderr_tail") if isinstance(reason, dict) else None
    if isinstance(tail, list) and tail:
        return " ".join(str(tail[-1]).split())
    adapter = reason.get("adapter") if isinstance(reason, dict) else None
    if isinstance(adapter, dict):
        detail = str(adapter.get("stderr") or adapter.get("stop") or "no reason recorded")
        return " ".join(detail.split())
    return "reason not recorded"


def _append_supervisor_event(topic_dir: str | Path, event: str, **payload) -> None:
    path = Path(topic_dir) / "work" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = checkpoint.load(topic_dir) or {}
    row = {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        "phase": state.get("phase"),
        "run_id": state.get("run_id"),
        **payload,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _other_adapter(name: str) -> str:
    alternatives = [candidate for candidate in sorted(ADAPTERS) if candidate != name]
    if not alternatives:
        raise ValueError("verification_adapter_not_independent")
    return alternatives[0]


def _observed_adapter(name: str, results: list | None):
    adapter = ADAPTERS[name]()
    return _ObservedAdapter(adapter, results) if results is not None else adapter


def _resume_once(
    topic_dir: str | Path, args, *, machine: bool,
    adapter_results: list | None = None, notify: bool = True,
) -> int:
    state = checkpoint.load(topic_dir)
    if isinstance(state, dict) and state.get("run_mode") == "import":
        from .course_import import CourseImportRunner

        stored = state.get("adapters")
        if not isinstance(stored, dict):
            raise ValueError("invalid_import_adapters")
        extract_name = args.adapter or stored.get("extract")
        verify_name = args.verify_adapter or stored.get("verify")
        if extract_name not in ADAPTERS or verify_name not in ADAPTERS:
            raise ValueError("invalid_import_adapters")
        extract_config, verify_config = _import_run_configs(args)
        runner = CourseImportRunner(
            topic_dir, wiki=_llmwiki(machine=machine),
            extract_adapter=_observed_adapter(extract_name, adapter_results),
            verify_adapter=_observed_adapter(verify_name, adapter_results),
            extract_config=extract_config,
            verify_config=verify_config,
            pages_config=_pages_run_config(args),
            notifier=_import_notifier(topic_dir) if notify else None,
            allow_same_adapter_verify=args.allow_same_adapter_verify,
        )
        return runner.run()
    return _run(
        topic_dir, args, machine=machine, adapter_results=adapter_results, notify=notify,
    )


def _supervisor_reason(stderr: str, results: list, code: int) -> dict:
    lines = [line for line in stderr.splitlines() if line.strip()][-3:]
    wanted = {
        EXIT_OK: "done",
        EXIT_QUOTA: "quota",
        EXIT_TRANSIENT: "transient",
        EXIT_FAIL: "fatal",
    }.get(code)
    matching = [result for result in results if result.stop == wanted]
    result = matching[-1] if matching else (results[-1] if results else None)
    adapter = None
    if result is not None:
        adapter = {"stop": result.stop, "stderr": result.stderr[:300]}
    return {"stderr_tail": lines, "adapter": adapter}


def _supervise_log_path(topic_dir: str | Path) -> Path:
    logs = Path(topic_dir) / "work" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    path = logs / f"supervise-{stamp}.log"
    path.touch(exist_ok=False)
    return path


def _supervise_resume(topic_dir: str | Path, args, *, machine: bool) -> int:
    """Repeats resume according to the exit contract without taking over the scheduler's job."""
    attempt = 0
    transient_retries = 0
    fatal_retries = 0
    previous_claims = _knowledge_claim_count(topic_dir)
    previous_progress = _supervisor_progress_token(topic_dir)
    loop_key = None
    loop_count = 0
    from .notifications import EventNotifier
    from .tool_config import load_tool_config
    notifier = EventNotifier(topic_dir, load_tool_config())
    terminal = {EXIT_OK, EXIT_PLAN_REVIEW, EXIT_GATE_REFUSED, EXIT_QUOTA}
    log_path = _supervise_log_path(topic_dir)
    log_relative = str(log_path.relative_to(Path(topic_dir)))
    while True:
        attempt += 1
        before = previous_claims
        progress_before = previous_progress
        stderr_capture = io.StringIO()
        adapter_results = []
        with log_path.open("a", encoding="utf-8") as log:
            stdout_streams = (log,) if machine else (log, sys.stdout)
            stderr_streams = (log, stderr_capture) if machine else (
                log, stderr_capture, sys.stderr,
            )
            with redirect_stdout(_TeeText(*stdout_streams)), redirect_stderr(
                _TeeText(*stderr_streams)
            ):
                code = _resume_once(
                    topic_dir, args, machine=machine,
                    adapter_results=adapter_results, notify=False,
                )
        after = _knowledge_claim_count(topic_dir)
        progress_after = _supervisor_progress_token(topic_dir)
        delta = after - before
        attempt_reason = _supervisor_reason(
            stderr_capture.getvalue(), adapter_results, code
        )
        _append_supervisor_event(
            topic_dir,
            "supervisor_attempt",
            attempt=attempt,
            exit_code=code,
            reason=attempt_reason,
            log=log_relative,
            claims_before=before,
            claims_after=after,
            claims_delta=delta,
            progress_before=list(progress_before),
            progress_after=list(progress_after),
            transient_retries=transient_retries,
            fatal_retries=fatal_retries,
        )
        previous_claims = after
        previous_progress = progress_after
        if code in terminal:
            _append_supervisor_event(
                topic_dir, "supervisor_stop", attempt=attempt, exit_code=code
            )
            if code in {EXIT_QUOTA, EXIT_PLAN_REVIEW}:
                _emit_wait_notice(notifier, topic_dir, code)
            return code
        if code == EXIT_TRANSIENT:
            if transient_retries >= args.transient_retries:
                _append_supervisor_event(
                    topic_dir,
                    "supervisor_stop",
                    attempt=attempt,
                    exit_code=EXIT_FAIL,
                    reason="transient_retry_limit",
                    transient_retries=transient_retries,
                )
                _emit_fail_notice(
                    notifier, topic_dir, "transient retry limit exhausted",
                    log_path=log_path,
                )
                return EXIT_FAIL
            transient_retries += 1
            _append_supervisor_event(
                topic_dir,
                "supervisor_wait",
                attempt=attempt,
                reason="transient",
                seconds=SUPERVISOR_TRANSIENT_WAIT_SECONDS,
                transient_retries=transient_retries,
            )
            time.sleep(SUPERVISOR_TRANSIENT_WAIT_SECONDS)
            continue
        if code != EXIT_FAIL:
            _append_supervisor_event(
                topic_dir,
                "supervisor_stop",
                attempt=attempt,
                exit_code=EXIT_FAIL,
                reason="invalid_exit_code",
            )
            _emit_fail_notice(
                notifier, topic_dir, f"unknown exit code {code}", log_path=log_path,
            )
            return EXIT_FAIL

        if progress_after != progress_before:
            loop_key = None
            loop_count = 0
        else:
            current_key = (code, progress_after)
            if current_key == loop_key:
                loop_count += 1
            else:
                loop_key = current_key
                loop_count = 1
        if loop_count >= 3:
            reason_text = _supervisor_reason_text(attempt_reason)
            phase, cycle = progress_after[:2]
            print(
                f"LOOP: {phase} {cycle} {code} x3 - {reason_text}",
                file=sys.stderr,
            )
            _append_supervisor_event(
                topic_dir, "supervisor_stop", attempt=attempt,
                exit_code=EXIT_FAIL, reason="loop",
            )
            _emit_fail_notice(
                notifier, topic_dir, f"3 attempts without progress - {reason_text}",
                title="stop: loop", log_path=log_path,
            )
            return EXIT_FAIL
        if fatal_retries >= args.fatal_retries:
            _append_supervisor_event(
                topic_dir,
                "supervisor_stop",
                attempt=attempt,
                exit_code=EXIT_FAIL,
                reason="fatal_retry_limit",
            )
            _emit_fail_notice(
                notifier, topic_dir, "fatal retry limit exhausted",
                log_path=log_path,
            )
            return EXIT_FAIL
        fatal_retries += 1
        _append_supervisor_event(
            topic_dir,
            "supervisor_wait",
            attempt=attempt,
            reason="fatal_retry",
            seconds=SUPERVISOR_FATAL_WAIT_SECONDS,
            fatal_retries=fatal_retries,
        )
        time.sleep(SUPERVISOR_FATAL_WAIT_SECONDS)


def _machine_object_no_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate checkpoint field")
        result[key] = value
    return result


def _cmd_pages(a) -> int:
    topic = Path(a.topic_dir)
    if not topic.is_dir():
        print(f"research pages: topic does not exist: {topic}", file=sys.stderr)
        return EXIT_FAIL
    try:
        with _exclusive_topic_lock(topic) as acquired:
            if not acquired:
                print("research pages: topic is locked by a live run (work/resume.lock)",
                      file=sys.stderr)
                return EXIT_FAIL
            return _cmd_pages_locked(a, topic)
    except OSError as error:
        print(f"research pages: topic lock unavailable: {error}", file=sys.stderr)
        return EXIT_FAIL


def _cmd_pages_locked(a, topic: Path) -> int:
    from .orchestrator import RunConfig
    from .pages import PagesError, PagesRunner
    from .tool_config import ToolConfigError

    try:
        notifier = _import_notifier(topic)
    except ToolConfigError as error:
        print(f"research pages: {error}", file=sys.stderr)
        return EXIT_FAIL
    if a.rebuild:
        plan = topic / "work" / "pages-plan.json"
        if plan.is_file():
            plan.unlink()
        wiki_dir = topic / "final" / "wiki"
        if wiki_dir.is_dir():
            for page in wiki_dir.glob("*.md"):
                page.unlink()
    config = RunConfig(
        model=a.model if a.model is not None else "sonnet", model_explicit=a.model is not None,
        effort=a.effort, timeout=a.timeout,
    )
    try:
        with _machine_silence() if a.json else nullcontext():
            runner = PagesRunner(
                topic, wiki=_llmwiki(machine=a.json), adapter=ADAPTERS[a.adapter](),
                config=config, notifier=notifier,
                log=lambda text: print(f"pages: {text}", file=sys.stderr),
            )
            code = runner.run()
    except (PagesError, _MachineSetupError, OSError, TypeError, ValueError) as error:
        print(f"research pages: {error}", file=sys.stderr)
        if a.json:
            _emit_machine(_machine_error("pages_failed"), operation="pages")
        return EXIT_FAIL
    if a.json:
        payload = {
            "schema_version": MACHINE_SCHEMA_VERSION, "topic_dir": str(topic),
            "exit_code": code, "stop_kind": runner.stop_kind, "stop_reason": runner.stop_reason,
        }
        emitted = _emit_machine(payload, operation="pages")
        return code if emitted == EXIT_OK else EXIT_FAIL
    return code


def _cmd_course_queue(a) -> int:
    from .course_import import (
        EXIT_PLAN_REVIEW as EXIT_IMPORT_PLAN_REVIEW,
        CourseImportRunner, ImportRunOptions, initialize_import,
    )
    from .course_queue import (
        EXIT_QUEUE_EMPTY, default_catalog, next_course, queue_rows, render_rows,
        reserve_next_course, snapshot, summary_json,
    )
    from .sources import SourceImportError
    from .tool_config import ToolConfigError

    queue = Path(a.queue).expanduser()
    dumps = Path(a.dumps).expanduser()
    base = Path(a.base)
    catalog = Path(a.catalog).expanduser() if a.catalog else default_catalog(dumps)
    if not queue.is_file():
        print(f"research course-queue: queue does not exist: {queue}", file=sys.stderr)
        return EXIT_FAIL

    def summary(rows, slug, code):
        return summary_json(rows, slug, code, snapshot=snapshot(rows, base, catalog))

    if a.dry_run:
        rows = queue_rows(queue, base, dumps)
        slug = next_course(rows)
        if a.json:
            print(summary(rows, None, None))
        else:
            print(render_rows(rows))
        return EXIT_OK

    with reserve_next_course(queue, base, dumps) as reservation:
        rows = reservation.rows
        slug = reservation.slug
        if slug is None:
            if a.json:
                print(summary(rows, None, None))
            else:
                print(render_rows(rows))
                if any(row["state"] == "running" for row in rows):
                    print("no available courses: remaining courses run in other workers")
                else:
                    print("course queue is empty: all entries are complete or belong to another flow")
            return EXIT_QUEUE_EMPTY
        extract_name = a.adapter or "claude"
        verify_name = a.verify_adapter or _other_adapter(extract_name)
        if not a.json:
            print(render_rows(rows))
            print(f"course: {slug} (import {extract_name}, verify {verify_name})")
        try:
            with _machine_silence() if a.json else nullcontext():
                wiki = _llmwiki(machine=a.json)
                extract_config, verify_config = _import_run_configs(a)
                options = ImportRunOptions(
                    dump=dumps / slug, base=base, slug=slug,
                    extract_adapter=extract_name, verify_adapter=verify_name,
                    auto_plan=True, link_dump=False,
                    extract_config=extract_config, verify_config=verify_config,
                    allow_same_adapter_verify=a.allow_same_adapter_verify,
                )
                root = initialize_import(wiki, options)
                code = CourseImportRunner(
                    root, wiki=wiki, extract_adapter=ADAPTERS[extract_name](),
                    verify_adapter=ADAPTERS[verify_name](),
                    extract_config=extract_config, verify_config=verify_config,
                    pages_config=_pages_run_config(a),
                    notifier=_import_notifier(root, catalog),
                    allow_same_adapter_verify=a.allow_same_adapter_verify,
                ).run()
        except (
            SourceImportError, CommonConfigError, ToolConfigError, _MachineSetupError,
            KeyError, OSError, TypeError, ValueError,
        ) as error:
            # str(error), not error.code: a code without details is one word in the job log
            # (the claim_replay_conflict incident of 26.08). The machine JSON stays a code.
            reason = str(error)
            print(f"research course-queue: {slug}: {reason}", file=sys.stderr)
            if a.json:
                _emit_machine(_machine_error("course_queue_failed"), operation="course-queue")
            return EXIT_FAIL

    rows = queue_rows(queue, base, dumps)
    if code == EXIT_IMPORT_PLAN_REVIEW:
        print(
            f"course {slug}: import plan is waiting for review (exit 76)",
            file=sys.stderr,
        )
    if a.json:
        print(summary(rows, slug, code))
    else:
        print(f"course {slug}: exit {code}")
        print(render_rows(rows))
    return code


def main(argv: list[str] | None = None) -> int:
    p = ArgumentParser(prog="research", description="turn a topic into a knowledge corpus")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("capabilities", help="canonical machine-readable CLI capabilities")
    sp.add_argument("--json", action="store_true", required=True)

    sp = sub.add_parser(
        "source-plan", help="read-only import plan for a local or coursedump corpus"
    )
    sp.add_argument("source")
    sp.add_argument(
        "--kind",
        choices=("auto", "local-markdown", "coursedump-manifest"),
        default="auto",
    )
    sp.add_argument("--json", action="store_true", required=True)

    sp = sub.add_parser("start", help="create a topic and start a run")
    sp.add_argument("topic")
    sp.add_argument("--base", help="topic database directory; defaults to paths.topics_root")
    sp.add_argument(
        "--no-run",
        action="store_true",
        help="create the topic without running the orchestrator",
    )
    _add_run_opts(sp)

    sp = sub.add_parser("import", help="import coursedump or local Markdown into a topic")
    sp.add_argument("dump", help="read-only corpus directory or Markdown file")
    sp.add_argument("--base", help="topic database directory; defaults to paths.topics_root")
    sp.add_argument("--slug", help="topic slug; defaults to the dump directory name")
    sp.add_argument(
        "--auto-plan", action="store_true",
        help="automatically approve the deterministic import plan",
    )
    sp.add_argument(
        "--link-dump", action="store_true",
        help="store absolute dump references instead of copying content into the topic",
    )
    _add_import_opts(sp)

    sp = sub.add_parser("status", help="show one run or summarize a topic database")
    sp.add_argument("topic_dir", nargs="?")
    sp.add_argument(
        "--base", nargs="?", const="",
        help="database summary; without a value use paths.topics_root",
    )
    sp.add_argument("--json", action="store_true", help="emit the database summary as JSON v1")
    sp.add_argument(
        "--short",
        action="store_true",
        help="compact human topic status with judge and coverage",
    )
    sp.add_argument(
        "--raw",
        action="store_true",
        help="full topic status with queue and rounds, without output-size limits",
    )
    sp.add_argument(
        "--stale-minutes", "--fresh-minutes",
        dest="stale_minutes",
        type=float,
        default=DEFAULT_STATUS_STALE_MINUTES,
        help="stale observable-activity threshold in minutes (default: 90)",
    )

    sp = sub.add_parser(
        "verify-status", help="promote candidate status through the verbatim-quote gate"
    )
    sp.add_argument("topic_dir")
    sp.add_argument("--dry-run", action="store_true", help="validate inputs without mutation")
    sp.add_argument("--json", action="store_true", help="emit one machine JSON object v1")

    sp = sub.add_parser("resume", help="continue a run from its checkpoint")
    sp.add_argument("topic_dir")
    _add_run_opts(sp, adapter_default=None)
    sp.add_argument(
        "--verify-adapter", choices=sorted(ADAPTERS), default=None,
        help="independent verification harness for an import topic",
    )
    sp.add_argument(
        "--allow-same-adapter-verify", action="store_true",
        help="explicitly allow the same harness to extract and verify",
    )
    sp.add_argument("--verify-model", default=None, help="verification model for imports")
    sp.add_argument(
        "--verify-effort", default=None,
        help="verification reasoning effort for imports",
    )
    sp.add_argument("--pages-model", default=None, help="page synthesis model for imports")
    sp.add_argument(
        "--pages-effort", default=None,
        help="page synthesis reasoning effort for imports",
    )
    sp.add_argument(
        "--supervise",
        action="store_true",
        help="repeat resume according to the exit contract until a terminal result",
    )
    sp.add_argument(
        "--fatal-retries",
        type=int,
        default=DEFAULT_SUPERVISOR_FATAL_RETRIES,
        help="maximum retries after fatal results with allowed progress (default: 6)",
    )
    sp.add_argument(
        "--transient-retries",
        type=int,
        default=DEFAULT_SUPERVISOR_TRANSIENT_RETRIES,
        help="maximum retries after transient results (default: 12)",
    )

    sp = sub.add_parser(
        "extend", help="extend a completed topic with a separate delta run"
    )
    sp.add_argument("topic_dir")
    _add_run_opts(sp)

    sp = sub.add_parser(
        "pages",
        help="build wiki pages from claims: concept map, synthesis, and audit",
    )
    sp.add_argument("topic_dir")
    sp.add_argument("--adapter", default="claude", choices=sorted(ADAPTERS))
    sp.add_argument("--model", default=None, help="model; default: adapter thinking-role model")
    sp.add_argument("--effort", default=None, help="synthesis reasoning effort")
    sp.add_argument("--timeout", type=float, default=None, help="harness wall timeout in seconds")
    sp.add_argument(
        "--rebuild", action="store_true",
        help="remove the page plan and generated pages, then rebuild them",
    )
    sp.add_argument("--json", action="store_true", help="emit one machine JSON object v1")

    sp = sub.add_parser(
        "course-queue",
        help=(
            "process the next incomplete course from queue.txt through import and pages; "
            "an empty queue exits 0 and plan review exits 76"
        ),
    )
    sp.add_argument("--queue", required=True, help="coursedump queue.txt with one slug per line")
    sp.add_argument("--dumps", required=True, help="coursedump output directory")
    sp.add_argument("--base", default=None, help="topic database (default: paths.topics_root)")
    sp.add_argument("--dry-run", action="store_true", help="show queue state without processing")
    sp.add_argument("--auto-plan", action="store_true", default=True,
                    help="approve the import plan automatically (default: enabled)")
    _add_import_opts(sp, adapter_default="claude")

    sp = sub.add_parser(
        "report",
        help="synthesize a completed topic into Markdown in one offline harness pass",
    )
    sp.add_argument("topic_dir")
    sp.add_argument("--out", required=True, help="Markdown output path; parent must exist")
    sp.add_argument("--name", default=None, help="report topic name; default: topic.json")
    sp.add_argument("--prompt", default=None,
                    help="prompt template with __TOPIC__, __OUT__, __NAME__, and "
                         "__TOPICS_ROOT__; default: built-in template")
    sp.add_argument("--topics-root", default=None,
                    help="topic database for cross-references; default: topic parent")
    sp.add_argument("--adapter", default="claude", choices=sorted(ADAPTERS))
    sp.add_argument("--model", default=None, help="model; default: adapter thinking-role model")
    sp.add_argument("--effort", default="xhigh", help="reasoning effort (default: xhigh)")
    sp.add_argument("--timeout", type=float, default=None, help="wall timeout in seconds")
    sp.add_argument("--guard", action="store_true",
                    help="guard job: an incomplete topic exits 0 without an error")

    sp = sub.add_parser(
        "doctor", help="diagnose harnesses and llmwiki without mutation"
    )
    sp.add_argument("--json", action="store_true")

    a = p.parse_args(argv)
    if a.cmd in {"start", "import", "course-queue"} or (a.cmd == "status" and a.base is not None):
        try:
            a.base = _resolve_base(a.base)
        except (CommonConfigError, OSError, TypeError, ValueError) as error:
            print(f"research {a.cmd}: base_config_failed: {error}", file=sys.stderr)
            if getattr(a, "json", False):
                _emit_machine(_machine_error("base_config_failed"), operation=a.cmd)
            return EXIT_FAIL
    if a.cmd == "capabilities":
        return _emit_machine(_capabilities(), operation="capabilities")
    if a.cmd == "source-plan":
        from .sources import SourceImportError, plan_import

        try:
            with _machine_silence():
                plan = plan_import(a.source, source_kind=a.kind)
        except SourceImportError as error:
            print(f"research source-plan: {error.code}", file=sys.stderr)
            _emit_machine(
                {
                    "schema_version": MACHINE_SCHEMA_VERSION,
                    "status": "error",
                    "error": {"code": error.code},
                },
                operation="source-plan",
            )
            return EXIT_FAIL
        except Exception:
            print("research source-plan: setup_failed", file=sys.stderr)
            _emit_machine(
                {
                    "schema_version": MACHINE_SCHEMA_VERSION,
                    "status": "error",
                    "error": {"code": "setup_failed"},
                },
                operation="source-plan",
            )
            return EXIT_FAIL
        emitted = _emit_machine(plan.to_dict(), operation="source-plan")
        return EXIT_PLAN_REVIEW if emitted == EXIT_OK else EXIT_FAIL
    if a.cmd == "doctor":
        return doctor(machine=a.json)
    if a.cmd == "pages":
        return _cmd_pages(a)
    if a.cmd == "course-queue":
        return _cmd_course_queue(a)
    if a.cmd == "report":
        from .notifications import EventNotifier
        from .report import run_report
        from .tool_config import ToolConfigError, load_tool_config

        template = None
        if a.prompt:
            try:
                template = Path(a.prompt).expanduser().read_text("utf-8")
            except (OSError, UnicodeError) as error:
                print(f"research report: prompt template is unreadable: {error}", file=sys.stderr)
                return EXIT_FAIL
        try:
            notifier = EventNotifier(a.topic_dir, load_tool_config())
        except ToolConfigError as error:
            print(f"research report: {error}", file=sys.stderr)
            return EXIT_FAIL
        return run_report(
            a.topic_dir, a.out, adapter=ADAPTERS[a.adapter](), live_pid=_live_lock_pid,
            notifier=notifier, name=a.name, template=template,
            topics_root=Path(a.topics_root).expanduser() if a.topics_root else None,
            model=a.model, effort=a.effort, timeout=a.timeout, guard=a.guard,
        )
    if a.cmd == "import":
        from .course_import import (
            CourseImportRunner,
            ImportRunOptions,
            initialize_import,
        )
        from .course_queue import default_catalog
        from .sources import SourceImportError

        try:
            dump_path = Path(a.dump)
            dump_name = (
                dump_path.stem
                if dump_path.suffix.casefold() == ".md"
                else dump_path.name
            )
            slug = a.slug or re.sub(r"[^a-z0-9]+", "-", dump_name.casefold()).strip("-")
            if not slug:
                raise ValueError("import_slug_required")
            prior = checkpoint.load(Path(a.base) / slug)
            prior_adapters = prior.get("adapters") if isinstance(prior, dict) else None
            extract_name = a.adapter or (
                prior_adapters.get("extract")
                if isinstance(prior_adapters, dict) else "claude"
            )
            verify_name = a.verify_adapter or (
                prior_adapters.get("verify")
                if isinstance(prior_adapters, dict) else _other_adapter(extract_name)
            )
            if extract_name not in ADAPTERS or verify_name not in ADAPTERS:
                raise ValueError("invalid_import_adapters")
            with _machine_silence() if a.json else nullcontext():
                wiki = _llmwiki(machine=a.json)
                extract_config, verify_config = _import_run_configs(a)
                options = ImportRunOptions(
                    dump=Path(a.dump), base=Path(a.base), slug=slug,
                    extract_adapter=extract_name, verify_adapter=verify_name,
                    auto_plan=a.auto_plan, link_dump=a.link_dump,
                    extract_config=extract_config, verify_config=verify_config,
                    allow_same_adapter_verify=a.allow_same_adapter_verify,
                )
                root = initialize_import(wiki, options)
                code = CourseImportRunner(
                    root, wiki=wiki, extract_adapter=ADAPTERS[extract_name](),
                    verify_adapter=ADAPTERS[verify_name](),
                    extract_config=options.extract_config,
                    verify_config=options.verify_config,
                    pages_config=_pages_run_config(a),
                    notifier=_import_notifier(
                        root,
                        Path(a.catalog).expanduser() if a.catalog
                        else default_catalog(dump_path.parent),
                    ),
                    allow_same_adapter_verify=a.allow_same_adapter_verify,
                ).run()
        except (
            SourceImportError, CommonConfigError, _MachineSetupError,
            KeyError, OSError, TypeError, ValueError,
        ) as error:
            # str(error), not error.code: a code without details is one word in the job log
            # (the claim_replay_conflict incident of 26.08). The machine JSON stays a code.
            reason = str(error)
            print(
                "research import: import_failed"
                if a.json else f"import interrupted: {reason}",
                file=sys.stderr,
            )
            if a.json:
                _emit_machine(_machine_error("import_failed"), operation="import")
            return EXIT_FAIL
        except Exception:
            if not a.json:
                raise
            print("research import: import_failed", file=sys.stderr)
            _emit_machine(_machine_error("import_failed"), operation="import")
            return EXIT_FAIL
        if a.json:
            return _emit_run_result(root, code, operation="import")
        print(root)
        if code == EXIT_PLAN_REVIEW:
            print(
                f"map is waiting for review: {root / 'work' / 'plan.md'}",
                file=sys.stderr,
            )
        return code
    if a.cmd == "verify-status":
        try:
            with _machine_silence() if a.json else nullcontext():
                report = _verify_status(a.topic_dir, dry_run=a.dry_run)
        except (OSError, TypeError, ValueError, UnicodeError) as error:
            print(
                "research verify-status: verification_failed"
                if a.json
                else f"verify-status interrupted: {error}",
                file=sys.stderr,
            )
            if a.json:
                _emit_machine(
                    _machine_error("verification_failed"),
                    operation="verify-status",
                )
            return EXIT_FAIL
        if a.json:
            return _emit_machine(
                {
                    **report,
                    "schema_version": MACHINE_SCHEMA_VERSION,
                    "status": "ok",
                },
                operation="verify-status",
            )
        _print_verify_report(report)
        return EXIT_OK
    if a.cmd == "status" and (bool(a.topic_dir) == bool(a.base)):
        p.error("status: pass exactly one of topic_dir or --base <root>")
    if a.cmd == "status" and a.base and a.raw:
        p.error("status: --raw applies only to one topic")
    if a.cmd == "status" and a.base and a.short:
        p.error("status: --short applies only to one topic")
    if a.cmd == "status" and a.short and (a.raw or a.json):
        p.error("status: --short is incompatible with --raw and --json")
    if a.cmd == "status" and a.stale_minutes <= 0:
        p.error("status: --stale-minutes must be positive")
    if a.cmd == "resume" and a.fatal_retries < 0:
        p.error("resume: --fatal-retries cannot be negative")
    if a.cmd == "resume" and a.transient_retries < 0:
        p.error("resume: --transient-retries cannot be negative")
    if a.cmd == "start":
        try:
            with _machine_silence() if a.json else nullcontext():
                lw = _llmwiki(machine=a.json)
        except SystemExit:
            if not a.json:
                raise
            _emit_machine(_machine_error("llmwiki_unavailable"), operation="start")
            return EXIT_FAIL
        except _MachineSetupError:
            print("research start: llmwiki_unavailable", file=sys.stderr)
            _emit_machine(_machine_error("llmwiki_unavailable"), operation="start")
            return EXIT_FAIL
        except Exception:
            if not a.json:
                raise
            print("research start: setup_failed", file=sys.stderr)
            _emit_machine(_machine_error("setup_failed"), operation="start")
            return EXIT_FAIL
        try:
            with _machine_silence() if a.json else nullcontext():
                root, err = _start_topic(lw, a, machine=a.json)
        except Exception:
            if not a.json:
                raise
            print("research start: start_failed", file=sys.stderr)
            _emit_machine(_machine_error("start_failed"), operation="start")
            return EXIT_FAIL
        if err:
            if a.json:
                print("research start: topic_conflict", file=sys.stderr)
                _emit_machine(_machine_error("topic_conflict"), operation="start")
            else:
                print(err, file=sys.stderr)
            return EXIT_FAIL
        _store_topic_name(root, a.name)
        if not a.json:
            print(root)
        if a.no_run:
            # We do not lie with "created": on an already started topic start resumes
            # (item 6) - we print the fact.
            if a.json:
                return _emit_run_result(root, EXIT_OK, operation="start")
            print(
                f"topic ready, checkpoint {(checkpoint.load(root) or {}).get('phase')} "
                f"(--no-run: orchestrator not started)"
            )
            return EXIT_OK
        code = _run(root, a, machine=True) if a.json else _run(root, a)
        if a.json:
            return _emit_run_result(root, code, operation="start")
        return code
    if a.cmd == "status":
        if a.base:
            try:
                with _machine_silence() if a.json else nullcontext():
                    rows = _status_base_rows(
                        a.base,
                        stale_minutes=a.stale_minutes,
                        wiki=_llmwiki(machine=a.json),
                    )
            except (_MachineSetupError, OSError, TypeError, ValueError):
                print("research status: base_status_failed", file=sys.stderr)
                if a.json:
                    _emit_machine(
                        {
                            "schema_version": MACHINE_SCHEMA_VERSION,
                            "status": "error",
                            "error": {"code": "base_status_failed"},
                        },
                        operation="status",
                    )
                return EXIT_FAIL
            if a.json:
                return _emit_machine(
                    {
                        "schema_version": MACHINE_SCHEMA_VERSION,
                        "status": "ok",
                        "base": str(Path(a.base)),
                        "stale_minutes": a.stale_minutes,
                        # v1 compatibility: older machine consumers read this name.
                        "fresh_minutes": a.stale_minutes,
                        "topics": rows,
                        "snapshot": _status_snapshot(a.base, rows),
                    },
                    operation="status",
                    max_bytes=None,
                )
            _print_base_table(rows)
            return EXIT_OK
        try:
            state = _load_machine_checkpoint(a.topic_dir)
        except ValueError:
            print("research status: invalid_checkpoint", file=sys.stderr)
            if not a.short:
                _emit_machine(_machine_error("invalid_checkpoint"), operation="status")
            return EXIT_FAIL
        if state is None:
            print(
                "checkpoint missing: this is not a researcher topic or the run never started",
                file=sys.stderr,
            )
            if not a.short:
                _emit_machine(_machine_error("checkpoint_not_found"), operation="status")
            return EXIT_FAIL
        try:
            with _machine_silence():
                payload = status_payload(a.topic_dir, state)
        except (OSError, TypeError, ValueError, UnicodeError, RecursionError):
            print("research status: invalid_checkpoint", file=sys.stderr)
            if not a.short:
                _emit_machine(_machine_error("invalid_checkpoint"), operation="status")
            return EXIT_FAIL
        phase = payload.get("phase")
        if phase not in checkpoint.PHASES:
            print("research status: invalid_checkpoint", file=sys.stderr)
            if not a.short:
                _emit_machine(_machine_error("invalid_checkpoint"), operation="status")
            return EXIT_FAIL
        if a.short:
            try:
                _print_short_status(
                    a.topic_dir, state, payload, stale_minutes=a.stale_minutes
                )
            except (OSError, TypeError, ValueError, UnicodeError, RecursionError):
                print("research status: short_status_failed", file=sys.stderr)
                return EXIT_FAIL
            return EXIT_OK
        payload = {
            **payload,
            "schema_version": MACHINE_SCHEMA_VERSION,
            "status": "done" if phase in {"done", "import:done"} else "running",
        }
        if not a.raw:
            payload.pop("queue", None)
            payload.pop("rounds", None)
            if payload.get("run_mode") == "import":
                for detail in ("documents", "registered", "decisions", "dump"):
                    payload.pop(detail, None)
        return _emit_machine(
            payload,
            operation="status",
            max_bytes=None if a.raw else MAX_MACHINE_OUTPUT_BYTES,
        )
    if a.cmd == "resume":
        try:
            state = (
                _load_machine_checkpoint(a.topic_dir)
                if a.json
                else checkpoint.load(a.topic_dir)
            )
        except ValueError:
            if not a.json:
                raise
            print("research resume: invalid_checkpoint", file=sys.stderr)
            _emit_machine(_machine_error("invalid_checkpoint"), operation="resume")
            return EXIT_FAIL
        if state is None:
            print("checkpoint missing: run start first", file=sys.stderr)
            if a.json:
                _emit_machine(
                    _machine_error("checkpoint_not_found"), operation="resume"
                )
            return EXIT_FAIL
        _store_topic_name(a.topic_dir, a.name)
        if a.supervise:
            try:
                code = _supervise_resume(a.topic_dir, a, machine=a.json)
            except (KeyError, OSError, TypeError, ValueError, UnicodeError) as error:
                print(
                    "research resume: supervisor_failed"
                    if a.json
                    else f"supervisor interrupted: {error}",
                    file=sys.stderr,
                )
                code = EXIT_FAIL
        else:
            try:
                code = _resume_once(a.topic_dir, a, machine=a.json)
            except Exception as error:
                if not a.json and not isinstance(error, (OSError, TypeError, ValueError)):
                    raise
                print(
                    "research resume: run_failed"
                    if a.json else f"run interrupted: {error}",
                    file=sys.stderr,
                )
                code = EXIT_FAIL
        if a.json:
            return _emit_run_result(a.topic_dir, code, operation="resume")
        return code
    if a.cmd == "extend":
        try:
            with _machine_silence() if a.json else nullcontext():
                from .orchestrator import begin_extend

                run_id, created = begin_extend(a.topic_dir)
        except Exception as e:
            if a.json:
                print("research extend: extend_not_started", file=sys.stderr)
                _emit_machine(_machine_error("extend_not_started"), operation="extend")
            else:
                if not isinstance(e, ValueError):
                    raise
                print(f"extension not started: {e}", file=sys.stderr)
            return EXIT_FAIL
        action = "started" if created else "continuing"
        _store_topic_name(a.topic_dir, a.name)
        if not a.json:
            print(f"extension {action}: run_id={run_id} | topic={Path(a.topic_dir)}")
        code = _run(a.topic_dir, a, machine=True) if a.json else _run(a.topic_dir, a)
        if a.json:
            return _emit_run_result(a.topic_dir, code, operation="extend")
        return code
    return EXIT_OK
