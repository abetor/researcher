"""Synthesize a Markdown report from a topic corpus without network access.

One THINK-role harness call writes the report through the standard researcher adapter.
The ``--guard`` option lets a scheduler wait quietly until the topic checkpoint is done
and no process holds ``resume.lock``. Notifications use the configured ``on_event``
hook rather than a scheduler-specific exit convention.

Exit 76 means the report was written or already existed and now awaits human review.
Exit 0 means a guarded topic is not terminal yet, 75 means quota exhaustion, 111 means a
transient failure, and 1 covers invalid input, a missing report, or an unguarded topic
that has not completed.
"""
from __future__ import annotations

import datetime
import fcntl
import os
import stat
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from . import checkpoint
from .adapters.base import ROLE_THINK, HarnessAdapter

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_QUOTA = 75
EXIT_REPORT_READY = 76
EXIT_TRANSIENT = 111

PLACEHOLDERS = ("__TOPIC__", "__OUT__", "__NAME__", "__TOPICS_ROOT__")

DEFAULT_TEMPLATE = """# Role: researcher corpus synthesizer over staging (one pass, no network)

Topic: __NAME__
Topic directory: __TOPIC__
Output: __OUT__ (markdown, in Russian, no emoji, hyphen instead of a dash, plain links in parentheses)

## Data (read only)

- `__TOPIC__/staging/claims.jsonl` and `__TOPIC__/final/claims.jsonl` - claims: `text`,
  `confidence`, `evidence[]` (source_id, quote, locator), status.
- `__TOPIC__/sources/sources.jsonl` - sources: `id`, `url`, `title`, `kind`, `fetched_at`.
- `__TOPIC__/work/checkpoint.json` - `stop`, `judge`, `gaps`, `breadth`, `search_map_progress`
  (which map clusters were covered) - for the data boundary.
- `__TOPIC__/work/search-map.json` - the agreed cluster map.
- Neighbouring topics of the base `__TOPICS_ROOT__` - may be cited as `[topic: <dir>]`.

The files are large - work through scripts (python/jq), do not read them whole into context.
First count: total claims, claims by confidence, how many carry numbers, sources by domain,
which map clusters were covered; that goes into the header.

## What to write

1. A "Data boundary" header: how many claims/sources, which map clusters were NOT executed,
   the stop reason verbatim, and what that means for the reader (where zeros = no data).
2. "The takeaway in two minutes" - 5-8 paragraphs with the main statements, each with 1-3
   links to sources (url from sources.jsonl).
3. Sections following the structure of the topic. Every number with a source and a date;
   show both sides of a contradiction, do not smooth it over.
4. "What this means for the reader" - concrete decisions.
5. "What can be skipped" and "Where the corpus does not support a recommendation".
6. "Gaps for a follow-up search": concrete wordings of clusters/queries (en/ru) - input for
   the next run.

Rules: only from the corpus, nothing from memory; every factual statement carries a link;
mark low-confidence and single-source items; length 40-70 KB. Write the file __OUT__ whole in
a single write at the end (atomic), do not create or modify any other file.
"""


def render_prompt(template: str, *, topic_dir: Path, out: Path, name: str,
                  topics_root: Path) -> str:
    values = {
        "__TOPIC__": str(topic_dir),
        "__OUT__": str(out),
        "__NAME__": name,
        "__TOPICS_ROOT__": str(topics_root),
    }
    for key, value in values.items():
        template = template.replace(key, value)
    return template


def topic_name(topic_dir: Path) -> str:
    """Read the topic name from its manifest, falling back to the directory name."""
    import json

    try:
        manifest = json.loads((topic_dir / "topic.json").read_text("utf-8"))
    except (OSError, ValueError, TypeError):
        return topic_dir.name
    value = manifest.get("topic") if isinstance(manifest, dict) else None
    return value if isinstance(value, str) and value.strip() else topic_dir.name


def topic_terminal(topic_dir: Path) -> tuple[bool, str]:
    """Inspect durable state for completion while the caller holds ``resume.lock``."""
    try:
        state = checkpoint.load(topic_dir)
    except ValueError as e:
        return False, f"checkpoint is unreadable: {e}"
    if state is None:
        return False, "no checkpoint - the topic never started"
    phase = state.get("phase")
    if phase != "done":
        return False, f"phase={phase!r}"
    return True, ""


@contextmanager
def _nonblocking_lock(path: Path, operation: int):
    """Yield the open locked fd, or None when a live competitor holds it."""
    with path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), operation | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
            return
        try:
            yield lock
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _temporary_report(out: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=out.parent, prefix=f".{out.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    return Path(name)


def _valid_report(path: Path) -> bool:
    """Minimal terminal-ready gate: a regular, non-empty UTF-8 markdown file."""
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            return False
        return bool(path.read_text("utf-8").strip())
    except (OSError, UnicodeError):
        return False


def _remove_temporary(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as error:
        print(f"report: temporary file not removed: {path}: {error}", file=sys.stderr)


def _log_path(topic_dir: Path) -> Path:
    logs = topic_dir / "work" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return logs / f"report-{stamp}.log"


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def run_report(topic_dir: str | Path, out: str | Path, *, adapter: HarnessAdapter,
               live_pid, notifier=None, name: Optional[str] = None,
               template: Optional[str] = None, topics_root: Optional[Path] = None,
               model: Optional[str] = None, effort: Optional[str] = None,
               timeout: Optional[float] = None, guard: bool = False,
               retries: int = 2, retry_backoff: float = 5.0) -> int:
    topic_dir = Path(topic_dir).resolve()
    out = Path(out).expanduser().resolve()
    if not out.parent.is_dir():
        print(f"report: report directory does not exist: {out.parent}", file=sys.stderr)
        return EXIT_FAIL
    output_lock = out.with_name(f".{out.name}.report.lock")
    try:
        with _nonblocking_lock(output_lock, fcntl.LOCK_EX) as output_guard:
            if output_guard is None:
                print(f"report: {out} is already being built by another process", file=sys.stderr)
                return EXIT_TRANSIENT
            if _size(out) > 0:
                print(f"report: {out} already exists ({_size(out)} bytes) - the report is "
                      "not overwritten, delete the file to rebuild", file=sys.stderr)
                return EXIT_REPORT_READY

            topic_lock = topic_dir / "work" / "resume.lock"
            with _nonblocking_lock(topic_lock, fcntl.LOCK_SH) as topic_guard:
                if topic_guard is None:
                    try:
                        pid = live_pid(topic_dir)
                    except (OSError, TypeError, ValueError):
                        pid = None
                    reason = (
                        f"resume.lock is held by pid {pid}"
                        if pid is not None else "resume.lock is held by a live process"
                    )
                    if guard:
                        print(f"report: the search has not finished yet ({reason}), waiting",
                              file=sys.stderr)
                        return EXIT_OK
                    print(
                        f"report: topic is not done ({reason}); for hourly waiting use "
                        "--guard", file=sys.stderr,
                    )
                    return EXIT_FAIL

                ready, reason = topic_terminal(topic_dir)
                if not ready:
                    if guard:
                        print(f"report: the search has not finished yet ({reason}), waiting",
                              file=sys.stderr)
                        return EXIT_OK
                    print(
                        f"report: topic is not done ({reason}); for hourly waiting use "
                        "--guard", file=sys.stderr,
                    )
                    return EXIT_FAIL

                temporary = _temporary_report(out)
                try:
                    report_name = name or topic_name(topic_dir)
                    prompt = render_prompt(
                        template or DEFAULT_TEMPLATE, topic_dir=topic_dir,
                        out=temporary, name=report_name,
                        topics_root=topics_root or topic_dir.parent,
                    )
                    selected_model = model or adapter.default_model_for_role(ROLE_THINK)
                    log_path = _log_path(topic_dir)
                    started = time.monotonic()
                    # The shared topic lock is held until the terminal replace: extend/resume/
                    # pages cannot change the corpus while the harness reads and synthesizes it.
                    result = adapter.run(
                        prompt, cwd=str(out.parent), timeout=timeout, retries=retries,
                        retry_backoff=retry_backoff, model=selected_model, effort=effort,
                        network=False,
                    )
                    minutes = max(1, round((time.monotonic() - started) / 60))
                    try:
                        log_path.write_text(
                            f"# report {report_name}\n# out={out}\n# stop={result.stop} "
                            f"exit={result.exit_code}\n{result.text}\n--- stderr ---\n"
                            f"{result.stderr}\n", "utf-8",
                        )
                    except OSError as error:
                        print(f"report: log not written: {error}", file=sys.stderr)
                    if result.stop == "quota":
                        print(f"report: harness quota exhausted, log {log_path}", file=sys.stderr)
                        return EXIT_QUOTA
                    if result.stop == "transient":
                        print(f"report: transient harness failure, log {log_path}",
                              file=sys.stderr)
                        return EXIT_TRANSIENT
                    if result.stop == "done" and _valid_report(temporary):
                        if _size(out) > 0:
                            print(f"report: {out} appeared during the build and was kept",
                                  file=sys.stderr)
                            return EXIT_REPORT_READY
                        os.replace(temporary, out)
                        size = _size(out)
                        print(
                            f"report: done {out} ({size} bytes, {minutes} min), "
                            f"log {log_path}"
                        )
                        if notifier is not None:
                            notifier.emit(
                                "report_done", f"report done {out}", kind="done",
                                title="report done",
                                lines=[f"{size // 1024} KB in {minutes} min",
                                       f"File: {out}"],
                            )
                        return EXIT_REPORT_READY
                    diagnostic = (result.stderr or "").strip().splitlines()
                    reason = diagnostic[-1] if diagnostic else (
                        f"harness exit {result.exit_code}, stop={result.stop}"
                    )
                    print(f"report: file did not appear ({reason}), log {log_path}",
                          file=sys.stderr)
                    if notifier is not None:
                        notifier.emit(
                            "report_failed", f"report failed {out}", kind="fail",
                            title="report failed",
                            lines=[reason[:200], f"Log: {log_path}",
                                   f"Action: research report {topic_dir} --out {out}"],
                        )
                    return EXIT_FAIL
                finally:
                    _remove_temporary(temporary)
    except OSError as error:
        print(f"report: lock unavailable: {error}", file=sys.stderr)
        return EXIT_FAIL
