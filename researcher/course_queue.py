"""Process the next unfinished course listed in ``queue.txt``.

The queue uses one slug per line and treats ``#`` as a comment. A dump lives at
``out/<slug>`` and its topic at ``<base>/<slug>``. Each invocation processes exactly
one course through import and page generation, then returns that course's exit code so
a scheduler can pick up the next item later. An empty queue is normal and returns zero.
Exit 76 means the import plan needs approval. Fatal errors, human waits, and refused
gates stop the queue for inspection instead of silently skipping incomplete material.
"""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import checkpoint

EXIT_OK = 0
# Local course-queue semantics: no available work is a success, not a scheduler
# stop. The shared exit 76 belongs to plan-review / waiting_human only.
EXIT_QUEUE_EMPTY = EXIT_OK
CATALOG_NAME = "catalog.tsv"
# Course stages in snapshot terms (canon v2 item 6): three stages; plan/register take
# seconds, so they are merged into extract.
_STAGES = {
    "import:plan": (1, "extract"),
    "import:register": (1, "extract"),
    "import:extract": (1, "extract"),
    "import:verify": (2, "verify"),
    "import:pages": (3, "pages"),
}
_STAGE_ORDER = ("extract", "verify", "pages")
# Timing model: seconds per transcript window, from the import receipts of seven courses
# on 23-25.08 (extract 15-36 s, verify 24-44 s, pages 51-79 s per window) - ~2 min total.
STAGE_SECONDS_PER_WINDOW = {"extract": 18, "verify": 30, "pages": 72}


@dataclass(frozen=True)
class CourseReservation:
    slug: str | None
    rows: list[dict[str, str]]


def read_queue(path: Path) -> list[str]:
    slugs: list[str] = []
    for raw in path.read_text("utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and line not in slugs:
            slugs.append(line)
    return slugs


def _lock_held(topic: Path) -> bool:
    """Is resume.lock live: the flock is held only by a running process, while a stale
    file left by a dead process is free."""
    path = topic / "work" / "resume.lock"
    try:
        with path.open("r", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            return False
    except OSError:
        return False


def _reservation_path(queue: Path, slug: str) -> Path:
    digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()
    return queue.parent / f".{queue.name}.reservations" / f"{digest}.lock"


def _reservation_held(queue: Path, slug: str) -> bool:
    path = _reservation_path(queue, slug)
    try:
        with path.open("r", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            return False
    except OSError:
        return False


def course_state(base: Path, dumps: Path, slug: str, *, queue: Path | None = None) -> str:
    """no-dump | done | foreign | running | stopped | pending."""
    dump = dumps / slug
    if not (dump / "INDEX.md").is_file():
        return "no-dump"
    if queue is not None and _reservation_held(queue, slug):
        return "running"
    topic = base / slug
    if not topic.exists():
        return "pending"
    state = checkpoint.load(topic)
    if state is None:
        return "foreign"
    if state.get("run_mode") != "import":
        return "foreign"
    # Busy beats stop/phase: a parallel worker (a second job of the same queue) must
    # neither take the course nor call it a STOP by checkpoint while the run is alive.
    if _lock_held(topic):
        return "running"
    if state.get("phase") == "import:done":
        return "done"
    if state.get("stop_kind") in {"gate_refused", "fatal"}:
        return "stopped"
    return "pending"


def queue_rows(queue: Path, base: Path, dumps: Path) -> list[dict[str, str]]:
    return [
        {"slug": slug, "state": course_state(base, dumps, slug, queue=queue)}
        for slug in read_queue(queue)
    ]


@contextmanager
def reserve_next_course(queue: Path, base: Path, dumps: Path):
    """Atomically claim one slug without serializing work on different courses."""
    selected_slug = None
    selected_state = None
    selected_lock = None
    try:
        for slug in read_queue(queue):
            path = _reservation_path(queue, slug)
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            state = course_state(base, dumps, slug)
            if state not in {"pending", "stopped"}:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                lock.close()
                continue
            selected_slug = slug
            selected_state = state
            selected_lock = lock
            break
        rows = queue_rows(queue, base, dumps)
        if selected_slug is not None:
            for row in rows:
                if row["slug"] == selected_slug:
                    row["state"] = selected_state
                    break
        yield CourseReservation(selected_slug, rows)
    finally:
        if selected_lock is not None:
            fcntl.flock(selected_lock.fileno(), fcntl.LOCK_UN)
            selected_lock.close()


def next_course(rows: list[dict[str, str]]) -> str | None:
    for row in rows:
        if row["state"] in {"pending", "stopped"}:
            return row["slug"]
    return None


def render_rows(rows: list[dict[str, str]]) -> str:
    labels = {
        "done": "done", "pending": "needs a run", "stopped": "STOP (investigate)",
        "running": "running (another worker) - skipped",
        "foreign": "topic not from import (archive flow) - skipped", "no-dump": "no transcript",
    }
    lines = [f"  {row['slug']} | {labels.get(row['state'], row['state'])}" for row in rows]
    remaining = sum(1 for row in rows if row["state"] in {"pending", "stopped", "running"})
    lines.append(f"remaining: {remaining}")
    return "\n".join(lines)


def summary_json(
    rows: list[dict[str, str]], slug: str | None, code: int | None,
    snapshot: dict | None = None,
) -> str:
    payload = {
        "schema_version": 1,
        "queue": rows,
        "course": slug,
        "exit_code": code,
        "queue_empty": next_course(rows) is None,
        "remaining": sum(
            1 for row in rows if row["state"] in {"pending", "stopped", "running"}
        ),
    }
    if snapshot is not None:
        payload["snapshot"] = snapshot
    return json.dumps(payload, ensure_ascii=False)


def default_catalog(dumps: Path) -> Path | None:
    """Return ``<dumps>/../catalog.tsv`` when present; treat it strictly as data."""
    path = dumps.parent / CATALOG_NAME
    return path if path.is_file() else None


def read_catalog(path: str | Path | None) -> dict[str, str]:
    """Read a ``slug -> title`` mapping from a coursedump ``catalog.tsv`` file.

    This is a data contract, not a package import. Lines starting with ``#`` are
    comments. The first non-comment row may be a header containing ``slug`` and
    ``title``; without such a header, the first two columns are used. A missing or
    unreadable file yields an empty mapping.
    """
    if path is None:
        return {}
    try:
        text = Path(path).read_text("utf-8")
    except (OSError, UnicodeError):
        return {}
    titles: dict[str, str] = {}
    slug_col, title_col = 0, 1
    header_seen = False
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cells = [cell.strip() for cell in raw.split("\t")]
        if not header_seen:
            header_seen = True
            if "slug" in cells and "title" in cells:
                slug_col, title_col = cells.index("slug"), cells.index("title")
                continue
        if len(cells) <= max(slug_col, title_col):
            continue
        slug, title = cells[slug_col], cells[title_col]
        if slug and title and title != "-":
            titles.setdefault(slug, title)
    return titles


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None


def _iso(timestamp: float) -> str:
    return datetime.datetime.fromtimestamp(
        timestamp, datetime.timezone.utc
    ).isoformat(timespec="seconds")


def _parse_iso(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.timezone.utc)
    return moment.timestamp()


def _worker_pid(topic: Path) -> int | None:
    heartbeat = _read_json(topic / "work" / "heartbeat.json")
    if isinstance(heartbeat, dict) and type(heartbeat.get("pid")) is int:
        return heartbeat["pid"]
    try:
        match = re.search(r"pid=(\d+)", (topic / "work" / "resume.lock").read_text("utf-8"))
    except (OSError, UnicodeError):
        return None
    return int(match.group(1)) if match else None


def _latest_mtime(paths) -> float | None:
    mtimes = []
    for path in paths:
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes) if mtimes else None


def _stage_progress(topic: Path, state: dict, stage: str) -> tuple[int, int | None, float | None]:
    """(done, total, stage start time) from the topic's durable artifacts.

    extract/verify: `extract:NNN`/`verify:NNN` in state["done"] against the selected (not
    DROP) documents; pages: pages with status=done in work/pages-plan.json against all.
    Stage start: extract - the checkpoint's started_at; verify - the mtime of the last
    extracted file; pages - the mtime of import-receipt.json (written when pages begins).
    """
    work = topic / "work"
    if stage == "pages":
        plan = _read_json(work / "pages-plan.json")
        if not isinstance(plan, list):
            return 0, None, _latest_mtime([work / "import-receipt.json"])
        done = sum(1 for page in plan if isinstance(page, dict) and page.get("status") == "done")
        return done, len(plan), _latest_mtime([work / "import-receipt.json"])
    decisions = state.get("decisions")
    total = None
    if isinstance(decisions, dict):
        total = sum(
            1 for row in state.get("documents", [])
            if isinstance(row, dict) and decisions.get(row.get("nnn")) != "DROP"
        )
    prefix = f"{stage}:"
    done = sum(1 for item in state.get("done", []) if isinstance(item, str) and item.startswith(prefix))
    if stage == "extract":
        started = _parse_iso(state.get("started_at"))
    else:
        extracted = work / "extracted"
        started = _latest_mtime(extracted.glob("*.json")) if extracted.is_dir() else None
    return done, total, started


def _selected_windows(topic: Path, state: dict) -> int | None:
    plan = _read_json(topic / "work" / "import-plan.json")
    documents = plan.get("documents") if isinstance(plan, dict) else None
    if not isinstance(documents, list):
        return None
    decisions = state.get("decisions")
    dropped = set()
    if isinstance(decisions, dict):
        dropped = {
            row.get("document_id") for row in state.get("documents", [])
            if isinstance(row, dict) and decisions.get(row.get("nnn")) == "DROP"
        }
    total = 0
    for document in documents:
        if not isinstance(document, dict) or document.get("document_id") in dropped:
            continue
        count = document.get("window_count")
        total += count if type(count) is int else 0
    return total


def _jsonl_rows(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text("utf-8").splitlines() if line.strip())
    except (OSError, UnicodeError):
        return 0


def _worker(topic: Path, slug: str, name: str | None, now: float) -> dict:
    worker = {
        "item": slug, "name": name, "pid": _worker_pid(topic), "job": None,
        "stage": None, "done": None, "total": None, "started_at": None,
        "eta_seconds": None, "eta_total_seconds": None,
        "state": "running", "note": None, "metrics": [],
    }
    try:
        state = checkpoint.load(topic)
    except (OSError, UnicodeError, ValueError):
        state = None
    if not isinstance(state, dict):
        return worker
    started_at = _parse_iso(state.get("started_at"))
    if started_at is not None:
        worker["started_at"] = _iso(started_at)
    stage = _STAGES.get(state.get("phase"))
    windows = _selected_windows(topic, state)
    if stage is None:
        if windows:
            worker["metrics"].append(f"windows {windows}")
        return worker
    index, stage_name = stage
    worker["stage"] = {"i": index, "n": len(_STAGE_ORDER), "name": stage_name}
    done, total, stage_started = _stage_progress(topic, state, stage_name)
    worker["done"], worker["total"] = done, total
    remaining_fraction = (total - done) / total if total else 1.0
    later = _STAGE_ORDER[_STAGE_ORDER.index(stage_name) + 1:]
    if done and total and stage_started is not None and now > stage_started:
        # Speed of this stage: elapsed/done * remaining.
        eta = (now - stage_started) / done * (total - done)
        worker["eta_seconds"] = int(round(eta))
        if windows:
            worker["eta_total_seconds"] = int(round(
                eta + sum(windows * STAGE_SECONDS_PER_WINDOW[name] for name in later)
            ))
    elif windows:
        # No speed data yet - fall back to the ~2 min per transcript window model.
        elapsed = max(0.0, now - started_at) if started_at is not None else 0.0
        eta_total = max(0.0, windows * sum(STAGE_SECONDS_PER_WINDOW.values()) - elapsed)
        eta = min(eta_total, windows * STAGE_SECONDS_PER_WINDOW[stage_name] * remaining_fraction)
        worker["eta_seconds"] = int(round(eta))
        worker["eta_total_seconds"] = int(round(eta_total))
    if windows:
        worker["metrics"].append(f"windows {windows}")
    claims = _jsonl_rows(topic / "staging" / "claims.jsonl") + _jsonl_rows(topic / "final" / "claims.jsonl")
    if claims:
        worker["metrics"].append(f"claims {claims}")
    return worker


def snapshot(
    rows: list[dict[str, str]], base: Path, catalog: str | Path | None, *,
    now: float | None = None,
) -> dict:
    """Build the coursedump status fragment consumed by a gateway snapshot.

    Stable schema, covered by ``tests/test_snapshot.py``:
      {"flow": "coursedump",
       "workers": [{"item": slug, "name": title|null, "pid": int|null, "job": null,
                    "stage": {"i": 1..3, "n": 3, "name": "extract|verify|pages"}|null,
                    "done": int|null, "total": int|null, "started_at": iso|null,
                    "eta_seconds": int|null, "eta_total_seconds": int|null,
                    "state": "running", "note": null, "metrics": ["windows N", "claims N"]}],
       "queue": [{"item": slug, "name": title|null, "state": "waiting"|"fail"}]}
    ``workers`` contains running courses with a live ``resume.lock`` flock or active
    reservation. ``queue`` contains pending and stopped courses in ``queue.txt`` order.
    ``name`` is the title from ``catalog.tsv`` or null when no match is available.
    """
    now = time.time() if now is None else now
    titles = read_catalog(catalog)
    workers = [
        _worker(base / row["slug"], row["slug"], titles.get(row["slug"]), now)
        for row in rows if row["state"] == "running"
    ]
    queue = [
        {
            "item": row["slug"], "name": titles.get(row["slug"]),
            "state": "fail" if row["state"] == "stopped" else "waiting",
        }
        for row in rows if row["state"] in {"pending", "stopped"}
    ]
    return {"flow": "coursedump", "workers": workers, "queue": queue}


__all__ = [
    "CourseReservation", "EXIT_QUEUE_EMPTY", "read_queue", "course_state",
    "queue_rows", "reserve_next_course", "next_course", "render_rows", "summary_json",
    "default_catalog", "read_catalog", "snapshot",
]
