"""Expose run observability from durable topic state.

Counters come from disk rather than orchestrator memory, so banners and status show
what survives interruption and remains visible to the next resume.
"""
from __future__ import annotations

import copy
import datetime
import json
import os
import tempfile
import threading
from pathlib import Path

from . import checkpoint


HEARTBEAT_ROLES = frozenset({"collect", "critic", "judge", "synth", "plan", "verify"})


def _heartbeat_path(topic_dir: str | Path) -> Path:
    return Path(topic_dir) / "work" / "heartbeat.json"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _write_heartbeat_atomic(path: Path, value: dict) -> None:
    """One replace instead of a log: the heartbeat reports only the current work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent,
        prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def read_heartbeat(topic_dir: str | Path) -> dict | None:
    """Read best effort; absence or a partial value must not break topic status."""
    try:
        value = json.loads(_heartbeat_path(topic_dir).read_text("utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("role") not in HEARTBEAT_ROLES:
        return None
    return value


class RoleHeartbeat:
    """Write an atomic active-role snapshot safe for a collector pool.

    Up to three collector calls may run while the file contract remains singular.
    Memory tracks all calls and disk shows the most recently started one. Completion
    switches to another live call; only the last completion clears call metadata.
    """

    def __init__(self, topic_dir: str | Path):
        self._path = _heartbeat_path(topic_dir)
        self._lock = threading.Lock()
        self._active: dict[object, dict] = {}

    def _write(self, value: dict) -> None:
        _write_heartbeat_atomic(self._path, {**value, "at": _now_iso()})

    def start(self, *, phase: str, role: str, cycle: int | None) -> object:
        if role not in HEARTBEAT_ROLES:
            raise ValueError(f"unknown heartbeat role: {role!r}")
        token = object()
        value = {
            "phase": phase,
            "role": role,
            "call_started": _now_iso(),
            "pid": os.getpid(),
            "harness_pid": None,
            "cycle": cycle,
        }
        with self._lock:
            self._active[token] = value
            self._write(value)
        return token

    def harness_pid(self, token: object, pid: int | None) -> None:
        with self._lock:
            value = self._active.get(token)
            if value is None:
                return
            value["harness_pid"] = pid
            self._write(value)

    def finish(self, token: object) -> None:
        with self._lock:
            finished = self._active.pop(token, None)
            if finished is None:
                return
            if self._active:
                self._write(next(reversed(self._active.values())))
                return
            self._write({**finished, "call_started": None, "harness_pid": None})


def _jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text("utf-8").splitlines() if line.strip())


def _claim_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        json.loads(line)["id"]
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    }


def topic_counts(topic_dir: str | Path) -> dict[str, int]:
    """Return aggregate topic counters for CLI output and progress."""
    root = Path(topic_dir)
    state = checkpoint.load(root) or {}
    wiki_dir = root / "final" / "wiki"
    work_dir = root / "work"
    if state.get("run_mode") == "extend" and isinstance(state.get("run_id"), str):
        work_dir = work_dir / "runs" / state["run_id"]
    found_dir = work_dir / "found"
    progress = search_map_progress(root, state)
    return {
        "claims": _jsonl_rows(root / "final" / "claims.jsonl"),
        "sources": _jsonl_rows(root / "sources" / "sources.jsonl"),
        "pages": len(list(wiki_dir.glob("*.md"))) if wiki_dir.is_dir() else 0,
        "cycles": state.get("cycles", 0),
        "found": sum(_jsonl_rows(path) for path in found_dir.glob("*.jsonl"))
        if found_dir.is_dir() else 0,
        "staging": _jsonl_rows(root / "staging" / "claims.jsonl"),
        "final": _jsonl_rows(root / "final" / "claims.jsonl"),
        "map_completed": progress["completed"],
        "map_total": progress["total"],
    }


def search_map_progress(topic_dir: str | Path, state: dict) -> dict:
    """Return durable progress for the reviewed search map, including old runs."""
    root = Path(topic_dir)
    work_dir = root / "work"
    if state.get("run_mode") == "extend" and isinstance(state.get("run_id"), str):
        work_dir = work_dir / "runs" / state["run_id"]
    disk_cluster_ids = None
    try:
        search_map = json.loads((work_dir / "search-map.json").read_text("utf-8"))
    except (OSError, ValueError, TypeError):
        pass
    else:
        clusters = search_map.get("clusters") if isinstance(search_map, dict) else None
        if isinstance(clusters, list):
            disk_cluster_ids = list(dict.fromkeys(
                row["id"]
                for row in clusters
                if isinstance(row, dict) and isinstance(row.get("id"), str)
            ))

    stored = state.get("search_map_progress")
    cluster_ids = stored.get("cluster_ids") if isinstance(stored, dict) else None
    if not isinstance(cluster_ids, list) or not all(
        isinstance(item, str) for item in cluster_ids
    ):
        cluster_ids = disk_cluster_ids or []
    cluster_ids = list(dict.fromkeys(cluster_ids))
    done = {item for item in state.get("done", []) if isinstance(item, str)}
    stored_completed = (
        stored.get("completed_ids") if isinstance(stored, dict) else None
    )
    if isinstance(stored_completed, list):
        done.update(item for item in stored_completed if isinstance(item, str))
    completed_ids = [item for item in cluster_ids if item in done]
    pending_ids = [item for item in cluster_ids if item not in done]
    return {
        "cluster_ids": cluster_ids,
        "completed_ids": completed_ids,
        "pending_ids": pending_ids,
        "completed": len(completed_ids),
        "total": len(cluster_ids),
        "on_disk": len(disk_cluster_ids) if disk_cluster_ids is not None else None,
        "disk_mismatch": (
            disk_cluster_ids is not None and disk_cluster_ids != cluster_ids
        ),
    }


def empty_funnel() -> dict:
    return {
        "found": 0,
        "staging": {"keep": 0, "drop": 0},
        "final": {"keep": 0, "drop": 0},
    }


def completed_funnel_from_disk(topic_dir: str | Path) -> dict:
    """Reconstruct the post-verification funnel from actual topic zones."""
    root = Path(topic_dir)
    counts = topic_counts(topic_dir)
    staging_ids = _claim_ids(root / "staging" / "claims.jsonl")
    final_ids = _claim_ids(root / "final" / "claims.jsonl")
    after_critic = len(staging_ids | final_ids)
    return {
        "found": counts["found"],
        "staging": {
            "keep": after_critic,
            "drop": max(0, counts["found"] - after_critic),
        },
        "final": {
            "keep": len(final_ids),
            "drop": len(staging_ids - final_ids),
        },
    }


def _pages_planned(root: Path) -> int | None:
    path = root / "work" / "pages-plan.json"
    if not path.is_file():
        return None
    try:
        plan = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return len(plan) if isinstance(plan, list) else None


def status_payload(topic_dir: str | Path, state: dict) -> dict:
    """Build status data, reconstructing old completed funnels from disk."""
    out = copy.deepcopy(state)
    if state.get("run_mode") == "import":
        root = Path(topic_dir)
        documents = [
            row for row in state.get("documents", []) if isinstance(row, dict)
        ]
        decisions = state.get("decisions")
        selected = None
        if isinstance(decisions, dict):
            selected = sum(
                1 for row in documents if decisions.get(row.get("nnn")) != "DROP"
            )
        extracted = root / "work" / "extracted"
        verified = root / "work" / "verified"
        out["import_progress"] = {
            "documents": len(documents),
            "selected": selected,
            "registered": len(state.get("registered", {}))
            if isinstance(state.get("registered"), dict) else 0,
            "extracted": len(list(extracted.glob("*.json"))) if extracted.is_dir() else 0,
            "verified": len(list(verified.glob("*.json"))) if verified.is_dir() else 0,
            "pages_planned": _pages_planned(root),
            "pages": len(list((root / "final" / "wiki").glob("*.md")))
            if (root / "final" / "wiki").is_dir() else 0,
        }
        return out
    out.setdefault("rounds", [])
    out["search_map_progress"] = search_map_progress(topic_dir, state)
    if isinstance(out.get("funnel"), dict):
        return out

    funnel = empty_funnel()
    if state.get("phase") == "done":
        funnel = completed_funnel_from_disk(topic_dir)
    else:
        funnel["found"] = topic_counts(topic_dir)["found"]
    out["funnel"] = funnel
    return out
