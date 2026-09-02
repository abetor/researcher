"""Emit researcher events through an independent shell hook."""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from . import checkpoint
from .course_queue import CATALOG_NAME, read_catalog
from .tool_config import ToolConfig


HOOK_TIMEOUT_SECONDS = 30
NOTICE_KINDS = frozenset({"progress", "done", "fail", "stall", "wait", "info"})
NOTICE_LINE_LIMIT = 200
NOTICE_LINES_LIMIT = 3
NOTICE_SUBJECT_LIMIT = 40
NOTICE_TITLE_LIMIT = 80
# v2 envelope class derived from the legacy kind (canon 25.08 item 5); kind stays in env unchanged.
NOTICE_CLASS_BY_KIND = {
    "progress": "event", "done": "event", "fail": "alert",
    "stall": "alert", "wait": "event", "info": "ack",
}
_ACTION_PREFIX = "Action:"
_COURSE_TOPIC_PREFIX = "Course: "

_DEFAULT_KINDS = {
    "run_started": "info",
    "cycle_done": "progress",
    "phase_changed": "progress",
    "done": "done",
    "gate": "fail",
    "stop": "fail",
}

# Titles: no caps, states in English (canon v2); regression guard - test_notifications.
_DEFAULT_TITLES = {
    "run_started": "start",
    "cycle_done": "progress",
    "phase_changed": "stage",
    "done": "done",
    "gate": "fail",
    "stop": "fail",
}


def _one_line(value: object) -> str:
    return " ".join(str(value).split())


def _decode_output(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def short_name(value: object, limit: int = NOTICE_SUBJECT_LIMIT) -> str:
    """Return the first complete words within the envelope or status limit."""
    text = _one_line(value)
    if len(text) <= limit:
        return text
    prefix = text[:limit].rstrip()
    # The cut landed inside a word - roll back to the boundary; on a space the word is already whole.
    if not text[limit].isspace() and not text[limit - 1].isspace() and " " in prefix:
        prefix = prefix.rsplit(" ", 1)[0]
    return prefix or text[:limit]


def _topic_text(topic_dir: Path) -> str:
    try:
        manifest = json.loads((topic_dir / "topic.json").read_text("utf-8"))
    except (OSError, TypeError, ValueError):
        return topic_dir.name
    value = manifest.get("topic") if isinstance(manifest, dict) else None
    if not isinstance(value, str) or not value.strip():
        return topic_dir.name
    return value


def _short_subject(topic_dir: Path) -> str:
    return short_name(_topic_text(topic_dir))


def _safe_state(topic_dir: Path) -> dict:
    try:
        state = checkpoint.load(topic_dir)
    except (OSError, TypeError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def _action(lines: Sequence[str]) -> str:
    for line in lines:
        if line.startswith(_ACTION_PREFIX):
            return line[len(_ACTION_PREFIX):].strip()
    return ""


def _lines(value: Sequence[object] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    raw = value.splitlines() if isinstance(value, str) else value
    result = []
    for item in raw:
        line = _one_line(item)
        if not line:
            continue
        result.append(line[:NOTICE_LINE_LIMIT])
        if len(result) == NOTICE_LINES_LIMIT:
            break
    return tuple(result)


def _hook_failed(topic_dir: Path, event: str, reason: str) -> None:
    state = checkpoint.load(topic_dir) or {}
    path = topic_dir / "work" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "event": "hook_failed",
        "phase": state.get("phase"),
        "run_id": state.get("run_id"),
        "hook_event": event,
        "reason": _one_line(reason)[:500],
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


class EventNotifier:
    """Invoke on_event with an envelope-v2 environment.

    RESEARCHER_FROM is coursedump for import topics identified by checkpoint run_mode
    or a Course prefix, and researcher otherwise. RESEARCHER_ABOUT is the course slug
    or short topic name. RESEARCHER_NAME is a course title from an explicit or adjacent
    catalog, falling back to the full topic. RESEARCHER_CLASS follows kind and
    RESEARCHER_ACTION comes from an Action line. Legacy envelope variables remain.
    """

    def __init__(
        self, topic_dir: str | Path, config: ToolConfig, *,
        catalog: str | Path | None = None,
    ):
        self.topic_dir = Path(topic_dir).resolve()
        self.command = config.on_event
        self.catalog = Path(catalog) if catalog is not None else None

    def _course_name(self, state: dict, slug: str) -> str:
        catalog = self.catalog
        if catalog is None and isinstance(state.get("dump"), str):
            catalog = Path(state["dump"]).parent.parent / CATALOG_NAME
        return read_catalog(catalog).get(slug, "") if catalog is not None else ""

    def envelope(self) -> dict[str, str]:
        """Resolve event-independent from, about, and name envelope fields."""
        state = _safe_state(self.topic_dir)
        topic = _one_line(_topic_text(self.topic_dir))
        if state.get("run_mode") == "import" or topic.startswith(_COURSE_TOPIC_PREFIX):
            slug = state.get("slug") if isinstance(state.get("slug"), str) else self.topic_dir.name
            return {
                "from": "coursedump", "about": slug,
                "name": _one_line(self._course_name(state, slug))[:NOTICE_LINE_LIMIT],
            }
        name = state.get("name")
        about = short_name(name) if isinstance(name, str) and name.strip() else short_name(topic)
        return {"from": "researcher", "about": about, "name": topic[:NOTICE_LINE_LIMIT]}

    def emit(
        self,
        event: str,
        text: str,
        *,
        kind: str | None = None,
        title: str | None = None,
        lines: Sequence[object] | str | None = None,
    ) -> bool:
        if not self.command:
            return False
        notice_kind = kind or _DEFAULT_KINDS.get(event, "info")
        if notice_kind not in NOTICE_KINDS:
            raise ValueError(
                f"unknown notice kind {notice_kind!r}; "
                f"allowed: {', '.join(sorted(NOTICE_KINDS))}"
            )
        subject = _short_subject(self.topic_dir)
        notice_title = _one_line(title or _DEFAULT_TITLES.get(event, event))[
            :NOTICE_TITLE_LIMIT
        ]
        notice_lines = _lines(lines)
        envelope = self.envelope()
        env = dict(os.environ)
        env.update({
            "RESEARCHER_EVENT": event,
            "RESEARCHER_KIND": notice_kind,
            "RESEARCHER_SUBJECT": subject,
            "RESEARCHER_TITLE": notice_title,
            "RESEARCHER_LINES": "\n".join(notice_lines),
            "RESEARCHER_FROM": envelope["from"],
            "RESEARCHER_ABOUT": envelope["about"],
            "RESEARCHER_NAME": envelope["name"],
            "RESEARCHER_CLASS": NOTICE_CLASS_BY_KIND[notice_kind],
            "RESEARCHER_ACTION": _action(notice_lines),
            # The legacy names stay a compatible contract for hooks already configured.
            "RESEARCHER_TOPIC": subject,
            "RESEARCHER_TEXT": _one_line(text),
            "RESEARCHER_TOPIC_DIR": str(self.topic_dir),
        })
        try:
            result = subprocess.run(
                ["/bin/sh", "-c", self.command],
                cwd=self.topic_dir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=HOOK_TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                return True
            stderr = _decode_output(result.stderr).strip()
            stdout = _decode_output(result.stdout).strip()
            reason = f"exit {result.returncode}: {stderr or stdout}"
        except subprocess.TimeoutExpired:
            reason = f"timeout after {HOOK_TIMEOUT_SECONDS} s"
        except (OSError, UnicodeError) as e:
            reason = str(e)
        print(f"research: hook_failed {event}: {_one_line(reason)}", file=sys.stderr)
        try:
            _hook_failed(self.topic_dir, event, reason)
        except OSError as e:
            print(
                f"research: hook_failed journal: {_one_line(e)}",
                file=sys.stderr,
            )
        return False
