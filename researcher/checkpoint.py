"""Store the logical run checkpoint under the topic's work/ directory.

Harness sessions are mutually incompatible, so portability between machines and
harnesses depends on repository-owned file state. Harness-specific session IDs are
only warm-resume optimizations; this checkpoint remains authoritative.
"""
import datetime
import json
import os
import tempfile
from pathlib import Path

SEARCH_PHASES = ("planned", "collecting", "verifying", "synthesizing", "done")
IMPORT_PHASES = (
    "import:plan",
    "import:register",
    "import:extract",
    "import:verify",
    "import:pages",
    "import:done",
)
PHASES = SEARCH_PHASES + IMPORT_PHASES


def _path(topic_dir: str | Path) -> Path:
    return Path(topic_dir) / "work" / "checkpoint.json"


def save(topic_dir: str | Path, state: dict) -> None:
    if state.get("phase") not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {state.get('phase')!r}")
    state = dict(state)
    state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    path = _path(topic_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False) as fh:
        fh.write(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        tmp = Path(fh.name)
    os.replace(tmp, path)


def load(topic_dir: str | Path) -> dict | None:
    path = _path(topic_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))
