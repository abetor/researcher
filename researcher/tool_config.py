"""Load fail-closed researcher configuration separately from shared config.toml."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


class ToolConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ToolConfig:
    path: Path
    exists: bool
    on_event: str | None = None


def researcher_home() -> Path:
    raw = os.environ.get("RESEARCHER_HOME")
    return Path(raw).expanduser() if raw else Path.home() / "tools-data" / "researcher-data"


def load_tool_config(root: str | Path | None = None) -> ToolConfig:
    home = researcher_home() if root is None else Path(root).expanduser()
    path = home / "config.toml"
    if not path.exists() and not path.is_symlink():
        return ToolConfig(path=path, exists=False)
    try:
        raw = tomllib.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValueError) as e:
        raise ToolConfigError(f"{path}: file is unreadable: {e}") from e
    if not isinstance(raw, dict):
        raise ToolConfigError(f"{path}: expected a TOML table")
    unknown = set(raw) - {"on_event"}
    if unknown:
        raise ToolConfigError(f"{path}: unknown fields {sorted(unknown)}")
    command = raw.get("on_event")
    if command is not None and (
            not isinstance(command, str) or not command.strip()
            or command != command.strip()):
        raise ToolConfigError(
            f"{path}: on_event must be a nonempty string without surrounding whitespace")
    return ToolConfig(path=path, exists=True, on_event=command)
