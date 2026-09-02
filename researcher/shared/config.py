"""Fail-closed TOML loading: unknown fields are errors, not ignored values.

Silently accepting a misspelled field produces configuration that appears applied
but has no effect. This vendored layer therefore rejects unknown fields.
"""
from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]{0,127}")
_HARNESS_NAMES = frozenset({"claude", "codex"})


class CommonConfigError(ValueError):
    """The shared config.toml exists but does not conform to schema v1."""


@dataclass(frozen=True)
class CommonPaths:
    vault: Path
    topics_root: Path
    sources_root: Path


@dataclass(frozen=True)
class HarnessCommand:
    name: str
    argv: tuple[str, ...]
    model_default: str | None = None
    effort_default: str | None = None


@dataclass(frozen=True)
class ToolCommand:
    name: str
    argv: tuple[str, ...]
    cwd: Path | None = None
    env_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommonConfig:
    """Shared configuration result; exists distinguishes absence from emptiness."""

    path: Path
    exists: bool
    schema_version: int | None = None
    paths: CommonPaths | None = None
    harnesses: tuple[HarnessCommand, ...] = ()
    tools: tuple[ToolCommand, ...] = ()

    def tool(self, name: str) -> ToolCommand | None:
        return next((tool for tool in self.tools if tool.name == name), None)


def load_toml(path: str | Path, *, known: set[str],
              required: frozenset[str] | set[str] = frozenset()) -> dict:
    """Load TOML, validate its field set, and return the raw dictionary.

    ValueError covers malformed syntax, unknown fields, and missing required fields.
    The caller owns defaults and type conversion; this layer enforces the schema.
    """
    path = Path(path)
    raw = tomllib.loads(path.read_text("utf-8"))
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"{path.name}: unknown fields {sorted(unknown)}")
    missing = set(required) - set(raw)
    if missing:
        raise ValueError(f"{path.name}: missing required fields {sorted(missing)}")
    return raw


def tools_data_root() -> Path:
    """Return the shared data root from TOOLS_DATA or its documented fallback."""
    raw = os.environ.get("TOOLS_DATA")
    return Path(raw).expanduser() if raw else Path.home() / "tools-data"


def _error(path: Path, key: str, message: str) -> CommonConfigError:
    return CommonConfigError(f"{path}: {key}: {message}")


def _table(value: object, *, path: Path, key: str, allowed: set[str],
           required: set[str] | frozenset[str] = frozenset()) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _error(path, key, "expected a TOML table")
    unknown = set(value) - allowed
    if unknown:
        raise _error(path, key, f"unknown keys {sorted(unknown)}")
    missing = required - set(value)
    if missing:
        raise _error(path, key, f"missing required keys {sorted(missing)}")
    return value


def _text(value: object, *, path: Path, key: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise _error(path, key, "expected a nonempty string without surrounding whitespace")
    return value


def _absolute_path(value: object, *, path: Path, key: str) -> Path:
    result = Path(_text(value, path=path, key=key)).expanduser()
    if not result.is_absolute():
        raise _error(path, key, "path must be absolute (a leading ~ is allowed)")
    return result


def _argv(value: object, *, path: Path, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise _error(path, key, "expected a nonempty list of strings")
    return tuple(_text(item, path=path, key=f"{key}[{i}]")
                 for i, item in enumerate(value))


def _env_names(value: object, *, path: Path, key: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _error(path, key, "expected a list of environment variable names")
    names = tuple(_text(item, path=path, key=f"{key}[{i}]")
                  for i, item in enumerate(value))
    if len(set(names)) != len(names):
        raise _error(path, key, "environment variable names must be unique")
    for i, name in enumerate(names):
        if not _ENV_NAME.fullmatch(name):
            raise _error(path, f"{key}[{i}]", "expected an environment name, not a value")
    return names


def load_common_config(root: str | Path | None = None) -> CommonConfig:
    """Read <root>/config.toml using the fail-closed shared schema v1.

    Absence is normal and yields CommonConfig.exists=False. An existing unreadable
    or invalid file raises CommonConfigError with its full path and schema key;
    configuration is never partially applied.
    """
    data_root = tools_data_root() if root is None else Path(root).expanduser()
    path = data_root / "config.toml"
    if not path.exists() and not path.is_symlink():
        return CommonConfig(path=path, exists=False)
    try:
        raw = tomllib.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValueError) as e:
        raise _error(path, "TOML", f"file is unreadable: {e}") from e

    root_table = _table(
        raw, path=path, key="<root>",
        allowed={"schema_version", "paths", "harness", "tools"},
        required={"schema_version", "paths"})
    schema_version = root_table["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise _error(path, "schema_version", "expected integer 1")

    raw_paths = _table(
        root_table["paths"], path=path, key="paths",
        allowed={"vault", "topics_root", "sources_root"},
        required={"vault", "topics_root", "sources_root"})
    paths = CommonPaths(*(
        _absolute_path(raw_paths[name], path=path, key=f"paths.{name}")
        for name in ("vault", "topics_root", "sources_root")
    ))

    raw_harnesses = root_table.get("harness", {})
    if not isinstance(raw_harnesses, dict):
        raise _error(path, "harness", "expected a TOML table")
    harnesses: list[HarnessCommand] = []
    for name, value in sorted(raw_harnesses.items()):
        if name not in _HARNESS_NAMES:
            raise _error(path, f"harness.{name}",
                         f"unknown harness; allowed values: {sorted(_HARNESS_NAMES)}")
        key = f"harness.{name}"
        table = _table(value, path=path, key=key,
                       allowed={"argv", "model_default", "effort_default"},
                       required={"argv"})
        harnesses.append(HarnessCommand(
            name=name,
            argv=_argv(table["argv"], path=path, key=key + ".argv"),
            model_default=_text(table["model_default"], path=path,
                                key=key + ".model_default")
            if "model_default" in table else None,
            effort_default=_text(table["effort_default"], path=path,
                                 key=key + ".effort_default")
            if "effort_default" in table else None,
        ))

    raw_tools = root_table.get("tools", {})
    if not isinstance(raw_tools, dict):
        raise _error(path, "tools", "expected a TOML table")
    tools: list[ToolCommand] = []
    for name, value in sorted(raw_tools.items()):
        if not _NAME.fullmatch(name):
            raise _error(path, f"tools.{name}", "invalid tool name")
        key = f"tools.{name}"
        table = _table(value, path=path, key=key,
                       allowed={"argv", "cwd", "env"}, required={"argv"})
        tools.append(ToolCommand(
            name=name,
            argv=_argv(table["argv"], path=path, key=key + ".argv"),
            cwd=_absolute_path(table["cwd"], path=path, key=key + ".cwd")
            if "cwd" in table else None,
            env_names=_env_names(table.get("env", []), path=path, key=key + ".env"),
        ))

    return CommonConfig(path=path, exists=True, schema_version=1, paths=paths,
                        harnesses=tuple(harnesses), tools=tuple(tools))
