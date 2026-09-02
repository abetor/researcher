"""Structured API source tools built on stdlib urllib.

The named registry and CLI entry point let orchestrator collectors invoke search and
fetch through Bash. Hacker News, arXiv, and GitHub are keyless high-signal sources.
The keyless web adapter is the prescribed web path for harnesses without native web,
a fallback for Claude, and the mandatory path in parity mode. Add a source by deriving
SourceTool, implementing search and fetch through _http, and registering the class.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from .arxiv import Arxiv
from .base import (
    ImportPlan,
    ImportReceipt,
    CoursedumpManifestSource,
    LocalMarkdownSource,
    PlannedDocument,
    SkippedDocument,
    SourceError,
    SourceImportError,
    SourceTool,
    SourceImportTool,
    SourceWindow,
    import_verified_corpus,
    iter_source_windows,
    normalize_extract_result,
    normalize_verify_result,
    plan_import,
    verified_claim_payload,
    write_verified_claims,
)
from .github import GitHub
from .hn import HackerNews
from .web import Web

REGISTRY: dict[str, type[SourceTool]] = {
    t.name: t for t in (HackerNews, Arxiv, GitHub, Web)
}
IMPORT_REGISTRY: dict[str, type[SourceImportTool]] = {
    tool.source_kind: tool for tool in (LocalMarkdownSource, CoursedumpManifestSource)
}


def cmd_prefix() -> str:
    """Return the source CLI Bash prefix and its harness allowlist rule.

    A collector runs in the topic directory rather than the repository, which may not
    be installed. The prefix supplies the repository root through PYTHONPATH and uses
    the orchestrator's interpreter. Both paths are shell-quoted so spaces or shell
    metacharacters cannot break the command. The form works in Claude's scoped Bash
    allowlist and inside the Codex sandbox.
    """
    root = Path(__file__).resolve().parents[2]
    return (
        f"PYTHONPATH={shlex.quote(str(root))} "
        f"{shlex.quote(sys.executable or 'python3')} -m researcher.sources"
    )


def get_tool(name: str) -> SourceTool:
    cls = REGISTRY.get(name)
    if cls is None:
        raise SourceError(
            f"unknown source tool {name!r}; available: {', '.join(sorted(REGISTRY))}"
        )
    return cls()


def tool_names() -> list[str]:
    return sorted(REGISTRY)


def get_import_tool(source_kind: str) -> SourceImportTool:
    if type(source_kind) is not str:
        raise SourceImportError("invalid_source_kind")
    cls = IMPORT_REGISTRY.get(source_kind)
    if cls is None:
        raise SourceImportError("invalid_source_kind")
    return cls()


def kind_for_tool(name: str | None) -> str | None:
    """Return the source kind without inventing a type for an unknown tool."""
    cls = REGISTRY.get((name or "").strip().lower())
    return cls.kind if cls is not None else None


def describe(names: list[str] | None = None) -> str:
    """Return '- name: description' lines for the collector prompt."""
    names = names if names is not None else tool_names()
    return "\n".join(f"- {n}: {REGISTRY[n].desc}" for n in names if n in REGISTRY)


__all__ = [
    "SourceTool",
    "SourceImportTool",
    "LocalMarkdownSource",
    "CoursedumpManifestSource",
    "SourceError",
    "SourceImportError",
    "PlannedDocument",
    "SkippedDocument",
    "ImportPlan",
    "SourceWindow",
    "ImportReceipt",
    "plan_import",
    "iter_source_windows",
    "import_verified_corpus",
    "normalize_extract_result",
    "normalize_verify_result",
    "verified_claim_payload",
    "write_verified_claims",
    "REGISTRY",
    "IMPORT_REGISTRY",
    "get_tool",
    "get_import_tool",
    "tool_names",
    "kind_for_tool",
    "describe",
    "cmd_prefix",
    "HackerNews",
    "Arxiv",
    "GitHub",
    "Web",
]
