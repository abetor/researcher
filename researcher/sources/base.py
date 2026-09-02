"""Source-tool interface and shared explicit failure.

Structured APIs such as Hacker News, arXiv, and GitHub form the core of this layer:
response-specific parsing yields more signal than raw pages and avoids reliance on one
web index. A keyless web adapter implements the same interface because Codex has no
native web tools. Claude may use native web by default, while this adapter is its
fallback and the required path in parity mode.

Contract:
- search(query) -> list[{"url", "title", "snippet"}]
- fetch(url) -> markdown (str)

Local corpora use the versioned import contract in ``sources.imports``: a deterministic
review plan followed by bounded windows from exactly approved files. This is neither
network access nor a hidden third form of fetch().

Implementations access the network only through _http, using stdlib urllib, transient
retries, and SourceError for explicit failures. The source layer adds no dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .imports import (
    ImportPlan,
    ImportReceipt,
    PlannedDocument,
    SkippedDocument,
    SourceImportError,
    SourceWindow,
    import_verified_corpus,
    iter_source_windows,
    normalize_extract_result,
    normalize_verify_result,
    plan_import,
    verified_claim_payload,
    write_verified_claims,
)


class SourceError(Exception):
    """Explicit source failure for network, HTTP, parsing, or argument errors.

    The CLI or orchestrator exposes a distinguishable failure instead of silence or
    invented data. _http absorbs retryable network errors; this exception represents
    a persistent HTTP response, malformed payload, or invalid argument."""


class SourceTool(ABC):
    """Own source-specific search and conversion of one document to Markdown."""

    name: str = "source"
    kind: str = "other"  # llm-wiki enum: the tool sets the type, not the model's self-report
    desc: str = ""  # one line for the "available sources" block in the collector prompt

    @abstractmethod
    def search(self, query: str, *, limit: int = 8) -> list[dict]:
        """Return at most limit results with url, title, and snippet fields."""

    @abstractmethod
    def fetch(self, url: str) -> str:
        """Fetch one URL and return the document as Markdown."""


class SourceImportTool:
    """Versioned non-network SourceTool variant with an explicit plan gate."""

    contract_version = 1
    source_kind: str

    def plan(self, source: str) -> ImportPlan:
        return plan_import(source, source_kind=self.source_kind)

    def windows(
        self, source: str, *, plan: ImportPlan, approval: str
    ) -> tuple[SourceWindow, ...]:
        return iter_source_windows(source, plan=plan, approval=approval)

    def import_verified(
        self,
        topic_dir: str,
        source: str,
        *,
        plan: ImportPlan,
        approval: str,
        extractor,
        verifier,
        wiki,
    ) -> ImportReceipt:
        return import_verified_corpus(
            topic_dir,
            source,
            plan=plan,
            approval=approval,
            extractor=extractor,
            verifier=verifier,
            wiki=wiki,
        )


class LocalMarkdownSource(SourceImportTool):
    source_kind = "local-markdown"


class CoursedumpManifestSource(SourceImportTool):
    source_kind = "coursedump-manifest"


__all__ = [
    "SourceError",
    "SourceTool",
    "SourceImportTool",
    "LocalMarkdownSource",
    "CoursedumpManifestSource",
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
]
