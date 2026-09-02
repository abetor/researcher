"""Versioned import contract for bounded local Markdown source corpora.

The source tree remains read-only and outside a knowledge topic.  A caller first
builds a deterministic plan, presents it for review, and may only open windows or
write verified claims when the exact plan id is supplied as approval.  Knowledge
writes are delegated to the injected llmwiki socket.
"""

from __future__ import annotations

import bisect
import errno
import hashlib
import difflib
import json
from functools import lru_cache
import os
import re
import stat
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol

# llm-wiki has no public normalization API yet. These private functions are imported
# deliberately from the owner of the final verify-quotes gate, so that extract/verify
# cannot accept a quote which that same gate would later fail to find.
from llmwiki.ext.evidence import _find_spans as _gate_find_spans
from llmwiki.ext.evidence import _norm as _gate_norm

SOURCE_IMPORT_SCHEMA_VERSION = 1
MAX_DOCUMENTS = 256
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
# A durable plan holds up to MAX_DOCUMENTS full relative paths and titles and is
# therefore naturally larger than the bounded machine envelope of the CLI (64 KiB). These
# limits must not be merged: a large plan is stored under work/, while machine stdout
# refuses separately with output_too_large through cli._emit_machine.
MAX_PLAN_BYTES = 512 * 1024
MAX_WINDOW_CHARS = 12_000
WINDOW_OVERLAP_CHARS = 512
MAX_WINDOWS_PER_DOCUMENT = 1 + MAX_DOCUMENT_BYTES // (
    MAX_WINDOW_CHARS // 2 - WINDOW_OVERLAP_CHARS
)
MAX_ROLE_RESULT_BYTES = 256 * 1024
MAX_CANDIDATES_PER_WINDOW = 64
MAX_QUOTE_CHARS = 8_000
MAX_SCAN_ENTRIES = 4_096
MAX_MANIFEST_SIZE_VALUE = 2**63 - 1
MAX_MANIFEST_INDEX = 2**31 - 1
MAX_EXISTING_CLAIMS_BYTES = 16 * 1024 * 1024
MAX_TOPIC_MANIFEST_BYTES = 256 * 1024
# This is a memory bound for an unvalidated string, not a model of the filesystem's
# NAME_MAX. APFS accepts up to 255 Unicode characters, which may legitimately take more
# than 255 UTF-8 bytes; a Linux threshold applied before os.open falsely rejected such names.
MAX_PATH_COMPONENT_BYTES = 1024

_SOURCE_KINDS = {"auto", "local-markdown", "coursedump-manifest"}
_CONFIDENCE = {"high", "medium", "low"}
_MANIFEST_FIELDS = {
    "rel",
    "kind",
    "size",
    "remote",
    "skip",
    "target",
    "index",
    "vid",
    "title",
}
_MANIFEST_V1_FIELDS = {"rel", "kind", "size", "remote", "skip", "target"}
_NON_TRANSCRIPT_KINDS = {"html"}
_SKIP_REASONS = {"document_too_large", "non_transcript"}
# A row with the same id is already in the topic: written either by this run (own) or by
# another lesson (foreign).
_REPLAY_OWN = "own"
_REPLAY_FOREIGN = "foreign"


class SourceImportError(ValueError):
    """Normalized source contract failure safe for a machine caller."""

    def __init__(
        self,
        code: str,
        *,
        relative_path: str | None = None,
        detail: str | None = None,
    ):
        message = code if relative_path is None else f"{code}:{relative_path}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.code = code
        self.relative_path = relative_path
        self.detail = detail


@dataclass(frozen=True)
class PlannedDocument:
    document_id: str
    relative_path: str
    title: str
    sha256: str
    size_bytes: int
    window_count: int

    def to_dict(self) -> dict[str, object]:
        _validate_planned_document(self)
        return {
            "document_id": self.document_id,
            "relative_path": self.relative_path,
            "title": self.title,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "window_count": self.window_count,
        }


@dataclass(frozen=True)
class SkippedDocument:
    relative_path: str
    title: str
    reason: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        _validate_skipped_document(self)
        return {
            "relative_path": self.relative_path,
            "title": self.title,
            "reason": self.reason,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class ImportPlan:
    schema_version: int
    plan_id: str
    status: str
    source_kind: str
    corpus_sha256: str
    documents: tuple[PlannedDocument, ...]
    skipped: tuple[SkippedDocument, ...] = ()
    window_chars: int = MAX_WINDOW_CHARS
    overlap_chars: int = WINDOW_OVERLAP_CHARS

    def to_dict(self) -> dict[str, object]:
        _validate_plan(self)
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "status": self.status,
            "source_kind": self.source_kind,
            "corpus_sha256": self.corpus_sha256,
            "window": {
                "max_chars": self.window_chars,
                "overlap_chars": self.overlap_chars,
            },
            "documents": [document.to_dict() for document in self.documents],
            "skipped": [document.to_dict() for document in self.skipped],
        }
        _bounded_json(payload, MAX_PLAN_BYTES, "plan_too_large")
        return payload

    @classmethod
    def from_dict(cls, value: object) -> ImportPlan:
        required = {
            "schema_version",
            "plan_id",
            "status",
            "source_kind",
            "corpus_sha256",
            "window",
            "documents",
        }
        if type(value) is not dict or frozenset(value) not in {
            frozenset(required), frozenset(required | {"skipped"})
        }:
            _fail("invalid_import_plan")
        window = value["window"]
        rows = value["documents"]
        skipped_rows = value.get("skipped", [])
        if type(window) is not dict or set(window) != {"max_chars", "overlap_chars"}:
            _fail("invalid_import_plan")
        if type(rows) is not list or len(rows) > MAX_DOCUMENTS:
            _fail("invalid_import_plan")
        if type(skipped_rows) is not list or len(rows) + len(skipped_rows) > MAX_DOCUMENTS:
            _fail("invalid_import_plan")
        documents: list[PlannedDocument] = []
        for row in rows:
            if type(row) is not dict or set(row) != {
                "document_id",
                "relative_path",
                "title",
                "sha256",
                "size_bytes",
                "window_count",
            }:
                _fail("invalid_import_plan")
            document = PlannedDocument(**row)
            _validate_planned_document(document)
            documents.append(document)
        skipped: list[SkippedDocument] = []
        for row in skipped_rows:
            if type(row) is not dict or set(row) != {
                "relative_path", "title", "reason", "size_bytes"
            }:
                _fail("invalid_import_plan")
            document = SkippedDocument(**row)
            _validate_skipped_document(document)
            skipped.append(document)
        plan = cls(
            schema_version=value["schema_version"],
            plan_id=value["plan_id"],
            status=value["status"],
            source_kind=value["source_kind"],
            corpus_sha256=value["corpus_sha256"],
            documents=tuple(documents),
            skipped=tuple(skipped),
            window_chars=window["max_chars"],
            overlap_chars=window["overlap_chars"],
        )
        _validate_plan(plan)
        _bounded_json(value, MAX_PLAN_BYTES, "plan_too_large")
        return plan


@dataclass(frozen=True)
class SourceWindow:
    schema_version: int
    window_id: str
    document_id: str
    relative_path: str
    start_offset: int
    end_offset: int
    start_line: int
    end_line: int
    text: str
    sha256: str

    def to_dict(self) -> dict[str, object]:
        _validate_window(self)
        return {
            "schema_version": self.schema_version,
            "window_id": self.window_id,
            "document_id": self.document_id,
            "relative_path": self.relative_path,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ImportReceipt:
    schema_version: int
    plan_id: str
    status: str
    documents: int
    windows: int
    candidates: int
    verified: int
    duplicates: int
    role_calls: dict[str, int] = field(default_factory=dict)
    usage: dict[str, object] = field(default_factory=dict)
    phase_seconds: dict[str, float] = field(default_factory=dict)
    total_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        values = (
            self.schema_version,
            self.documents,
            self.windows,
            self.candidates,
            self.verified,
            self.duplicates,
        )
        if (
            any(type(value) is not int or value < 0 for value in values)
            or self.schema_version != SOURCE_IMPORT_SCHEMA_VERSION
            or not 1 <= self.documents <= MAX_DOCUMENTS
            or self.windows > self.documents * MAX_WINDOWS_PER_DOCUMENT
            or self.candidates > self.windows * MAX_CANDIDATES_PER_WINDOW
            or self.verified + self.duplicates > self.candidates
            or type(self.plan_id) is not str
            or not _is_prefixed_hex(self.plan_id, "plan_", 24)
            or self.status != "imported"
            or type(self.role_calls) is not dict
            or any(
                type(name) is not str
                or not name
                or type(count) is not int
                or count < 0
                for name, count in self.role_calls.items()
            )
            or type(self.usage) is not dict
            or type(self.phase_seconds) is not dict
            or any(
                type(name) is not str
                or not name
                or type(seconds) not in {int, float}
                or seconds < 0
                for name, seconds in self.phase_seconds.items()
            )
            or type(self.total_seconds) not in {int, float}
            or self.total_seconds < 0
        ):
            _fail("invalid_import_receipt")
        payload = {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "status": self.status,
            "documents": self.documents,
            "windows": self.windows,
            "candidates": self.candidates,
            "verified": self.verified,
            "duplicates": self.duplicates,
            "role_calls": dict(sorted(self.role_calls.items())),
            "usage": self.usage,
            "phase_seconds": dict(sorted(self.phase_seconds.items())),
            "total_seconds": self.total_seconds,
        }
        _bounded_json(payload, MAX_ROLE_RESULT_BYTES, "invalid_import_receipt")
        return payload


@dataclass(frozen=True)
class _Document:
    planned: PlannedDocument
    text: str


@dataclass(frozen=True)
class _ClaimZoneSnapshot:
    zone: str
    directory_fd: int
    directory_before: os.stat_result
    file_fd: int | None
    file_before: os.stat_result | None
    raw: bytes


class ExtractRole(Protocol):
    role: str

    def __call__(self, window: Mapping[str, object]) -> object: ...


class VerifyRole(Protocol):
    role: str

    def __call__(
        self, candidate: Mapping[str, object], window: Mapping[str, object]
    ) -> object: ...


def plan_import(source: str | Path, *, source_kind: str = "auto") -> ImportPlan:
    """Read a bounded source snapshot and return a review-only deterministic plan."""

    kind, documents, skipped = _load_documents(source, source_kind)
    return _plan_for_documents(kind, documents, skipped)


def _plan_for_documents(
    kind: str,
    documents: tuple[_Document, ...],
    skipped: tuple[SkippedDocument, ...] = (),
) -> ImportPlan:
    document_rows = tuple(document.planned for document in documents)
    plan_body = {
        "schema_version": SOURCE_IMPORT_SCHEMA_VERSION,
        "source_kind": kind,
        "documents": [document.to_dict() for document in document_rows],
        "window": {
            "max_chars": MAX_WINDOW_CHARS,
            "overlap_chars": WINDOW_OVERLAP_CHARS,
        },
    }
    if skipped:
        plan_body["skipped"] = [document.to_dict() for document in skipped]
    corpus_sha256 = _digest_json(plan_body)
    plan_id = (
        "plan_"
        + hashlib.sha256(
            ("source-import-v1:" + corpus_sha256).encode("ascii")
        ).hexdigest()[:24]
    )
    return ImportPlan(
        schema_version=SOURCE_IMPORT_SCHEMA_VERSION,
        plan_id=plan_id,
        status="waiting_human",
        source_kind=kind,
        corpus_sha256=corpus_sha256,
        documents=document_rows,
        skipped=skipped,
    )


def iter_source_windows(
    source: str | Path,
    *,
    plan: ImportPlan,
    approval: str,
) -> tuple[SourceWindow, ...]:
    """Open bounded windows only after exact approval and snapshot reconciliation."""

    documents = _approved_documents(source, plan=plan, approval=approval)
    return tuple(window for document in documents for window in _windows_for(document))


def _approved_documents(
    source: str | Path, *, plan: ImportPlan, approval: str
) -> tuple[_Document, ...]:
    """Open one fd-anchored snapshot and bind its exact bytes to an approved plan."""

    _validate_plan(plan)
    if type(approval) is not str or approval != plan.plan_id:
        _fail("plan_not_approved")
    kind, documents, skipped = _load_documents(source, plan.source_kind)
    if _plan_for_documents(kind, documents, skipped) != plan:
        _fail("source_plan_conflict")
    return documents


def import_verified_corpus(
    topic_dir: str | Path,
    source: str | Path,
    *,
    plan: ImportPlan,
    approval: str,
    extractor: ExtractRole,
    verifier: VerifyRole,
    wiki: object,
    on_conflict=None,
) -> ImportReceipt:
    """Run separate extract/verify roles and write only through the llmwiki socket.

    ``on_conflict`` follows the evidence-reporting contract of
    ``write_verified_claims``.
    """

    extract_role = _role_name(extractor)
    verify_role = _role_name(verifier)
    if extract_role == verify_role:
        _fail("verification_role_not_independent")
    documents = _approved_documents(source, plan=plan, approval=approval)
    windows = tuple(
        window for document in documents for window in _windows_for(document)
    )

    validate_topic = getattr(wiki, "validate_topic", None)
    add_source = getattr(wiki, "add_source", None)
    add_claim = getattr(wiki, "add_claim", None)
    duplicate_type = getattr(wiki, "DuplicateClaim", None)
    if (
        not callable(validate_topic)
        or not callable(add_source)
        or not callable(add_claim)
        or not isinstance(duplicate_type, type)
    ):
        _fail("invalid_llmwiki_socket")
    try:
        before_errors = validate_topic(topic_dir)
    except Exception:
        _fail("llmwiki_validation_failed")
    if type(before_errors) is not list or before_errors:
        _fail("llmwiki_validation_failed")

    source_ids: dict[str, str] = {}
    try:
        for document in documents:
            planned = document.planned
            source_id = add_source(
                topic_dir,
                kind="local",
                title=planned.title,
                url=f"urn:researcher-source:{planned.document_id}",
                tool="researcher-source-v1",
                meta={
                    "document_id": planned.document_id,
                    "relative_path": planned.relative_path,
                    "sha256": planned.sha256,
                    "source_kind": plan.source_kind,
                },
            )
            if type(source_id) is not str or not source_id.startswith("src_"):
                _fail("invalid_llmwiki_result")
            source_ids[planned.document_id] = source_id
    except SourceImportError:
        raise
    except Exception:
        _fail("llmwiki_write_failed")

    candidates_count = 0
    verified_count = 0
    duplicates = 0
    for window in windows:
        window_payload = window.to_dict()
        try:
            extracted = extractor(window_payload)
        except Exception:
            _fail("extract_role_failed")
        candidates = _normalize_candidates(extracted, window)
        candidates_count += len(candidates)
        for candidate in candidates:
            try:
                verified = verifier(candidate, window_payload)
            except Exception:
                _fail("verify_role_failed")
            verdict = _normalize_verdict(verified)
            if not verdict:
                continue
            evidence = []
            for item in candidate["evidence"]:
                local_start = item["start_offset"] - window.start_offset
                local_end = item["end_offset"] - window.start_offset
                start_line = window.start_line + window.text.count("\n", 0, local_start)
                end_line = window.start_line + window.text.count(
                    "\n", 0, max(local_start, local_end - 1)
                )
                evidence.append(
                    {
                        "source_id": source_ids[window.document_id],
                        "quote": item["quote"],
                        "stance": "supports",
                        "locator": {
                            "type": "line",
                            "value": f"{start_line}-{end_line}",
                        },
                    }
                )
            confidence, single_source = _single_source_confidence(
                candidate["confidence"], evidence
            )
            tags = candidate["tags"]
            meta = {
                "source_window": window.window_id,
                "single_source": single_source,
            }
            claim_row = {
                "text": candidate["text"],
                "confidence": confidence,
                "status": "verified",
                "evidence": evidence,
                "tags": tags,
                "meta": meta,
            }
            claim_id, existing = _replay_claim_snapshot(
                wiki, topic_dir, candidate["text"]
            )
            if existing:
                _absorb_existing_claim(
                    wiki, topic_dir, claim_row, claim_id, existing, on_conflict
                )
                duplicates += 1
                continue
            try:
                add_claim(
                    topic_dir,
                    "staging",
                    text=candidate["text"],
                    confidence=confidence,
                    status="verified",
                    evidence=evidence,
                    tags=tags,
                    meta=meta,
                )
                verified_count += 1
            except duplicate_type as error:
                duplicate_id = error.args[0] if error.args else None
                if duplicate_id != claim_id:
                    _fail(
                        "claim_replay_conflict",
                        f"the socket returned duplicate {duplicate_id}, expected {claim_id}",
                    )
                replay_id, replay_rows = _replay_claim_snapshot(
                    wiki, topic_dir, candidate["text"]
                )
                if replay_id != claim_id:
                    _fail(
                        "claim_replay_conflict",
                        f"claim id changed on re-read: {claim_id} -> {replay_id}",
                    )
                _absorb_existing_claim(
                    wiki, topic_dir, claim_row, claim_id, replay_rows, on_conflict
                )
                duplicates += 1
            except Exception:
                _fail("llmwiki_write_failed")

    try:
        after_errors = validate_topic(topic_dir)
    except Exception:
        _fail("llmwiki_validation_failed")
    if type(after_errors) is not list or after_errors:
        _fail("llmwiki_validation_failed")
    return ImportReceipt(
        schema_version=SOURCE_IMPORT_SCHEMA_VERSION,
        plan_id=plan.plan_id,
        status="imported",
        documents=len(documents),
        windows=len(windows),
        candidates=candidates_count,
        verified=verified_count,
        duplicates=duplicates,
        role_calls={"extract": len(windows), "verify": candidates_count},
    )


def _load_documents(
    source: str | Path, source_kind: str
) -> tuple[str, tuple[_Document, ...], tuple[SkippedDocument, ...]]:
    if type(source_kind) is not str or source_kind not in _SOURCE_KINDS:
        _fail("invalid_source_kind")
    source_fd, source_name = _open_source(source)
    try:
        source_metadata = os.fstat(source_fd)
        kind = source_kind
        if kind == "auto":
            kind = (
                "coursedump-manifest"
                if stat.S_ISDIR(source_metadata.st_mode)
                and _regular_entry_exists(source_fd, "manifest.jsonl")
                else "local-markdown"
            )
        if kind == "coursedump-manifest":
            if not stat.S_ISDIR(source_metadata.st_mode):
                _fail("invalid_coursedump_source")
            entries, skipped = _coursedump_entries(source_fd)
        else:
            entries, skipped = _markdown_entries(
                source_fd, source_name, source_metadata
            )
    finally:
        os.close(source_fd)
    if not entries and not skipped:
        _fail("source_corpus_empty")
    if len(entries) + len(skipped) > MAX_DOCUMENTS:
        _fail("too_many_source_documents")
    documents: list[_Document] = []
    total_bytes = 0
    seen_paths: set[str] = set()
    for relative, data in entries:
        normalized = _normalized_relative(relative)
        path_key = normalized.casefold()
        if path_key in seen_paths:
            _fail("source_path_collision")
        seen_paths.add(path_key)
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_BYTES:
            _fail("source_corpus_too_large")
        try:
            text = (
                data.decode("utf-8", errors="strict")
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .replace("\u2028", "\n")
                .replace("\u2029", "\n")
            )
        except UnicodeDecodeError:
            _fail_document("source_document_not_utf8", normalized)
        if _has_forbidden_text(text, allow_layout=True):
            _fail_document("source_document_has_control", normalized)
        sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        document_id = (
            "doc_"
            + hashlib.sha256((normalized + "\0" + sha256).encode("utf-8")).hexdigest()[
                :24
            ]
        )
        title = _title_for(normalized)
        probe = PlannedDocument(
            document_id=document_id,
            relative_path=normalized,
            title=title,
            sha256=sha256,
            size_bytes=len(text.encode("utf-8")),
            window_count=0,
        )
        window_count = len(_window_ranges(text))
        planned = PlannedDocument(
            document_id=probe.document_id,
            relative_path=probe.relative_path,
            title=probe.title,
            sha256=probe.sha256,
            size_bytes=probe.size_bytes,
            window_count=window_count,
        )
        documents.append(_Document(planned=planned, text=text))
    documents.sort(key=lambda document: document.planned.relative_path.casefold())
    skipped.sort(key=lambda document: document.relative_path.casefold())
    return kind, tuple(documents), tuple(skipped)


def _open_source(source: str | Path) -> tuple[int, str]:
    try:
        path = Path(source)
        rendered = os.fspath(path)
    except (TypeError, ValueError, UnicodeError):
        _fail("invalid_source_path")
    if not rendered or _has_forbidden_text(rendered):
        _fail("invalid_source_path")
    components = list(path.parts)
    absolute = path.is_absolute()
    if absolute:
        components = components[1:]
    if len(components) > 64:
        _fail("invalid_source_path")
    base = "/" if absolute else "."
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        current_fd = os.open(base, flags)
    except OSError:
        _fail("source_read_failed")
    try:
        for index, component in enumerate(components):
            _path_component(component, "source path component")
            child_flags = (
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            )
            if index < len(components) - 1:
                child_flags |= getattr(os, "O_DIRECTORY", 0)
            try:
                child_fd = os.open(component, child_flags, dir_fd=current_fd)
            except FileNotFoundError:
                _fail("source_path_not_found")
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    _fail("source_path_not_regular")
                _fail("source_read_failed")
            try:
                child_metadata = os.fstat(child_fd)
            except OSError:
                os.close(child_fd)
                _fail("source_read_failed")
            if index < len(components) - 1 and not stat.S_ISDIR(child_metadata.st_mode):
                os.close(child_fd)
                _fail("source_path_not_regular")
            os.close(current_fd)
            current_fd = child_fd
        return current_fd, components[-1] if components else ""
    except BaseException:
        os.close(current_fd)
        raise


def _path_component(value: object, label: str) -> str:
    if (
        type(value) is not str
        or value in {"", ".", ".."}
        or value != unicodedata.normalize("NFC", value)
        or "/" in value
        or "\\" in value
        or _has_forbidden_text(value)
    ):
        _fail(f"invalid_{label.replace(' ', '_')}")
    try:
        if len(value.encode("utf-8")) > MAX_PATH_COMPONENT_BYTES:
            _fail(f"invalid_{label.replace(' ', '_')}")
    except UnicodeError:
        _fail(f"invalid_{label.replace(' ', '_')}")
    return value


def _regular_entry_exists(directory_fd: int, name: str) -> bool:
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError:
        _fail("source_read_failed")
    return stat.S_ISREG(metadata.st_mode)


def _open_directory_at(directory_fd: int, name: str, code: str) -> int:
    _path_component(name, "source path component")
    try:
        child_fd = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
    except OSError:
        _fail(code)
    try:
        metadata = os.fstat(child_fd)
    except OSError:
        os.close(child_fd)
        _fail(code)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(child_fd)
        _fail(code)
    return child_fd


def _open_regular_at(directory_fd: int, name: str, code: str) -> int:
    _path_component(name, "source path component")
    fd: int | None = None
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        opened = os.fstat(fd)
    except OSError:
        if fd is not None:
            os.close(fd)
        _fail(code)
    assert fd is not None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or not stat.S_ISREG(opened.st_mode)
        or (metadata.st_dev, metadata.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        os.close(fd)
        _fail(code)
    return fd


def _read_open_file(fd: int, limit: int, too_large: str) -> bytes:
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            _fail("source_path_not_regular")
        if before.st_size > limit:
            _fail(too_large)
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(fd, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(fd)
    except SourceImportError:
        raise
    except OSError:
        _fail("source_read_failed")
    if len(data) > limit:
        _fail(too_large)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or len(data) != after.st_size:
        _fail("source_changed_while_read")
    return data


def _markdown_entries(
    source_fd: int, source_name: str, metadata: os.stat_result
) -> tuple[list[tuple[str, bytes]], list[SkippedDocument]]:
    if stat.S_ISREG(metadata.st_mode):
        if not source_name.casefold().endswith(".md"):
            _fail("source_corpus_empty")
        relative = _normalized_relative(source_name)
        if metadata.st_size > MAX_DOCUMENT_BYTES:
            return [], [
                SkippedDocument(
                    relative_path=relative,
                    title=_title_for(relative),
                    reason="document_too_large",
                    size_bytes=metadata.st_size,
                )
            ]
        try:
            data = _read_open_file(
                source_fd, MAX_DOCUMENT_BYTES, "source_document_too_large"
            )
        except SourceImportError as error:
            _fail_document(error.code, relative)
        return [(relative, data)], []
    if not stat.S_ISDIR(metadata.st_mode):
        _fail("source_path_not_regular")
    result: list[tuple[str, bytes]] = []
    skipped: list[SkippedDocument] = []
    scanned = [0]
    rechecked = [0]
    total_bytes = [0]
    _scan_markdown_directory(
        source_fd, (), result, skipped, scanned, rechecked, total_bytes
    )
    return (
        sorted(result, key=lambda item: item[0].casefold()),
        sorted(skipped, key=lambda item: item.relative_path.casefold()),
    )


def _scan_markdown_directory(
    directory_fd: int,
    prefix: tuple[str, ...],
    result: list[tuple[str, bytes]],
    skipped: list[SkippedDocument],
    scanned: list[int],
    rechecked: list[int],
    total_bytes: list[int],
) -> None:
    if len(prefix) > 64:
        _fail("invalid_source_path")
    try:
        before = os.fstat(directory_fd)
        names = _bounded_directory_names(directory_fd, MAX_SCAN_ENTRIES - scanned[0])
    except OSError:
        _fail("source_read_failed")
    scanned[0] += len(names)
    for name in names:
        _path_component(name, "source path component")
        try:
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError:
            _fail("source_read_failed")
        if stat.S_ISLNK(metadata.st_mode):
            _fail("source_path_not_regular")
        relative_parts = (*prefix, name)
        if stat.S_ISDIR(metadata.st_mode):
            child_fd = _open_directory_at(directory_fd, name, "source_path_not_regular")
            try:
                _scan_markdown_directory(
                    child_fd,
                    relative_parts,
                    result,
                    skipped,
                    scanned,
                    rechecked,
                    total_bytes,
                )
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(metadata.st_mode) and name.casefold().endswith(".md"):
            if len(result) + len(skipped) >= MAX_DOCUMENTS:
                _fail("too_many_source_documents")
            relative = "/".join(relative_parts)
            if metadata.st_size > MAX_DOCUMENT_BYTES:
                skipped.append(
                    SkippedDocument(
                        relative_path=relative,
                        title=_title_for(relative),
                        reason="document_too_large",
                        size_bytes=metadata.st_size,
                    )
                )
                continue
            file_fd = _open_regular_at(directory_fd, name, "source_path_not_regular")
            try:
                try:
                    data = _read_open_file(
                        file_fd, MAX_DOCUMENT_BYTES, "source_document_too_large"
                    )
                except SourceImportError as error:
                    _fail_document(error.code, relative)
            finally:
                os.close(file_fd)
            total_bytes[0] += len(data)
            if total_bytes[0] > MAX_TOTAL_BYTES:
                _fail("source_corpus_too_large")
            result.append((relative, data))
        elif not stat.S_ISREG(metadata.st_mode):
            _fail("source_path_not_regular")
    try:
        after_names = _bounded_directory_names(
            directory_fd, MAX_SCAN_ENTRIES - rechecked[0]
        )
        rechecked[0] += len(after_names)
        after = os.fstat(directory_fd)
    except OSError:
        _fail("source_read_failed")
    if names != after_names or (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        _fail("source_changed_while_read")


def _bounded_directory_names(directory_fd: int, read_budget: int) -> list[str]:
    """Collect a deterministic directory snapshot without an iterator read-past."""

    if type(read_budget) is not int or read_budget <= 0:
        _fail("too_many_source_entries")
    iterator = None
    try:
        iterator = os.scandir(directory_fd)
        names: list[str] = []
        for _ in range(read_budget):
            try:
                entry = next(iterator)
            except StopIteration:
                return sorted(
                    names,
                    key=lambda name: unicodedata.normalize("NFC", name).casefold(),
                )
            if type(entry.name) is not str:
                _fail("source_read_failed")
            names.append(entry.name)
        _fail("too_many_source_entries")
    except SourceImportError:
        raise
    except (OSError, TypeError, ValueError):
        _fail("source_read_failed")
    finally:
        if iterator is not None:
            try:
                iterator.close()
            except OSError:
                _fail("source_read_failed")


def _coursedump_entries(
    root_fd: int,
) -> tuple[list[tuple[str, bytes]], list[SkippedDocument]]:
    manifest_fd = _open_regular_at(root_fd, "manifest.jsonl", "invalid_source_manifest")
    try:
        raw = _read_open_file(
            manifest_fd, MAX_MANIFEST_BYTES, "source_manifest_too_large"
        )
    finally:
        os.close(manifest_fd)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _fail("invalid_source_manifest")
    text_fd = _open_directory_at(root_fd, "text", "invalid_coursedump_source")
    try:
        entries: list[tuple[str, bytes]] = []
        skipped: list[SkippedDocument] = []
        targets: set[str] = set()
        total_bytes = 0
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line_number > MAX_SCAN_ENTRIES or not line.strip():
                _fail("invalid_source_manifest")
            row = _decode_manifest_line(line)
            target = row["target"]
            if not target:
                continue
            normalized = _normalized_relative(target)
            target_key = normalized.casefold()
            if not target_key.endswith(".md") or target_key in targets:
                _fail("invalid_source_manifest")
            if len(entries) + len(skipped) >= MAX_DOCUMENTS:
                _fail("too_many_source_documents")
            targets.add(target_key)
            try:
                file_fd = _open_relative_regular(
                    text_fd, normalized, "source_document_missing"
                )
            except SourceImportError as error:
                _fail_document(error.code, normalized)
            try:
                try:
                    size_bytes = os.fstat(file_fd).st_size
                except OSError:
                    _fail_document("source_read_failed", normalized)
                reason = None
                if size_bytes > MAX_DOCUMENT_BYTES:
                    reason = "document_too_large"
                elif row["kind"].casefold() in _NON_TRANSCRIPT_KINDS:
                    reason = "non_transcript"
                if reason is not None:
                    skipped.append(
                        SkippedDocument(
                            relative_path=normalized,
                            title=_title_for(normalized),
                            reason=reason,
                            size_bytes=size_bytes,
                        )
                    )
                    continue
                try:
                    data = _read_open_file(
                        file_fd, MAX_DOCUMENT_BYTES, "source_document_too_large"
                    )
                except SourceImportError as error:
                    _fail_document(error.code, normalized)
            finally:
                os.close(file_fd)
            total_bytes += len(data)
            if total_bytes > MAX_TOTAL_BYTES:
                _fail("source_corpus_too_large")
            entries.append((normalized, data))
        return (
            sorted(entries, key=lambda item: item[0].casefold()),
            sorted(skipped, key=lambda item: item.relative_path.casefold()),
        )
    finally:
        os.close(text_fd)


def _decode_manifest_line(line: str) -> dict[str, object]:
    try:
        row = json.loads(line, object_pairs_hook=_object_no_duplicates)
    except SourceImportError:
        raise
    except (ValueError, RecursionError):
        _fail("invalid_source_manifest")
    if type(row) is not dict or frozenset(row) not in {
        frozenset(_MANIFEST_V1_FIELDS), frozenset(_MANIFEST_FIELDS)
    }:
        _fail("invalid_source_manifest")
    for field, default in {"index": 0, "vid": "", "title": ""}.items():
        row.setdefault(field, default)
    if (
        type(row["rel"]) is not str
        or type(row["kind"]) is not str
        or type(row["size"]) is not int
        or not 0 <= row["size"] <= MAX_MANIFEST_SIZE_VALUE
        or type(row["remote"]) is not str
        or type(row["skip"]) is not str
        or type(row["target"]) is not str
        or type(row["index"]) is not int
        or not 0 <= row["index"] <= MAX_MANIFEST_INDEX
        or type(row["vid"]) is not str
        or type(row["title"]) is not str
        or any(
            _has_forbidden_text(row[field])
            for field in ("rel", "kind", "remote", "skip", "target", "vid", "title")
        )
    ):
        _fail("invalid_source_manifest")
    if row["rel"]:
        _normalized_relative(row["rel"])
    return row


def _object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("invalid_source_manifest")
        result[key] = value
    return result


def _open_relative_regular(directory_fd: int, relative: str, code: str) -> int:
    parts = relative.split("/")
    if not parts or len(parts) > 64:
        _fail(code)
    try:
        current_fd = os.dup(directory_fd)
    except OSError:
        _fail(code)
    try:
        for component in parts[:-1]:
            next_fd = _open_directory_at(current_fd, component, code)
            os.close(current_fd)
            current_fd = next_fd
        file_fd = _open_regular_at(current_fd, parts[-1], code)
    finally:
        os.close(current_fd)
    return file_fd


def _normalized_relative(value: str) -> str:
    if type(value) is not str or not value or _has_forbidden_text(value):
        _fail("invalid_source_relative_path")
    normalized = unicodedata.normalize("NFC", value.replace("\\", "/"))
    path = Path(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _fail("invalid_source_relative_path")
    rendered = path.as_posix()
    if len(rendered.encode("utf-8")) > 1024:
        _fail("invalid_source_relative_path")
    return rendered


def _title_for(relative: str) -> str:
    title = Path(relative).name
    while Path(title).suffix.casefold() in {".md", ".mp4", ".pdf", ".html", ".txt"}:
        title = Path(title).stem
    title = " ".join(title.split()).strip()
    if not title:
        _fail("invalid_source_title")
    return title[:300]


def _window_ranges(text: str) -> tuple[tuple[int, int], ...]:
    if not text:
        return ()
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + MAX_WINDOW_CHARS)
        if end < len(text):
            boundary = text.rfind("\n", start + MAX_WINDOW_CHARS // 2, end)
            if boundary > start:
                end = boundary + 1
        ranges.append((start, end))
        if end == len(text):
            break
        start = max(start + 1, end - WINDOW_OVERLAP_CHARS)
    return tuple(ranges)


def _windows_for(document: _Document) -> tuple[SourceWindow, ...]:
    newlines = [index for index, char in enumerate(document.text) if char == "\n"]
    windows: list[SourceWindow] = []
    for start, end in _window_ranges(document.text):
        text = document.text[start:end]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        window_id = (
            "win_"
            + hashlib.sha256(
                f"{document.planned.document_id}:{start}:{end}:{digest}".encode("ascii")
            ).hexdigest()[:24]
        )
        window = SourceWindow(
            schema_version=SOURCE_IMPORT_SCHEMA_VERSION,
            window_id=window_id,
            document_id=document.planned.document_id,
            relative_path=document.planned.relative_path,
            start_offset=start,
            end_offset=end,
            start_line=bisect.bisect_left(newlines, start) + 1,
            end_line=bisect.bisect_left(newlines, max(start, end - 1)) + 1,
            text=text,
            sha256=digest,
        )
        _validate_window(window)
        windows.append(window)
    return tuple(windows)


def _normalize_candidates(
    value: object, window: SourceWindow, on_drop=None
) -> list[dict[str, object]]:
    _bounded_json(value, MAX_ROLE_RESULT_BYTES, "extract_result_too_large")
    if type(value) is not list or len(value) > MAX_CANDIDATES_PER_WINDOW:
        _fail("invalid_extract_result")
    normalized: list[dict[str, object]] = []
    dropped = 0
    for candidate in value:
        if type(candidate) is not dict or set(candidate) != {
            "text",
            "confidence",
            "evidence",
            "tags",
        }:
            _fail("invalid_extract_result")
        text = candidate["text"]
        confidence = candidate["confidence"]
        evidence = candidate["evidence"]
        tags = candidate["tags"]
        if (
            type(text) is not str
            or not text.strip()
            or len(text) > 4_000
            or _has_forbidden_text(text, allow_layout=True)
            or type(confidence) is not str
            or confidence not in _CONFIDENCE
            or type(evidence) is not list
            or not evidence
            or len(evidence) > 8
            or type(tags) is not list
            or len(tags) > 8
            or any(
                type(tag) is not str
                or not tag.strip()
                or len(tag) > 100
                or _has_forbidden_text(tag)
                for tag in tags
            )
        ):
            _fail("invalid_extract_result")
        # With on_drop set, a claim whose quote cannot be recovered (a paraphrase even
        # after fuzzy search) does NOT fail the whole answer: a retry does not help - codex
        # repeats the same quote (runs bc-pkainthrs and bc-gicourse, 23-24.08), and three
        # retries used to kill an entire course. Such a claim is handed to on_drop and
        # skipped; every other error still fails as before.
        checked_evidence = []
        for item in evidence:
            try:
                checked_evidence.append(_normalize_evidence(item, window))
            except SourceImportError as error:
                if on_drop is None or error.args[:1] != ("evidence_quote_mismatch",):
                    raise
        if not checked_evidence:
            dropped += 1
            on_drop(candidate)
            continue
        normalized.append(
            {
                "text": text.strip(),
                "confidence": confidence,
                "evidence": checked_evidence,
                "tags": [tag.strip() for tag in tags],
            }
        )
    if dropped and not normalized:
        _fail("evidence_quote_mismatch")
    return normalized


@lru_cache(maxsize=64)
def _fold(text: str) -> tuple[str, tuple[int, ...]]:
    """Text -> (folded form, map from folded indices back to the original). The folded form
    is lowercase letters and digits, with any run of anything else (spaces, line breaks,
    punctuation) collapsed into a single space. It is needed to match a quote up to
    cosmetics: ASR transcripts punctuate arbitrarily, and the model "fixes" the punctuation
    while copying (run bc-pkainthrs, 23.08)."""
    chars: list[str] = []
    index: list[int] = []
    pending_space = False
    for pos, ch in enumerate(text):
        if not ch.isalnum():
            pending_space = bool(chars)
            continue
        if pending_space:
            chars.append(" ")
            index.append(pos - 1)
            pending_space = False
        low = ch.lower()
        chars.append(low if len(low) == 1 else ch)
        index.append(pos)
    index.append(len(text))
    return "".join(chars), tuple(index)


@lru_cache(maxsize=64)
def _fold_tokens(text: str) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    """Tokens of the folded form (see _fold) with their (start, end) bounds in the original."""
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    chars: list[str] = []
    start = 0
    for pos, ch in enumerate(text):
        if ch.isalnum():
            if not chars:
                start = pos
            low = ch.lower()
            chars.append(low if len(low) == 1 else ch)
            continue
        if chars:
            tokens.append("".join(chars))
            spans.append((start, pos))
            chars = []
    if chars:
        tokens.append("".join(chars))
        spans.append((start, len(text)))
    return tuple(tokens), tuple(spans)


@lru_cache(maxsize=64)
def _trigram_index(tokens: tuple[str, ...]) -> dict[tuple[str, ...], tuple[int, ...]]:
    index: dict[tuple[str, ...], list[int]] = {}
    for pos in range(len(tokens) - 2):
        index.setdefault(tokens[pos:pos + 3], []).append(pos)
    return {key: tuple(val) for key, val in index.items()}


_FUZZY_MIN_TOKENS = 5  # anything shorter requires an exact match (after folding)
_FUZZY_MAX_CANDIDATES = 128


def _locate_quote_fuzzy(window_text: str, quote: str, hint_local) -> tuple[int, int] | None:
    """The quote as a pointer: while copying from an ASR transcript the model swaps a word
    or drops a filler (run bc-gicourse, 24.08) and repeats that deterministically on
    retries. We look for a span with an edit budget of about 1 per 8 words; the returned
    text is still taken from the original window, and the claim is checked further by an
    independent critic."""
    qtoks, _ = _fold_tokens(quote)
    if len(qtoks) < _FUZZY_MIN_TOKENS:
        return None
    budget = max(1, len(qtoks) // 8)
    wtoks, wspans = _fold_tokens(window_text)
    index = _trigram_index(wtoks)
    starts: list[int] = []
    for qpos in range(len(qtoks) - 2):
        for wpos in index.get(qtoks[qpos:qpos + 3], ()):
            start = max(0, wpos - qpos - budget)
            if start not in starts:
                starts.append(start)
            if len(starts) >= _FUZZY_MAX_CANDIDATES:
                break
        if len(starts) >= _FUZZY_MAX_CANDIDATES:
            break
    best: tuple[int, int, int] | None = None  # (edits, start_char, end_char)
    for start in starts:
        region = list(wtoks[start:start + len(qtoks) + 2 * budget])
        blocks = [
            block for block in difflib.SequenceMatcher(
                None, list(qtoks), region, autojunk=False
            ).get_matching_blocks() if block.size
        ]
        matched = sum(block.size for block in blocks)
        if not blocks:
            continue
        first = start + blocks[0].b
        last = start + blocks[-1].b + blocks[-1].size - 1
        edits = max(len(qtoks), last - first + 1) - matched
        if edits > budget:
            continue
        span = (wspans[first][0], wspans[last][1])
        key = (edits, abs(span[0] - hint_local) if hint_local is not None else span[0])
        if best is None or key < (best[0], abs(best[1] - hint_local) if hint_local is not None else best[1]):
            best = (edits, span[0], span[1])
    if best is None:
        return None
    return best[1], best[2]


def _trim_incomplete_markup_tail(
    quote: str, source_after_quote: str,
) -> str:
    """Strip a tag or entity tail cut off by the model, if the source continues it.

    Only provably unfinished markup is stripped: the continuation right after the quote
    must close a tag with ``>`` or an entity with ``;``. Ordinary ``<`` and ``&`` in text
    without such a continuation are left alone.
    """
    tag = re.search(r"<[^<>\n]*$", quote)
    if tag is not None:
        close = source_after_quote.find(">", 0, 129)
        if close >= 0 and "\n" not in source_after_quote[:close]:
            completed = tag.group(0) + source_after_quote[:close + 1]
            if (
                re.fullmatch(r"</?[A-Za-z][^<>\n]{0,128}>", completed)
                or re.fullmatch(r"<!--[^<>\n]{0,128}-->", completed)
            ):
                return quote[:tag.start()].rstrip()
    entity = re.search(r"&(?:#[xX]?[0-9A-Fa-f]*|[A-Za-z][A-Za-z0-9]*)$", quote)
    if entity is not None and source_after_quote.startswith(";"):
        return quote[:entity.start()].rstrip()
    return quote


def _checked_gate_quote(
    window: SourceWindow, window_id: str, local_start: int, local_end: int,
) -> dict[str, object]:
    """Build evidence only if the final llm-wiki matcher can find the quote."""
    quote = window.text[local_start:local_end]
    quote = _trim_incomplete_markup_tail(quote, window.text[local_end:])
    if not quote:
        _fail("evidence_quote_mismatch")
    local_end = local_start + len(quote)
    lines = window.text.split("\n")
    if lines[-1:] == [""]:
        lines.pop()
    needle = _gate_norm(quote)
    if not needle or not _gate_find_spans(lines, needle):
        _fail("evidence_quote_mismatch")
    return {
        "window_id": window_id,
        "quote": quote,
        "start_offset": window.start_offset + local_start,
        "end_offset": window.start_offset + local_end,
    }


def _normalize_evidence(value: object, window: SourceWindow) -> dict[str, object]:
    """Role evidence -> exact offsets. The offsets are computed by CODE from the quote: the
    model (live run 23.08, codex on Cyrillic text) copies the quote correctly but not the
    positions; the start/end_offset it sends is only a hint when the quote repeats inside
    the window. The quote is matched against the window up to cosmetics (case, spaces and
    line breaks, punctuation: in run bc-pkainthrs, 23.08, the rejections were a capital
    letter, a collapsed line break, a dropped comma), and what goes into evidence is a
    substring of the ORIGINAL window, not the model's text - so the quote stays verbatim
    with respect to the source."""
    if type(value) is not dict or not {"window_id", "quote"} <= set(value) or not set(value) <= {
        "window_id",
        "quote",
        "start_offset",
        "end_offset",
    }:
        _fail("invalid_evidence")
    window_id = value["window_id"]
    quote = value["quote"]
    hint = value.get("start_offset")
    if (
        window_id != window.window_id
        or type(quote) is not str
        or not quote
        or len(quote) > MAX_QUOTE_CHARS
        or _has_forbidden_text(quote, allow_layout=True)
        or (hint is not None and type(hint) is not int)
        or ("end_offset" in value and type(value["end_offset"]) is not int)
    ):
        _fail("invalid_evidence")
    folded_text, index = _fold(window.text)
    folded_quote, _ = _fold(quote)
    if not folded_quote:
        _fail("evidence_quote_mismatch")
    positions: list[int] = []
    cursor = folded_text.find(folded_quote)
    while cursor >= 0 and len(positions) < 64:
        positions.append(cursor)
        cursor = folded_text.find(folded_quote, cursor + 1)
    if not positions:
        span = _locate_quote_fuzzy(
            window.text, quote,
            hint - window.start_offset if hint is not None else None,
        )
        if span is None:
            _fail("evidence_quote_mismatch")
        return _checked_gate_quote(window, window_id, span[0], span[1])
    folded_start = positions[0]
    if hint is not None and len(positions) > 1:
        local_hint = hint - window.start_offset
        folded_start = min(positions, key=lambda pos: abs(index[pos] - local_hint))
    local_start = index[folded_start]
    local_end = index[folded_start + len(folded_quote) - 1] + 1
    return _checked_gate_quote(window, window_id, local_start, local_end)


def _normalize_verdict(value: object) -> bool:
    _bounded_json(value, MAX_ROLE_RESULT_BYTES, "verify_result_too_large")
    if type(value) is not dict or set(value) != {"accepted", "reason"}:
        _fail("invalid_verify_result")
    accepted = value["accepted"]
    reason = value["reason"]
    if (
        type(accepted) is not bool
        or type(reason) is not str
        or not reason.strip()
        or len(reason) > 2_000
        or _has_forbidden_text(reason, allow_layout=True)
    ):
        _fail("invalid_verify_result")
    return accepted


def normalize_extract_result(
    value: object, window: SourceWindow, *, on_drop=None
) -> tuple[dict[str, object], ...]:
    """Validate one CLI extract-role answer at a strict public boundary.

    When supplied, ``on_drop`` receives a candidate whose quotations cannot be matched
    even fuzzily in the source window. Such a claim is skipped rather than rejecting the
    whole answer. The answer is still rejected when every candidate is dropped, because
    that indicates wholly unusable output.
    """

    _validate_window(window)
    return tuple(_normalize_candidates(value, window, on_drop=on_drop))


def normalize_verify_result(value: object) -> bool:
    """Public strict boundary for one CLI verify-role answer."""

    return _normalize_verdict(value)


def verified_claim_payload(
    candidate: Mapping[str, object],
    window: SourceWindow,
    *,
    source_id: str,
) -> dict[str, object]:
    """Convert one normalized candidate into the llmwiki staging shape."""

    if type(source_id) is not str or not source_id.startswith("src_"):
        _fail("invalid_llmwiki_result")
    checked = _normalize_candidates([dict(candidate)], window)[0]
    evidence: list[dict[str, object]] = []
    for item in checked["evidence"]:
        local_start = item["start_offset"] - window.start_offset
        local_end = item["end_offset"] - window.start_offset
        start_line = window.start_line + window.text.count("\n", 0, local_start)
        end_line = window.start_line + window.text.count(
            "\n", 0, max(local_start, local_end - 1)
        )
        evidence.append(
            {
                "source_id": source_id,
                "quote": item["quote"],
                "stance": "supports",
                "locator": {"type": "line", "value": f"{start_line}-{end_line}"},
            }
        )
    confidence, single_source = _single_source_confidence(
        checked["confidence"], evidence
    )
    return {
        "text": checked["text"],
        "confidence": confidence,
        "status": "verified",
        "evidence": evidence,
        "tags": checked["tags"],
        "meta": {
            "source_window": window.window_id,
            "single_source": single_source,
        },
    }


def write_verified_claims(
    topic_dir: str | Path,
    claims: list[dict[str, object]],
    *,
    wiki: object,
    on_conflict=None,
) -> tuple[int, int]:
    """Batch-write verified claims through llmwiki with strict replay checks.

    When supplied, ``on_conflict`` receives evidence for each claim that was not written
    as a new row. A cross-document duplicate is non-fatal and extends the existing
    evidence; a genuine replay mismatch is fatal, but its evidence is recorded before
    the error is raised.
    """

    if type(claims) is not list or not claims:
        _fail("invalid_verified_claims")
    add_claims = getattr(wiki, "add_claims", None)
    if not callable(add_claims):
        _fail("invalid_llmwiki_socket")

    pending: list[dict[str, object]] = []
    duplicate_count = 0
    expected_ids: dict[str, dict[str, object]] = {}
    for claim in claims:
        if type(claim) is not dict or set(claim) != {
            "text",
            "confidence",
            "status",
            "evidence",
            "tags",
            "meta",
        }:
            _fail("invalid_verified_claims")
        if claim["status"] != "verified":
            _fail("invalid_verified_claims")
        claim_id, existing = _replay_claim_snapshot(wiki, topic_dir, claim["text"])
        prior = expected_ids.get(claim_id)
        if prior is not None:
            # The same text twice in one batch is the same cross-window repeat as between
            # lessons (id = hash(text)): we keep the first record and write an evidence note.
            if prior != claim:
                _report_claim_conflict(
                    on_conflict,
                    _claim_conflict_report(
                        reason="duplicate_text_in_batch",
                        claim_id=claim_id,
                        text=claim["text"],
                        zone=None,
                        row=prior,
                        evidence=claim["evidence"],
                        tags=claim["tags"],
                        meta=claim["meta"],
                        diverged=_claim_row_divergence(
                            prior,
                            text=claim["text"],
                            confidence=claim["confidence"],
                            evidence=claim["evidence"],
                            tags=claim["tags"],
                            meta=claim["meta"],
                        ),
                    ),
                )
            duplicate_count += 1
            continue
        expected_ids[claim_id] = claim
        if existing:
            _absorb_existing_claim(
                wiki, topic_dir, claim, claim_id, existing, on_conflict
            )
            duplicate_count += 1
        else:
            pending.append(claim)

    if not pending:
        return 0, duplicate_count
    try:
        results = add_claims(topic_dir, "staging", pending)
    except SourceImportError:
        raise
    except Exception:
        _fail("llmwiki_write_failed")
    if type(results) is not list or len(results) != len(pending):
        _fail("invalid_llmwiki_result")

    written = 0
    for result, claim in zip(results, pending):
        expected_id, _ = _replay_claim_snapshot(wiki, topic_dir, claim["text"])
        if (
            type(result) is not dict
            or set(result) != {"id", "result"}
            or result["result"] not in {"written", "duplicate"}
            or result["id"] != expected_id
        ):
            _fail("invalid_llmwiki_result")
        if result["result"] == "written":
            written += 1
            continue
        _, existing = _replay_claim_snapshot(wiki, topic_dir, claim["text"])
        _absorb_existing_claim(
            wiki, topic_dir, claim, expected_id, existing, on_conflict
        )
        duplicate_count += 1
    return written, duplicate_count


def _single_source_confidence(
    confidence: object, evidence: list[dict[str, object]]
) -> tuple[str, bool]:
    if type(confidence) is not str or confidence not in _CONFIDENCE:
        _fail("invalid_extract_result")
    source_ids = {
        item["source_id"] for item in evidence if type(item.get("source_id")) is str
    }
    single_source = len(source_ids) == 1
    if single_source and confidence == "high":
        confidence = "medium"
    return confidence, single_source


def _replay_claim_snapshot(
    wiki: object, topic_dir: str | Path, text: object
) -> tuple[str, list[tuple[str, dict[str, object]]]]:
    """Read topic identity and both claim zones from one anchored topic fd."""

    topic_fd = _open_replay_topic(topic_dir)
    manifest_fd: int | None = None
    zone_snapshots: list[_ClaimZoneSnapshot] = []
    try:
        try:
            topic_before = os.fstat(topic_fd)
            manifest_fd = _open_regular_at(
                topic_fd, "topic.json", "claim_replay_lookup_failed"
            )
            manifest_before = os.fstat(manifest_fd)
            raw_manifest = _read_open_file(
                manifest_fd,
                MAX_TOPIC_MANIFEST_BYTES,
                "claim_replay_lookup_failed",
            )
        except (OSError, SourceImportError):
            _fail("claim_replay_lookup_failed")
        slug = _decode_topic_slug(raw_manifest)
        claim_id = _expected_claim_id(wiki, slug, text)
        existing = _read_existing_claims_at(
            topic_fd, claim_id, snapshots=zone_snapshots
        )
        for snapshot in zone_snapshots:
            _assert_claim_zone_stable(topic_fd, snapshot)
        _assert_regular_entry_stable(
            topic_fd, "topic.json", manifest_fd, manifest_before
        )
        _assert_topic_path_stable(topic_dir, topic_fd, topic_before)
        return claim_id, existing
    finally:
        for snapshot in reversed(zone_snapshots):
            if snapshot.file_fd is not None:
                os.close(snapshot.file_fd)
            os.close(snapshot.directory_fd)
        if manifest_fd is not None:
            os.close(manifest_fd)
        os.close(topic_fd)


def _open_replay_topic(topic_dir: str | Path) -> int:
    try:
        path = Path(topic_dir)
        rendered = os.fspath(path)
    except (TypeError, ValueError, UnicodeError):
        _fail("claim_replay_lookup_failed")
    if not rendered or _has_forbidden_text(rendered):
        _fail("claim_replay_lookup_failed")
    components = list(path.parts)
    if path.is_absolute():
        components = components[1:]
    if len(components) > 64:
        _fail("claim_replay_lookup_failed")
    base = "/" if path.is_absolute() else "."
    current_fd: int | None = None
    try:
        current_fd = os.open(
            base,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        for component in components:
            next_fd = _open_directory_at(
                current_fd, component, "claim_replay_lookup_failed"
            )
            os.close(current_fd)
            current_fd = next_fd
        metadata = os.fstat(current_fd)
        if not stat.S_ISDIR(metadata.st_mode):
            _fail("claim_replay_lookup_failed")
        result = current_fd
        current_fd = None
        return result
    except (OSError, SourceImportError):
        _fail("claim_replay_lookup_failed")
    finally:
        if current_fd is not None:
            os.close(current_fd)


def _decode_topic_slug(raw: bytes) -> str:
    try:
        manifest = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_claim_object_no_duplicates,
        )
    except SourceImportError:
        raise
    except (UnicodeError, ValueError, RecursionError):
        _fail("claim_replay_lookup_failed")
    slug = manifest.get("slug") if type(manifest) is dict else None
    if (
        type(slug) is not str
        or not slug
        or len(slug) > 1024
        or slug[0] not in "abcdefghijklmnopqrstuvwxyz0123456789"
        or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in slug)
    ):
        _fail("claim_replay_lookup_failed")
    return slug


def _expected_claim_id(wiki: object, slug: str, text: object) -> str:
    make_claim_id = getattr(wiki, "make_claim_id", None)
    if not callable(make_claim_id) or type(text) is not str:
        _fail("invalid_llmwiki_socket")
    try:
        claim_id = make_claim_id(slug, text)
    except Exception:
        _fail("claim_replay_lookup_failed")
    if (
        type(claim_id) is not str
        or len(claim_id) > 128
        or _has_forbidden_text(claim_id)
    ):
        _fail("claim_replay_lookup_failed")
    return claim_id


def _claim_origin(meta: object) -> str | None:
    """The window a claim was written from.

    In llm-wiki a claim id is hash(text), so the id alone does NOT distinguish our earlier
    write from the same fact stated in another lesson: only the window does.
    """

    if type(meta) is not dict:
        return None
    origin = meta.get("source_window")
    return origin if type(origin) is str else None


def _evidence_source_ids(evidence: object) -> list[str]:
    if type(evidence) is not list:
        return []
    return sorted(
        {
            item["source_id"]
            for item in evidence
            if type(item) is dict and type(item.get("source_id")) is str
        }
    )


def _claim_row_divergence(
    row: Mapping[str, object],
    *,
    text: object,
    confidence: str,
    evidence: list[dict[str, object]],
    tags: object,
    meta: dict[str, object],
) -> list[str]:
    """Fields of an already written row that diverge from what we are about to write.

    evidence is compared by CONTAINMENT rather than byte for byte: the socket's evidence
    list grows append-only (merge_claim_evidence), so extra entries in a row are legitimate
    growth from another lesson, while a missing entry of ours is a real divergence.
    """

    stored_evidence = row.get("evidence")
    diverged: list[str] = []
    if row.get("text") != text:
        diverged.append("text")
    if row.get("confidence") != confidence:
        diverged.append("confidence")
    if row.get("status") not in {"verified", "final"}:
        diverged.append("status")
    if type(stored_evidence) is not list or any(
        item not in stored_evidence for item in evidence
    ):
        diverged.append("evidence")
    if row.get("tags") != (tags if tags else None):
        diverged.append("tags")
    if row.get("meta") != meta:
        diverged.append("meta")
    return diverged


def _claim_conflict_report(
    *,
    reason: str,
    claim_id: str,
    text: object,
    zone: str | None,
    row: Mapping[str, object],
    evidence: list[dict[str, object]],
    tags: object,
    meta: dict[str, object],
    diverged: list[str],
) -> dict[str, object]:
    """An evidence note for a claim that did not land as a new row: both sides and what diverged."""

    return {
        "reason": reason,
        "claim_id": claim_id,
        "zone": zone,
        "text": text,
        "diverged": diverged,
        "incoming": {
            "source_window": _claim_origin(meta),
            "source_ids": _evidence_source_ids(evidence),
            "tags": list(tags) if type(tags) is list else tags,
        },
        "existing": {
            "source_window": _claim_origin(row.get("meta")),
            "source_ids": _evidence_source_ids(row.get("evidence")),
            "tags": row.get("tags"),
        },
    }


def _resolve_claim_replay(
    existing: list[tuple[str, dict[str, object]]],
    *,
    claim_id: str,
    text: object,
    confidence: str,
    evidence: list[dict[str, object]],
    tags: object,
    meta: dict[str, object],
    on_conflict=None,
) -> tuple[str, str | None, Mapping[str, object] | None, list[str]]:
    """Who wrote the row that already sits under this id: us or another lesson.

    The strict replay assert guards the idempotency of resume ("the run was restarted, so a
    previously written claim must match"), and it must apply ONLY to what this same run
    wrote. Our own record is recognized by meta.source_window: it carries the document and
    the window offsets, so another lesson has a different one. The same fact from a
    different window is a normal cross-document duplicate, not a failure; what stays fatal
    is a divergence within one window (overwriting knowledge already recorded).
    Returns (_REPLAY_OWN|_REPLAY_FOREIGN, zone, row, diverged fields).
    """

    foreign: tuple[str, dict[str, object], list[str]] | None = None
    for zone, row in existing:
        diverged = _claim_row_divergence(
            row,
            text=text,
            confidence=confidence,
            evidence=evidence,
            tags=tags,
            meta=meta,
        )
        if not diverged:
            return _REPLAY_OWN, zone, row, []
        if _claim_origin(row.get("meta")) != _claim_origin(meta):
            if foreign is None:
                foreign = (zone, row, diverged)
            continue
        _report_claim_conflict(
            on_conflict,
            _claim_conflict_report(
                reason="claim_replay_conflict",
                claim_id=claim_id,
                text=text,
                zone=zone,
                row=row,
                evidence=evidence,
                tags=tags,
                meta=meta,
                diverged=diverged,
            ),
        )
        _fail(
            "claim_replay_conflict",
            f"{claim_id} in {zone}: same window {_claim_origin(meta)}, "
            f"diverged {', '.join(diverged)}",
        )
    if foreign is not None:
        return _REPLAY_FOREIGN, foreign[0], foreign[1], foreign[2]
    _report_claim_conflict(
        on_conflict,
        _claim_conflict_report(
            reason="claim_replay_missing",
            claim_id=claim_id,
            text=text,
            zone=None,
            row={},
            evidence=evidence,
            tags=tags,
            meta=meta,
            diverged=["missing"],
        ),
    )
    _fail("claim_replay_conflict", f"{claim_id}: the socket reports a duplicate but there is no row")


def _report_claim_conflict(on_conflict, report: dict[str, object]) -> None:
    if on_conflict is not None:
        on_conflict(report)


def _merge_foreign_claim_evidence(
    wiki: object,
    topic_dir: str | Path,
    zone: str | None,
    claim_id: str,
    evidence: list[dict[str, object]],
) -> dict[str, object]:
    """Append another lesson's evidence to an already written claim - through the socket only.

    The same choice as `_reconcile_kept_candidates` makes during search: the same fact from
    a second source is not discarded but grows onto the claim. The socket deduplicates
    evidence writes by exact match, so a repeat after an interruption grows nothing - the
    merge is idempotent, as resume requires. A socket failure does not fail the course: the
    knowledge is already written, only the link to the second lesson is lost, and that link
    stays in the evidence note on disk.
    """

    merge = getattr(wiki, "merge_claim_evidence", None)
    if zone is None or not callable(merge):
        return {"merge": "unavailable"}
    try:
        results = merge(topic_dir, zone, claim_id, [dict(item) for item in evidence])
    except Exception as error:  # noqa: BLE001 - a socket failure must not fail the course
        return {"merge": "failed", "merge_error": str(error)[:500]}
    if type(results) is not list:
        return {"merge": "failed", "merge_error": "invalid_llmwiki_result"}
    return {
        "merge": "merged",
        "merge_results": [
            item.get("result") for item in results if type(item) is dict
        ],
    }


def _absorb_existing_claim(
    wiki: object,
    topic_dir: str | Path,
    claim: Mapping[str, object],
    claim_id: str,
    existing: list[tuple[str, dict[str, object]]],
    on_conflict=None,
) -> str:
    """A row with this id already exists: either our idempotent replay or another lesson."""

    kind, zone, row, diverged = _resolve_claim_replay(
        existing,
        claim_id=claim_id,
        text=claim["text"],
        confidence=claim["confidence"],
        evidence=claim["evidence"],
        tags=claim["tags"],
        meta=claim["meta"],
        on_conflict=on_conflict,
    )
    if kind != _REPLAY_FOREIGN:
        return kind
    report = _claim_conflict_report(
        reason="cross_document_duplicate",
        claim_id=claim_id,
        text=claim["text"],
        zone=zone,
        row=row if row is not None else {},
        evidence=claim["evidence"],
        tags=claim["tags"],
        meta=claim["meta"],
        diverged=diverged,
    )
    report.update(
        _merge_foreign_claim_evidence(
            wiki, topic_dir, zone, claim_id, claim["evidence"]
        )
    )
    _report_claim_conflict(on_conflict, report)
    return kind


def _read_existing_claims_at(
    topic_fd: int,
    claim_id: str,
    *,
    snapshots: list[_ClaimZoneSnapshot],
) -> list[tuple[str, dict[str, object]]]:
    """Rows of both zones together with their zone: an evidence merge is addressed by the row's zone."""

    result: list[tuple[str, dict[str, object]]] = []
    for zone in ("staging", "final"):
        snapshot = _open_claim_zone_snapshot(topic_fd, zone)
        snapshots.append(snapshot)
        result.extend((zone, row) for row in _claim_rows(snapshot.raw, claim_id))
    return result


def _open_claim_zone_snapshot(topic_fd: int, zone: str) -> _ClaimZoneSnapshot:
    zone_fd: int | None = None
    file_fd: int | None = None
    try:
        zone_fd = _open_directory_at(topic_fd, zone, "claim_replay_lookup_failed")
        zone_before = os.fstat(zone_fd)
        file_fd, file_before, raw = _open_optional_claim_file(zone_fd)
        result = _ClaimZoneSnapshot(
            zone=zone,
            directory_fd=zone_fd,
            directory_before=zone_before,
            file_fd=file_fd,
            file_before=file_before,
            raw=raw,
        )
        zone_fd = None
        file_fd = None
        return result
    except (OSError, SourceImportError):
        _fail("claim_replay_lookup_failed")
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if zone_fd is not None:
            os.close(zone_fd)


def _claim_rows(raw: bytes, claim_id: str) -> list[dict[str, object]]:
    if raw and not raw.endswith(b"\n"):
        _fail("claim_replay_lookup_failed")
    found: list[dict[str, object]] = []
    for line in raw.splitlines():
        if not line or len(line) > MAX_ROLE_RESULT_BYTES:
            _fail("claim_replay_lookup_failed")
        try:
            row = json.loads(line, object_pairs_hook=_claim_object_no_duplicates)
        except SourceImportError:
            raise
        except (UnicodeError, ValueError, RecursionError):
            _fail("claim_replay_lookup_failed")
        if type(row) is not dict:
            _fail("claim_replay_lookup_failed")
        if row.get("id") == claim_id:
            if found:
                _fail("claim_replay_lookup_failed")
            found.append(row)
    return found


def _open_optional_claim_file(
    zone_fd: int,
) -> tuple[int | None, os.stat_result | None, bytes]:
    try:
        path_before = os.stat("claims.jsonl", dir_fd=zone_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None, None, b""
    except OSError:
        _fail("claim_replay_lookup_failed")
    if not stat.S_ISREG(path_before.st_mode):
        _fail("claim_replay_lookup_failed")
    file_fd: int | None = None
    try:
        file_fd = _open_regular_at(
            zone_fd, "claims.jsonl", "claim_replay_lookup_failed"
        )
        opened_before = os.fstat(file_fd)
        raw = _read_open_file(
            file_fd, MAX_EXISTING_CLAIMS_BYTES, "claim_replay_lookup_failed"
        )
        if _regular_signature(path_before) != _regular_signature(opened_before):
            _fail("claim_replay_lookup_failed")
        result = (file_fd, opened_before, raw)
        file_fd = None
        return result
    except (OSError, SourceImportError):
        _fail("claim_replay_lookup_failed")
    finally:
        if file_fd is not None:
            os.close(file_fd)


def _assert_claim_zone_stable(topic_fd: int, snapshot: _ClaimZoneSnapshot) -> None:
    if snapshot.file_fd is not None:
        if snapshot.file_before is None:
            _fail("claim_replay_lookup_failed")
        _assert_regular_entry_stable(
            snapshot.directory_fd,
            "claims.jsonl",
            snapshot.file_fd,
            snapshot.file_before,
        )
    _assert_directory_entry_stable(
        topic_fd,
        snapshot.zone,
        snapshot.directory_fd,
        snapshot.directory_before,
    )


def _assert_regular_entry_stable(
    parent_fd: int,
    name: str,
    file_fd: int,
    before: os.stat_result,
) -> None:
    try:
        after = os.fstat(file_fd)
        linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        _fail("claim_replay_lookup_failed")
    if (
        not stat.S_ISREG(after.st_mode)
        or not stat.S_ISREG(linked.st_mode)
        or _regular_signature(before) != _regular_signature(after)
        or _regular_signature(before) != _regular_signature(linked)
    ):
        _fail("claim_replay_lookup_failed")


def _assert_directory_entry_stable(
    parent_fd: int,
    name: str,
    directory_fd: int,
    before: os.stat_result,
) -> None:
    try:
        after = os.fstat(directory_fd)
        linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        _fail("claim_replay_lookup_failed")
    if (
        not stat.S_ISDIR(after.st_mode)
        or not stat.S_ISDIR(linked.st_mode)
        or _directory_signature(before) != _directory_signature(after)
        or _directory_signature(before) != _directory_signature(linked)
    ):
        _fail("claim_replay_lookup_failed")


def _assert_topic_path_stable(
    topic_dir: str | Path,
    topic_fd: int,
    before: os.stat_result,
) -> None:
    try:
        after = os.fstat(topic_fd)
    except OSError:
        _fail("claim_replay_lookup_failed")
    if _directory_signature(before) != _directory_signature(after):
        _fail("claim_replay_lookup_failed")
    reopened_fd = _open_replay_topic(topic_dir)
    try:
        try:
            linked = os.fstat(reopened_fd)
        except OSError:
            _fail("claim_replay_lookup_failed")
        if _directory_signature(before) != _directory_signature(linked):
            _fail("claim_replay_lookup_failed")
    finally:
        os.close(reopened_fd)


def _regular_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _claim_object_no_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("claim_replay_lookup_failed")
        result[key] = value
    return result


def _validate_planned_document(document: PlannedDocument) -> None:
    if type(document) is not PlannedDocument:
        _fail("invalid_import_plan")
    if (
        type(document.document_id) is not str
        or not _is_prefixed_hex(document.document_id, "doc_", 24)
        or type(document.relative_path) is not str
        or _normalized_relative(document.relative_path) != document.relative_path
        or type(document.title) is not str
        or not document.title
        or len(document.title) > 300
        or _has_forbidden_text(document.title)
        or not _is_sha256(document.sha256)
        or type(document.size_bytes) is not int
        or not 0 <= document.size_bytes <= MAX_DOCUMENT_BYTES
        or type(document.window_count) is not int
        or not 0 <= document.window_count <= MAX_WINDOWS_PER_DOCUMENT
    ):
        _fail("invalid_import_plan")
    expected_id = (
        "doc_"
        + hashlib.sha256(
            (document.relative_path + "\0" + document.sha256).encode("utf-8")
        ).hexdigest()[:24]
    )
    if document.document_id != expected_id:
        _fail("invalid_import_plan")


def _validate_skipped_document(document: SkippedDocument) -> None:
    if type(document) is not SkippedDocument:
        _fail("invalid_import_plan")
    if (
        type(document.relative_path) is not str
        or _normalized_relative(document.relative_path) != document.relative_path
        or type(document.title) is not str
        or not document.title
        or len(document.title) > 300
        or _has_forbidden_text(document.title)
        or type(document.reason) is not str
        or document.reason not in _SKIP_REASONS
        or type(document.size_bytes) is not int
        or not 0 <= document.size_bytes <= MAX_MANIFEST_SIZE_VALUE
    ):
        _fail("invalid_import_plan")


def _validate_plan(plan: ImportPlan) -> None:
    if type(plan) is not ImportPlan:
        _fail("invalid_import_plan")
    if (
        type(plan.schema_version) is not int
        or plan.schema_version != SOURCE_IMPORT_SCHEMA_VERSION
        or type(plan.plan_id) is not str
        or not _is_prefixed_hex(plan.plan_id, "plan_", 24)
        or plan.status != "waiting_human"
        or type(plan.source_kind) is not str
        or plan.source_kind not in _SOURCE_KINDS - {"auto"}
        or not _is_sha256(plan.corpus_sha256)
        or type(plan.documents) is not tuple
        or (not plan.documents and not plan.skipped)
        or len(plan.documents) + len(plan.skipped) > MAX_DOCUMENTS
        or type(plan.window_chars) is not int
        or plan.window_chars != MAX_WINDOW_CHARS
        or type(plan.overlap_chars) is not int
        or plan.overlap_chars != WINDOW_OVERLAP_CHARS
    ):
        _fail("invalid_import_plan")
    for document in plan.documents:
        _validate_planned_document(document)
    for document in plan.skipped:
        _validate_skipped_document(document)
    relative_paths = [document.relative_path for document in plan.documents]
    skipped_paths = [document.relative_path for document in plan.skipped]
    document_ids = [document.document_id for document in plan.documents]
    if (
        relative_paths != sorted(relative_paths, key=str.casefold)
        or skipped_paths != sorted(skipped_paths, key=str.casefold)
        or len({path.casefold() for path in relative_paths}) != len(relative_paths)
        or len({path.casefold() for path in skipped_paths}) != len(skipped_paths)
        or set(map(str.casefold, relative_paths)) & set(map(str.casefold, skipped_paths))
        or len(set(document_ids)) != len(document_ids)
        or sum(document.size_bytes for document in plan.documents) > MAX_TOTAL_BYTES
    ):
        _fail("invalid_import_plan")
    body = {
        "schema_version": SOURCE_IMPORT_SCHEMA_VERSION,
        "source_kind": plan.source_kind,
        "documents": [document.to_dict() for document in plan.documents],
        "window": {
            "max_chars": plan.window_chars,
            "overlap_chars": plan.overlap_chars,
        },
    }
    if plan.skipped:
        body["skipped"] = [document.to_dict() for document in plan.skipped]
    corpus_sha256 = _digest_json(body)
    plan_id = (
        "plan_"
        + hashlib.sha256(
            ("source-import-v1:" + corpus_sha256).encode("ascii")
        ).hexdigest()[:24]
    )
    if plan.corpus_sha256 != corpus_sha256 or plan.plan_id != plan_id:
        _fail("invalid_import_plan")


def _validate_window(window: SourceWindow) -> None:
    if type(window) is not SourceWindow:
        _fail("invalid_source_window")
    if (
        type(window.schema_version) is not int
        or window.schema_version != SOURCE_IMPORT_SCHEMA_VERSION
        or type(window.window_id) is not str
        or not _is_prefixed_hex(window.window_id, "win_", 24)
        or type(window.document_id) is not str
        or not _is_prefixed_hex(window.document_id, "doc_", 24)
        or type(window.relative_path) is not str
        or _normalized_relative(window.relative_path) != window.relative_path
        or type(window.start_offset) is not int
        or type(window.end_offset) is not int
        or not 0 <= window.start_offset < window.end_offset <= MAX_DOCUMENT_BYTES
        or type(window.start_line) is not int
        or type(window.end_line) is not int
        or not 1 <= window.start_line <= window.end_line <= MAX_DOCUMENT_BYTES + 1
        or type(window.text) is not str
        or not window.text
        or len(window.text) > MAX_WINDOW_CHARS
        or _has_forbidden_text(window.text, allow_layout=True)
        or window.end_line - window.start_line != window.text[:-1].count("\n")
        or window.end_offset - window.start_offset != len(window.text)
        or not _is_sha256(window.sha256)
        or hashlib.sha256(window.text.encode("utf-8")).hexdigest() != window.sha256
    ):
        _fail("invalid_source_window")
    expected_id = (
        "win_"
        + hashlib.sha256(
            f"{window.document_id}:{window.start_offset}:{window.end_offset}:{window.sha256}".encode(
                "ascii"
            )
        ).hexdigest()[:24]
    )
    if window.window_id != expected_id:
        _fail("invalid_source_window")


def _role_name(role: object) -> str:
    try:
        name = getattr(role, "role", None)
        callable_role = callable(role)
    except Exception:
        _fail("invalid_role")
    if (
        type(name) is not str
        or not name.strip()
        or len(name) > 100
        or _has_forbidden_text(name)
        or not callable_role
    ):
        _fail("invalid_role")
    return name.strip().casefold()


def _bounded_json(value: object, limit: int, code: str) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeError):
        _fail(code.replace("too_large", "invalid"))
    if len(rendered) > limit:
        _fail(code)
    return rendered


def _digest_json(value: object) -> str:
    return hashlib.sha256(
        _bounded_json(value, MAX_PLAN_BYTES, "plan_too_large")
    ).hexdigest()


def _has_forbidden_text(value: str, *, allow_layout: bool = False) -> bool:
    if not isinstance(value, str):
        return True
    allowed = {"\n", "\r", "\t"} if allow_layout else set()
    return any(
        char not in allowed
        and (
            unicodedata.category(char).startswith("C")
            or unicodedata.category(char) in {"Zl", "Zp"}
        )
        for char in value
    )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_prefixed_hex(value: object, prefix: str, digits: int) -> bool:
    return (
        type(value) is str
        and value.startswith(prefix)
        and len(value) == len(prefix) + digits
        and all(char in "0123456789abcdef" for char in value[len(prefix) :])
    )


def _fail(code: str, detail: str | None = None):
    raise SourceImportError(code, detail=detail) from None


def _fail_document(code: str, relative_path: str):
    raise SourceImportError(code, relative_path=relative_path) from None


__all__ = [
    "SOURCE_IMPORT_SCHEMA_VERSION",
    "MAX_DOCUMENTS",
    "MAX_DOCUMENT_BYTES",
    "MAX_TOTAL_BYTES",
    "MAX_WINDOW_CHARS",
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
