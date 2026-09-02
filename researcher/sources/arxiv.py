"""Keyless arXiv Atom API adapter for AI and computer-science sources.

The adapter returns metadata, abstract, and a PDF link rather than full text. Downloading
PDF content is a separate step. Requests respect arXiv's interval guidance, while _http
backs off from sporadic 429 responses.
"""
from __future__ import annotations

import re
import urllib.parse
import xml.etree.ElementTree as ET

from . import _http
from .base import SourceError, SourceTool

_API = "http://export.arxiv.org/api/query?"
_ATOM = "{http://www.w3.org/2005/Atom}"
_WS = re.compile(r"\s+")
_DOCTYPE = re.compile(r"<!DOCTYPE", re.I)


class Arxiv(SourceTool):
    name = "arxiv"
    kind = "paper"
    desc = "arXiv API (no key): AI/CS/physics preprints - titles, authors, abstracts, PDF link"

    def search(self, query: str, *, limit: int = 8) -> list[dict]:
        q = urllib.parse.urlencode({"search_query": f"all:{query}", "start": 0,
                                    "max_results": max(1, min(limit, 30)),
                                    "sortBy": "relevance"})
        entries = _parse_entries(_http.get_text(_API + q))
        return [{"url": e["url"], "title": e["title"], "snippet": _snippet(e)} for e in entries]

    def fetch(self, url: str) -> str:
        aid = _arxiv_id(url)
        q = urllib.parse.urlencode({"id_list": aid, "max_results": 1})
        entries = _parse_entries(_http.get_text(_API + q))
        if not entries:
            raise SourceError(f"arXiv returned no entry for {aid!r}")
        return _render_entry(entries[0])


def _ws(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _parse_entries(xml_text: str) -> list[dict]:
    # The stdlib ElementTree is safe against XXE/DTD retrieval but vulnerable to entity
    # expansion (billion laughs / quadratic blowup). That vector needs an internal DTD with
    # <!ENTITY>, which is only possible with a <!DOCTYPE. Legitimate arXiv Atom contains no
    # DOCTYPE, so we cut it off at the entrance (no defusedxml: the workspace is
    # stdlib-only, with no external dependencies).
    if _DOCTYPE.search(xml_text):
        raise SourceError("arXiv XML with a DOCTYPE rejected (entity-expansion guard)")
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise SourceError(f"arXiv returned something that is not XML: {e}") from e
    out = []
    for e in root.findall(f"{_ATOM}entry"):
        authors = [n.strip() for a in e.findall(f"{_ATOM}author")
                   if (n := a.findtext(f"{_ATOM}name"))]
        pdf = ""
        for link in e.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf":
                pdf = link.get("href") or ""
        out.append({
            "url": (e.findtext(f"{_ATOM}id") or "").strip(),
            "title": _ws(e.findtext(f"{_ATOM}title") or ""),
            "summary": _ws(e.findtext(f"{_ATOM}summary") or ""),
            "published": (e.findtext(f"{_ATOM}published") or "").strip(),
            "authors": authors,
            "pdf": pdf,
        })
    return out


def _snippet(e: dict) -> str:
    who = ", ".join(e["authors"][:3]) + (" et al." if len(e["authors"]) > 3 else "")
    summ = e["summary"]
    summ = summ[:300] + "..." if len(summ) > 300 else summ
    return " | ".join(p for p in (who, e["published"][:10], summ) if p)


def _arxiv_id(url: str) -> str:
    """The id from arxiv.org/abs/NNNN, /pdf/NNNN(.pdf) or a bare id (the vN version is kept)."""
    u = url.strip()
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+)", u, re.I)
    aid = (m.group(1) if m else u).rstrip("/")
    if aid.lower().endswith(".pdf"):
        aid = aid[:-4]
    if not aid:
        raise SourceError(f"could not parse an arXiv id out of {url!r}")
    return aid


def _render_entry(e: dict) -> str:
    lines = [f"# {e['title']}", ""]
    if e["authors"]:
        lines.append("Authors: " + ", ".join(e["authors"]))
    if e["published"]:
        lines.append("Published: " + e["published"][:10])
    if e["url"]:
        lines.append("arXiv: " + e["url"])
    if e["pdf"]:
        lines.append("PDF: " + e["pdf"])
    lines += ["", "## Abstract", "", e["summary"]]
    return "\n".join(lines).strip() + "\n"
