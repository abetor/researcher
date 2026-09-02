"""GitHub REST adapter for repository search and README retrieval.

Repository search provides engineering signal; fetch returns an existing README as
Markdown. GITHUB_TOKEN, when present, raises the API limit and is never hard-coded.
Issues, pull requests, discussions, GraphQL, and code search remain outside this adapter.
"""
from __future__ import annotations

import base64
import os
import re
import urllib.parse

from . import _http
from .base import SourceError, SourceTool

_SEARCH = "https://api.github.com/search/repositories?"
_API = "https://api.github.com"


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"  # raises the limit from 60 to 5000 req/hr
    return h


class GitHub(SourceTool):
    name = "github"
    kind = "repo"
    desc = "GitHub API: repository search + README (a token in env GITHUB_TOKEN raises the rate limit)"

    def search(self, query: str, *, limit: int = 8) -> list[dict]:
        q = urllib.parse.urlencode({"q": query, "per_page": max(1, min(limit, 50)),
                                    "sort": "stars", "order": "desc"})
        data = _http.get_json(_SEARCH + q, headers=_headers())
        out = []
        for r in data.get("items", []):
            stars = r.get("stargazers_count")
            bits = [b for b in (r.get("description") or "",
                                f"{stars}*" if stars is not None else "",
                                r.get("language") or "") if b]
            out.append({"url": r.get("html_url") or "", "title": r.get("full_name") or "",
                        "snippet": " | ".join(bits)})
        return out

    def fetch(self, url: str) -> str:
        owner, repo = _owner_repo(url)
        data = _http.get_json(f"{_API}/repos/{owner}/{repo}/readme", headers=_headers())
        content = data.get("content") or ""
        if data.get("encoding") == "base64" and content:
            try:
                text = base64.b64decode(content).decode("utf-8", "replace")
            except Exception as e:
                raise SourceError(f"could not decode the README of {owner}/{repo}: {e}") from e
        else:
            text = content
        return f"# {owner}/{repo} README\n\n{text.strip()}\n"


def _owner_repo(url: str) -> tuple[str, str]:
    """owner/repo from github.com/owner/repo(/...) or from a short owner/repo."""
    u = url.strip()
    m = re.search(r"github\.com/([^/\s]+)/([^/\s#?]+)", u, re.I)
    if m:
        owner, repo = m.group(1), m.group(2)
    elif "/" in u and " " not in u:
        owner, rest = u.split("/", 1)
        repo = rest.split("/", 1)[0]
    else:
        raise SourceError(f"expected github.com/owner/repo or owner/repo, got {url!r}")
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo:
        raise SourceError(f"empty owner/repo in {url!r}")
    return owner, repo
