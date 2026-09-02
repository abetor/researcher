"""Keyless Hacker News adapter built on the Algolia HN API.

It provides inexpensive engineering and AI signal. Search returns stories, while fetch
renders the complete post and comment tree as Markdown.
"""
from __future__ import annotations

import html
import re
import urllib.parse

from . import _http
from .base import SourceError, SourceTool

_SEARCH = "https://hn.algolia.com/api/v1/search?"
_ITEM = "https://hn.algolia.com/api/v1/items/"
_TAG = re.compile(r"<[^>]+>")


class HackerNews(SourceTool):
    name = "hn"
    kind = "other"  # the llm-wiki source.schema has no separate forum/community type
    desc = "Hacker News (Algolia API, no key): threads and comments on engineering/AI topics"

    def search(self, query: str, *, limit: int = 8) -> list[dict]:
        q = urllib.parse.urlencode({"query": query, "tags": "story",
                                    "hitsPerPage": max(1, min(limit, 50))})
        data = _http.get_json(_SEARCH + q)
        out = []
        for h in data.get("hits", []):
            oid = h.get("objectID")
            if not oid:
                continue
            title = h.get("title") or h.get("story_title") or "(untitled)"
            bits = []
            if h.get("points") is not None:
                bits.append(f"{h['points']} points")
            if h.get("num_comments") is not None:
                bits.append(f"{h['num_comments']} comments")
            if h.get("author"):
                bits.append(f"by {h['author']}")
            if h.get("url"):
                bits.append(h["url"])  # the story's external link, if any
            out.append({"url": f"https://news.ycombinator.com/item?id={oid}",
                        "title": title, "snippet": ", ".join(bits)})
        return out

    def fetch(self, url: str) -> str:
        data = _http.get_json(_ITEM + _item_id(url))
        return _render_item(data)


def _item_id(url: str) -> str:
    """The id from news.ycombinator.com/item?id=NNN, .../items/NNN or a bare number."""
    u = url.strip()
    if u.isdigit():
        return u
    parsed = urllib.parse.urlparse(u)
    qs = urllib.parse.parse_qs(parsed.query)
    if qs.get("id"):
        return qs["id"][0]
    tail = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    if tail.isdigit():
        return tail
    raise SourceError(f"could not parse an HN item id out of {url!r}")


def _detag(s: str) -> str:
    """HN comment HTML -> text: <p> becomes a newline, tags are dropped, entities decoded."""
    s = s.replace("<p>", "\n\n").replace("</p>", "")
    return html.unescape(_TAG.sub("", s)).strip()


def _render_item(data: dict, *, max_comments: int = 120) -> str:
    title = data.get("title") or data.get("story_title") or "HN item"
    lines = [f"# {title}", ""]
    if data.get("url"):
        lines.append(f"Link: {data['url']}")
    if data.get("author"):
        lines.append(f"Author: {data['author']}")
    if data.get("points") is not None:
        lines.append(f"Points: {data['points']}")
    if data.get("text"):
        lines += ["", _detag(data["text"])]
    lines += ["", "## Comments", ""]
    count = [0]

    def walk(node: dict, depth: int) -> None:
        for c in node.get("children") or []:
            if count[0] >= max_comments:
                return
            txt = _detag(c.get("text") or "")
            if txt:
                count[0] += 1
                lines.append(f"{'  ' * depth}- **{c.get('author') or '(deleted)'}**: {txt}")
            walk(c, depth + 1)

    walk(data, 0)
    return "\n".join(lines).strip() + "\n"
