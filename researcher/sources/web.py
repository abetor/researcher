"""Keyless web search and page retrieval for harnesses without native web tools.

One SourceTool interface supports interchangeable backends so adding Exa changes
configuration rather than collector prompts. Paid search is optional and never the
default. RESEARCHER_WEB_BACKEND selects ddg, searxng, or exa. DuckDuckGo Lite is the
keyless default; SearXNG requires a user-operated JSON endpoint; Exa requires explicit
selection and EXA_API_KEY. Without an explicit selection, a configured SearXNG URL wins,
otherwise DuckDuckGo is used.

DuckDuckGo Lite uses POST because live checks found GET returned a long-lived anti-bot
page while the equivalent form POST returned results. A transparent user agent proved
more reliable than browser impersonation. Public SearXNG instances frequently returned
429 or CAPTCHA responses, so this backend targets a user-operated instance. Anti-bot
pages can arrive with a successful HTTP status and are explicitly converted to
SourceError rather than misreported as an empty result set.

fetch(url) is independent of the search backend. It tries a direct request, then
browser-like headers, keyless Jina Reader, and Wayback by default. The ordered chain is
configured by RESEARCHER_FETCH_FALLBACKS; an empty value keeps only direct retrieval.
"""
from __future__ import annotations

import html
import ipaddress
import json
import os
import re
import urllib.parse
from html.parser import HTMLParser

from . import _http
from .base import SourceError, SourceTool

_DDG_LITE = "https://lite.duckduckgo.com/lite/"
_EXA_SEARCH = "https://api.exa.ai/search"
_JINA_READER = "https://r.jina.ai/"
_WAYBACK_AVAILABLE = "https://archive.org/wayback/available"
_FETCH_FALLBACKS_DEFAULT = "browserlike,jina,wayback"
_FETCH_FALLBACK_NAMES = frozenset({"browserlike", "jina", "wayback"})
_MANAGED_FALLBACK_NAMES = frozenset({"jina", "wayback"})
_SECRET_QUERY_FIELDS = frozenset({
    "token", "sig", "signature", "key", "expires", "auth", "session",
})
# A challenge stub is usually small and its markers live in the head or the start of the
# DOM. We look only for structural signs inside a bounded window: words from an ordinary
# article about captchas must not turn a source into an anti-bot page. Search and fetch
# have different signatures.
_ANTIBOT_SCAN_BYTES = 64 * 1024
_ANTIBOT_STUB_MAX_BYTES = 256 * 1024
_FETCH_BLOCKED = re.compile(
    r"<title\b[^>]*>[^<]{0,160}captcha|"
    r"<(?:form|div)\b[^>]*(?:id|class)=[\"'][^\"']*(?:challenge-form|cf[-_]chl)[^\"']*[\"']|"
    r"<(?:script|iframe)\b[^>]*(?:src|href)=[\"'][^\"']*/cdn-cgi/challenge-platform/|"
    r"\bclass=[\"'][^\"']*\b(?:g-recaptcha|h-captcha)\b[^\"']*[\"']|"
    r"\bdata-sitekey\s*=",
    re.I | re.S,
)
_DDG_BLOCKED = re.compile(
    r"<(?:div|form)\b[^>]*(?:id|class)=[\"'][^\"']*(?:anomaly-modal|challenge-form)[^\"']*[\"']|"
    r"<title\b[^>]*>[^<]{0,160}captcha|"
    r"\bclass=[\"'][^\"']*\b(?:g-recaptcha|h-captcha)\b[^\"']*[\"']|"
    r"\bdata-sitekey\s*=",
    re.I | re.S,
)
_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/127.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}
_FETCH_LIMIT = 40000  # characters per page: the collector needs text, not the whole DOM
# The READ ceiling for a response, in bytes. _FETCH_LIMIT trims already extracted text - by
# then the whole page is in memory, and nobody on the web promised us a response size. 3 MB
# of raw HTML covers any meaningful article with room to spare (40k characters of text is
# about 1-2% of such a DOM).
_FETCH_BYTES = 3_000_000
_SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "nav", "footer", "header",
              "aside", "form", "iframe", "template"}
_BLOCK_TAGS = {"p", "div", "section", "article", "br", "tr", "table", "ul", "ol", "blockquote",
               "pre", "figure", "main"}
_HEADINGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}


class _FetchAttemptFailed(SourceError):
    """A transport or anti-bot failure after which the next stage may be tried."""


class Web(SourceTool):
    name = "web"
    kind = "web"
    desc = ("keyless web search and page fetch (DuckDuckGo lite; your own SearXNG or paid "
            "Exa as a config option)")

    def __init__(self, backend: str | None = None):
        self.backend = (backend or os.environ.get("RESEARCHER_WEB_BACKEND") or
                        ("searxng" if os.environ.get("RESEARCHER_SEARXNG_URL") else "ddg")).strip()

    def search(self, query: str, *, limit: int = 8) -> list[dict]:
        q = (query or "").strip()
        if not q:
            raise SourceError("empty search query")
        limit = max(1, min(limit, 25))
        fn = {"ddg": _ddg_search, "searxng": _searxng_search, "exa": _exa_search}.get(self.backend)
        if fn is None:
            raise SourceError(f"no web search backend {self.backend!r}; available: ddg, searxng, exa")
        return fn(q, limit)

    def fetch(self, url: str) -> str:
        u = (url or "").strip()
        if not u.lower().startswith(("http://", "https://")):
            raise SourceError(f"expected an http(s) url, got {url!r}")
        fallbacks = _fetch_fallbacks()
        attempts = [("direct", lambda: _fetch_page(u))]
        for name in fallbacks:
            if name in _MANAGED_FALLBACK_NAMES and not _managed_fallback_allowed(u):
                continue
            attempts.append((name, {
                "browserlike": lambda: _fetch_page(u, headers=_BROWSER_HEADERS),
                "jina": lambda: _fetch_jina(u),
                "wayback": lambda: _fetch_wayback(u),
            }[name]))

        failures: list[str] = []
        for name, attempt in attempts:
            try:
                return attempt()
            except _FetchAttemptFailed as e:
                failures.append(f"{name}: {e}")
        raise SourceError(f"could not fetch {u}; attempts: " + " | ".join(failures))


def _fetch_fallbacks() -> list[str]:
    raw = os.environ.get("RESEARCHER_FETCH_FALLBACKS")
    value = _FETCH_FALLBACKS_DEFAULT if raw is None else raw
    names = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in _FETCH_FALLBACK_NAMES]
    if unknown:
        raise SourceError(
            "RESEARCHER_FETCH_FALLBACKS: unknown stages "
            f"{', '.join(unknown)}; available: browserlike,jina,wayback")
    return names


def _managed_fallback_allowed(url: str) -> bool:
    """Whether the full URL may be disclosed to the external Jina/Wayback services.

    Parsing is fail-closed: an ambiguous URL, userinfo, a local address or a
    secret-looking query parameter name disables both managed stages.
    """
    try:
        parsed = urllib.parse.urlsplit(url)
        host = parsed.hostname
        if not host or parsed.username is not None or parsed.password is not None:
            return False
    except (TypeError, ValueError):
        return False
    lowered_host = host.rstrip(".").casefold()
    if lowered_host == "localhost" or lowered_host.endswith(".localhost"):
        return False
    try:
        address = ipaddress.ip_address(lowered_host)
    except ValueError:
        pass
    else:
        if address.is_private or address.is_loopback or address.is_link_local:
            return False
    if re.search(r"%(?![0-9A-Fa-f]{2})", parsed.query):
        return False
    try:
        fields = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    except (TypeError, ValueError):
        return False
    for name, _value in fields:
        parts = {
            part for part in re.split(r"[^a-z0-9]+", name.casefold()) if part
        }
        if name.casefold() in _SECRET_QUERY_FIELDS or parts & _SECRET_QUERY_FIELDS:
            return False
    return True


def _fetch_page(url: str, *, headers: dict | None = None) -> str:
    return _render_fetched(url, _fetch_bytes(url, headers=headers, limit=_FETCH_BYTES))


def _fetch_jina(url: str) -> str:
    headers = {"Accept": "text/markdown"}
    key = (os.environ.get("JINA_API_KEY") or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return _render_fetched(url, _fetch_bytes(
        f"{_JINA_READER}{url}", headers=headers, limit=_FETCH_BYTES))


def _fetch_wayback(url: str) -> str:
    available_url = f"{_WAYBACK_AVAILABLE}?" + urllib.parse.urlencode({"url": url})
    try:
        data = _http.get_json(available_url)
    except SourceError as e:
        raise _FetchAttemptFailed(str(e)) from e
    archived = data.get("archived_snapshots") if isinstance(data, dict) else None
    closest = archived.get("closest") if isinstance(archived, dict) else None
    if not isinstance(closest, dict) or not closest.get("available"):
        raise _FetchAttemptFailed(f"Wayback found no available snapshot for {url}")
    snapshot_url = str(closest.get("url") or "").strip()
    timestamp = str(closest.get("timestamp") or "").strip()
    if not snapshot_url.lower().startswith(("http://", "https://")) or not timestamp:
        raise _FetchAttemptFailed(f"Wayback returned an incomplete snapshot for {url}")
    body = _render_fetched(url, _fetch_bytes(snapshot_url, limit=_FETCH_BYTES))
    return f"[wayback {timestamp}]\n{body}"


def _fetch_bytes(url: str, **kw) -> bytes:
    try:
        return _http.get_bytes(url, **kw)
    except SourceError as e:
        raise _FetchAttemptFailed(str(e)) from e


def _render_fetched(url: str, raw: bytes) -> str:
    if raw[:5] == b"%PDF-":
        raise SourceError(f"{url} is a PDF, this tool does not parse it (use arxiv for papers)")
    cut = len(raw) > _FETCH_BYTES  # read(limit+1) returned an extra byte - the page is larger
    text = raw[:_FETCH_BYTES].decode("utf-8", "replace")
    blocked = _blocked_marker(raw, _FETCH_BLOCKED)
    if blocked:
        marker = " ".join(blocked.group(0).split())[:80]
        raise _FetchAttemptFailed(f"{url} returned an anti-bot page (marker {marker!r})")
    body = _html_to_markdown(text) if _looks_html(text) else text.strip()
    if not body.strip():
        raise SourceError(f"no text extracted from {url} (empty after stripping markup)")
    if len(body) > _FETCH_LIMIT:
        body = body[:_FETCH_LIMIT] + "\n\n[...truncated at 40000 characters...]"
    elif cut:
        # The text fit, but the page itself was larger than the read ceiling - the collector
        # must know it is not seeing everything (silent truncation is a quiet lie about how
        # complete the source is).
        body += f"\n\n[...page truncated at {_FETCH_BYTES} bytes on read...]"
    return f"# {url}\n\n{body}\n"


# --- search backends --------------------------------------------------------

def _ddg_search(query: str, limit: int) -> list[dict]:
    """DuckDuckGo lite via a POST form. kl=wt-wt means no regional bias in the results."""
    data = urllib.parse.urlencode({"q": query, "kl": "wt-wt"}).encode()
    raw = _http.get_bytes(_DDG_LITE, data=data)
    page = raw.decode("utf-8", "replace")
    if _blocked_marker(raw, _DDG_BLOCKED):
        raise SourceError(
            "DuckDuckGo returned an anti-bot challenge instead of results (a common per-IP "
            "limit). Retry later, or point RESEARCHER_SEARXNG_URL at your own instance, or "
            "search with the structured tools (hn/arxiv/github)")
    p = _DdgLiteParser()
    p.feed(page)
    return p.results[:limit]


def _blocked_marker(raw: bytes, pattern: re.Pattern[str]):
    """The marker of a small challenge stub within the first 64 KiB of the response."""
    if len(raw) > _ANTIBOT_STUB_MAX_BYTES:
        return None
    head = raw[:_ANTIBOT_SCAN_BYTES].decode("utf-8", "replace")
    return pattern.search(head)


def _searxng_search(query: str, limit: int) -> list[dict]:
    """Your own SearXNG instance: the JSON API (format=json must be enabled in its settings.yml).

    Almost every public instance blocks both json and non-browser requests (live check:
    429/captcha on 12 of 15), which is why the URL comes from configuration and there is
    no list of public instances here.
    """
    base = (os.environ.get("RESEARCHER_SEARXNG_URL") or "").strip().rstrip("/")
    if not base:
        raise SourceError("the searxng backend requires RESEARCHER_SEARXNG_URL (the address of your own instance)")
    url = f"{base}/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "safesearch": "0"})
    data = _http.get_json(url)
    out = []
    for r in (data.get("results") or [])[:limit]:
        u = (r.get("url") or "").strip()
        if not u:
            continue
        out.append({"url": u, "title": (r.get("title") or u).strip(),
                    "snippet": " ".join((r.get("content") or "").split())})
    return out


def _exa_search(query: str, limit: int) -> list[dict]:
    """A paid backend - an OPTION behind the same interface (never the default, key from env only).

    NOT verified live (there is no key): the request form is taken from the Exa
    documentation (POST /search, an x-api-key header, body {query, numResults}). Whoever
    turns EXA_API_KEY on first must check the response against docs.exa.ai - unlike the ddg
    backend this is unverified code (the repo rule: a live smoke test or an honest note).
    """
    key = (os.environ.get("EXA_API_KEY") or "").strip()
    if not key:
        raise SourceError("the exa backend requires EXA_API_KEY (paid; the default stays keyless)")
    body = json.dumps({"query": query, "numResults": limit}).encode()
    data = _http.get_json(_EXA_SEARCH, data=body,
                          headers={"x-api-key": key, "Content-Type": "application/json"})
    out = []
    for r in (data.get("results") or [])[:limit]:
        u = (r.get("url") or "").strip()
        if not u:
            continue
        snippet = r.get("summary") or r.get("text") or ""
        out.append({"url": u, "title": (r.get("title") or u).strip(),
                    "snippet": " ".join(str(snippet).split())[:400]})
    return out


# --- parsing DDG lite results -----------------------------------------------

class _DdgLiteParser(HTMLParser):
    """<a class=result-link href=URL>title</a> + <td class=result-snippet>snippet</td>.

    Ads (links to duckduckgo.com/y.js) are dropped, and the redirect wrapper /l/?uddg=<url>
    (returned by the GET flavor of the results page) is unwrapped into the real url.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._in_title = False
        self._in_snippet = False
        self._buf: list[str] = []
        self._href = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        if tag == "a" and "result-link" in cls:
            self._in_title, self._buf, self._href = True, [], a.get("href") or ""
        elif tag in ("td", "div") and "result-snippet" in cls:
            self._in_snippet, self._buf = True, []

    def handle_endtag(self, tag):
        if self._in_title and tag == "a":
            self._in_title = False
            raw, url = self._href, _unwrap(self._href)
            title = " ".join("".join(self._buf).split())
            if url and not (_is_ad(raw) or _is_ad(url)):
                self.results.append({"url": url, "title": title or url, "snippet": ""})
        elif self._in_snippet and tag in ("td", "div"):
            self._in_snippet = False
            if self.results and not self.results[-1]["snippet"]:
                self.results[-1]["snippet"] = " ".join("".join(self._buf).split())

    def handle_data(self, data):
        if self._in_title or self._in_snippet:
            self._buf.append(data)


def _is_ad(url: str) -> bool:
    return "duckduckgo.com/y.js" in url or url.startswith("//duckduckgo.com/y.js")


def _unwrap(href: str) -> str:
    """//duckduckgo.com/l/?uddg=<percent-encoded url> -> the real url."""
    h = (href or "").strip()
    if not h:
        return ""
    if h.startswith("//"):
        h = "https:" + h
    if "uddg=" in h:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(h).query)
        if qs.get("uddg"):
            return qs["uddg"][0]
    return h


# --- html -> markdown (a replacement for the native WebFetch) ---------------

def _looks_html(text: str) -> bool:
    head = text[:4000].lower()
    return "<html" in head or "<!doctype html" in head or "<body" in head or "<div" in head


class _TextParser(HTMLParser):
    """Crude but predictable text extraction: headings, paragraphs and lists, no scripts or menus."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            if tag == "head":
                return  # <title> inside head is needed; the rest of the head is not text anyway
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "title":
            self._in_title = True
        elif tag in _HEADINGS:
            self.out.append(f"\n\n{_HEADINGS[tag]} ")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in _BLOCK_TAGS:
            self.out.append("\n\n" if tag != "br" else "\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and tag != "head":
            self._skip = max(0, self._skip - 1)
            return
        if tag == "title":
            self._in_title = False
        elif tag in _HEADINGS or tag in _BLOCK_TAGS:
            self.out.append("\n\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title += data
            return
        if data.strip():
            self.out.append(re.sub(r"\s+", " ", data))


def _html_to_markdown(text: str) -> str:
    p = _TextParser()
    p.feed(text)
    body = "".join(p.out)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    lines = [ln.rstrip() for ln in body.splitlines()]
    body = "\n".join(lines).strip()
    title = " ".join(html.unescape(p.title).split())
    return f"## {title}\n\n{body}" if title else body
