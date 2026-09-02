"""Tests for the source tools (R2): NO network at all - every HTTP call is mocked through
the _http._urlopen seam.

We mock at the level of the single real network call (_urlopen) rather than at the requests
level: that way both response parsing and our retry/error classification in _http are covered.
"""
import io
import json
import sys
import urllib.error
from base64 import b64encode
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import researcher.sources.__main__ as sources_main  # noqa: E402
from researcher.sources import _http, cmd_prefix, describe, get_tool, tool_names  # noqa: E402
from researcher.sources.base import SourceError  # noqa: E402
from researcher.sources.web import Web  # noqa: E402


class _Resp:
    """A mini response instead of http.client.HTTPResponse: read([n])+close().

    read(n) returns NO MORE than n bytes, like a real HTTPResponse; the read ceiling of
    _http.get_bytes(limit=...) rests on that: without the argument the whole page is pulled.
    """
    def __init__(self, data: bytes):
        self._data = data

    def read(self, n: int | None = None):
        return self._data if n is None else self._data[:n]

    def close(self):
        pass


def _mock_http(monkeypatch, routes: dict, *, seen: list | None = None):
    """routes: {url_substring: bytes|str}. Installs a fake _urlopen and silences sleep (retries).
    seen (when passed) accumulates the Request objects themselves - that is how the method,
    body and headers are checked."""
    def fake_urlopen(req, timeout):
        if seen is not None:
            seen.append(req)
        for key, val in routes.items():
            if key in req.full_url:
                return _Resp(val if isinstance(val, bytes) else val.encode())
        raise AssertionError(f"no mock for {req.full_url}")
    monkeypatch.setattr(_http, "_urlopen", fake_urlopen)
    monkeypatch.setattr(_http.time, "sleep", lambda *_: None)


# --- response fixtures ------------------------------------------------------

_HN_SEARCH = json.dumps({"hits": [
    {"objectID": "111", "title": "Rust async runtime", "url": "https://ex.com/a",
     "points": 42, "num_comments": 10, "author": "alice"},
    {"objectID": "222", "story_title": "Tokio internals", "points": 5, "num_comments": 2},
]})
_HN_ITEM = json.dumps({
    "title": "Rust async runtime", "url": "https://ex.com/a", "author": "alice", "points": 42,
    "text": "<p>Opening post</p>",
    "children": [{"author": "bob", "text": "<p>A comment with a <a href=\"x\">link</a></p>",
                  "children": [{"author": "carol", "text": "reply", "children": []}]}],
})
_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.01234v1</id>
    <title>Retrieval Augmented Generation</title>
    <summary>We propose a RAG method for language models.</summary>
    <published>2024-01-05T00:00:00Z</published>
    <author><name>A. Author</name></author>
    <author><name>B. Author</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2401.01234v1"/>
  </entry>
</feed>"""
_GH_SEARCH = json.dumps({"items": [
    {"full_name": "owner/repo", "html_url": "https://github.com/owner/repo",
     "description": "deep research agent", "stargazers_count": 1234, "language": "Python"},
]})
_GH_README = json.dumps({"encoding": "base64",
                         "content": b64encode("# Repo\n\nreadme body".encode()).decode()})


# --- _http: retries and honest errors ---------------------------------------

def test_http_retry_then_success(monkeypatch):
    calls = {"n": 0}

    def fake(req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("temp fail")  # transient -> retry
        return _Resp(b'{"ok": true}')
    monkeypatch.setattr(_http, "_urlopen", fake)
    monkeypatch.setattr(_http.time, "sleep", lambda *_: None)
    assert _http.get_json("http://x") == {"ok": True}
    assert calls["n"] == 2


def test_http_permanent_404_is_source_error(monkeypatch):
    def fake(req, timeout):
        raise urllib.error.HTTPError("http://x", 404, "nope", {}, io.BytesIO(b"nope"))
    monkeypatch.setattr(_http, "_urlopen", fake)
    with pytest.raises(SourceError):
        _http.get_bytes("http://x")


def test_http_transient_exhausted_is_source_error(monkeypatch):
    def fake(req, timeout):
        raise urllib.error.HTTPError("http://x", 503, "busy", {}, io.BytesIO(b"busy"))
    monkeypatch.setattr(_http, "_urlopen", fake)
    monkeypatch.setattr(_http.time, "sleep", lambda *_: None)
    with pytest.raises(SourceError):
        _http.get_bytes("http://x", retries=2)


# --- HN ---------------------------------------------------------------------

def test_hn_search(monkeypatch):
    _mock_http(monkeypatch, {"hn.algolia.com/api/v1/search": _HN_SEARCH})
    res = get_tool("hn").search("rust async")
    assert len(res) == 2
    assert res[0]["url"] == "https://news.ycombinator.com/item?id=111"
    assert res[0]["title"] == "Rust async runtime"
    assert "42 points" in res[0]["snippet"] and "by alice" in res[0]["snippet"]
    assert res[1]["title"] == "Tokio internals"  # the story_title fallback


def test_hn_fetch_renders_thread(monkeypatch):
    _mock_http(monkeypatch, {"hn.algolia.com/api/v1/items/111": _HN_ITEM})
    md = get_tool("hn").fetch("https://news.ycombinator.com/item?id=111")
    assert md.startswith("# Rust async runtime")
    assert "Opening post" in md
    assert "**bob**" in md and "link" in md and "<a" not in md  # html stripped
    assert "carol" in md  # a nested comment


# --- arXiv ------------------------------------------------------------------

def test_arxiv_search(monkeypatch):
    _mock_http(monkeypatch, {"export.arxiv.org/api/query": _ARXIV_XML})
    res = get_tool("arxiv").search("rag")
    assert res[0]["url"] == "http://arxiv.org/abs/2401.01234v1"
    assert res[0]["title"] == "Retrieval Augmented Generation"
    assert "A. Author" in res[0]["snippet"] and "2024-01-05" in res[0]["snippet"]


def test_arxiv_fetch(monkeypatch):
    _mock_http(monkeypatch, {"export.arxiv.org/api/query": _ARXIV_XML})
    md = get_tool("arxiv").fetch("https://arxiv.org/abs/2401.01234")
    assert "# Retrieval Augmented Generation" in md
    assert "## Abstract" in md and "RAG" in md
    assert "A. Author, B. Author" in md
    assert "PDF: http://arxiv.org/pdf/2401.01234v1" in md


def test_arxiv_rejects_doctype(monkeypatch):
    bomb = '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY a "x">]><feed></feed>'
    _mock_http(monkeypatch, {"export.arxiv.org/api/query": bomb})
    with pytest.raises(SourceError):
        get_tool("arxiv").search("x")


# --- GitHub -----------------------------------------------------------------

def test_github_search(monkeypatch):
    _mock_http(monkeypatch, {"api.github.com/search/repositories": _GH_SEARCH})
    res = get_tool("github").search("deep research")
    assert res[0]["url"] == "https://github.com/owner/repo"
    assert res[0]["title"] == "owner/repo"
    assert "1234*" in res[0]["snippet"] and "Python" in res[0]["snippet"]


def test_github_fetch_readme(monkeypatch):
    _mock_http(monkeypatch, {"/repos/owner/repo/readme": _GH_README})
    md = get_tool("github").fetch("https://github.com/owner/repo")
    assert "owner/repo README" in md and "readme body" in md


def test_github_fetch_bad_url_is_source_error():
    with pytest.raises(SourceError):
        get_tool("github").fetch("not a url at all")


# --- web: keyless search and fetch (tasks item 1) ---------------------------

# The lite.duckduckgo.com/lite/ markup is a cut-down live response from 2026-08-10 plus two
# cases absent from it but present at DDG: an ad link (y.js) and the redirect wrapper
# /l/?uddg= (returned by the GET flavor of the results page).
_DDG_LITE = """<html><body><table>
  <tr><td><a rel="nofollow" href="//duckduckgo.com/y.js?ad_provider=x" class='result-link'>Ad</a></td></tr>
  <tr><td class='result-snippet'>ad copy</td></tr>
  <tr><td><a rel="nofollow" href="https://tokio.rs/" class='result-link'>Tokio</a></td></tr>
  <tr><td class='result-snippet'>An <b>async</b> runtime for   Rust</td></tr>
  <tr><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fsmol.rs%2F&amp;rut=1" class='result-link'>Smol</a></td></tr>
  <tr><td class='result-snippet'>a small runtime</td></tr>
</table></body></html>"""
_DDG_BLOCKED = """<html><head><title>DuckDuckGo</title></head><body>
  <div class="anomaly-modal__mask"><form id="challenge-form"></form></div></body></html>"""
_SEARX_JSON = json.dumps({"results": [
    {"url": "https://tokio.rs/", "title": "Tokio", "content": "async  runtime\nfor Rust"},
    {"url": "", "title": "broken", "content": "no url - dropped"},
]})
_EXA_JSON = json.dumps({"results": [
    {"url": "https://tokio.rs/", "title": "Tokio", "text": "async runtime"},
]})
_PAGE_HTML = """<html><head><title>  Page  title </title>
  <style>.x{color:red}</style></head>
  <body><nav>the menu is not wanted here</nav>
  <h1>The main thing</h1><p>First paragraph with a <a href="/x">link</a>.</p>
  <script>alert('no')</script>
  <ul><li>item one</li><li>item two</li></ul>
  <footer>the footer is not needed</footer></body></html>"""


def test_web_ddg_search_post_form(monkeypatch):
    seen = []
    _mock_http(monkeypatch, {"lite.duckduckgo.com": _DDG_LITE}, seen=seen)
    res = Web(backend="ddg").search("rust async runtime", limit=5)
    # a POST form rather than GET: a live GET on lite/ runs into the anti-bot page (checked 2026-08-10)
    assert seen[0].get_method() == "POST" and b"q=rust+async+runtime" in seen[0].data
    # the ad (y.js) is dropped, the uddg wrapper is unwrapped, the snippet is whitespace-collapsed
    assert [r["url"] for r in res] == ["https://tokio.rs/", "https://smol.rs/"]
    assert res[0]["title"] == "Tokio" and res[0]["snippet"] == "An async runtime for Rust"


def test_web_ddg_limit(monkeypatch):
    _mock_http(monkeypatch, {"lite.duckduckgo.com": _DDG_LITE})
    assert len(Web(backend="ddg").search("x", limit=1)) == 1


def test_web_ddg_antibot_page_is_error_not_empty(monkeypatch):
    """An anti-bot page arrives with HTTP 200/202, which _http treats as 'success'. An empty
    list here would be a lie ('nothing was found' instead of 'we were not let in')."""
    _mock_http(monkeypatch, {"lite.duckduckgo.com": _DDG_BLOCKED})
    with pytest.raises(SourceError, match="anti-bot"):
        Web(backend="ddg").search("rust")


def test_web_ddg_result_about_recaptcha_is_not_an_antibot_page(monkeypatch):
    page = """<html><head><title>DuckDuckGo</title></head><body>
      <a class='result-link' href='https://example.com/recaptcha'>reCAPTCHA internals</a>
      <div class='result-snippet'>How captcha challenges work</div>
    </body></html>"""
    _mock_http(monkeypatch, {"lite.duckduckgo.com": page})
    assert Web(backend="ddg").search("recaptcha") == [{
        "url": "https://example.com/recaptcha",
        "title": "reCAPTCHA internals",
        "snippet": "How captcha challenges work",
    }]


def test_web_empty_query_is_error():
    with pytest.raises(SourceError):
        Web(backend="ddg").search("   ")


def test_web_searxng_backend(monkeypatch):
    monkeypatch.setenv("RESEARCHER_SEARXNG_URL", "https://searx.example.org/")
    seen = []
    _mock_http(monkeypatch, {"searx.example.org": _SEARX_JSON}, seen=seen)
    res = Web(backend="searxng").search("rust")
    assert "format=json" in seen[0].full_url and seen[0].full_url.startswith(
        "https://searx.example.org/search?")
    assert [r["url"] for r in res] == ["https://tokio.rs/"]  # the entry without a url is dropped
    assert res[0]["snippet"] == "async runtime for Rust"


def test_web_searxng_without_url_is_error(monkeypatch):
    monkeypatch.delenv("RESEARCHER_SEARXNG_URL", raising=False)
    with pytest.raises(SourceError, match="RESEARCHER_SEARXNG_URL"):
        Web(backend="searxng").search("rust")


def test_web_exa_is_opt_in_paid_backend(monkeypatch):
    """A paid backend is an option behind the SAME interface: an honest error without a key,
    the same [{url,title,snippet}] contract with one. Not verified live (no key) - see web.py."""
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    with pytest.raises(SourceError, match="EXA_API_KEY"):
        Web(backend="exa").search("rust")
    monkeypatch.setenv("EXA_API_KEY", "k-secret")
    seen = []
    _mock_http(monkeypatch, {"api.exa.ai": _EXA_JSON}, seen=seen)
    res = Web(backend="exa").search("rust", limit=3)
    assert seen[0].get_method() == "POST"
    assert json.loads(seen[0].data) == {"query": "rust", "numResults": 3}
    assert seen[0].get_header("X-api-key") == "k-secret"  # the key comes from env only, never from code
    assert res == [{"url": "https://tokio.rs/", "title": "Tokio", "snippet": "async runtime"}]


def test_web_backend_default_is_keyless(monkeypatch):
    """The default is keyless (the AGENTS rule): ddg; your own SearXNG is picked up from env;
    the paid exa only through an explicit RESEARCHER_WEB_BACKEND."""
    monkeypatch.delenv("RESEARCHER_WEB_BACKEND", raising=False)
    monkeypatch.delenv("RESEARCHER_SEARXNG_URL", raising=False)
    monkeypatch.setenv("EXA_API_KEY", "k")  # a key exists, but on its own it does not enable the paid path
    assert Web().backend == "ddg"
    monkeypatch.setenv("RESEARCHER_SEARXNG_URL", "https://searx.example.org")
    assert Web().backend == "searxng"
    monkeypatch.setenv("RESEARCHER_WEB_BACKEND", "exa")
    assert Web().backend == "exa"


def test_web_unknown_backend_is_error(monkeypatch):
    with pytest.raises(SourceError, match="no web search backend"):
        Web(backend="google").search("rust")


def test_web_fetch_html_to_markdown(monkeypatch):
    _mock_http(monkeypatch, {"example.com": _PAGE_HTML})
    md = get_tool("web").fetch("https://example.com/a")
    assert md.startswith("# https://example.com/a")
    assert "## Page title" in md                   # <title> as a subheading
    assert "# The main thing" in md                # h1 -> a markdown heading
    assert "First paragraph with a link." in md    # the link text stays in the flow
    assert "- item one" in md and "- item two" in md
    assert "alert" not in md and "color:red" not in md   # script/style are cut out
    assert "the menu is not wanted here" not in md and "the footer is not needed" not in md  # nav/footer too


def test_web_fetch_rejects_pdf_and_nonhttp(monkeypatch):
    _mock_http(monkeypatch, {"example.com": b"%PDF-1.7 ..."})
    with pytest.raises(SourceError, match="PDF"):
        get_tool("web").fetch("https://example.com/a.pdf")
    with pytest.raises(SourceError, match="http"):
        get_tool("web").fetch("file.txt")


def test_web_fetch_plain_text_passthrough(monkeypatch):
    _mock_http(monkeypatch, {"example.com": "just text without markup"})
    assert "just text without markup" in get_tool("web").fetch("https://example.com/a.txt")


def test_web_fetch_truncates_huge_page(monkeypatch):
    _mock_http(monkeypatch, {"example.com": "<html><body><p>" + "a" * 60000 + "</p></body></html>"})
    md = get_tool("web").fetch("https://example.com/big")
    assert "truncated" in md and len(md) < 41000


def test_web_fetch_limits_read_itself_not_only_output(monkeypatch):
    """The ceiling is applied while READING the response, not afterwards (codex review, fix-5).

    Trimming after the fact does not help: by then a page of arbitrary size is already fully
    in the process memory. We capture the read() argument itself: it must be the ceiling, not None.
    """
    from researcher.sources import web as web_mod
    huge = b"<html><body><p>" + b"a" * 5_000_000 + b"</p></body></html>"
    asked: list = []

    class _SpyResp(_Resp):
        def read(self, n=None):
            asked.append(n)
            return super().read(n)

    monkeypatch.setattr(_http, "_urlopen", lambda req, timeout: _SpyResp(huge))
    md = get_tool("web").fetch("https://example.com/huge")
    assert asked == [web_mod._FETCH_BYTES + 1]  # the +1 byte is the "page exceeds the ceiling" marker
    assert "truncated" in md


def test_web_fetch_direct_success_stops_fallback_chain(monkeypatch):
    calls = []

    def get_bytes(url, **kw):
        calls.append((url, kw))
        return b"direct body"

    monkeypatch.setattr(_http, "get_bytes", get_bytes)
    assert "direct body" in Web().fetch("https://example.com/article")
    assert calls == [("https://example.com/article",
                      {"headers": None, "limit": 3_000_000})]


def test_web_fetch_browserlike_after_blocked_direct(monkeypatch):
    monkeypatch.setenv("RESEARCHER_FETCH_FALLBACKS", "browserlike,jina")
    calls = []

    def get_bytes(url, **kw):
        calls.append((url, kw))
        if kw.get("headers") is None:
            return b"<html><div class='cf-chl'>challenge</div></html>"
        return b"<html><body><p>browser success</p></body></html>"

    monkeypatch.setattr(_http, "get_bytes", get_bytes)
    md = Web().fetch("https://example.com/article")
    assert "browser success" in md
    assert len(calls) == 2
    headers = calls[1][1]["headers"]
    assert headers["Accept-Language"] == "en-US,en;q=0.9"
    assert {name for name in headers if name.startswith("Sec-Fetch-")} == {
        "Sec-Fetch-Dest", "Sec-Fetch-Mode", "Sec-Fetch-Site", "Sec-Fetch-User"}


def test_web_fetch_jina_after_direct_failure_and_sends_optional_key(monkeypatch):
    monkeypatch.setenv("RESEARCHER_FETCH_FALLBACKS", "jina,wayback")
    monkeypatch.setenv("JINA_API_KEY", "jina-secret")
    calls = []

    def get_bytes(url, **kw):
        calls.append((url, kw))
        if url == "https://example.com/article":
            raise SourceError("HTTP 403 direct")
        assert url == "https://r.jina.ai/https://example.com/article"
        return b"# Reader markdown\n\nJina success"

    monkeypatch.setattr(_http, "get_bytes", get_bytes)
    md = Web().fetch("https://example.com/article")
    assert "Jina success" in md
    assert len(calls) == 2
    assert calls[1][1]["headers"] == {
        "Accept": "text/markdown", "Authorization": "Bearer jina-secret"}


def test_web_fetch_jina_without_key_is_still_keyless(monkeypatch):
    monkeypatch.setenv("RESEARCHER_FETCH_FALLBACKS", "jina")
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    calls = []

    def get_bytes(url, **kw):
        calls.append((url, kw))
        if url.startswith("https://r.jina.ai/"):
            return b"keyless jina"
        raise SourceError("HTTP 403 direct")

    monkeypatch.setattr(_http, "get_bytes", get_bytes)
    assert "keyless jina" in Web().fetch("https://example.com/article")
    assert calls[1][1]["headers"] == {"Accept": "text/markdown"}


def test_web_fetch_wayback_marks_snapshot_on_first_line(monkeypatch):
    monkeypatch.setenv("RESEARCHER_FETCH_FALLBACKS", "wayback")
    calls = []

    def get_bytes(url, **kw):
        calls.append(url)
        if url == "https://example.com/article":
            raise SourceError("HTTP 403 direct")
        assert url == "https://web.archive.org/web/20240102030405/https://example.com/article"
        return b"<html><body><p>archived body</p></body></html>"

    def get_json(url, **kw):
        assert url == ("https://archive.org/wayback/available?"
                       "url=https%3A%2F%2Fexample.com%2Farticle")
        return {"archived_snapshots": {"closest": {
            "available": True,
            "url": "https://web.archive.org/web/20240102030405/https://example.com/article",
            "timestamp": "20240102030405",
        }}}

    monkeypatch.setattr(_http, "get_bytes", get_bytes)
    monkeypatch.setattr(_http, "get_json", get_json)
    md = Web().fetch("https://example.com/article")
    assert md.splitlines()[0] == "[wayback 20240102030405]"
    assert "archived body" in md
    assert calls == ["https://example.com/article",
                     "https://web.archive.org/web/20240102030405/https://example.com/article"]


def test_web_fetch_full_failure_lists_every_attempt_and_reason(monkeypatch):
    monkeypatch.setenv("RESEARCHER_FETCH_FALLBACKS", "browserlike,jina,wayback")

    def get_bytes(url, **kw):
        if url.startswith("https://r.jina.ai/"):
            raise SourceError("HTTP 503 reader")
        if kw.get("headers"):
            raise SourceError("HTTP 403 browser")
        raise SourceError("HTTP 403 direct")

    monkeypatch.setattr(_http, "get_bytes", get_bytes)
    monkeypatch.setattr(_http, "get_json", lambda *a, **kw: {"archived_snapshots": {}})
    with pytest.raises(SourceError) as exc:
        Web().fetch("https://example.com/article")
    message = str(exc.value)
    assert all(name in message for name in ("direct:", "browserlike:", "jina:", "wayback:"))
    assert all(reason in message for reason in (
        "HTTP 403 direct", "HTTP 403 browser", "HTTP 503 reader", "found no"))


@pytest.mark.parametrize("page", [
    "<html><script src='/cdn-cgi/challenge-platform/x'></script></html>",
    "<html><title>Captcha verification</title></html>",
    "<html><div class='g-recaptcha' data-sitekey='x'></div></html>",
    "<html><form id='challenge-form'></form></html>",
    "<html><div class='cf-chl-widget'></div></html>",
])
def test_web_fetch_antibot_markers_are_failure_even_on_http_success(monkeypatch, page):
    monkeypatch.setenv("RESEARCHER_FETCH_FALLBACKS", "")
    monkeypatch.setattr(_http, "get_bytes", lambda *a, **kw: page.encode())
    with pytest.raises(SourceError, match="anti-bot"):
        Web().fetch("https://example.com/article")


def test_web_fetch_article_about_antibot_terms_is_not_blocked(monkeypatch):
    monkeypatch.setattr(
        _http, "get_bytes",
        lambda *a, **kw: (
            b"<html><head><title>Bot protection review</title></head><body><article>"
            b"This article compares reCAPTCHA and hCaptcha. It quotes Just a moment, "
            b"Attention Required, Cloudflare challenge, verify you are human, checking "
            b"your browser, g-recaptcha, data-sitekey, cf-chl, anomaly-modal and "
            b"challenge-form as prose.</article></body></html>"
        ),
    )
    body = Web().fetch("https://example.com/article")
    assert "compares reCAPTCHA and hCaptcha" in body
    assert "checking your browser" in body


def test_web_fetch_fallback_config_default_empty_order_and_validation(monkeypatch):
    from researcher.sources.web import _fetch_fallbacks

    monkeypatch.delenv("RESEARCHER_FETCH_FALLBACKS", raising=False)
    assert _fetch_fallbacks() == ["browserlike", "jina", "wayback"]
    monkeypatch.setenv("RESEARCHER_FETCH_FALLBACKS", "")
    assert _fetch_fallbacks() == []
    monkeypatch.setenv("RESEARCHER_FETCH_FALLBACKS", " wayback, jina ")
    assert _fetch_fallbacks() == ["wayback", "jina"]
    monkeypatch.setenv("RESEARCHER_FETCH_FALLBACKS", "browserlike,magic")
    with pytest.raises(SourceError, match="magic"):
        _fetch_fallbacks()


def test_web_fetch_respects_configured_order(monkeypatch):
    monkeypatch.setenv("RESEARCHER_FETCH_FALLBACKS", "jina,browserlike")
    calls = []

    def get_bytes(url, **kw):
        calls.append(url)
        if url.startswith("https://r.jina.ai/"):
            return b"jina won"
        raise SourceError("direct blocked")

    monkeypatch.setattr(_http, "get_bytes", get_bytes)
    assert "jina won" in Web().fetch("https://example.com/article")
    assert calls == ["https://example.com/article",
                     "https://r.jina.ai/https://example.com/article"]


@pytest.mark.parametrize("url", [
    "http://10.0.0.7/private",
    "http://127.0.0.1/loopback",
    "http://169.254.12.3/link-local",
    "http://[::1]/ipv6-loopback",
    "http://localhost/local-name",
])
def test_web_fetch_never_sends_local_hosts_to_managed_fallbacks(
    monkeypatch, url
):
    monkeypatch.setenv("RESEARCHER_FETCH_FALLBACKS", "jina,wayback")
    calls = []

    def get_bytes(request_url, **kw):
        calls.append(request_url)
        raise SourceError("direct failed")

    monkeypatch.setattr(_http, "get_bytes", get_bytes)
    monkeypatch.setattr(
        _http, "get_json", lambda *a, **kw: pytest.fail("Wayback received a private URL")
    )
    with pytest.raises(SourceError):
        Web().fetch(url)
    assert calls == [url]


def test_web_fetch_never_sends_userinfo_url_to_managed_fallbacks(monkeypatch):
    monkeypatch.setenv("RESEARCHER_FETCH_FALLBACKS", "jina,wayback")
    url = "https://alice:private@example.com/article"
    calls = []
    monkeypatch.setattr(
        _http, "get_bytes",
        lambda request_url, **kw: calls.append(request_url) or (_ for _ in ()).throw(
            SourceError("direct failed")
        ),
    )
    monkeypatch.setattr(
        _http, "get_json", lambda *a, **kw: pytest.fail("Wayback received a userinfo URL")
    )
    with pytest.raises(SourceError):
        Web().fetch(url)
    assert calls == [url]


@pytest.mark.parametrize(
    "field", ["token", "sig", "signature", "key", "expires", "auth", "session"]
)
def test_web_fetch_never_sends_secret_query_to_managed_fallbacks(
    monkeypatch, field
):
    monkeypatch.setenv("RESEARCHER_FETCH_FALLBACKS", "jina,wayback")
    url = f"https://example.com/article?x=1&{field.upper()}=private"
    calls = []
    monkeypatch.setattr(
        _http, "get_bytes",
        lambda request_url, **kw: calls.append(request_url) or (_ for _ in ()).throw(
            SourceError("direct failed")
        ),
    )
    monkeypatch.setattr(
        _http, "get_json", lambda *a, **kw: pytest.fail("Wayback received a signed URL")
    )
    with pytest.raises(SourceError):
        Web().fetch(url)
    assert calls == [url]


# --- the source CLI command prefix ------------------------------------------

def test_cmd_prefix_carries_pythonpath_of_repo():
    """The collector works inside the TOPIC DIRECTORY and usually cannot see the repo - without
    PYTHONPATH the command fails with ModuleNotFoundError (live check 2026-08-10)."""
    prefix = cmd_prefix()
    repo = str(Path(__file__).parent.parent)
    assert prefix.startswith(f"PYTHONPATH={repo} ")
    assert prefix.endswith(" -m researcher.sources")


def test_cmd_prefix_quotes_paths_with_spaces(monkeypatch):
    """A checkout or an interpreter in a directory with a space must not break the collector
    command (codex review, fix-4). We check by shell-lexing the result rather than against a
    reference string."""
    import shlex

    import researcher.sources as sources_pkg
    # a deliberately nonexistent path: .resolve() must not substitute a symlink (/tmp -> /private/tmp)
    fake = "/no such/my repo/tool-researcher/researcher/sources/__init__.py"
    monkeypatch.setattr(sources_pkg, "__file__", fake)
    monkeypatch.setattr(sources_pkg.sys, "executable", "/opt/py 3.13/bin/python3")
    parts = shlex.split(sources_pkg.cmd_prefix())
    assert parts == ["PYTHONPATH=/no such/my repo/tool-researcher",
                     "/opt/py 3.13/bin/python3", "-m", "researcher.sources"]


# --- registry ---------------------------------------------------------------

def test_registry_names_and_describe():
    assert set(tool_names()) == {"hn", "arxiv", "github", "web"}
    d = describe(["hn", "arxiv"])
    assert "hn:" in d and "arxiv:" in d and "github" not in d


def test_registry_unknown_tool():
    with pytest.raises(SourceError):
        get_tool("bogus")


# --- CLI --------------------------------------------------------------------

def test_cli_search_prints_json(monkeypatch, capsys):
    _mock_http(monkeypatch, {"hn.algolia.com/api/v1/search": _HN_SEARCH})
    assert sources_main.main(["hn", "search", "rust", "async"]) == 0  # a multi-word query without quotes
    out = json.loads(capsys.readouterr().out)
    assert out[0]["title"] == "Rust async runtime"


def test_cli_fetch_prints_markdown(monkeypatch, capsys):
    _mock_http(monkeypatch, {"api.github.com": _GH_README})
    assert sources_main.main(["github", "fetch", "https://github.com/owner/repo"]) == 0
    assert "readme body" in capsys.readouterr().out


def test_cli_source_error_exits_nonzero(monkeypatch, capsys):
    def boom(req, timeout):
        raise urllib.error.HTTPError("http://x", 500, "err", {}, io.BytesIO(b""))
    monkeypatch.setattr(_http, "_urlopen", boom)
    monkeypatch.setattr(_http.time, "sleep", lambda *_: None)
    assert sources_main.main(["arxiv", "search", "x"]) == 1
    assert "arxiv" in capsys.readouterr().err


def test_cli_unknown_tool_argparse_exit():
    with pytest.raises(SystemExit):
        sources_main.main(["bogus", "search", "x"])
