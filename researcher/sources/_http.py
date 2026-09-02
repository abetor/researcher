"""HTTP helper using stdlib urllib, transient retries, and explicit failures.

_urlopen is the only real network call and the mock seam used by hermetic tests.
Timeouts, disconnects, 408, 425, 429, and 5xx responses use exponential backoff.
Other 4xx responses and malformed payloads raise SourceError in the source layer.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .base import SourceError

# An honest User-Agent: some APIs (GitHub) answer 403 without one; we do not pose as a browser.
_UA = "researcher-source-tool/0.1 (deep-research engine; stdlib urllib)"
_DEFAULT_HEADERS = {"User-Agent": _UA, "Accept-Encoding": "identity"}
# Status codes that a retry can cure (unlike 401/403/404, which are permanent and pointless to retry).
_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def _urlopen(req, timeout):
    """The real network call. A separate function is the seam for monkeypatching in tests."""
    return urllib.request.urlopen(req, timeout=timeout)


def get_bytes(url: str, *, headers: dict | None = None, timeout: float = 30.0,
              retries: int = 2, backoff: float = 0.5, data: bytes | None = None,
              limit: int | None = None) -> bytes:
    """Issue POST when ``data`` is provided and cap response reads at ``limit``.

    urllib supplies the form content type when the caller does not. POST is a query form
    here, not a mutation: DuckDuckGo Lite rejects GET with an anti-bot check but returns
    results for the equivalent form POST, so retrying it is as safe as retrying GET.

    ``limit`` caps bytes read rather than trimming after an unbounded read. One extra
    byte lets callers distinguish an exact fit from a truncated response.
    """
    h = dict(_DEFAULT_HEADERS)
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h, data=data)
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = _urlopen(req, timeout)
        except urllib.error.HTTPError as e:  # a subclass of URLError - must be caught first
            if e.code in _TRANSIENT_STATUS and attempt < retries:
                last = e
                time.sleep(backoff * (2 ** attempt))
                continue
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            raise SourceError(f"HTTP {e.code} on {url}: {e.reason} {body}".strip()) from e
        except (urllib.error.URLError, OSError) as e:  # broken link/timeout/DNS - transient
            if attempt < retries:
                last = e
                time.sleep(backoff * (2 ** attempt))
                continue
            raise SourceError(f"network failed for {url}: {e}") from e
        try:
            return resp.read(limit + 1) if limit is not None else resp.read()
        finally:
            closer = getattr(resp, "close", None)
            if callable(closer):
                closer()
    raise SourceError(f"retries exhausted for {url}: {last}")  # pragma: no cover


def get_json(url: str, **kw):
    raw = get_bytes(url, **kw)
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise SourceError(f"the response from {url} is not JSON: {e}") from e


def get_text(url: str, **kw) -> str:
    return get_bytes(url, **kw).decode("utf-8", "replace")
