#!/usr/bin/env python3
"""Opt-in live probe of the direct and Jina stages of web fetch.

Without `--run` it touches no network and returns 77. Run by hand:
`python3 smoke/smoke_fetch_fallback.py --run [https://example.com/]`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from researcher.sources.base import SourceError  # noqa: E402
from researcher.sources.web import _fetch_jina, _fetch_page  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--run" not in args:
        print("SKIP: the live network is off; pass --run")
        return 77
    args.remove("--run")
    url = args[0] if args else "https://example.com/"
    rows = []
    for name, fetch in (("direct", _fetch_page), ("jina", _fetch_jina)):
        try:
            body = fetch(url)
        except SourceError as error:
            rows.append((name, "FAIL", f"SourceError: {error}"))
        else:
            rows.append((name, "OK", f"{len(body)} characters, {body.splitlines()[0]!r}"))
    print(f"URL: {url}")
    print("| Stage | Result | Fact |")
    print("|---|---|---|")
    for name, result, fact in rows:
        print(f"| {name} | {result} | {fact} |")
    if all(result == "OK" for _, result, _ in rows):
        return 0
    if all("network failed for" in fact for _, _, fact in rows):
        return 77
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
