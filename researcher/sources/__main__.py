"""Source CLI entry point: python3 -m researcher.sources <tool> search|fetch <arg...>

Search prints JSON records with url, title, and snippet; fetch prints Markdown.
SourceError goes to stderr with exit 1, so collectors see an explicit failure. This is
the fixed Bash command boundary used by orchestrator collectors.

Examples from the repository; sources.cmd_prefix() adds PYTHONPATH for collectors:
  python3 -m researcher.sources hn search "rust async runtime"
  python3 -m researcher.sources arxiv fetch "https://arxiv.org/abs/2401.01234"
  python3 -m researcher.sources github search "deep research agent" --limit 5
  python3 -m researcher.sources web search "deep research agent architecture"
  python3 -m researcher.sources web fetch "https://example.com/article"
"""
from __future__ import annotations

import argparse
import json
import sys

from . import get_tool, tool_names
from .base import SourceError


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m researcher.sources",
                                description="source tools: structured API sources")
    p.add_argument("tool", choices=tool_names())
    p.add_argument("op", choices=["search", "fetch"])
    p.add_argument("arg", nargs="+", help="search: the query (quotes optional); fetch: url")
    p.add_argument("--limit", type=int, default=8, help="max results for search")
    a = p.parse_args(argv)
    arg = " ".join(a.arg).strip()
    try:
        tool = get_tool(a.tool)
        if a.op == "search":
            print(json.dumps(tool.search(arg, limit=a.limit), ensure_ascii=False, indent=2))
        else:
            sys.stdout.write(tool.fetch(arg))
    except SourceError as e:
        print(f"source tool {a.tool} {a.op}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
