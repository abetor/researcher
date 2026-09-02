#!/usr/bin/env python3
"""Live probe of the keyless web tool and of the collector capability profile (tasks 1 and 2).

Run by hand: `python3 smoke/smoke_web_collect.py [--with-harness claude,codex]`.
Network access is required; the harness step (a real CLI call on a subscription) runs only
behind the flag because it burns quota. Repo convention: exit 77 = skip (no network, binary
or login), any other nonzero code = failure.

Steps:
1. live web search - a real search query through our tool.
2. live web fetch - the first link from the results turned into markdown.
3. anti-vacuum: the same tool with a deliberately dead backend MUST go red (SourceError),
   otherwise green results in steps 1-2 are worth nothing.
4. (--with-harness) a real collector in PARITY mode on each named harness: native web is
   off and the PRESCRIBED path to the web is our source tool (enforcement exists only in
   claude; in codex the sandbox has network access and parity there is prompt discipline).
   We check that the model returned claims with urls AND labelled sources[].tool as our
   tool itself. This also exercises the command built by the ADAPTER (in codex, opening the
   network for the sandbox) rather than one assembled by hand.

WHAT THIS STEP PROVES AND WHAT IT DOES NOT (important, it is easy to overrate):
- in claude parity is held by MECHANICS: native web is absent from the physical --tools set
  (the model simply does not have it), and other Bash commands are cut off by the closed
  permission mode;
- in codex there is NO enforcement: the sandbox has network access during the collection
  phase, and the source tool is merely PRESCRIBED by the prompt. sources[].tool is the
  model's self-report, not a trace of executed commands: it catches sloppiness ("found it
  some other way and honestly wrote that down"), but it does not prove that the model never
  went to the network around the tool. Our --output-format json does not return a command
  trace (that needs stream-json with tool_use events) - a deliberate limit of this probe.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from researcher import sources  # noqa: E402
from researcher.adapters import ADAPTERS  # noqa: E402
from researcher.orchestrator import _prompt_collect, _sources_block  # noqa: E402
from researcher.sources.base import SourceError  # noqa: E402

QUERY = "deep research agent architecture"
rows: list[tuple[str, str, str, str]] = []


def step(name: str, expect: str, fn) -> object:
    try:
        val, fact = fn()
        rows.append((name, expect, fact, "OK"))
        return val
    except Exception as e:  # noqa: BLE001 - this is a probe, any failure is a fact we want
        rows.append((name, expect, f"{type(e).__name__}: {e}", "FAIL"))
        return None


def _harnesses(argv: list[str]) -> list[str]:
    """--with-harness [claude,codex] (no value means claude)."""
    if "--with-harness" not in argv:
        return []
    i = argv.index("--with-harness")
    val = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("-") else "claude"
    return [n.strip() for n in val.split(",") if n.strip()]


def main() -> int:
    with_harness = _harnesses(sys.argv)
    web = sources.get_tool("web")
    print(f"web search backend: {web.backend}")

    res = step("1. web search", "a non-empty result list with http urls",
               lambda: (lambda r: (r, f"{len(r)} results, first: {r[0]['url']}"))(
                   web.search(QUERY, limit=5)))
    if not res:
        print(_table())
        return 77 if not _network_ok() else 1

    step("2. web fetch", "markdown of the first link, > 500 characters",
         lambda: (lambda md: (md, f"{len(md)} characters, starts with: {md.splitlines()[0][:60]!r}"))(
             web.fetch(res[0]["url"])))

    def dead():
        os.environ["RESEARCHER_SEARXNG_URL"] = "https://searxng.invalid-host-for-smoke.example"
        try:
            from researcher.sources.web import Web
            Web(backend="searxng").search(QUERY)
        except SourceError as e:
            return True, f"as expected, SourceError: {str(e)[:80]}"
        finally:
            os.environ.pop("RESEARCHER_SEARXNG_URL", None)
        raise AssertionError("the dead backend did not go red - a defect in the probe")

    step("3. anti-vacuum", "a dead backend -> SourceError", dead)

    if with_harness:
        for name in with_harness:
            step(f"4. collector {name}, parity", "JSON with claims and sources[].tool=web",
                 lambda n=name: _collector_live(n))
    else:
        rows.append(("4. collector, parity", "JSON with claims and sources[].tool=web",
                     "skipped (no --with-harness)", "SKIP"))

    print(_table())
    return 1 if any(r[3] == "FAIL" for r in rows) else 0


def _collector_live(harness: str):
    ad = ADAPTERS[harness]()
    prefix = sources.cmd_prefix()
    profile = ad.collect_profile(source_cmd_prefix=prefix, parity=True)
    block = _sources_block(["web"], profile, prefix)
    prompt = _prompt_collect("deep research agent architecture", QUERY, block)
    print(f"--- {harness}: collector tool set / allowlist ---")
    print(f"tools (physical):  {profile.tools}")
    print(f"allowed_tools:     {profile.allowed_tools}")
    print(f"--- {harness}: adapter command ---")
    print(" ".join(ad.build_cmd("<prompt>", model="haiku", tools=profile.tools,
                                allowed_tools=profile.allowed_tools,
                                network=profile.network)))
    res = ad.run(prompt, cwd=os.getcwd(), timeout=600, model="haiku", tools=profile.tools,
                 allowed_tools=profile.allowed_tools, network=profile.network)
    print("--- harness answer (first 1500 characters) ---")
    print(res.text[:1500])
    if res.stop != "done":
        raise AssertionError(f"stop={res.stop}: {res.stderr[:200]}")
    data = json.loads(res.text[res.text.find("{"):res.text.rfind("}") + 1])
    claims = data.get("claims") or []
    urls = {e.get("url") for c in claims for e in (c.get("evidence") or [])}
    if not claims or not urls:
        raise AssertionError(f"the collector returned nothing: {res.text[:200]}")
    # What the model itself reports the sources were obtained with. Not a command trace (the
    # json output does not give one), but an empty or foreign tool here is already a signal:
    # the search went through something other than what was prescribed.
    tools_used = sorted({(s.get("tool") or "?") for s in (data.get("sources") or [])})
    if tools_used != ["web"]:
        raise AssertionError(f"sources[].tool = {tools_used}, expected only ['web'] "
                             f"(in parity web is the prescribed path to the network; in "
                             f"claude it is also the only physical one, in codex there is "
                             f"no enforcement)")
    return data, (f"{len(claims)} claims, {len(urls)} urls, sources[].tool={tools_used}, "
                  f"stop={res.stop}")


def _network_ok() -> bool:
    try:
        sources.get_tool("hn").search("test", limit=1)
        return True
    except SourceError:
        return False


def _table() -> str:
    out = ["", "| Step | Expected | Fact | Result |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    out += ["", "Limits of what was not checked: the searxng backend (no own instance) and exa",
            "(paid, no key) were never run live - only mocked; a harness not named in",
            "--with-harness is not checked at all; a full start/resume run (plan, verification,",
            "synthesis, writing into the socket) is not touched by this script - that is",
            "acceptance, tasks item 4. Step 4 on codex does NOT prove that the model never",
            "went to the network around the tool: there is no enforcement there and",
            "sources[].tool is the model's self-report. In claude the path is closed by",
            "mechanics."]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
