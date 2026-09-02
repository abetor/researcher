"""Orchestrator tests: a fake harness plus a fake wiki socket through DI (units), plus an
integration test against the REAL llm-wiki socket (sys.path points at ../tool-llm-wiki, and
the test is skipped if it is absent).

The harness is faked at the run() level - that is the DI seam of the port; the error
classifier is NOT duplicated here (it lives in adapters/base and is covered by
test_adapters), we only supply a ready RunResult.stop as if the harness had already run.
"""
import datetime
import hashlib
import fcntl
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from researcher import checkpoint  # noqa: E402
from researcher.adapters.base import Capabilities, HarnessAdapter, RunResult  # noqa: E402
from researcher.orchestrator import (  # noqa: E402
    EXIT_DONE, EXIT_PLAN_REVIEW, EXIT_QUOTA, EXIT_TRANSIENT,
    Orchestrator, RunConfig as _RunConfig, _SUBSTRATE_SECTIONS,
    _append_system_section, _extract_json, _prompt_judge, _without_system_section,
    _validate_collector_sources, begin_extend,
)
from researcher.observability import RoleHeartbeat, read_heartbeat  # noqa: E402

# The expected static collector tool set on a CLAUDE-CLASS harness (native web plus a
# per-tool allowlist). Spelled out as a literal rather than read from the profile: otherwise
# the test would be tautological.
CLAUDE_LIKE_COLLECT_TOOLS = "WebSearch WebFetch Read Grep Glob"

_LLMWIKI = Path(__file__).parent.parent.parent / "tool-llm-wiki"


def RunConfig(*args, **kwargs):
    """The older pipeline tests explicitly model the auto mode of a nightly job.

    The interactive default is covered by separate plan-gate tests that use the real
    `_RunConfig`, so introducing the new default stop does not turn every earlier phase
    regression into a two-run scenario.
    """
    kwargs.setdefault("auto_plan", True)
    # The older fail-closed regressions check the last line of defense without pauses.
    # The production default (2 retries) is pinned by separate tests below.
    kwargs.setdefault("retries", 0)
    return _RunConfig(*args, **kwargs)


# --- fakes -----------------------------------------------------------------

def _role(prompt: str) -> str:
    for key, role in (("planner", "plan"), ("collector", "collect"),
                      ("critic", "verify"), ("synthesizer", "synth"),
                      ("judge", "judge")):
        if key in prompt:
            return role
    return "?"


class FakeHarness(HarnessAdapter):
    """A harness without processes: run() returns whatever responder(role, prompt) returns.

    Capabilities are set by parameters: the default is a CLAUDE-CLASS harness (native web
    plus a per-tool allowlist); native_web=False, tool_allowlist=False gives a codex-class
    one (no web, the boundary is the sandbox). The collector profile is derived from them by
    the base collect_profile.
    """
    name = "fake"

    def __init__(self, responder, *, native_web=True, tool_allowlist=True,
                 static_tool_set=True, judge_responder=None):
        self.responder = responder
        self.native_web = native_web
        self.tool_allowlist = tool_allowlist
        self.static_tool_set = static_tool_set
        self.judge_responder = judge_responder
        self.calls = []  # (role, prompt, build_kw) - for asserting on prompts and tools

    def capabilities(self):
        return Capabilities(False, False, False, False, False,
                            native_web=self.native_web, tool_allowlist=self.tool_allowlist,
                            static_tool_set=self.static_tool_set)

    def build_cmd(self, prompt, **kw):
        return ["true"]

    def parse_output(self, stdout, exit_code):
        return RunResult(ok=exit_code == 0, text=stdout, exit_code=exit_code)

    def run(self, prompt, *, cwd, timeout=None, **build_kw):
        observer = build_kw.pop("on_harness_pid", None)
        role = _role(prompt)
        self.calls.append((role, prompt, build_kw))
        if observer is not None:
            observer(9876)
        if role == "judge":
            out = self.judge_responder(prompt) if self.judge_responder else json.dumps({
                "verdict": "enough",
                "why": "what was collected is enough for synthesis",
                "coverage_md": "## Closed\n\nThe basic test corpus is covered.",
                "gaps": [],
                "new_clusters": [],
            })
        else:
            out = self.responder(role, prompt)
        if observer is not None:
            observer(None)
        return out if isinstance(out, RunResult) else RunResult(True, out, 0, stop="done")


class FakeWiki:
    """Fake wiki socket: claims on disk (like the real one - synthesis and promote read them
    straight from the files), pages in memory. A claim id is a hash of its text (as in the
    real socket: stable and IDENTICAL in both zones), so that promote_claim reconciles by id
    and an idempotent resume creates no duplicates. Schemas and FKs are not enforced (that is
    the real socket's job, and the integration test covers it)."""

    class DuplicateClaim(Exception):
        pass

    FrontmatterError = ValueError

    def __init__(self):
        self.sources = {}
        self.claims = {"staging": [], "final": []}  # a mirror of the disk, for assertions
        self.pages = {}  # concept slug -> {"title","claims","body"} (the rewritable W1 layer)
        # Seams for simulating a HARD interruption in the middle of a publication (codex review
        # block-1): None/False means off, otherwise a RuntimeError at exactly the right point.
        self.crash_after_promotes = None  # fail on the (N+1)-th promote
        self.crash_on_page = False        # fail on the first page write
        self.promotes = 0
        self.merges = []                  # (zone, claim_id, evidence) of each merge call
        self.merge_error = None           # an exception instead of the merge (socket failure)

    @staticmethod
    def _cid(text):
        return "clm_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _path(topic_dir, zone):
        return Path(topic_dir) / zone / "claims.jsonl"

    def _read(self, topic_dir, zone):
        p = self._path(topic_dir, zone)
        return [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()] \
            if p.exists() else []

    def _write(self, topic_dir, zone, rows):
        p = self._path(topic_dir, zone)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")

    def add_source(self, topic_dir, *, kind, title, url=None, tool=None, **_):
        key = url or f"{kind}:{title}"
        if key in self.sources:
            return self.sources[key]
        sid = f"src_{len(self.sources):012d}"
        self.sources[key] = sid
        row = {"id": sid, "kind": kind, "title": title, "url": url, "tool": tool,
               "meta": _.get("meta") or {}}
        path = Path(topic_dir) / "sources" / "sources.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return sid

    def add_claim(self, topic_dir, zone, *, text, confidence, status, evidence, run_id=None,
                  meta=None, **_):
        rows = self._read(topic_dir, zone)
        cid = self._cid(text)
        if any(r["id"] == cid for r in rows):
            raise self.DuplicateClaim(cid)
        row = {"id": cid, "text": text, "confidence": confidence, "status": status,
               "evidence": evidence}
        if run_id:
            row["run_id"] = run_id
        if meta:
            row["meta"] = meta
        rows.append(row)
        self._write(topic_dir, zone, rows)
        self.claims[zone].append(row)
        return cid

    def merge_claim_evidence(self, topic_dir, zone, claim_id, evidence):
        """Like the real socket (llm-wiki): evidence grows in place, with exact dedup."""
        if self.merge_error is not None:
            raise self.merge_error
        rows = self._read(topic_dir, zone)
        idx = next((i for i, r in enumerate(rows) if r["id"] == claim_id), None)
        if idx is None:
            raise KeyError(f"claim {claim_id} not found in {zone}")
        out = []
        for entry in evidence:
            if entry in rows[idx]["evidence"]:
                out.append({"source_id": entry.get("source_id"), "result": "duplicate"})
                continue
            rows[idx]["evidence"].append(entry)
            out.append({"source_id": entry.get("source_id"), "result": "added"})
        self._write(topic_dir, zone, rows)
        self.merges.append((zone, claim_id, list(evidence)))
        return out

    def promote_claim(self, topic_dir, claim_id):
        """verified staging -> final by moving the row (it disappears from staging). The status
        is NOT changed. Reconciliation: already in final means the append is skipped, but the
        row is still removed from staging (just like the real socket)."""
        if self.crash_after_promotes is not None and self.promotes >= self.crash_after_promotes:
            raise RuntimeError("process interrupted during promote")
        self.promotes += 1
        staging = self._read(topic_dir, "staging")
        idx = next((i for i, r in enumerate(staging) if r["id"] == claim_id), None)
        if idx is None:
            raise KeyError(f"claim {claim_id} is not in staging")
        row = staging[idx]
        if row["status"] != "verified":
            raise ValueError(f"promote accepts only verified: {claim_id} {row['status']!r}")
        final = self._read(topic_dir, "final")
        if claim_id not in {r["id"] for r in final}:
            final.append(row)
            self._write(topic_dir, "final", final)
            self.claims["final"].append(row)
        self._write(topic_dir, "staging", [r for i, r in enumerate(staging) if i != idx])
        self.claims["staging"] = [r for r in self.claims["staging"] if r["id"] != claim_id]
        return claim_id

    def add_wiki_page(self, topic_dir, *, title, claims, body=""):
        if self.crash_on_page:
            raise RuntimeError("process interrupted while writing a page")
        slug = title.strip().lower()
        if slug in self.pages:
            raise FileExistsError(slug)
        self.pages[slug] = {"title": title, "claims": list(claims), "body": body}
        self._write_page(topic_dir, slug)
        return "wik_" + self._cid(title)[4:]

    def update_wiki_page(self, topic_dir, *, title, claims, body=""):
        slug = title.strip().lower()
        if slug not in self.pages:
            raise FileNotFoundError(slug)
        self.pages[slug] = {"title": title, "claims": list(claims), "body": body}
        self._write_page(topic_dir, slug)
        return "wik_" + self._cid(title)[4:]

    def _write_page(self, topic_dir, slug):
        """On-disk mirror of a page: pending finalization compares the topic contents."""
        page = self.pages[slug]
        path = Path(topic_dir) / "final" / "wiki" / f"{slug}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(page, ensure_ascii=False, sort_keys=True) + "\n", "utf-8")

    @staticmethod
    def parse_page(text):
        page = json.loads(text)
        return {"title": page["title"], "claims": page["claims"]}, page["body"]


def _src(n=1):
    return [{"kind": "web", "title": f"Src {i}", "url": f"https://e.com/{i}", "tool": "web",
             "published_at": f"2026-08-{i + 1:02d}", "published_at_reason": None}
            for i in range(n)]


def _claim(text, url="https://e.com/0"):
    return {"text": text, "confidence": "high",
            "evidence": [{"url": url, "quote": "quote", "stance": "supports"}]}


def _page_body(marker="coherent analysis"):
    return (
        f"## What is known\n\n{marker}.\n\n"
        "## Why it matters\n\nThe page claim explains what the fact means.\n\n"
        "## What it is for\n\nA practical application.\n\n"
        "## Confidence and why\n\nThe page source confirms the evidence.\n\n"
        "## Alternatives\n\nConsider an alternative approach."
    )


def _topic(tmp_path, phase="planned", **extra):
    root = tmp_path / "topic"
    (root / "work").mkdir(parents=True)
    st = {"phase": phase, "topic": "why pytest fails", "queue": [], "done": [],
          "sessions": {}}
    st.update(extra)
    checkpoint.save(root, st)
    return root


def _scripted(plan_tasks, collect_map, keep_all=True, final_all=True, page_ready=True):
    """Build a responder: plan -> tasks; collect -> collect_map[query]; verify/synth -> keep all.
    synth returns final indexes (0..19) plus one concept page covering every final claim.
    The orchestrator itself filters out the extra indexes by the real number of staged claims.

    By default this positive scripted pipeline enriches every non-empty answer up to the
    current synthesis contract: at least 3 facts, 2 domains and 3 kinds. Tests of the compiler
    itself use `_synth_only` and raw fixtures, so nothing here bypasses the production gate."""
    prepared = json.loads(json.dumps(collect_map, ensure_ascii=False))
    if page_ready:
        for query, payload in prepared.items():
            claims = payload.get("claims") if isinstance(payload.get("claims"), list) else []
            sources = payload.get("sources") if isinstance(payload.get("sources"), list) else []
            if not claims:
                continue
            suffix = hashlib.sha256(query.encode()).hexdigest()[:8]
            supports = [
                {"kind": "paper", "title": f"Supporting paper {query}",
                 "url": f"https://fixture-paper-{suffix}.example/source", "tool": "arxiv",
                 "published_at": "2026-08-11", "published_at_reason": None},
                {"kind": "repo", "title": f"Supporting repository {query}",
                 "url": f"https://fixture-repo-{suffix}.example/source", "tool": "github",
                 "published_at": "2026-08-11", "published_at_reason": None},
            ]
            existing_urls = {source.get("url") for source in sources if isinstance(source, dict)}
            sources.extend(source for source in supports if source["url"] not in existing_urls)
            while len(claims) < 3:
                support = supports[len(claims) - 1]
                claims.append(_claim(
                    f"supporting fact {query} {len(claims) + 1}", support["url"]))
            for claim, support in zip(claims[-2:], supports):
                if not any(evidence.get("url") == support["url"]
                           for evidence in claim.get("evidence", [])
                           if isinstance(evidence, dict)):
                    claim.setdefault("evidence", []).append(
                        {"url": support["url"], "quote": "supporting quote",
                         "stance": "supports"})
            payload["sources"] = sources
            payload["claims"] = claims

    def responder(role, prompt):
        if role == "plan":
            return json.dumps({"plan_md": "# plan", "tasks": plan_tasks})
        if role == "collect":
            q = prompt.split('Query: "', 1)[1].split('"', 1)[0]
            if q.startswith("Saturation control"):
                return json.dumps({"sources": [], "claims": []})
            return json.dumps(prepared[q])
        if role == "verify":
            keep = [{"index": i, "keep": keep_all, "confidence": "high"} for i in range(20)]
            return json.dumps({"verdicts": keep, "gaps": []})
        if role == "synth":
            fin = [{"index": i} for i in range(20)] if final_all else []
            pages = ([{"title": "Combined concept", "claim_indexes": list(range(20)),
                       "body_md": _page_body()}] if final_all else [])
            return json.dumps({"final": fin, "pages": pages, "summary_md": "# synthesis",
                               "gaps": []})
        raise AssertionError(role)
    return responder


# --- units: full pass -------------------------------------------------------

def test_fake_role_writes_fresh_complete_heartbeat(tmp_path):
    root = _topic(tmp_path, phase="planned", cycles=0)
    orch = Orchestrator(
        root, adapter=FakeHarness(lambda *_: "ok"), wiki=FakeWiki(),
    )

    result = orch._call(
        "planner", role="think", tools="", allowed_tools="",
        heartbeat_role="plan",
    )

    heartbeat = read_heartbeat(root)
    assert result.ok and heartbeat is not None
    assert set(heartbeat) == {
        "at", "phase", "role", "call_started", "pid", "harness_pid", "cycle",
    }
    assert heartbeat == {
        **heartbeat,
        "phase": "planned",
        "role": "plan",
        "call_started": None,
        "pid": os.getpid(),
        "harness_pid": None,
        "cycle": 0,
    }
    at = datetime.datetime.fromisoformat(heartbeat["at"])
    now = datetime.datetime.now(datetime.timezone.utc)
    assert 0 <= (now - at).total_seconds() < 5
    assert not list((root / "work").glob("heartbeat.json.*.tmp"))


def test_heartbeat_keeps_other_parallel_collector_visible(tmp_path):
    heartbeat = RoleHeartbeat(tmp_path)
    first = heartbeat.start(phase="collecting", role="collect", cycle=3)
    heartbeat.harness_pid(first, 111)
    second = heartbeat.start(phase="collecting", role="collect", cycle=3)
    heartbeat.harness_pid(second, 222)
    heartbeat.finish(second)

    visible = read_heartbeat(tmp_path)
    assert visible is not None
    assert visible["call_started"] is not None and visible["harness_pid"] == 111

    heartbeat.finish(first)
    ended = read_heartbeat(tmp_path)
    assert ended is not None
    assert ended["call_started"] is None and ended["harness_pid"] is None

def test_plan_gate_stops_by_default_and_persists_editable_map(tmp_path, capsys):
    root = _topic(tmp_path)
    ad = FakeHarness(lambda role, _prompt: json.dumps({
        "plan_md": "# plan",
        "clusters": [{"id": "c1", "query": "first angle"}],
    }) if role == "plan" else pytest.fail("collection must not start after plan"))

    code = Orchestrator(root, adapter=ad, wiki=FakeWiki(), config=_RunConfig()).run()

    assert code == EXIT_PLAN_REVIEW
    state = checkpoint.load(root)
    assert state["phase"] == "planned" and state["queue"] == []
    assert state["plan_gate"]["status"] == "waiting"
    search_map = root / "work" / "search-map.json"
    assert json.loads(search_map.read_text("utf-8"))["clusters"] == [
        {"id": "c1", "query": "first angle"}
    ]
    assert capsys.readouterr().out.strip() == (
        f"the map is awaiting approval: {search_map}; if it looks right run resume, "
        "otherwise edit the file"
    )


def test_resume_uses_edited_search_map_without_replanning(tmp_path):
    root = _topic(tmp_path)
    first = FakeHarness(lambda role, _prompt: json.dumps({
        "plan_md": "# plan",
        "clusters": [
            {"id": "drop", "query": "to delete"},
            {"id": "edit", "query": "before the edit"},
        ],
    }) if role == "plan" else pytest.fail(role))
    orch = Orchestrator(root, adapter=first, wiki=FakeWiki(), config=_RunConfig())
    assert orch._phase_plan() == "plan_review"

    path = root / "work" / "search-map.json"
    search_map = json.loads(path.read_text("utf-8"))
    search_map["clusters"] = [
        {"id": "edit", "query": "after the edit"},
        {"id": "added", "query": "added cluster"},
    ]
    path.write_text(json.dumps(search_map, ensure_ascii=False, indent=2), "utf-8")

    resumed = FakeHarness(lambda role, _prompt: pytest.fail(
        f"resuming the map must not call the planner again: {role}"))
    resumed_orch = Orchestrator(root, adapter=resumed, wiki=FakeWiki(), config=_RunConfig())
    assert resumed_orch._phase_plan() == "done"

    state = checkpoint.load(root)
    assert state["phase"] == "collecting"
    assert state["queue"] == [
        {"id": "edit", "query": "after the edit"},
        {"id": "added", "query": "added cluster"},
    ]
    assert state["plan_gate"]["status"] == "approved"
    assert resumed.calls == []


def test_auto_plan_is_explicit_and_skips_review_stop(tmp_path):
    root = _topic(tmp_path)
    ad = FakeHarness(lambda role, _prompt: json.dumps({
        "plan_md": "# plan",
        "clusters": [{"id": "c1", "query": "q1"}],
    }) if role == "plan" else pytest.fail(role))
    cfg = _RunConfig(auto_plan=True)

    assert Orchestrator(root, adapter=ad, wiki=FakeWiki(), config=cfg)._phase_plan() == "done"
    state = checkpoint.load(root)
    assert state["phase"] == "collecting"
    assert state["queue"] == [{"id": "c1", "query": "q1"}]
    assert state["plan_gate"]["status"] == "approved"
    assert _RunConfig().auto_plan is False

def test_full_pipeline_fake(tmp_path):
    root = _topic(tmp_path)
    collect = {
        "q1": {"sources": _src(1), "claims": [_claim("fact1")]},
        "q2": {
            "sources": [
                {"kind": "paper", "title": "Another domain",
                 "url": "https://other.example/1", "tool": "arxiv",
                 "published_at": "2026-08-02", "published_at_reason": None},
                {"kind": "repo", "title": "Third kind",
                 "url": "https://repo.example/1", "tool": "github",
                 "published_at": "2026-08-02", "published_at_reason": None},
            ],
            "claims": [_claim("fact2", "https://other.example/1"),
                       _claim("fact3", "https://repo.example/1")],
        },
    }
    ad = FakeHarness(_scripted(
        [{"id": "t1", "query": "q1"}, {"id": "t2", "query": "q2"}], collect,
        page_ready=False))
    wiki = FakeWiki()
    code = Orchestrator(root, adapter=ad, wiki=wiki, config=RunConfig()).run()

    assert code == EXIT_DONE
    assert checkpoint.load(root)["phase"] == "done"
    # the final set is filled by promote (a move out of staging), not by add_claim(final):
    assert {c["text"] for c in wiki.claims["final"]} == {"fact1", "fact2", "fact3"}
    assert wiki.claims["final"][0]["status"] == "verified"  # promote does NOT change the status
    assert wiki.claims["staging"] == []  # both promoted, so staging is empty (move, not copy)
    assert not wiki._read(root, "staging")  # and it is empty on disk too
    # the synthesis concept page is written (W1) and references only final claims:
    assert len(wiki.pages) == 1
    page = next(iter(wiki.pages.values()))
    final_ids = {c["id"] for c in wiki.claims["final"]}
    assert page["claims"] and set(page["claims"]) <= final_ids
    assert (root / "work" / "plan.md").read_text("utf-8").strip() == "# plan"
    synthesis = (root / "work" / "synthesis.md").read_text("utf-8")
    assert synthesis.startswith("# synthesis") and "## Coverage" in synthesis
    assert (root / "work" / "found" / "t1.jsonl").exists()


def test_single_source_high_is_downgraded_and_marked_by_code(tmp_path):
    root = _topic(tmp_path)
    collect = {"q1": {"sources": _src(1), "claims": [_claim("single-source fact")]}}
    wiki = FakeWiki()

    assert Orchestrator(
        root,
        adapter=FakeHarness(_scripted([{"id": "t1", "query": "q1"}], collect)),
        wiki=wiki,
        config=RunConfig(pool_size=1),
    ).run() == EXIT_DONE

    claim = wiki._read(root, "final")[0]
    assert claim["confidence"] == "medium"
    assert claim["meta"]["single_source"] is True


def test_collector_downgrades_single_source_before_found_staging_path(tmp_path):
    root = _topic(tmp_path, phase="collecting")
    wiki = FakeWiki()
    orch = Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=wiki)
    (root / "work" / "found").mkdir()

    orch._ingest_collector(
        {"id": "t1"},
        {"sources": _src(1), "claims": [_claim("raw single-source high")]},
        set(),
    )

    found = json.loads((root / "work" / "found" / "t1.jsonl").read_text("utf-8"))
    assert found["confidence"] == "medium"
    assert found["meta"]["single_source"] is True


def test_promote_rejects_legacy_single_source_high_before_final(tmp_path):
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    sid = wiki.add_source(root, kind="web", title="Legacy",
                          url="https://legacy.example/source", tool="web")
    wiki.add_claim(
        root, "staging", text="legacy high", confidence="high", status="verified",
        evidence=[{"source_id": sid, "quote": "q", "stance": "supports"}],
    )
    other_sid = wiki.add_source(root, kind="paper", title="Another domain",
                                url="https://other.example/source", tool="arxiv")
    wiki.add_source(root, kind="repo", title="Third kind",
                    url="https://repo.example/source", tool="github")
    for text in ("extra fact 1", "extra fact 2"):
        wiki.add_claim(
            root, "staging", text=text, confidence="medium", status="verified",
            evidence=[{"source_id": other_sid, "quote": "q", "stance": "supports"}],
        )
    page = [{"title": "Legacy concept", "claim_indexes": [0, 1, 2],
             "body_md": _page_body()}]

    with pytest.raises(ValueError, match="legacy staging claim.*single source"):
        Orchestrator(
            root, adapter=FakeHarness(_synth_only([0, 1, 2], page)), wiki=wiki,
            config=RunConfig(max_cycles=0),
        ).run()

    assert wiki._read(root, "final") == []
    assert (root / "work" / _PENDING).exists()


def test_two_source_high_stays_high_and_is_not_marked_single_source(tmp_path):
    root = _topic(tmp_path)
    sources = _src(2)
    claim = _claim("cross-confirmed fact", sources[0]["url"])
    claim["evidence"].append({"url": sources[1]["url"], "quote": "second quote",
                              "stance": "supports"})
    wiki = FakeWiki()

    assert Orchestrator(
        root,
        adapter=FakeHarness(_scripted(
            [{"id": "t1", "query": "q1"}], {"q1": {"sources": sources, "claims": [claim]}})),
        wiki=wiki,
        config=RunConfig(pool_size=1),
    ).run() == EXIT_DONE

    final = wiki._read(root, "final")[0]
    assert final["confidence"] == "high"
    assert final["meta"]["single_source"] is False


def test_progress_goes_to_stderr_with_phase_cycle_task_and_counts(tmp_path, capsys):
    root = _topic(tmp_path)
    collect = {"q1": {"sources": _src(1), "claims": [_claim("fact1")]}}
    ad = FakeHarness(_scripted([{"id": "t1", "query": "q1"}], collect))

    assert Orchestrator(
        root, adapter=ad, wiki=FakeWiki(), config=RunConfig(max_cycles=3, pool_size=1)
    ).run() == EXIT_DONE

    out = capsys.readouterr()
    assert out.out == ""
    assert "progress: phase=planned" in out.err
    started = out.err.index("phase=collecting | cycle=1 of 3 | task=t1 (started)")
    finished = out.err.index("phase=collecting | cycle=1 of 3 | task=t1 (completed)")
    assert started < finished
    assert "claims found=3 staging=0 final=0" in out.err
    assert "progress: phase=verifying" in out.err
    assert "progress: phase=synthesizing" in out.err
    assert "progress: phase=done" in out.err


@pytest.mark.parametrize(("tool", "expected_kind"), [
    ("web", "web"),
    ("github", "repo"),
    ("arxiv", "paper"),
    ("hn", "other"),
])
def test_source_kind_comes_from_tool_not_model_report(tmp_path, tool, expected_kind):
    """A defect seen in production: the model wrote kind=web even for HN and GitHub."""
    root = _topic(tmp_path, phase="collecting")
    wiki = FakeWiki()
    captured = []
    original = wiki.add_source

    def add_source(topic_dir, **kwargs):
        captured.append(kwargs)
        return original(topic_dir, **kwargs)

    wiki.add_source = add_source
    orch = Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=wiki)
    (root / "work" / "found").mkdir()
    orch._ingest_collector(
        {"id": "t1"},
        {"sources": [{"kind": "web", "title": "source",
                      "url": "https://example.com/x", "tool": tool,
                      "published_at": "2026-08-11", "published_at_reason": None}],
         "claims": []},
        set(),
    )

    assert captured[0]["kind"] == expected_kind


@pytest.mark.parametrize("source", [
    {"kind": "web", "title": "no date", "url": "https://e.com/x", "tool": "web"},
    {"kind": "web", "title": "broken date", "url": "https://e.com/x", "tool": "web",
     "published_at": "yesterday", "published_at_reason": None},
    {"kind": "web", "title": "no reason", "url": "https://e.com/x", "tool": "web",
     "published_at": None, "published_at_reason": None},
])
def test_collector_source_requires_publication_date_or_explicit_reason(tmp_path, source):
    task = {"id": "t1", "query": "q1"}
    root = _topic(tmp_path, phase="collecting", queue=[task], cycles=0,
                  rounds_without_claim=0, run_id="run_test")
    responder = _scripted(
        [task], {"q1": {"sources": [source], "claims": [_claim("fact")]}})

    with pytest.raises(ValueError, match="published_at"):
        Orchestrator(root, adapter=FakeHarness(responder), wiki=FakeWiki(),
                     config=RunConfig(max_cycles=1, pool_size=1)).run()
    assert checkpoint.load(root)["phase"] == "collecting"
    assert (root / "work" / "bad-answers" / "collect-t1-attempt-1.txt").exists()


def test_published_at_reason_survives_validation_into_source_meta(tmp_path):
    root = _topic(tmp_path, phase="collecting")
    reason = "the source document carries no date"
    data = {
        "sources": [{
            "kind": "paper", "title": "Undated archive",
            "url": "https://archive.example/undated", "tool": "arxiv",
            "published_at": None, "published_at_reason": reason,
        }],
        "claims": [],
    }
    _validate_collector_sources(data)
    orch = Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=FakeWiki())
    (root / "work" / "found").mkdir()

    orch._ingest_collector({"id": "t1"}, data, set())

    source = json.loads(
        (root / "sources" / "sources.jsonl").read_text("utf-8").splitlines()[0])
    assert source["meta"] == {"published_at": None, "published_at_reason": reason}


def test_funnel_round_history_and_events_are_durable(tmp_path):
    root = _topic(tmp_path)

    def responder(role, _prompt):
        if role == "plan":
            return json.dumps({"plan_md": "# plan", "tasks": [{"id": "t1", "query": "q"}]})
        if role == "collect":
            sources = _src(5)
            sources[3]["url"] = "https://other.example/3"
            sources[3]["kind"] = "paper"
            sources[3]["tool"] = "arxiv"
            sources[4]["url"] = "https://other.example/4"
            sources[4]["kind"] = "repo"
            sources[4]["tool"] = "github"
            return json.dumps({
                "sources": sources,
                "claims": [_claim(f"fact{i + 1}", source["url"])
                           for i, source in enumerate(sources)],
            })
        if role == "verify":
            return json.dumps({"verdicts": [
                {"index": 0, "keep": True, "confidence": "high"},
                {"index": 1, "keep": True, "confidence": "medium"},
                {"index": 2, "keep": True, "confidence": "medium"},
                {"index": 3, "keep": True, "confidence": "medium"},
                {"index": 4, "keep": False, "confidence": "low"},
            ], "gaps": []})
        if role == "synth":
            return json.dumps({
                "final": [{"index": 0}, {"index": 1}, {"index": 3}],
                "pages": [{"title": "Concept", "claim_indexes": [0, 1, 3],
                           "body_md": _page_body()}],
                "summary_md": "summary",
                "gaps": [],
            })
        raise AssertionError(role)

    assert Orchestrator(
        root, adapter=FakeHarness(responder), wiki=FakeWiki(),
        config=RunConfig(max_cycles=1, pool_size=1),
    ).run() == EXIT_DONE

    state = checkpoint.load(root)
    assert state["funnel"] == {
        "found": 5,
        "staging": {"keep": 4, "drop": 1},
        "final": {"keep": 3, "drop": 1},
    }
    assert state["rounds"] == [{
        "cycle": 1, "tasks": ["t1"], "new_claims": 5, "found": 5,
        "accepted_claims": 4,
    }]
    events = [json.loads(line) for line in
              (root / "work" / "events.jsonl").read_text("utf-8").splitlines()]
    event_names = [event["event"] for event in events]
    assert event_names == [
        "search_map_approved", "collection_round", "judge_verdict",
        "phase_changed", "verification_funnel", "breadth_snapshot",
        "synthesis_funnel", "coverage_note",
    ]
    assert events[1]["round"]["new_claims"] == 5
    synthesis_event = next(row for row in events if row["event"] == "synthesis_funnel")
    assert synthesis_event["funnel"] == state["funnel"]


def test_single_synth_call_and_no_subagents(tmp_path):
    root = _topic(tmp_path)
    collect = {"q1": {"sources": _src(1), "claims": [_claim("fact1")]}}
    ad = FakeHarness(_scripted([{"id": "t1", "query": "q1"}], collect))
    Orchestrator(root, adapter=ad, wiki=FakeWiki(), config=RunConfig()).run()

    synth_calls = [c for c in ad.calls if c[0] == "synth"]
    assert len(synth_calls) == 1  # synthesis is strictly one call
    for role, prompt, kw in ad.calls:
        if role == "collect":
            assert "Task/Agent" in prompt and "do NOT use" in prompt  # sub-agents are forbidden
            assert kw["allowed_tools"] == CLAUDE_LIKE_COLLECT_TOOLS
            assert "Task" not in kw["allowed_tools"] and "Agent" not in kw["allowed_tools"]


def test_critic_gaps_are_ignored_and_do_not_continue_collection(tmp_path):
    root = _topic(tmp_path)
    collected = []
    verify_calls = 0

    def responder(role, prompt):
        nonlocal verify_calls
        if role == "plan":
            return json.dumps({"plan_md": "# plan", "tasks": [{"id": "t1", "query": "q0"}]})
        if role == "collect":
            query = prompt.split('Query: "', 1)[1].split('"', 1)[0]
            collected.append(query)
            index = len(collected)
            kind, tool = (("web", "web"), ("paper", "arxiv"),
                          ("repo", "github"))[(index - 1) % 3]
            source = {"kind": kind, "title": query, "url": f"https://d{index}.example/x",
                      "tool": tool, "published_at": "2026-08-11",
                      "published_at_reason": None}
            return json.dumps({"sources": [source],
                               "claims": [_claim(f"fact {query}", source["url"])]})
        if role == "verify":
            verify_calls += 1
            gaps = ([
                {"query": "q-low", "reason": "no data on the low-priority angle",
                 "priority": "low"},
                {"query": "q-high", "reason": "an important conclusion rests on one source",
                 "priority": "high"},
            ] if verify_calls == 1 else [])
            return json.dumps({
                "verdicts": [{"index": i, "keep": True, "confidence": "high"}
                              for i in range(20)],
                "gaps": gaps,
            })
        if role == "synth":
            return json.dumps({"final": [{"index": i} for i in range(20)],
                               "pages": [{"title": "Gaps concept",
                                          "claim_indexes": list(range(20)),
                                          "body_md": _page_body()}],
                               "summary_md": "summary", "gaps": []})
        raise AssertionError(role)

    assert Orchestrator(root, adapter=FakeHarness(responder), wiki=FakeWiki(),
                        config=RunConfig(max_cycles=3, pool_size=1)).run() == EXIT_DONE
    state = checkpoint.load(root)
    assert collected == ["q0"]
    assert state["cycles"] == 1 and state["gaps"] == []
    events = [json.loads(line) for line in
              (root / "work" / "events.jsonl").read_text("utf-8").splitlines()]
    assert not [event for event in events if event["event"] == "gaps_queued"]
    funnel = next(event for event in events if event["event"] == "verification_funnel")
    assert "gaps" not in funnel


def test_repeated_critic_gap_is_never_collected(tmp_path):
    root = _topic(tmp_path)
    collected = []

    def responder(role, prompt):
        if role == "plan":
            return json.dumps({"plan_md": "# plan", "tasks": [{"id": "t1", "query": "q0"}]})
        if role == "collect":
            query = prompt.split('Query: "', 1)[1].split('"', 1)[0]
            collected.append(query)
            if query.startswith("Saturation control"):
                return json.dumps({"sources": [], "claims": []})
            if query == "q0":
                source = {"kind": "web", "title": "primary",
                          "url": "https://one.example/source", "tool": "web",
                          "published_at": "2026-08-11", "published_at_reason": None}
                claims = [_claim("primary fact 1", source["url"]),
                          _claim("primary fact 2", source["url"])]
            else:
                source = {"kind": "paper", "title": "follow-up",
                          "url": "https://two.example/source", "tool": "arxiv",
                          "published_at": "2026-08-11", "published_at_reason": None}
                repo = {"kind": "repo", "title": "third kind",
                        "url": "https://three.example/source", "tool": "github",
                        "published_at": "2026-08-11", "published_at_reason": None}
                claims = [_claim("follow-up fact", source["url"])]
                return json.dumps({"sources": [source, repo], "claims": claims})
            return json.dumps({"sources": [source], "claims": claims})
        if role == "verify":
            return json.dumps({
                "verdicts": [{"index": i, "keep": True, "confidence": "high"}
                              for i in range(20)],
                "gaps": [{"query": "q-gap", "reason": "the reviewer repeats the same gap",
                          "priority": "high"}],
            })
        if role == "synth":
            return json.dumps({
                "final": [{"index": i} for i in range(20)],
                "pages": [{"title": "Dedup concept", "claim_indexes": list(range(20)),
                           "body_md": _page_body()}],
                "summary_md": "summary", "gaps": [],
            })
        raise AssertionError(role)

    assert Orchestrator(
        root, adapter=FakeHarness(responder), wiki=FakeWiki(),
        config=RunConfig(max_cycles=3, saturation_rounds=1, pool_size=1),
    ).run() == EXIT_DONE

    assert collected.count("q-gap") == 0
    assert checkpoint.load(root)["gaps"] == []


def test_synthesizer_gap_is_recorded_but_never_restarts_collection(tmp_path):
    root = _topic(tmp_path)
    synth_calls = 0
    collected = []

    def responder(role, prompt):
        nonlocal synth_calls
        if role == "plan":
            return json.dumps({"plan_md": "# plan", "tasks": [{"id": "t1", "query": "q0"}]})
        if role == "collect":
            query = prompt.split('Query: "', 1)[1].split('"', 1)[0]
            collected.append(query)
            if query.startswith("Saturation control"):
                return json.dumps({"sources": [], "claims": []})
            source = {"kind": "web", "title": query,
                      "url": f"https://{query}.example/x", "tool": "web",
                      "published_at": "2026-08-11", "published_at_reason": None}
            if query == "q0":
                repo = {"kind": "repo", "title": "repository",
                        "url": "https://repo.example/q0", "tool": "github",
                        "published_at": "2026-08-11", "published_at_reason": None}
                paper = {"kind": "paper", "title": "paper",
                         "url": "https://paper.example/q0", "tool": "arxiv",
                         "published_at": "2026-08-11", "published_at_reason": None}
                return json.dumps({"sources": [source, repo, paper],
                                   "claims": [_claim("fact q0 0", source["url"]),
                                              _claim("fact q0 1", repo["url"])]})
            source.update(kind="paper", tool="arxiv")
            return json.dumps({"sources": [source],
                               "claims": [_claim(f"fact {query}", source["url"])]})
        if role == "verify":
            return json.dumps({
                "verdicts": [{"index": i, "keep": True, "confidence": "high"}
                              for i in range(20)],
                "gaps": [],
            })
        if role == "synth":
            synth_calls += 1
            gaps = [{"query": "q-synth-gap", "reason": "no alternative",
                     "priority": "high"}]
            pages = [{"title": "Synth-gap concept", "claim_indexes": list(range(20)),
                      "body_md": _page_body()}]
            return json.dumps({"final": [{"index": i} for i in range(20)], "pages": pages,
                               "summary_md": "summary", "gaps": gaps})
        raise AssertionError(role)

    assert Orchestrator(root, adapter=FakeHarness(responder), wiki=FakeWiki(),
                        config=RunConfig(max_cycles=3, saturation_rounds=1,
                                         pool_size=1)).run() == EXIT_DONE
    assert synth_calls == 1
    assert collected == ["q0"]
    state = checkpoint.load(root)
    gap = state["gaps"][0]
    assert gap["query"] == "q-synth-gap" and gap["status"] == "unqueued"
    assert gap["source"] == "synth" and gap["claims_brought"] == 0
    coverage = (root / "work" / "synthesis.md").read_text("utf-8")
    assert "Open questions (synthesis)" in coverage
    assert "q-synth-gap" in coverage
    events = [json.loads(line) for line in
              (root / "work" / "events.jsonl").read_text("utf-8").splitlines()]
    assert sum(event["event"] == "collection_round" for event in events) == 1
    assert not any(event["event"] == "gaps_queued" and event.get("phase") == "synthesis"
                   for event in events)


def test_judge_enough_then_synth_gap_finishes_without_collection_round(tmp_path):
    root = _topic(
        tmp_path, phase="synthesizing",
        stop={"reason": "enough", "why": "the judge finished"},
    )
    wiki = FakeWiki()
    _stage(wiki, root, "fact 1", "fact 2", "fact 3")

    def synth(role, _prompt):
        assert role == "synth"
        return json.dumps({
            "final": [{"index": 0}, {"index": 1}, {"index": 2}],
            "pages": [{
                "title": "Result", "claim_indexes": [0, 1, 2],
                "body_md": _page_body(),
            }],
            "summary_md": "result",
            "gaps": [{
                "query": "follow-up question", "reason": "not closed by synthesis",
                "priority": "high",
            }],
        })

    assert Orchestrator(
        root, adapter=FakeHarness(synth), wiki=wiki, config=RunConfig(),
    ).run() == EXIT_DONE

    state = checkpoint.load(root)
    assert state["phase"] == "done"
    assert state["gaps"][0]["status"] == "unqueued"
    assert "follow-up question" in (root / "work" / "synthesis.md").read_text("utf-8")
    events = [json.loads(line) for line in
              (root / "work" / "events.jsonl").read_text("utf-8").splitlines()]
    assert not any(event["event"] == "collection_round" for event in events)


def test_accepted_gap_continuation_clears_stale_gate_stop(tmp_path):
    root = _topic(
        tmp_path,
        phase="synthesizing",
        cycles=1,
        rounds_without_claim=0,
        stop_kind="gate_refused",
        stop_detail={"reasons": ["an old refusal"]},
    )
    orch = Orchestrator(
        root,
        adapter=FakeHarness(lambda role, prompt: pytest.fail("harness forbidden")),
        wiki=FakeWiki(),
        config=RunConfig(max_cycles=3, saturation_rounds=2),
    )
    state = checkpoint.load(root)

    assert orch._queue_gaps(
        state,
        [{"query": "new gap", "reason": "a source is missing"}],
        source="synthesizer",
    )

    saved = checkpoint.load(root)
    assert saved["phase"] == "collecting"
    assert saved["queue"]
    assert "stop_kind" not in saved
    assert "stop_detail" not in saved


# --- units: synthesis through the W1/W2 socket ------------------------------

def _ev():
    return [{"source_id": "s0", "quote": "q", "stance": "supports"}]


def _stage(wiki, root, *texts):
    """Put verified claims straight into staging (as after the verify phase) and set
    phase=synthesizing."""
    for index, t in enumerate(texts):
        domain = f"source-{index % 2}.example"
        kinds = (("web", "web"), ("paper", "arxiv"), ("repo", "github"))
        kind, tool = kinds[index % len(kinds)]
        sid = wiki.add_source(root, kind=kind, title=f"Source {index}",
                              url=f"https://{domain}/{index}", tool=tool,
                              meta={"published_at": "2026-08-11",
                                    "published_at_reason": None})
        evidence = [{"source_id": sid, "quote": "q", "stance": "supports"}]
        wiki.add_claim(root, "staging", text=t, confidence="medium", status="verified",
                       evidence=evidence)
    st = checkpoint.load(root)
    st["phase"] = "synthesizing"
    checkpoint.save(root, st)


def _synth_only(final_idx, pages):
    def responder(role, prompt):
        assert role == "synth", role  # from the synthesizing phase only the synthesizer is called
        return json.dumps({"final": [{"index": i} for i in final_idx],
                           "pages": pages, "summary_md": "# synthesis", "gaps": []})
    return responder


def _done_topic_for_extend(tmp_path):
    root = _topic(tmp_path, phase="done", run_id="run_initial", run_mode="initial",
                  cycles=2, funnel={
                      "found": 3,
                      "staging": {"keep": 3, "drop": 0},
                      "final": {"keep": 3, "drop": 0},
                  }, rounds=[{"cycle": 1, "tasks": ["old"], "new_claims": 3,
                              "found": 3}], stop={"reason": "saturated"})
    wiki = FakeWiki()
    claim_ids = []
    for index, text in enumerate(("fact 1", "fact 2", "fact 3")):
        kind, tool = (("web", "web"), ("paper", "arxiv"), ("repo", "github"))[index]
        sid = wiki.add_source(
            root, kind=kind, title=f"Old source {index}",
            url=f"https://source-{index % 2}.example/{index}", tool=tool,
            meta={"published_at": "2026-08-01", "published_at_reason": None})
        claim_ids.append(wiki.add_claim(
            root, "final", text=text, confidence="medium", status="verified",
            evidence=[{"source_id": sid, "quote": "q", "stance": "supports"}],
            run_id="run_initial"))
    wiki.add_wiki_page(root, title="Concept A", claims=claim_ids,
                       body=_page_body("old analysis A"))
    wiki.add_wiki_page(root, title="Concept B", claims=claim_ids,
                       body=_page_body("old analysis B"))
    found = root / "work" / "found"
    found.mkdir(parents=True)
    (found / "old.jsonl").write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n"
                for text in ("fact 1", "fact 2", "fact 3")), "utf-8")
    (root / "work" / "synthesis.md").write_text("old synthesis", "utf-8")
    return root, wiki, claim_ids


def test_begin_extend_ignores_legacy_narrow_source_and_uses_own_work_dir(tmp_path):
    root, wiki, _ = _done_topic_for_extend(tmp_path)
    old_found = (root / "work" / "found" / "old.jsonl").read_bytes()
    previous = checkpoint.load(root)
    previous["narrow_source"] = {
        "status": "narrow-source", "reason": "the sources of the previous run are exhausted"
    }
    checkpoint.save(root, previous)

    run_id, created = begin_extend(root)

    assert created is True and run_id != "run_initial"
    state = checkpoint.load(root)
    assert state["phase"] == "planned" and state["run_mode"] == "extend"
    assert state["run_id"] == run_id and state["parent_run_id"] == "run_initial"
    assert state["funnel"] == {
        "found": 0, "staging": {"keep": 0, "drop": 0},
        "final": {"keep": 0, "drop": 0},
    }
    assert state["run_history"][0]["run_id"] == "run_initial"
    assert "narrow_source" not in state["run_history"][0]
    assert "inherited_narrow_sources" not in state
    snapshot = root / "work" / "runs" / "run_initial" / "checkpoint-final.json"
    saved = json.loads(snapshot.read_text("utf-8"))
    assert saved["funnel"]["final"]["keep"] == 3
    assert "narrow_source" not in saved
    assert (root / "work" / "runs" / run_id).is_dir()
    assert (root / "work" / "found" / "old.jsonl").read_bytes() == old_found
    assert wiki._read(root, "final")
    assert begin_extend(root) == (run_id, False)


def test_legacy_narrow_source_field_is_ignored_on_resume(tmp_path):
    root = _topic(
        tmp_path, phase="synthesizing",
        narrow_source={"status": "narrow-source", "reason": "legacy"},
    )
    wiki = FakeWiki()
    _stage(wiki, root, "fact 1", "fact 2", "fact 3")

    code = Orchestrator(
        root,
        adapter=FakeHarness(_synth_only(
            [0, 1, 2], [{
                "title": "Legacy", "claim_indexes": [0, 1, 2],
                "body_md": _page_body(),
            }],
        )),
        wiki=wiki,
        config=RunConfig(),
    ).run()

    assert code == EXIT_DONE
    assert checkpoint.load(root)["phase"] == "done"

def test_begin_extend_snapshots_legacy_funnel_from_disk(tmp_path):
    root = _topic(
        tmp_path, phase="done", run_id="run_legacy", run_mode="initial",
        stop={"reason": "saturated"},
    )
    wiki = FakeWiki()
    sid = wiki.add_source(
        root, kind="web", title="legacy", url="https://legacy.example/source", tool="web"
    )
    evidence = [{"source_id": sid, "quote": "q", "stance": "supports"}]
    for text in ("A", "B"):
        wiki.add_claim(
            root, "staging", text=text, confidence="medium", status="verified",
            evidence=evidence,
        )
    wiki.add_claim(
        root, "final", text="A", confidence="medium", status="verified",
        evidence=evidence,
    )
    found = root / "work" / "found"
    found.mkdir(parents=True)
    (found / "legacy.jsonl").write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in ("A", "B", "C")),
        "utf-8",
    )

    begin_extend(root)

    expected = {
        "found": 3,
        "staging": {"keep": 2, "drop": 1},
        "final": {"keep": 1, "drop": 1},
    }
    state = checkpoint.load(root)
    assert state["run_history"][0]["funnel"] == expected
    snapshot = root / "work" / "runs" / "run_legacy" / "checkpoint-final.json"
    assert json.loads(snapshot.read_text("utf-8"))["funnel"] == expected


def test_begin_extend_respects_topic_lock(tmp_path):
    root, _wiki, _ = _done_topic_for_extend(tmp_path)
    lock_path = root / "work" / "resume.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Extend cannot start while a resume"):
            begin_extend(root)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    assert checkpoint.load(root)["phase"] == "done"


def test_extend_plan_also_waits_for_review_in_its_run_directory(tmp_path, capsys):
    root, wiki, _ = _done_topic_for_extend(tmp_path)
    run_id, _ = begin_extend(root)
    ad = FakeHarness(lambda role, prompt: json.dumps({
        "plan_md": "# extend plan",
        "clusters": [{"id": "delta", "query": "collect more"}],
    }) if role == "plan" else pytest.fail(role))

    assert Orchestrator(root, adapter=ad, wiki=wiki, config=_RunConfig()).run() \
        == EXIT_PLAN_REVIEW

    path = root / "work" / "runs" / run_id / "search-map.json"
    state = checkpoint.load(root)
    assert state["phase"] == "planned" and state["plan_gate"]["status"] == "waiting"
    assert state["plan_gate"]["path"] == str(path.relative_to(root))
    assert path.exists()
    assert "extension of an already completed topic" in ad.calls[0][1]
    assert f"the map is awaiting approval: {path}" in capsys.readouterr().out


def test_extend_dedups_final_before_critic_and_delta_merges_only_touched_page(tmp_path):
    root, wiki, old_claim_ids = _done_topic_for_extend(tmp_path)
    untouched_path = root / "final" / "wiki" / "concept b.md"
    untouched_before = untouched_path.read_bytes()
    run_id, _ = begin_extend(root)
    prompts = {}

    def responder(role, prompt):
        prompts[role] = prompt
        if role == "plan":
            return json.dumps({
                "plan_md": "# extend plan",
                "clusters": [{"id": "delta", "query": "new angle"}],
            })
        if role == "collect":
            return json.dumps({
                "sources": [
                    {"kind": "web", "title": "Old source 0",
                     "url": "https://source-0.example/0", "tool": "web",
                     "published_at": "2026-08-01", "published_at_reason": None},
                    {"kind": "web", "title": "New source",
                     "url": "https://delta.example/4", "tool": "web",
                     "published_at": "2026-08-11", "published_at_reason": None},
                ],
                "claims": [
                    _claim("  FACT   1 ", "https://source-0.example/0"),
                    _claim("fact 4", "https://delta.example/4"),
                ],
            })
        if role == "verify":
            assert "fact 4" in prompt and "FACT   1" not in prompt
            return json.dumps({
                "verdicts": [{"index": 0, "keep": True, "confidence": "medium"}],
                "gaps": [],
            })
        if role == "synth":
            assert "DELTA MERGE" in prompt
            assert "Concept A" in prompt and "Concept B" in prompt
            return json.dumps({
                "final": [{"index": 0}],
                "pages": [{"title": "Concept A", "claim_indexes": [0],
                           "body_md": _page_body("updated analysis A")}],
                "summary_md": "delta synthesis",
                "gaps": [],
            })
        raise AssertionError(role)

    assert Orchestrator(
        root, adapter=FakeHarness(responder), wiki=wiki,
        config=_RunConfig(auto_plan=True, max_cycles=1, pool_size=1),
    ).run() == EXIT_DONE

    state = checkpoint.load(root)
    final = wiki._read(root, "final")
    assert len(final) == 4
    delta = next(claim for claim in final if claim["text"] == "fact 4")
    assert delta["run_id"] == run_id
    assert state["funnel"] == {
        "found": 1,
        "staging": {"keep": 1, "drop": 0},
        "final": {"keep": 1, "drop": 0},
    }
    new_found = root / "work" / "runs" / run_id / "found" / "delta.jsonl"
    assert [json.loads(line)["text"] for line in new_found.read_text("utf-8").splitlines()] \
        == ["fact 4"]
    updated = wiki.pages["concept a"]
    assert updated["claims"] == [*old_claim_ids, delta["id"]]
    assert "updated analysis A" in updated["body"]
    assert untouched_path.read_bytes() != untouched_before
    assert "## Coverage" in wiki.pages["concept b"]["body"]
    assert (root / "work" / "synthesis.md").read_text("utf-8") == "old synthesis"
    delta_synthesis = (root / "work" / "runs" / run_id / "synthesis.md").read_text("utf-8")
    assert delta_synthesis.startswith("delta synthesis")
    assert "## Coverage" in delta_synthesis


def test_extend_adds_coverage_to_untouched_pages(tmp_path):
    root = _topic(tmp_path)

    def initial_responder(role, prompt):
        if role == "plan":
            return json.dumps({
                "plan_md": "# plan",
                "tasks": [{"id": "initial", "query": "primary collection"}],
            })
        if role == "collect":
            query = prompt.split('Query: "', 1)[1].split('"', 1)[0]
            if query != "primary collection":
                return json.dumps({"sources": [], "claims": []})
            sources = [
                {"kind": "web", "title": "Narrow source",
                 "url": "https://only.example/a", "tool": "web",
                 "published_at": "2026-08-01", "published_at_reason": None},
                {"kind": "web", "title": "Source B1",
                 "url": "https://b1.example/b", "tool": "web",
                 "published_at": "2026-08-02", "published_at_reason": None},
                {"kind": "paper", "title": "Source B2",
                 "url": "https://b2.example/b", "tool": "arxiv",
                 "published_at": "2026-08-03", "published_at_reason": None},
                {"kind": "repo", "title": "Source B3",
                 "url": "https://b3.example/b", "tool": "github",
                 "published_at": "2026-08-04", "published_at_reason": None},
            ]
            claims = [
                *[_claim(f"narrow fact A{index}", sources[0]["url"])
                  for index in range(1, 4)],
                *[_claim(f"broad fact B{index}", source["url"])
                  for index, source in enumerate(sources[1:], 1)],
            ]
            return json.dumps({"sources": sources, "claims": claims})
        if role == "verify":
            return json.dumps({
                "verdicts": [{"index": i, "keep": True, "confidence": "medium"}
                             for i in range(20)],
                "gaps": [],
            })
        if role == "synth":
            return json.dumps({
                "final": [{"index": i} for i in range(6)],
                "pages": [
                    {"title": "Narrow concept A", "claim_indexes": [0, 1, 2],
                     "body_md": _page_body("primary narrow analysis A")},
                    {"title": "Broad concept B", "claim_indexes": [3, 4, 5],
                     "body_md": _page_body("primary broad analysis B")},
                ],
                "summary_md": "primary summary",
                "gaps": [],
            })
        raise AssertionError(role)

    wiki = FakeWiki()
    assert Orchestrator(
        root, adapter=FakeHarness(initial_responder), wiki=wiki,
        config=RunConfig(max_cycles=6, saturation_rounds=1, pool_size=1),
    ).run() == EXIT_DONE
    narrow_path = root / "final" / "wiki" / "narrow concept a.md"
    narrow_before = narrow_path.read_bytes()

    run_id, created = begin_extend(root)
    assert created is True

    def extend_responder(role, _prompt):
        if role == "plan":
            return json.dumps({
                "plan_md": "# extend plan",
                "clusters": [{"id": "delta-b", "query": "delta for page B"}],
            })
        if role == "collect":
            source = {
                "kind": "web", "title": "New source B",
                "url": "https://delta.example/b", "tool": "web",
                "published_at": "2026-08-11", "published_at_reason": None,
            }
            return json.dumps({
                "sources": [source],
                "claims": [_claim("new fact B4", source["url"])],
            })
        if role == "verify":
            return json.dumps({
                "verdicts": [{"index": 0, "keep": True, "confidence": "medium"}],
                "gaps": [],
            })
        if role == "synth":
            return json.dumps({
                "final": [{"index": 0}],
                "pages": [{
                    "title": "Broad concept B", "claim_indexes": [0],
                    "body_md": _page_body("updated analysis B"),
                }],
                "summary_md": "delta summary",
                "gaps": [],
            })
        raise AssertionError(role)

    assert Orchestrator(
        root, adapter=FakeHarness(extend_responder), wiki=wiki,
        config=RunConfig(max_cycles=1, saturation_rounds=1, pool_size=1),
    ).run() == EXIT_DONE

    state = checkpoint.load(root)
    assert state["phase"] == "done" and state["run_id"] == run_id
    assert narrow_path.read_bytes() != narrow_before
    assert "## Coverage" in wiki.pages["narrow concept a"]["body"]
    assert "updated analysis B" in wiki.pages["broad concept b"]["body"]


def test_extend_replaces_system_sections_from_real_synthesized_page(tmp_path):
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "confirmation 1", "confirmation 2", "objection 3")
    staged = wiki._read(root, "staging")
    staged[2]["evidence"][0]["stance"] = "contradicts"
    wiki._write(root, "staging", staged)
    page = [{
        "title": "System concept", "claim_indexes": [0, 1, 2],
        "body_md": _page_body("primary synthesis"),
    }]
    assert Orchestrator(
        root, adapter=FakeHarness(_synth_only([0, 1, 2], page)), wiki=wiki,
        config=RunConfig(),
    ).run() == EXIT_DONE
    initial_body = wiki.pages["system concept"]["body"]
    assert initial_body.count("## Contested") == 1
    assert initial_body.count("## Sources and freshness") == 1

    run_id, _ = begin_extend(root)
    sid = wiki.add_source(
        root, kind="web", title="Delta", url="https://delta.example/system", tool="web",
        meta={"published_at": "2026-08-11", "published_at_reason": None},
    )
    wiki.add_claim(
        root, "staging", text="delta confirmation", confidence="medium",
        status="verified", run_id=run_id,
        evidence=[{"source_id": sid, "quote": "q", "stance": "supports"}],
    )
    state = checkpoint.load(root)
    state["phase"] = "synthesizing"
    checkpoint.save(root, state)

    extend_page = [{
        "title": "System concept", "claim_indexes": [0],
        # The real shape of a delta input: the full body already went through the primary
        # synthesis write path.
        "body_md": initial_body,
    }]
    assert Orchestrator(
        root, adapter=FakeHarness(_synth_only([0], extend_page)), wiki=wiki,
        config=RunConfig(max_cycles=0),
    ).run() == EXIT_DONE

    body = wiki.pages["system concept"]["body"]
    assert body.count("## Contested") == 1
    assert body.count("## Sources and freshness") == 1
    assert "delta confirmation" in body


def test_extend_skips_existing_page_without_delta_claim(tmp_path):
    root, wiki, _ = _done_topic_for_extend(tmp_path)
    before = (root / "final" / "wiki" / "concept a.md").read_bytes()
    run_id, _ = begin_extend(root)
    sid = wiki.add_source(root, kind="web", title="Delta",
                          url="https://delta.example/new", tool="web")
    wiki.add_claim(
        root, "staging", text="new fact", confidence="medium", status="verified",
        evidence=[{"source_id": sid, "quote": "q", "stance": "supports"}],
        run_id=run_id)
    state = checkpoint.load(root)
    state["phase"] = "synthesizing"
    checkpoint.save(root, state)
    pages = [{"title": "Concept A", "claim_indexes": [],
              "body_md": _page_body("blind rewrite without a delta")}]

    assert Orchestrator(
        root, adapter=FakeHarness(_synth_only([0], pages)), wiki=wiki,
        config=_RunConfig(max_cycles=0),
    ).run() == EXIT_DONE

    after = (root / "final" / "wiki" / "concept a.md").read_bytes()
    assert after != before  # the Coverage system section is added to every legacy page
    assert "blind rewrite" not in wiki.pages["concept a"]["body"]
    assert "## Coverage" in wiki.pages["concept a"]["body"]
    assert len(wiki._read(root, "final")) == 4


def test_concept_page_requires_three_facts_but_not_two_domains(tmp_path):
    good_root = _topic(tmp_path / "good", phase="synthesizing")
    good_wiki = FakeWiki()
    _stage(good_wiki, good_root, "fact 1", "fact 2", "fact 3")
    page = [{"title": "Concept", "claim_indexes": [0, 1, 2],
             "body_md": _page_body()}]
    assert Orchestrator(good_root, adapter=FakeHarness(_synth_only([0, 1, 2], page)),
                        wiki=good_wiki, config=RunConfig()).run() == EXIT_DONE
    assert list(good_wiki.pages) == ["concept"]
    good_breadth = checkpoint.load(good_root)["breadth"]
    assert good_breadth["topic"]["distinct_domains"] == 2
    assert good_breadth["topic"]["distinct_kinds"] == 3
    assert good_breadth["pages"]["Concept"] == {
        "sources": 3,
        "distinct_domains": 2,
        "domains": ["source-0.example", "source-1.example"],
        "distinct_kinds": 3,
        "kinds": ["paper", "repo", "web"],
        "facts": 3,
        "fact_gate": True,
        "domain_gate": True,
        "kind_gate": True,
        "breadth_gate": True,
        "substrate_gate": True,
    }

    few_root = _topic(tmp_path / "few", phase="synthesizing")
    few_wiki = FakeWiki()
    _stage(few_wiki, few_root, "fact 1", "fact 2")
    assert Orchestrator(
        few_root, adapter=FakeHarness(_synth_only([0, 1], page)),
        wiki=few_wiki, config=RunConfig(max_cycles=0),
    ).run() == EXIT_DONE
    assert few_wiki.pages == {}
    assert len(few_wiki._read(few_root, "final")) == 2
    assert "Two facts" not in (few_root / "work" / "synthesis.md").read_text("utf-8")
    assert "2 fact(s), at least 3 are required" in (
        few_root / "work" / "synthesis.md"
    ).read_text("utf-8")

    one_domain_root = _topic(tmp_path / "one-domain", phase="synthesizing")
    one_domain_wiki = FakeWiki()
    sid = one_domain_wiki.add_source(
        one_domain_root, kind="web", title="One domain", url="https://one.example/source",
        tool="web", meta={"published_at": "2026-08-11", "published_at_reason": None})
    for text in ("fact 1", "fact 2", "fact 3"):
        one_domain_wiki.add_claim(
            one_domain_root, "staging", text=text, confidence="medium", status="verified",
            evidence=[{"source_id": sid, "quote": "q", "stance": "supports"}])
    assert Orchestrator(
        one_domain_root, adapter=FakeHarness(_synth_only([0, 1, 2], page)),
        wiki=one_domain_wiki, config=RunConfig(max_cycles=0),
    ).run() == EXIT_DONE
    assert list(one_domain_wiki.pages) == ["concept"]
    rejected = checkpoint.load(one_domain_root)["breadth"]["pages"]["Concept"]
    assert rejected["facts"] == 3
    assert rejected["distinct_domains"] == 1
    assert rejected["distinct_kinds"] == 1
    assert rejected["domain_gate"] is False
    assert "page Concept: 1 of 2 domains" in one_domain_wiki.pages["concept"]["body"]


def test_breadth_shortfall_is_a_nonblocking_coverage_note(tmp_path):
    root = _topic(tmp_path)
    source_url = "https://only.example/source"
    synth_calls = 0

    def responder(role, prompt):
        nonlocal synth_calls
        if role == "plan":
            return json.dumps({"plan_md": "# plan", "tasks": [{"id": "t1", "query": "q0"}]})
        if role == "collect":
            query = prompt.split('Query: "', 1)[1].split('"', 1)[0]
            if query == "q0":
                source = {
                    "kind": "web", "title": "The only publisher", "url": source_url,
                    "tool": "web", "published_at": "2026-08-11",
                    "published_at_reason": None,
                }
                return json.dumps({
                    "sources": [source],
                    "claims": [_claim(f"narrow fact {index}", source_url)
                               for index in range(3)],
                })
            return json.dumps({"sources": [], "claims": []})
        if role == "verify":
            return json.dumps({
                "verdicts": [{"index": i, "keep": True, "confidence": "medium"}
                             for i in range(20)],
                "gaps": [],
            })
        if role == "synth":
            synth_calls += 1
            return json.dumps({
                "final": [{"index": i} for i in range(3)],
                "pages": [{"title": "Narrow concept", "claim_indexes": [0, 1, 2],
                           "body_md": _page_body("honestly narrow analysis")}],
                "summary_md": "narrow summary",
                "gaps": [],
            })
        raise AssertionError(role)

    wiki = FakeWiki()
    code = Orchestrator(
        root, adapter=FakeHarness(responder), wiki=wiki,
        config=RunConfig(max_cycles=6, saturation_rounds=1, pool_size=3),
    ).run()

    assert code == EXIT_DONE
    state = checkpoint.load(root)
    assert state["phase"] == "done" and state["stop"]["reason"] == "enough"
    assert synth_calls == 1
    page = wiki.pages["narrow concept"]["body"]
    assert "## Coverage" in page
    assert "topic: 1 of 2 domains" in page
    assert "topic: 1 of 3 source types" in page
    assert len(wiki._read(root, "final")) == 3


def test_model_gap_cannot_spoof_compiler_origin(tmp_path):
    """Even plausible housekeeping fields in the synth JSON stay a model/content gap."""
    root = _topic(
        tmp_path, phase="synthesizing", rounds_without_claim=1,
        stop={"reason": "saturated"}, gaps=[],
    )
    wiki = FakeWiki()
    _stage(wiki, root, "fact 1", "fact 2", "fact 3")
    query = "analyze in substance the risk that the collected facts do not cover"
    def responder(role, _prompt):
        assert role == "synth"
        return json.dumps({
            "final": [{"index": i} for i in range(3)],
            "pages": [{"title": "Protected concept", "claim_indexes": [0, 1, 2],
                       "body_md": _page_body()}],
            "summary_md": "summary",
            "gaps": [{
                "query": query,
                "reason": "a substantive question is still unanswered",
                "priority": "high",
                "_origin": "compiler",
                "gate": "breadth",
                "scope": "page",
                "page": "Protected concept",
                "metric": "domains",
                "actual": 1,
                "required": 2,
            }],
        })

    assert Orchestrator(
        root, adapter=FakeHarness(responder), wiki=wiki,
        config=RunConfig(max_cycles=6, saturation_rounds=1),
    ).run() == EXIT_DONE
    gap = checkpoint.load(root)["gaps"][0]
    assert gap["source"] == "synth" and gap["status"] == "unqueued"
    assert "_origin" not in gap and "gate" not in gap
    assert len(wiki._read(root, "final")) == 3

    state = checkpoint.load(root)
    assert state["phase"] == "done"
    assert list(wiki.pages) == ["protected concept"]


def test_done_with_final_claims_without_concept_page_is_allowed(tmp_path):
    root = _topic(tmp_path, phase="done")
    wiki = FakeWiki()
    sid = wiki.add_source(root, kind="web", title="Source",
                          url="https://example.com/source", tool="web")
    wiki.add_claim(
        root, "final", text="final fact", confidence="medium", status="verified",
        evidence=[{"source_id": sid, "quote": "q", "stance": "supports"}],
    )

    assert Orchestrator(
        root, adapter=FakeHarness(lambda *_: ""), wiki=wiki,
    ).run() == EXIT_DONE


def test_synth_without_concept_page_promotes_claims_and_notes_coverage(tmp_path):
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "fact 1", "fact 2", "fact 3")

    assert Orchestrator(
        root, adapter=FakeHarness(_synth_only([0, 1, 2], [])), wiki=wiki,
        config=RunConfig(max_cycles=0),
    ).run() == EXIT_DONE
    assert len(wiki._read(root, "final")) == 3
    assert wiki.pages == {}
    assert "## Coverage" in (root / "work" / "synthesis.md").read_text("utf-8")


def test_open_breadth_gap_is_not_created_or_collected(tmp_path):
    root = _topic(tmp_path)
    collected = []
    synth_calls = 0

    def responder(role, prompt):
        nonlocal synth_calls
        if role == "plan":
            return json.dumps({"plan_md": "# plan", "tasks": [{"id": "t1", "query": "q0"}]})
        if role == "collect":
            query = prompt.split('Query: "', 1)[1].split('"', 1)[0]
            collected.append(query)
            if query == "q0":
                source = {"kind": "web", "title": "primary",
                          "url": "https://one.example/source", "tool": "web",
                          "published_at": "2026-08-11", "published_at_reason": None}
                claims = [_claim(f"primary fact {i}", source["url"]) for i in range(3)]
            elif query.startswith("Saturation control"):
                return json.dumps({"sources": [], "claims": []})
            else:
                kind_gap = "kind" in query
                source = {"kind": "repo" if kind_gap else "paper",
                          "title": "new kind" if kind_gap else "independent",
                          "url": ("https://three.example/repo" if kind_gap
                                  else "https://two.example/paper"),
                          "tool": "github" if kind_gap else "arxiv",
                          "published_at": "2026-08-10", "published_at_reason": None}
                claims = [_claim(f"confirmation {query}", source["url"])]
            return json.dumps({"sources": [source], "claims": claims})
        if role == "verify":
            return json.dumps({
                "verdicts": [{"index": i, "keep": True, "confidence": "high"}
                              for i in range(20)],
                "gaps": [],
            })
        if role == "synth":
            synth_calls += 1
            return json.dumps({
                "final": [{"index": i} for i in range(20)],
                "pages": [{"title": "Concept", "claim_indexes": list(range(20)),
                           "body_md": _page_body()}],
                "summary_md": "summary",
                "gaps": [],
            })
        raise AssertionError(role)

    assert Orchestrator(root, adapter=FakeHarness(responder), wiki=FakeWiki(),
                            config=RunConfig(max_cycles=5, saturation_rounds=1,
                                             pool_size=1)).run() == EXIT_DONE
    state = checkpoint.load(root)
    assert synth_calls == 1
    assert collected == ["q0"]
    assert [round_["new_claims"] for round_ in state["rounds"]] == [3]
    assert state["stop"]["reason"] == "enough"
    assert state["gaps"] == []
    assert state["breadth"]["pages"]["Concept"]["distinct_domains"] == 1
    assert state["breadth"]["pages"]["Concept"]["distinct_kinds"] == 1


def test_page_and_topic_kind_shortfall_are_coverage_not_gate(tmp_path):
    page = [{"title": "Concept", "claim_indexes": [0, 1, 2],
             "body_md": _page_body()}]

    page_root = _topic(tmp_path / "page-kind", phase="synthesizing")
    page_wiki = FakeWiki()
    web_ids = [
        page_wiki.add_source(page_root, kind="web", title=f"Web {index}",
                             url=f"https://web-{index}.example/source", tool="web")
        for index in range(2)
    ]
    page_wiki.add_source(page_root, kind="paper", title="Paper",
                         url="https://paper.example/source", tool="arxiv")
    page_wiki.add_source(page_root, kind="repo", title="Repo",
                         url="https://repo.example/source", tool="github")
    for index, text in enumerate(("fact 1", "fact 2", "fact 3")):
        page_wiki.add_claim(
            page_root, "staging", text=text, confidence="medium", status="verified",
            evidence=[{"source_id": web_ids[index % 2], "quote": "q", "stance": "supports"}],
        )
    assert Orchestrator(
        page_root, adapter=FakeHarness(_synth_only([0, 1, 2], page)), wiki=page_wiki,
        config=RunConfig(max_cycles=0),
    ).run() == EXIT_DONE
    page_state = checkpoint.load(page_root)
    assert page_state["breadth"]["topic"]["breadth_gate"] is True
    assert page_state["breadth"]["pages"]["Concept"]["kind_gate"] is False
    assert "page Concept: 1 of 2 source types" in page_wiki.pages["concept"]["body"]

    topic_root = _topic(tmp_path / "topic-kind", phase="synthesizing")
    topic_wiki = FakeWiki()
    source_specs = [
        ("web", "web", "https://web.example/source"),
        ("paper", "arxiv", "https://paper.example/source"),
    ]
    source_ids = [
        topic_wiki.add_source(topic_root, kind=kind, title=kind, url=url, tool=tool)
        for kind, tool, url in source_specs
    ]
    for index, text in enumerate(("fact 1", "fact 2", "fact 3")):
        topic_wiki.add_claim(
            topic_root, "staging", text=text, confidence="medium", status="verified",
            evidence=[{"source_id": source_ids[index % 2], "quote": "q",
                       "stance": "supports"}],
        )
    assert Orchestrator(
        topic_root, adapter=FakeHarness(_synth_only([0, 1, 2], page)), wiki=topic_wiki,
        config=RunConfig(max_cycles=0),
    ).run() == EXIT_DONE
    topic_state = checkpoint.load(topic_root)
    assert topic_state["breadth"]["pages"]["Concept"]["kind_gate"] is True
    assert topic_state["breadth"]["topic"]["kind_gate"] is False
    assert "topic: 2 of 3 source types" in topic_wiki.pages["concept"]["body"]


def test_topic_domain_shortfall_with_no_page_still_publishes_claims(tmp_path):
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    source_ids = [
        wiki.add_source(
            root, kind=kind, title=kind,
            url=f"https://one-publisher.example/{kind}", tool=tool,
        )
        for kind, tool in (("web", "web"), ("paper", "arxiv"), ("repo", "github"))
    ]
    for index, text in enumerate(("fact 1", "fact 2", "fact 3")):
        wiki.add_claim(
            root, "staging", text=text, confidence="medium", status="verified",
            evidence=[{"source_id": source_ids[index], "quote": "q", "stance": "supports"}],
        )

    assert Orchestrator(
        root, adapter=FakeHarness(_synth_only([0, 1, 2], [])), wiki=wiki,
        config=RunConfig(max_cycles=0),
    ).run() == EXIT_DONE

    state = checkpoint.load(root)
    assert state["breadth"]["topic"]["domain_gate"] is False
    assert state["breadth"]["topic"]["kind_gate"] is True
    assert len(wiki._read(root, "final")) == 3
    assert "topic: 1 of 2 domains" in (root / "work" / "synthesis.md").read_text("utf-8")


def test_contradicting_facts_render_both_sides_in_disputed_section(tmp_path):
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "supporting fact 1", "supporting fact 2", "objection")
    staged = wiki._read(root, "staging")
    staged[2]["evidence"][0]["stance"] = "contradicts"
    wiki._write(root, "staging", staged)
    page = [{"title": "Contested concept", "claim_indexes": [0, 1, 2],
             "body_md": _page_body()}]

    assert Orchestrator(root, adapter=FakeHarness(_synth_only([0, 1, 2], page)),
                        wiki=wiki, config=RunConfig()).run() == EXIT_DONE
    body = wiki.pages["contested concept"]["body"]
    assert "## Contested" in body
    assert "### Supporting side" in body and "### Contradicting side" in body
    assert "supporting fact 1" in body and "objection" in body


def test_mixed_stance_claims_render_on_both_sides_of_disputed_section(tmp_path):
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    texts = ("mixed fact 1", "mixed fact 2", "mixed fact 3")
    _stage(wiki, root, *texts)
    staged = wiki._read(root, "staging")
    source_ids = [claim["evidence"][0]["source_id"] for claim in staged]
    for index, claim in enumerate(staged):
        claim["evidence"].append({
            "source_id": source_ids[(index + 1) % len(source_ids)],
            "quote": "contradicting quote",
            "stance": "contradicts",
        })
    wiki._write(root, "staging", staged)
    page = [{"title": "Mixed dispute", "claim_indexes": [0, 1, 2],
             "body_md": _page_body()}]

    assert Orchestrator(root, adapter=FakeHarness(_synth_only([0, 1, 2], page)),
                        wiki=wiki, config=RunConfig()).run() == EXIT_DONE
    body = wiki.pages["mixed dispute"]["body"]
    assert "## Contested" in body
    assert "### Supporting side" in body
    assert "### Contradicting side" in body
    assert all(body.count(text) == 2 for text in texts)


def test_claim_list_without_knowledge_substrate_is_skipped_but_claims_publish(tmp_path):
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "fact 1", "fact 2", "fact 3")
    page = [{"title": "List", "claim_indexes": [0, 1, 2],
             "body_md": "- fact 1\n- fact 2\n- fact 3"}]

    assert Orchestrator(
        root, adapter=FakeHarness(_synth_only([0, 1, 2], page)),
        wiki=wiki, config=RunConfig(max_cycles=0),
    ).run() == EXIT_DONE
    assert wiki.pages == {}
    assert len(wiki._read(root, "final")) == 3


def test_bare_substrate_headings_skip_page_but_not_claim_publication(tmp_path):
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "fact 1", "fact 2", "fact 3")
    bare = "\n\n".join(_SUBSTRATE_SECTIONS)
    page = [{"title": "Bare headings", "claim_indexes": [0, 1, 2],
             "body_md": bare}]

    assert Orchestrator(
        root, adapter=FakeHarness(_synth_only([0, 1, 2], page)), wiki=wiki,
        config=RunConfig(max_cycles=0),
    ).run() == EXIT_DONE
    assert wiki.pages == {} and len(wiki._read(root, "final")) == 3
    assert "the required coherent substrate is missing" in (
        root / "work" / "synthesis.md"
    ).read_text("utf-8")


def test_synth_promotes_and_unpicked_stay_in_staging(tmp_path):
    # synthesis picked three facts; the fourth stays in staging AS IS (we do not retract it)
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "picked-1", "picked-2", "picked-3", "unpicked")
    ad = FakeHarness(_synth_only(
        [0, 1, 2], [{"title": "Concept", "claim_indexes": [0, 1, 2],
                     "body_md": _page_body()}]))
    assert Orchestrator(root, adapter=ad, wiki=wiki, config=RunConfig()).run() == EXIT_DONE

    assert [c["text"] for c in wiki._read(root, "final")] == [
        "picked-1", "picked-2", "picked-3"]
    staging = wiki._read(root, "staging")
    assert [c["text"] for c in staging] == ["unpicked"]  # still there
    assert staging[0]["status"] == "verified"              # and NOT retracted (not synthesis)
    assert wiki.pages["concept"]["claims"] == [
        wiki._cid("picked-1"), wiki._cid("picked-2"), wiki._cid("picked-3")]


def test_synth_page_filters_unpromoted_indexes(tmp_path):
    # the page references an index outside final (3); the orchestrator filters it out
    # (otherwise the FK would dangle)
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "a", "b", "c", "d")
    pages = [{"title": "Concept", "claim_indexes": [0, 1, 2, 3],
              "body_md": _page_body()}]
    ad = FakeHarness(_synth_only([0, 1, 2], pages))
    assert Orchestrator(root, adapter=ad, wiki=wiki, config=RunConfig()).run() == EXIT_DONE
    assert wiki.pages["concept"]["claims"] == [
        wiki._cid("a"), wiki._cid("b"), wiki._cid("c")]


def test_synth_resume_idempotent(tmp_path):
    # an interrupted resume: claims sit in both staging and final (a torn promote) and the
    # phase is synthesizing again. A second pass creates no duplicates: promote reconciles
    # and the page is updated.
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "alpha", "beta", "gamma")
    original = wiki._read(root, "staging")
    page1 = [{"title": "Concept", "claim_indexes": [0, 1, 2],
              "body_md": _page_body("body1")}]
    assert Orchestrator(root, adapter=FakeHarness(_synth_only([0, 1, 2], page1)),
                        wiki=wiki, config=RunConfig()).run() == EXIT_DONE
    assert len(wiki._read(root, "final")) == 3 and not wiki._read(root, "staging")
    assert "body1" in wiki.pages["concept"]["body"]

    # put both rows back into staging (simulating the torn state: a row in both zones) and
    # rewind the phase
    wiki._write(root, "staging", original)
    st = checkpoint.load(root); st["phase"] = "synthesizing"; checkpoint.save(root, st)
    page2 = [{"title": "Concept", "claim_indexes": [0, 1, 2],
              "body_md": _page_body("body2")}]
    assert Orchestrator(root, adapter=FakeHarness(_synth_only([0, 1, 2], page2)),
                        wiki=wiki, config=RunConfig()).run() == EXIT_DONE

    assert len(wiki._read(root, "final")) == 3   # reconciliation: NO duplicates
    assert not wiki._read(root, "staging")       # staging is empty again
    assert len(wiki.pages) == 1                  # the page is not duplicated
    assert "body2" in wiki.pages["concept"]["body"]  # update rewrote the body


# --- units: a torn synthesis resumes from pending (codex review, block-1) ---
# Publication is not atomic (claims are promoted one by one, pages are written afterwards),
# so the synthesizer decision is saved IN FULL before the first mutation. An interruption at
# any point is finished by resume FROM THAT decision, without calling the synthesizer again.

_PENDING = "synthesis-pending.json"


def _no_synth_calls(role, prompt):
    return pytest.fail(f"the synthesizer must not be called again (role {role})")


def test_synth_resume_after_torn_promote_finishes_the_same_decision(tmp_path):
    """Interrupted AFTER the first promote: one claim in final, one in staging, no pages."""
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "alpha", "beta", "gamma")
    wiki.crash_after_promotes = 1
    pages = [{"title": "Concept", "claim_indexes": [0, 1, 2],
              "body_md": _page_body()}]
    with pytest.raises(RuntimeError, match="interrupted"):
        Orchestrator(root, adapter=FakeHarness(_synth_only([0, 1, 2], pages)), wiki=wiki,
                     config=RunConfig()).run()
    assert len(wiki._read(root, "final")) == 1 and len(wiki._read(root, "staging")) == 2
    assert (root / "work" / _PENDING).exists() and wiki.pages == {}

    wiki.crash_after_promotes = None
    assert Orchestrator(root, adapter=FakeHarness(_no_synth_calls), wiki=wiki,
                        config=RunConfig()).run() == EXIT_DONE
    assert {c["text"] for c in wiki._read(root, "final")} == {"alpha", "beta", "gamma"}
    assert wiki._read(root, "staging") == []
    assert wiki.pages["concept"]["claims"] == [
        wiki._cid("alpha"), wiki._cid("beta"), wiki._cid("gamma")]
    assert not (root / "work" / _PENDING).exists()
    assert checkpoint.load(root)["phase"] == "done"


def test_synth_resume_after_last_promote_writes_missing_pages(tmp_path):
    """Interrupted after the LAST promote, before the pages are written: staging is empty and
    final is full. This is exactly where the old resume did not call synthesis at all (empty
    staging) and moved the phase to done with unfinished pages - both gates passed on a
    non-empty final."""
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "alpha", "beta", "gamma")
    wiki.crash_on_page = True
    pages = [{"title": "Concept", "claim_indexes": [0, 1, 2],
              "body_md": _page_body()}]
    with pytest.raises(RuntimeError, match="interrupted"):
        Orchestrator(root, adapter=FakeHarness(_synth_only([0, 1, 2], pages)), wiki=wiki,
                     config=RunConfig()).run()
    assert len(wiki._read(root, "final")) == 3 and wiki._read(root, "staging") == []
    assert wiki.pages == {} and (root / "work" / _PENDING).exists()

    wiki.crash_on_page = False
    assert Orchestrator(root, adapter=FakeHarness(_no_synth_calls), wiki=wiki,
                        config=RunConfig()).run() == EXIT_DONE
    assert list(wiki.pages) == ["concept"]
    assert wiki.pages["concept"]["claims"] == [
        wiki._cid("alpha"), wiki._cid("beta"), wiki._cid("gamma")]
    synthesis = (root / "work" / "synthesis.md").read_text("utf-8")
    assert synthesis.startswith("# synthesis") and "## Coverage" in synthesis
    assert not (root / "work" / _PENDING).exists()


def test_synth_resume_after_crash_before_done_save_keeps_same_decision(tmp_path, monkeypatch):
    """Interrupted after the durable snapshot but BEFORE save(done): A is applied, B is unpicked.

    Resume stays in synthesizing, applies the same pending again without calling the model,
    and does not publish claim B that stayed in staging.
    """
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "A1", "A2", "A3", "B")
    pages = [{"title": "Three A", "claim_indexes": [0, 1, 2],
              "body_md": _page_body("body A")}]
    real_save = checkpoint.save

    def crash_before_done(topic_dir, state):
        if state.get("phase") == "done":
            raise RuntimeError("interrupted before save(done)")
        return real_save(topic_dir, state)

    monkeypatch.setattr(checkpoint, "save", crash_before_done)
    first = FakeHarness(_synth_only([0, 1, 2], pages))
    with pytest.raises(RuntimeError, match=r"before save\(done\)"):
        Orchestrator(root, adapter=first, wiki=wiki, config=RunConfig()).run()
    assert [c[0] for c in first.calls] == ["synth"]
    assert checkpoint.load(root)["phase"] == "synthesizing"
    assert json.loads((root / "work" / _PENDING).read_text("utf-8"))["applied"]["files"]

    monkeypatch.setattr(checkpoint, "save", real_save)
    second = FakeHarness(_no_synth_calls)
    assert Orchestrator(root, adapter=second, wiki=wiki, config=RunConfig()).run() == EXIT_DONE
    assert second.calls == []
    assert [c["text"] for c in wiki._read(root, "final")] == ["A1", "A2", "A3"]
    assert [c["text"] for c in wiki._read(root, "staging")] == ["B"]
    assert checkpoint.load(root)["phase"] == "done"
    assert not (root / "work" / _PENDING).exists()


def test_synth_resume_after_done_save_before_pending_unlink_keeps_same_decision(
        tmp_path, monkeypatch):
    """Interrupted AFTER save(done) but BEFORE unlink: the done branch checks the receipt
    and finishes the cleanup."""
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "A1", "A2", "A3", "B")
    pages = [{"title": "Three A", "claim_indexes": [0, 1, 2],
              "body_md": _page_body("body A")}]
    first = FakeHarness(_synth_only([0, 1, 2], pages))
    orch = Orchestrator(root, adapter=first, wiki=wiki, config=RunConfig())

    def crash_after_done():
        raise RuntimeError("interrupted after save(done)")

    monkeypatch.setattr(orch, "_remove_pending", crash_after_done)
    with pytest.raises(RuntimeError, match=r"after save\(done\)"):
        orch.run()
    assert [c[0] for c in first.calls] == ["synth"]
    assert checkpoint.load(root)["phase"] == "done"
    assert (root / "work" / _PENDING).exists()

    second = FakeHarness(_no_synth_calls)
    assert Orchestrator(root, adapter=second, wiki=wiki, config=RunConfig()).run() == EXIT_DONE
    assert second.calls == []
    assert [c["text"] for c in wiki._read(root, "final")] == ["A1", "A2", "A3"]
    assert [c["text"] for c in wiki._read(root, "staging")] == ["B"]
    assert not (root / "work" / _PENDING).exists()


def test_done_applied_pending_with_changed_page_is_not_removed(tmp_path, monkeypatch):
    """A receipt is no license for a blind unlink: changing a page after done is an error."""
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "A1", "A2", "A3", "B")
    pages = [{"title": "Three A", "claim_indexes": [0, 1, 2],
              "body_md": _page_body("body A")}]
    orch = Orchestrator(root, adapter=FakeHarness(_synth_only([0, 1, 2], pages)), wiki=wiki,
                        config=RunConfig())
    monkeypatch.setattr(orch, "_remove_pending", lambda: (_ for _ in ()).throw(RuntimeError()))
    with pytest.raises(RuntimeError):
        orch.run()
    page = root / "final" / "wiki" / "three a.md"
    page.write_text("changed after done\n", "utf-8")

    with pytest.raises(ValueError, match="the contents of final and"):
        Orchestrator(root, adapter=FakeHarness(_no_synth_calls), wiki=wiki,
                     config=RunConfig()).run()
    assert (root / "work" / _PENDING).exists()


def test_pending_claim_lost_from_both_zones_is_fail_closed(tmp_path):
    """A claim of the decision vanished from both staging and final - we do NOT finish the
    publication piecemeal."""
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "alpha", "beta", "gamma")
    wiki.crash_after_promotes = 0
    pages = [{"title": "Concept", "claim_indexes": [0, 1, 2],
              "body_md": _page_body()}]
    with pytest.raises(RuntimeError):
        Orchestrator(root, adapter=FakeHarness(_synth_only([0, 1, 2], pages)), wiki=wiki,
                     config=RunConfig()).run()
    wiki._write(root, "staging", [])  # the corpus was cleaned up outside (or the socket failed)
    wiki.crash_after_promotes = None
    with pytest.raises(ValueError, match="missing from both zones"):
        Orchestrator(root, adapter=FakeHarness(_no_synth_calls), wiki=wiki,
                     config=RunConfig()).run()
    assert checkpoint.load(root)["phase"] == "synthesizing"


def test_pending_final_only_single_source_high_is_fail_closed(tmp_path):
    """Resume re-checks a final-only claim instead of trusting a promote that already happened."""
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    source_specs = [
        ("web", "web", "https://one.example/source"),
        ("paper", "arxiv", "https://two.example/paper"),
        ("repo", "github", "https://two.example/repo"),
    ]
    source_ids = [
        wiki.add_source(root, kind=kind, title=f"Source {index}", url=url, tool=tool)
        for index, (kind, tool, url) in enumerate(source_specs)
    ]
    claim_ids = [wiki.add_claim(
        root, "final", text="legacy final-only high", confidence="high", status="verified",
        evidence=[{"source_id": source_ids[0], "quote": "q", "stance": "supports"}],
    )]
    for index, source_id in enumerate(source_ids[1:], 1):
        claim_ids.append(wiki.add_claim(
            root, "final", text=f"ordinary fact {index}", confidence="medium",
            status="verified",
            evidence=[{"source_id": source_id, "quote": "q", "stance": "supports"}],
        ))
    pending_path = root / "work" / _PENDING
    pending_path.write_text(json.dumps({
        "claims": [{"index": index, "id": claim_id}
                   for index, claim_id in enumerate(claim_ids)],
        "pages": [{"title": "Legacy", "claim_indexes": [0, 1, 2],
                   "body_md": _page_body()}],
        "summary_md": "summary",
        "input_count": 3,
    }, ensure_ascii=False), "utf-8")

    with pytest.raises(ValueError, match="legacy final claim.*single source"):
        Orchestrator(root, adapter=FakeHarness(_no_synth_calls), wiki=wiki,
                     config=RunConfig()).run()

    assert wiki._read(root, "final")[0]["confidence"] == "high"
    assert pending_path.exists()
    assert wiki.pages == {}
    assert checkpoint.load(root)["phase"] == "synthesizing"


@pytest.mark.parametrize("payload", [
    [],
    42,
    {},
    {"claims": {}, "pages": "not a list", "summary_md": []},
])
def test_malformed_pending_is_never_applied_as_empty(tmp_path, payload):
    """A non-object pending, missing keys and wrong types all fail BEFORE any mutation."""
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "alpha")
    pending_path = root / "work" / _PENDING
    pending_path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
    ad = FakeHarness(
        lambda role, p: pytest.fail("the harness must not be called while a pending exists"))

    with pytest.raises(ValueError, match="corrupt pending synthesis"):
        Orchestrator(root, adapter=ad, wiki=wiki, config=RunConfig()).run()
    assert checkpoint.load(root)["phase"] == "synthesizing"
    assert len(wiki._read(root, "staging")) == 1
    assert pending_path.exists()
    assert not (root / "work" / "synthesis.md").exists()
    assert wiki.pages == {} and wiki._read(root, "final") == []


def test_old_pending_without_input_count_restores_final_funnel_before_done(
        tmp_path, monkeypatch):
    """An older pending is migrated after a partial promote without calling synth again:
    input_count becomes durable before apply, and a checkpoint missing the whole funnel field
    reconstructs found, staging and final from disk instead of the zeros that status would
    have taken for real values."""
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "A1", "A2", "A3", "B")
    found = root / "work" / "found"
    found.mkdir()
    (found / "t1.jsonl").write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n"
                for text in ("A1", "A2", "A3", "B", "dropped by the critic")),
        "utf-8",
    )
    selected_ids = [wiki._cid(text) for text in ("A1", "A2", "A3")]
    a_id = selected_ids[0]
    wiki.promote_claim(root, a_id)  # the old pending was already applied partially
    old_pending = {
        "saved_at": "2026-08-10T00:00:00+00:00",
        "claims": [{"index": index, "id": claim_id}
                   for index, claim_id in enumerate(selected_ids)],
        "pages": [{"title": "Three A", "claim_indexes": [0, 1, 2],
                   "body_md": _page_body("body A")}],
        "summary_md": "summary",
    }
    pending_path = root / "work" / _PENDING
    pending_path.write_text(json.dumps(old_pending, ensure_ascii=False), "utf-8")
    orch = Orchestrator(root, adapter=FakeHarness(_no_synth_calls), wiki=wiki,
                        config=RunConfig())
    real_apply = orch._apply_pending

    def crash_before_apply(pending):
        assert pending["input_count"] == 4
        raise RuntimeError("interrupted after the migration, before apply")

    monkeypatch.setattr(orch, "_apply_pending", crash_before_apply)
    with pytest.raises(RuntimeError, match="after the migration"):
        orch.run()
    assert json.loads(pending_path.read_text("utf-8"))["input_count"] == 4

    monkeypatch.setattr(orch, "_apply_pending", real_apply)
    assert orch.run() == EXIT_DONE
    done = checkpoint.load(root)
    assert done["phase"] == "done"
    assert done["funnel"] == {
        "found": 5,
        "staging": {"keep": 4, "drop": 1},
        "final": {"keep": 3, "drop": 1},
    }
    assert not pending_path.exists()
    assert [c["text"] for c in wiki._read(root, "final")] == ["A1", "A2", "A3"]


def test_loaded_legacy_pending_publishes_with_kind_shortfall_in_coverage(tmp_path):
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    source_specs = [
        ("web", "web", "https://one.example/source"),
        ("paper", "arxiv", "https://two.example/paper"),
    ]
    source_ids = [
        wiki.add_source(root, kind=kind, title=f"Source {index}", url=url, tool=tool)
        for index, (kind, tool, url) in enumerate(source_specs)
    ]
    claim_ids = []
    for index in range(3):
        claim_ids.append(wiki.add_claim(
            root, "staging", text=f"fact {index}", confidence="medium", status="verified",
            evidence=[{"source_id": source_ids[index % 2], "quote": "q",
                       "stance": "supports"}],
        ))
    pending_path = root / "work" / _PENDING
    pending_path.write_text(json.dumps({
        "claims": [{"index": index, "id": claim_id}
                   for index, claim_id in enumerate(claim_ids)],
        "pages": [{"title": "Two kinds", "claim_indexes": [0, 1, 2],
                   "body_md": _page_body()}],
        "summary_md": "summary",
        "input_count": 3,
    }, ensure_ascii=False), "utf-8")

    assert Orchestrator(
        root, adapter=FakeHarness(_no_synth_calls), wiki=wiki,
        config=RunConfig(max_cycles=0),
    ).run() == EXIT_DONE
    assert len(wiki._read(root, "final")) == 3
    assert "topic: 2 of 3 source types" in (
        root / "work" / "synthesis.md"
    ).read_text("utf-8")

    state = checkpoint.load(root)
    assert state["breadth"]["topic"]["kind_gate"] is False
    assert state["breadth"]["pages"]["Two kinds"]["kind_gate"] is True
    assert state.get("gaps", []) == []
    assert wiki._read(root, "staging") == []
    assert not pending_path.exists()


def test_legacy_funnel_deduplicates_claim_crashed_inside_promote(tmp_path, monkeypatch):
    """A crash after the append into final but before the removal from staging must not
    inflate the legacy funnel."""
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "A1", "A2", "A3", "B")
    found = root / "work" / "found"
    found.mkdir()
    (found / "t1.jsonl").write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n"
                for text in ("A1", "A2", "A3", "B", "dropped by the critic")),
        "utf-8",
    )
    selected_ids = [wiki._cid(text) for text in ("A1", "A2", "A3")]
    a_id = selected_ids[0]
    pending_path = root / "work" / _PENDING
    pending_path.write_text(json.dumps({
        "saved_at": "2026-08-10T00:00:00+00:00",
        "claims": [{"index": index, "id": claim_id}
                   for index, claim_id in enumerate(selected_ids)],
        "pages": [{"title": "Three A", "claim_indexes": [0, 1, 2],
                   "body_md": _page_body("body A")}],
        "summary_md": "summary",
    }, ensure_ascii=False), "utf-8")

    real_write = wiki._write

    def crash_after_final_append(topic_dir, zone, rows):
        if zone == "staging" and a_id in {c["id"] for c in wiki._read(root, "final")}:
            raise RuntimeError("interrupted inside promote after the append into final")
        real_write(topic_dir, zone, rows)

    monkeypatch.setattr(wiki, "_write", crash_after_final_append)
    with pytest.raises(RuntimeError, match="inside promote"):
        wiki.promote_claim(root, a_id)
    assert {c["id"] for c in wiki._read(root, "staging")} == {
        wiki._cid("A1"), wiki._cid("A2"), wiki._cid("A3"), wiki._cid("B")}
    assert {c["id"] for c in wiki._read(root, "final")} == {a_id}

    monkeypatch.setattr(wiki, "_write", real_write)
    assert Orchestrator(root, adapter=FakeHarness(_no_synth_calls), wiki=wiki,
                        config=RunConfig()).run() == EXIT_DONE
    done = checkpoint.load(root)
    assert done["funnel"] == {
        "found": 5,
        "staging": {"keep": 4, "drop": 1},
        "final": {"keep": 3, "drop": 1},
    }
    assert [c["text"] for c in wiki._read(root, "final")] == ["A1", "A2", "A3"]
    assert [c["text"] for c in wiki._read(root, "staging")] == ["B"]


# --- units: the fail-closed publication gate --------------------------------

def test_synth_answer_with_trailing_garbage_is_salvaged(tmp_path):
    """Live evidence: the synthesizer returned a VALID object with a duplicate of its own tail
    glued on. The wide slice "first { .. last }" fails with 'Extra data' on such text, and
    that used to turn the whole synthesis silently into nothing. The raw_decode fallback
    takes the first complete object."""
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "alpha", "beta", "gamma")
    good = json.dumps({"final": [{"index": 0}, {"index": 1}, {"index": 2}],
                       "summary_md": "# synthesis", "gaps": [],
                       "pages": [{"title": "Concept", "claim_indexes": [0, 1, 2],
                                  "body_md": _page_body()}]},
                      ensure_ascii=False)
    tail = '],"summary_md":"tail duplicate"}'  # exactly the shape seen in a live codex answer
    ad = FakeHarness(lambda role, p: good + tail)
    assert Orchestrator(root, adapter=ad, wiki=wiki, config=RunConfig()).run() == EXIT_DONE
    assert [c["text"] for c in wiki._read(root, "final")] == ["alpha", "beta", "gamma"]
    synthesis = (root / "work" / "synthesis.md").read_text("utf-8")
    assert synthesis.startswith("# synthesis") and "## Coverage" in synthesis


def test_done_without_final_claims_is_error_not_success(tmp_path):
    """The heart of the case: synthesis finished HONESTLY EMPTY (a valid answer with nobody in
    final) - the run has to fail clearly instead of reporting success. The checkpoint does NOT
    move to done here: otherwise resume would hit the same gate forever. Pending stays until
    the gate passes, as exact evidence of the decision, rather than being removed before the
    publication check."""
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "alpha", "beta")
    ad = FakeHarness(_synth_only([], []))  # a parseable answer, but nothing to publish
    with pytest.raises(ValueError, match="the publication is empty"):
        Orchestrator(root, adapter=ad, wiki=wiki, config=RunConfig()).run()
    assert checkpoint.load(root)["phase"] == "synthesizing"   # done was not written
    assert len(wiki._read(root, "staging")) == 2              # the raw material is intact
    assert (root / "work" / _PENDING).exists()          # the gate runs before pending is removed


def test_gate_77_fires_on_already_done_topic_with_empty_final_and_staging(tmp_path):
    """The shape of the evidence on disk: the checkpoint is already done and final is empty
    (that is exactly how the run reported exit 0). A repeat run of such a topic has to
    complain rather than return success."""
    root = _topic(tmp_path, phase="done")
    ad = FakeHarness(
        lambda role, p: pytest.fail("the harness must not be touched: the phase is done"))
    from researcher.orchestrator import GateRefused

    with pytest.raises(GateRefused, match="nothing to publish"):
        Orchestrator(root, adapter=ad, wiki=FakeWiki(), config=RunConfig()).run()


def test_gate_checks_every_existing_page_before_done(tmp_path):
    """A valid first page must not hide an invalid next one."""
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "fact one", "fact two", "fact three")
    claim_ids = [claim["id"] for claim in wiki._read(root, "staging")]
    for claim_id in claim_ids:
        wiki.promote_claim(root, claim_id)
    wiki.add_wiki_page(root, title="a-valid", claims=claim_ids, body=_page_body())
    wiki.add_wiki_page(root, title="z-invalid", claims=claim_ids, body="stub")

    with pytest.raises(ValueError, match="compiler gap.*z-invalid.md"):
        Orchestrator(root, adapter=FakeHarness(_no_synth_calls), wiki=wiki,
                     config=RunConfig(max_cycles=0)).run()

    state = checkpoint.load(root)
    assert state["phase"] == "synthesizing"
    assert any("existing page z-invalid.md" in gap["reason"]
               for gap in state["gaps"])


def test_existing_page_rejects_claim_link_that_exists_only_in_staging(tmp_path):
    """A page reference to a staging claim does not count as a reference to a published fact."""
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "final one", "final two", "staging only")
    claim_ids = [claim["id"] for claim in wiki._read(root, "staging")]
    for claim_id in claim_ids[:2]:
        wiki.promote_claim(root, claim_id)
    wiki.add_wiki_page(root, title="staging-link", claims=claim_ids, body=_page_body())

    with pytest.raises(ValueError, match="claim references.*final claims"):
        Orchestrator(root, adapter=FakeHarness(_no_synth_calls), wiki=wiki,
                     config=RunConfig(max_cycles=0))._gate_published()

    state = checkpoint.load(root)
    assert state["phase"] == "synthesizing"
    assert [claim["id"] for claim in wiki._read(root, "staging")] == claim_ids[2:]
    assert any("the page claim references do not match the final claims" in gap["reason"]
               for gap in state["gaps"])


def test_existing_placeholder_page_creates_compiler_gap_before_done(tmp_path):
    """The mere presence of an .md file does not replace the content gate of an existing
    concept page."""
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    source_specs = [
        ("web", "web", "https://one.example/source"),
        ("paper", "arxiv", "https://two.example/paper"),
        ("repo", "github", "https://two.example/repo"),
    ]
    claim_ids = []
    for index, (kind, tool, url) in enumerate(source_specs):
        source_id = wiki.add_source(
            root, kind=kind, title=f"Source {index}", url=url, tool=tool)
        claim_ids.append(wiki.add_claim(
            root, "final", text=f"final fact {index}", confidence="medium",
            status="verified",
            evidence=[{"source_id": source_id, "quote": "q", "stance": "supports"}],
        ))
    wiki.add_wiki_page(root, title="Stub", claims=claim_ids, body="substrate")

    with pytest.raises(ValueError, match="compiler gap.*no coherent substrate"):
        Orchestrator(root, adapter=FakeHarness(_no_synth_calls), wiki=wiki,
                     config=RunConfig(max_cycles=0)).run()

    state = checkpoint.load(root)
    assert state["phase"] == "synthesizing"
    assert any("existing page stub.md" in gap["reason"]
               for gap in state["gaps"])
    assert (root / "final" / "wiki" / "stub.md").exists()


def test_done_with_pending_is_error_even_when_final_is_not_empty(tmp_path):
    """done + pending are contradictory: a non-empty final may not hide a second decision."""
    root = _topic(tmp_path, phase="done")
    wiki = FakeWiki()
    wiki.add_claim(root, "final", text="old result", confidence="high",
                   status="verified", evidence=_ev())
    pending = {"saved_at": "2026-08-10T00:00:00+00:00", "claims": [],
               "pages": [], "summary_md": "another decision"}
    (root / "work" / _PENDING).write_text(json.dumps(pending), "utf-8")
    ad = FakeHarness(lambda role, p: pytest.fail("the harness must not be touched when done"))
    with pytest.raises(ValueError, match="cannot be done.*pending"):
        Orchestrator(root, adapter=ad, wiki=wiki, config=RunConfig()).run()
    assert (root / "work" / _PENDING).exists()


def test_resume_lock_rejects_second_process_honestly(tmp_path):
    """An already held topic flock gives the second resume an immediate ValueError."""
    root = _topic(tmp_path, phase="done")
    wiki = FakeWiki()
    _stage(wiki, root, "result 1", "result 2", "result 3")
    claim_ids = [claim["id"] for claim in wiki._read(root, "staging")]
    for claim_id in claim_ids:
        wiki.promote_claim(root, claim_id)
    state = checkpoint.load(root)
    state["phase"] = "done"
    checkpoint.save(root, state)
    wiki.add_wiki_page(root, title="Existing concept", claims=claim_ids,
                       body=_page_body())
    lock_path = root / "work" / "resume.lock"
    with lock_path.open("a+", encoding="utf-8") as first:
        fcntl.flock(first.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        first.seek(0); first.truncate(); first.write("pid=111\n"); first.flush()
        with pytest.raises(ValueError, match="another process.*pid=111"):
            Orchestrator(root, adapter=FakeHarness(lambda r, p: pytest.fail("no call")),
                         wiki=wiki, config=RunConfig()).run()
    assert Orchestrator(root, adapter=FakeHarness(lambda r, p: pytest.fail("no call")),
                        wiki=wiki, config=RunConfig()).run() == EXIT_DONE


def test_pending_temp_names_are_unique(tmp_path, monkeypatch):
    """Two concurrent pending writers do not share one fixed synthesis-pending.tmp."""
    root = _topic(tmp_path, phase="synthesizing")
    orch = Orchestrator(root, adapter=FakeHarness(lambda r, p: "{}"), wiki=FakeWiki())
    sources = []

    def boom(src, dst):
        sources.append(Path(src))
        raise OSError("stop before replace")

    monkeypatch.setattr("researcher.orchestrator.os.replace", boom)
    for summary in ("one", "two"):
        with pytest.raises(OSError):
            orch._save_pending({"claims": [], "pages": [], "summary_md": summary})
    assert len(set(sources)) == 2 and all(p.exists() for p in sources)


# --- units: answer parsing - exactly one JSON object (codex review, fix-1) --

def test_extract_json_finds_object_after_garbage_brace():
    """A stray '{' in the prose BEFORE the answer no longer eats the valid JSON that follows
    (the old slice "first { .. last }" used to fail on such text)."""
    good = json.dumps({"final": [{"index": 0}], "summary_md": "summary"}, ensure_ascii=False)
    parsed = _extract_json("my thinking: {this is not json} and here is the answer:\n" + good)
    assert parsed["summary_md"] == "summary"


def test_extract_json_tolerates_known_incomplete_tail():
    """The live shape of the bug: a DUPLICATE of its own tail (not an object) is glued to a
    valid object."""
    good = json.dumps({"final": [{"index": 0}], "summary_md": "summary"}, ensure_ascii=False)
    assert _extract_json(good + '],"summary_md":"tail duplicate"}')["summary_md"] == "summary"


def test_extract_json_two_objects_is_fail_closed():
    """Two STANDALONE objects are ambiguous: silently picking the first means picking a draft."""
    a = json.dumps({"final": [], "summary_md": "draft"})
    b = json.dumps({"final": [{"index": 0}], "summary_md": "final copy"})
    with pytest.raises(ValueError, match="TWO standalone"):
        _extract_json(a + "\n\n" + b)


def test_extract_json_discards_only_known_preamble_objects_before_real_answer():
    """Both live preamble shapes (the plan object and the status ping) are recognized before
    the "exactly one" rule. The basis is an analysis of 11 real bad answers: those two shapes
    caused 4 of the 11 failures. Everything else stays ambiguous: mutating any branch of the
    filter to False breaks the first two assertions, and relaxing it to "take the last one"
    breaks the rest.
    """
    plan = json.dumps({"plan": [
        {"step": "Find sources", "status": "in_progress"},
        {"step": "Assemble JSON", "status": "pending"},
    ]}, ensure_ascii=False)
    answer = {"sources": [], "claims": []}
    assert _extract_json(plan + "\n" + json.dumps(answer)) == answer
    assert _extract_json('{"status":"searching"}\n' + json.dumps(answer)) == answer
    # The status key is there, but the object is not single-key - that is a possible answer,
    # not a ping.
    with pytest.raises(ValueError, match="TWO standalone"):
        _extract_json('{"status":"searching","claims":[]}\n' + json.dumps(answer))
    # A single-key status with a non-string value is not a ping either.
    with pytest.raises(ValueError, match="TWO standalone"):
        _extract_json('{"status":{"phase":1}}\n' + json.dumps(answer))
    # A draft answer placed before the answer is still ambiguous.
    with pytest.raises(ValueError, match="TWO standalone"):
        _extract_json(json.dumps({"sources": [], "claims": [{"text": "x"}]})
                      + "\n" + json.dumps(answer))


def test_extract_json_reports_truncated_answer_as_truncation():
    """A truncated answer is called truncated, not "two objects".

    The live shape (3 cases out of 11 in one night): the model did not finish the JSON, the
    scanner collected inner fragments of the same object and reported ambiguity - a diagnosis
    that led away from the cause. Mutating the closeness-to-end threshold breaks the first
    assertion, losing the `first` flag breaks the second.
    """
    body = ('{"sources":[{"kind":"web","title":"A","url":"https://a"},'
            '{"kind":"web","title":"B","url":"https://b"}],"claims":[{"text":"t",'
            '"confidence":"medium","evidence":[{"url":"https://a","quote":"q"')
    with pytest.raises(ValueError, match="TRUNCATED"):
        _extract_json(body)
    # A stray '{' in the prose BEFORE a valid answer does not count as truncation.
    answer = {"sources": [], "claims": []}
    assert _extract_json('text { not json\n' + json.dumps(answer)) == answer


def test_extract_json_non_dict_is_error():
    """A valid JSON array or scalar is a parse error here, not an AttributeError on .get below."""
    with pytest.raises(ValueError, match="not an object"):
        _extract_json(json.dumps([{"final": []}]))
    with pytest.raises(ValueError, match="not an object"):
        _extract_json("42")


# --- units: structural junk does not turn into emptiness (codex review, block-2) --
# Invariant: an answer that does not parse structurally is NOT the same as "the work is done,
# the result is empty". The task stays in the queue, the phase stays in the checkpoint, the
# raw answer goes to work/bad-answers/, and resume calls exactly the role that failed.


def test_two_objects_retry_same_collector_then_succeeds_and_counts_failure(tmp_path):
    task = {"id": "t1", "query": "q1"}
    root = _topic(tmp_path, phase="collecting", queue=[task], cycles=0,
                  rounds_without_claim=0, run_id="run_test")
    good = json.dumps({"sources": _src(1), "claims": [_claim("fact1")]},
                      ensure_ascii=False)
    calls = 0

    # Ambiguous junk: a DRAFT of the same shape precedes the answer (not a preamble ping, which
    # the engine now recognizes and discards silently) - it cannot be chosen, so a retry is needed.
    draft = json.dumps({"sources": _src(1), "claims": [_claim("draft")]},
                       ensure_ascii=False)

    def responder(role, prompt):
        nonlocal calls
        assert role == "collect"
        calls += 1
        if calls == 1:
            return draft + "\n" + good
        return good

    ad = FakeHarness(responder)
    cfg = RunConfig(max_cycles=1, pool_size=1, retries=2, retry_backoff=0)
    assert Orchestrator(root, adapter=ad, wiki=FakeWiki(), config=cfg)._phase_collect() == "done"

    state = checkpoint.load(root)
    assert state["phase"] == "verifying" and state["structural_failures"] == 1
    assert [call[0] for call in ad.calls] == ["collect", "collect"]
    assert ad.calls[0][2] == ad.calls[1][2]  # the same model/effort/tools of the same role
    assert "no multiple objects" not in ad.calls[0][1]
    assert "no multiple objects" in ad.calls[1][1]
    raw = root / "work" / "bad-answers" / "collect-t1-attempt-1.txt"
    assert raw.exists() and "draft" in raw.read_text("utf-8")


def test_three_structural_failures_are_fatal_and_keep_every_attempt(tmp_path, monkeypatch):
    task = {"id": "t1", "query": "q1"}
    root = _topic(tmp_path, phase="collecting", queue=[task], cycles=0,
                  rounds_without_claim=0, run_id="run_test")
    calls = 0

    def responder(role, prompt):
        nonlocal calls
        calls += 1
        return json.dumps({"status": f"attempt-{calls}"}) + "\n" + json.dumps({})

    ad = FakeHarness(responder)
    sleeps = []
    monkeypatch.setattr("researcher.orchestrator.time.sleep", sleeps.append)
    cfg = RunConfig(max_cycles=1, pool_size=1, retries=2)
    with pytest.raises(ValueError, match="collector"):
        Orchestrator(root, adapter=ad, wiki=FakeWiki(), config=cfg)._phase_collect()

    state = checkpoint.load(root)
    assert state["phase"] == "collecting" and state["structural_failures"] == 3
    assert state["queue"] == [task] and state["done"] == []
    paths = sorted((root / "work" / "bad-answers").glob("collect-t1-attempt-*.txt"))
    assert [path.name for path in paths] == [
        "collect-t1-attempt-1.txt",
        "collect-t1-attempt-2.txt",
        "collect-t1-attempt-3.txt",
    ]
    assert all(f'"status": "attempt-{index}"' in path.read_text("utf-8")
               for index, path in enumerate(paths, 1))
    assert len(ad.calls) == 3
    assert sleeps == [5.0, 10.0]
    assert all("no multiple objects" in prompt
               for _, prompt, _ in ad.calls[1:])

def test_collector_garbage_keeps_task_in_queue_and_is_not_done(tmp_path):
    root = _topic(tmp_path)
    tasks = [{"id": "t1", "query": "q1"}, {"id": "t2", "query": "q2"}]
    collect = {"q1": {"sources": _src(1), "claims": [_claim("fact1")]},
               "q2": {"sources": _src(1), "claims": [_claim("fact2")]}}
    good = _scripted(tasks, collect)

    def responder(role, prompt):
        if role == "collect" and 'Query: "q2"' in prompt:
            return "the collector writes prose instead of json"
        return good(role, prompt)

    with pytest.raises(ValueError, match="collector"):
        Orchestrator(root, adapter=FakeHarness(responder), wiki=FakeWiki(),
                     config=RunConfig(pool_size=2)).run()
    st = checkpoint.load(root)
    assert st["phase"] == "collecting"
    assert st["done"] == ["t1"]                          # the successful task is recorded
    assert [q["id"] for q in st["queue"]] == ["t2"]      # the junk one is NOT marked as done
    raw = (root / "work" / "bad-answers" / "collect-t2-attempt-1.txt").read_text("utf-8")
    assert "the collector writes prose instead of json" in raw     # the raw answer is saved


def test_resume_after_collector_garbage_repeats_that_task(tmp_path):
    """Resume repeats EXACTLY the role and task that failed instead of starting over."""
    root = _topic(tmp_path)
    tasks = [{"id": "t1", "query": "q1"}, {"id": "t2", "query": "q2"}]
    collect = {
        "q1": {"sources": _src(1),
               "claims": [_claim("fact1"), _claim("fact1-addendum")]},
        "q2": {"sources": [{"kind": "paper", "title": "Another domain",
                              "url": "https://other.example/q2", "tool": "arxiv",
                              "published_at": "2026-08-11", "published_at_reason": None},
                            {"kind": "repo", "title": "Third kind",
                             "url": "https://repo.example/q2", "tool": "github",
                             "published_at": "2026-08-11", "published_at_reason": None}],
               "claims": [_claim("fact2", "https://other.example/q2")]},
    }
    good = _scripted(tasks, collect, page_ready=False)
    with pytest.raises(ValueError):
        Orchestrator(root, adapter=FakeHarness(
            lambda role, p: "prose" if role == "collect" and 'Query: "q2"' in p else good(role, p)),
            wiki=FakeWiki(), config=RunConfig(pool_size=2)).run()

    ad2 = FakeHarness(good)
    wiki2 = FakeWiki()
    assert Orchestrator(root, adapter=ad2, wiki=wiki2, config=RunConfig(pool_size=2)).run() == EXIT_DONE
    queries = [p.split('Query: "', 1)[1].split('"', 1)[0] for r, p, _ in ad2.calls if r == "collect"]
    assert [query for query in queries if not query.startswith("Saturation control")] == ["q2"]
    assert {c["text"] for c in wiki2.claims["final"]} == {
        "fact1", "fact1-addendum", "fact2"}


def test_collector_garbage_at_last_cycle_does_not_consume_gates(tmp_path):
    """Junk on the last max_cycles/saturation attempt spends no gate: the checkpoint stays
    collecting and resume really does repeat the task."""
    task = {"id": "t1", "query": "q1"}
    root = _topic(tmp_path, phase="collecting", queue=[task], cycles=1,
                  rounds_without_claim=1, run_id="run_test")
    cfg = RunConfig(max_cycles=2, saturation_rounds=2, pool_size=1)
    with pytest.raises(ValueError, match="collector"):
        Orchestrator(root, adapter=FakeHarness(lambda role, p: "not json"), wiki=FakeWiki(),
                     config=cfg).run()

    failed = checkpoint.load(root)
    assert failed["phase"] == "collecting"
    assert failed["cycles"] == 1
    assert failed["rounds_without_claim"] == 1
    assert failed["queue"] == [task] and failed["done"] == []

    collect = {"q1": {"sources": _src(1), "claims": [_claim("fact1")]}}
    resumed = FakeHarness(_scripted([task], collect))
    wiki = FakeWiki()
    assert Orchestrator(root, adapter=resumed, wiki=wiki, config=cfg).run() == EXIT_DONE
    assert [role for role, _, _ in resumed.calls][0] == "collect"
    done = checkpoint.load(root)
    assert done["phase"] == "done" and done["cycles"] == 2
    assert done["queue"] == [] and done["done"] == ["t1"]


def test_critic_garbage_keeps_phase_verifying_and_candidates(tmp_path):
    """Junk from the critic used to mean `{}` -> ALL candidates were dropped silently and the
    checkpoint moved to synthesizing with an empty staging. Now the phase holds and the
    candidates are intact."""
    root = _topic(tmp_path)
    tasks = [{"id": "t1", "query": "q1"}]
    collect = {"q1": {"sources": _src(1), "claims": [_claim("fact1")]}}
    good = _scripted(tasks, collect)
    ad = FakeHarness(
        lambda role, p: "the critic answers with prose" if role == "verify" else good(role, p))
    wiki = FakeWiki()
    with pytest.raises(ValueError, match="critic"):
        Orchestrator(root, adapter=ad, wiki=wiki, config=RunConfig()).run()

    assert checkpoint.load(root)["phase"] == "verifying"      # it did not move to synthesizing
    assert wiki._read(root, "staging") == []                  # there were no mutations
    assert (root / "work" / "found" / "t1.jsonl").exists()    # the candidates are intact
    # resume calls exactly the critic and drives the run to the end
    ad2 = FakeHarness(good)
    wiki2 = FakeWiki()
    assert Orchestrator(root, adapter=ad2, wiki=wiki2, config=RunConfig()).run() == EXIT_DONE
    assert [r for r, _, _ in ad2.calls][0] == "verify"
    assert {c["text"] for c in wiki2.claims["final"]} == {
        "fact1", "supporting fact q1 2", "supporting fact q1 3"}


def test_synth_garbage_does_not_reach_gate_and_keeps_staging(tmp_path):
    """Junk from the synthesizer never reaches the publication gate: with a torn final (a claim
    from an earlier pass already in final) a weak gate would let the phase reach done on
    SOMEONE ELSE'S result."""
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "alpha", "beta")
    wiki._write(root, "final", [{"id": "clm_foreign", "text": "old claim", "confidence": "high",
                                 "status": "verified", "evidence": _ev()}])
    ad = FakeHarness(lambda role, p: "the synthesizer answers with prose")
    with pytest.raises(ValueError, match="synthesizer"):
        Orchestrator(root, adapter=ad, wiki=wiki, config=RunConfig()).run()
    assert checkpoint.load(root)["phase"] == "synthesizing"
    assert len(wiki._read(root, "staging")) == 2
    assert (root / "work" / "bad-answers" / "synth-attempt-1.txt").exists()


@pytest.mark.parametrize("payload", [
    {},
    {"sources": "not a list", "claims": []},
])
def test_collector_role_schema_is_fail_closed(tmp_path, payload):
    """Valid JSON without the role fields, or with a wrong type, does not complete the task."""
    root = _topic(tmp_path)
    task = {"id": "t1", "query": "q1"}
    good = _scripted([task], {"q1": {"sources": _src(1), "claims": [_claim("fact1")]}})

    def responder(role, prompt):
        return json.dumps(payload, ensure_ascii=False) if role == "collect" else good(role, prompt)

    with pytest.raises(ValueError, match="collector.*schema"):
        Orchestrator(root, adapter=FakeHarness(responder), wiki=FakeWiki(),
                     config=RunConfig()).run()
    st = checkpoint.load(root)
    assert st["phase"] == "collecting" and st["queue"] == [task] and st["done"] == []
    assert (root / "work" / "bad-answers" / "collect-t1-attempt-1.txt").exists()


@pytest.mark.parametrize("payload", [{}, {"verdicts": "not a list"}])
def test_critic_role_schema_is_fail_closed(tmp_path, payload):
    """An incomplete verdict object does not mean a silent "drop everything"."""
    root = _topic(tmp_path)
    task = {"id": "t1", "query": "q1"}
    collect = {"q1": {"sources": _src(1), "claims": [_claim("fact1")]}}
    good = _scripted([task], collect)

    def responder(role, prompt):
        return json.dumps(payload, ensure_ascii=False) if role == "verify" else good(role, prompt)

    wiki = FakeWiki()
    with pytest.raises(ValueError, match="critic.*schema"):
        Orchestrator(root, adapter=FakeHarness(responder), wiki=wiki,
                     config=RunConfig()).run()
    assert checkpoint.load(root)["phase"] == "verifying"
    assert wiki._read(root, "staging") == []
    assert (root / "work" / "found" / "t1.jsonl").exists()
    assert (root / "work" / "bad-answers" / "verify-attempt-1.txt").exists()


@pytest.mark.parametrize("payload", [
    {},
    {"final": [], "pages": "not a list", "summary_md": []},
])
def test_synth_role_schema_is_fail_closed(tmp_path, payload):
    """A wrong synth schema creates no empty pending and never reaches publication."""
    root = _topic(tmp_path, phase="synthesizing")
    wiki = FakeWiki()
    _stage(wiki, root, "alpha")
    ad = FakeHarness(lambda role, p: json.dumps(payload, ensure_ascii=False))
    with pytest.raises(ValueError, match="synthesizer.*schema"):
        Orchestrator(root, adapter=ad, wiki=wiki, config=RunConfig()).run()
    assert checkpoint.load(root)["phase"] == "synthesizing"
    assert len(wiki._read(root, "staging")) == 1
    assert not (root / "work" / _PENDING).exists()
    assert (root / "work" / "bad-answers" / "synth-attempt-1.txt").exists()


# --- units: source tools in the collector prompt ----------------------------

def _collect_calls(tmp_path, *, cfg, **fake_kw):
    """Run one pass and return the collector calls as (role, prompt, build_kw)."""
    root = _topic(tmp_path)
    collect = {"q1": {"sources": _src(1), "claims": [_claim("fact1")]}}
    ad = FakeHarness(_scripted([{"id": "t1", "query": "q1"}], collect), **fake_kw)
    Orchestrator(root, adapter=ad, wiki=FakeWiki(), config=cfg).run()
    calls = [c for c in ad.calls if c[0] == "collect"]
    assert calls
    return calls


def test_sources_block_and_bash_tool(tmp_path):
    from researcher import sources as sources_pkg
    for role, prompt, kw in _collect_calls(tmp_path, cfg=RunConfig(sources=["hn", "arxiv"])):
        assert sources_pkg.cmd_prefix() in prompt        # strict command prefix (with PYTHONPATH)
        assert "hn:" in prompt and "arxiv:" in prompt      # the available-sources block
        # Bash is not bare: EXACTLY the source-CLI prefix is allowed (a scoped allowlist rule)
        assert f"Bash({sources_pkg.cmd_prefix()}:*)" in kw["allowed_tools"]
        assert "Task/Agent" in prompt                      # sub-agents are still forbidden


def test_no_sources_means_no_bash_and_no_block(tmp_path):
    for role, prompt, kw in _collect_calls(tmp_path, cfg=RunConfig()):  # the default: no sources
        assert "Bash" not in kw["allowed_tools"]
        assert "researcher.sources" not in prompt


# --- units: the harness capability profile in the collector prompt ----------

def test_prompt_says_native_web_on_claude_like_harness(tmp_path):
    for role, prompt, kw in _collect_calls(tmp_path, cfg=RunConfig(sources=["web"])):
        assert "NATIVE WebSearch/WebFetch tools" in prompt
        assert kw["allowed_tools"].startswith("WebSearch WebFetch")
        assert kw["network"] is True  # the collector needs the network (a sandbox has to open it)


def test_prompt_has_no_native_web_on_codex_like_harness(tmp_path):
    """On a harness without web the prompt does NOT promise WebSearch/WebFetch, and the harness
    has neither an allowlist nor a physical tool set at all (its boundary is the sandbox):
    None, not an invented string."""
    calls = _collect_calls(tmp_path, cfg=RunConfig(sources=["web", "hn"]),
                           native_web=False, tool_allowlist=False, static_tool_set=False)
    for role, prompt, kw in calls:
        assert "You have NO native web search" in prompt
        assert "WebSearch" not in prompt
        assert kw["allowed_tools"] is None and kw["tools"] is None
        assert kw["network"] is True
        # and the prompt does not lie about blocking: a harness without an allowlist does not block
        assert "the boundary here rests on you" in prompt
        assert "forbidden by harness permissions" not in prompt


def test_parity_mode_turns_off_native_web_on_claude_like(tmp_path):
    """Parity (VISION §9.1): native web is switched off PHYSICALLY - it is absent from the tool
    set and from the allowlist, not only from the prompt text."""
    calls = _collect_calls(tmp_path, cfg=RunConfig(sources=["web"], parity=True))
    for role, prompt, kw in calls:
        assert "WebSearch" not in kw["allowed_tools"] and "WebFetch" not in kw["allowed_tools"]
        assert "WebSearch" not in kw["tools"] and "WebFetch" not in kw["tools"]
        assert "You have NO native web search" in prompt
        assert "Bash(" in kw["allowed_tools"]  # but the source CLI stays
        assert "Bash" in kw["tools"]           # (in the physical set it appears as a plain name)


def test_collector_gets_physical_tool_set_same_for_all(tmp_path):
    """The physical tool set is passed to the collector separately from the allowlist and is
    IDENTICAL for every collector of a run (a different set per call breaks the prefix cache)."""
    from researcher import sources as sources_pkg
    tasks = [{"id": f"t{i}", "query": f"q{i}"} for i in (1, 2)]
    root = _topic(tmp_path)
    collect = {"q1": {"sources": _src(1), "claims": [_claim("f1")]},
               "q2": {"sources": _src(1), "claims": [_claim("f2")]}}
    ad = FakeHarness(_scripted(tasks, collect))
    Orchestrator(root, adapter=ad, wiki=FakeWiki(),
                 config=RunConfig(sources=["web"])).run()
    sets = {kw["tools"] for role, _, kw in ad.calls if role == "collect"}
    assert sets == {"WebSearch WebFetch Read Grep Glob Bash"}
    # the scoped rule lives ONLY in the allowlist: the prefix never enters the physical set
    assert sources_pkg.cmd_prefix() not in next(iter(sets))


def test_no_web_and_no_web_tool_prompt_says_so_plainly(tmp_path):
    """A harness without native web and without the web tool: the prompt header says "there is
    no web at all" rather than a conditional "if it is in the list below" (claude review)."""
    calls = _collect_calls(tmp_path, cfg=RunConfig(sources=["hn"]),
                           native_web=False, tool_allowlist=False, static_tool_set=False)
    for role, prompt, kw in calls:
        assert "You have NO general web access AT ALL" in prompt
        assert "if it is in the list below" not in prompt
        assert "hn:" in prompt  # the structured sources are still there


def test_parity_without_web_tool_is_config_error(tmp_path):
    root = _topic(tmp_path)
    ad = FakeHarness(
        lambda role, p: pytest.fail("the harness must not be touched with a broken config"))
    with pytest.raises(ValueError, match="parity mode"):
        Orchestrator(root, adapter=ad, wiki=FakeWiki(),
                     config=RunConfig(sources=["hn"], parity=True))


def test_think_phases_get_no_network_and_no_web(tmp_path):
    """Web and network go to the collection phase only: thinking roles use what was collected."""
    root = _topic(tmp_path)
    collect = {"q1": {"sources": _src(1), "claims": [_claim("fact1")]}}
    ad = FakeHarness(_scripted([{"id": "t1", "query": "q1"}], collect))
    Orchestrator(root, adapter=ad, wiki=FakeWiki(), config=RunConfig(sources=["web"])).run()
    think = [c for c in ad.calls if c[0] != "collect"]
    assert {c[0] for c in think} == {"plan", "verify", "judge", "synth"}
    for role, prompt, kw in think:
        assert kw["network"] is False
        assert "WebSearch" not in kw["allowed_tools"] and "Bash" not in kw["allowed_tools"]
        # physically too: a thinking phase gets neither web nor Bash in its tool set
        assert "WebSearch" not in kw["tools"] and "Bash" not in kw["tools"]


# --- units: model routing by role, and effort -------------------------------

def _routed_calls(tmp_path, cfg):
    """A full run with the given config; returns {role: [(model, effort), ...]}."""
    root = _topic(tmp_path)
    collect = {"q1": {"sources": _src(1), "claims": [_claim("fact1")]},
               "q2": {"sources": _src(1), "claims": [_claim("fact2")]}}
    tasks = [{"id": "t1", "query": "q1"}, {"id": "t2", "query": "q2"}]
    ad = FakeHarness(_scripted(tasks, collect))
    assert Orchestrator(root, adapter=ad, wiki=FakeWiki(), config=cfg).run() == EXIT_DONE
    out: dict[str, list] = {}
    for role, _, kw in ad.calls:
        out.setdefault(role, []).append((kw["model"], kw["effort"]))
    return out


def test_routing_cheap_model_collects_strong_model_thinks(tmp_path):
    """The target mode for nightly jobs: the cheap model collects while the strong one plans,
    criticizes and synthesizes. The role is an engine abstraction; model names come from the
    config."""
    calls = _routed_calls(tmp_path, RunConfig(model="strong", model_collect="cheap",
                                              effort="high", effort_collect="low"))
    assert set(calls) == {"plan", "collect", "verify", "judge", "synth"}
    assert len(calls["collect"]) == 2
    assert set(calls["collect"]) == {("cheap", "low")}
    assert calls["plan"] == calls["judge"] == calls["synth"] == [("strong", "high")]
    assert set(calls["verify"]) == {("strong", "high")}


def test_routing_not_configured_keeps_single_model_everywhere(tmp_path):
    """Backward compatibility: without --model-collect/--effort-collect the roles are not split
    - every phase runs on one model and effort is not invented (None = the harness default)."""
    calls = _routed_calls(tmp_path / "with-effort", RunConfig(model="one-model", effort="medium"))
    assert {v for vals in calls.values() for v in vals} == {("one-model", "medium")}
    plain = _routed_calls(tmp_path / "without", RunConfig(model="one-model"))
    assert {v for vals in plain.values() for v in vals} == {("one-model", None)}


def test_model_priority_explicit_generic_beats_adapter_role_default(tmp_path):
    """Precedence: role flag > explicit generic > adapter role default > implicit.
    The harness here is a fake: model names are the adapter's knowledge and the engine does
    not hardcode them."""
    class HarnessWithRoleDefault(FakeHarness):
        def default_model_for_role(self, role):
            return "light-from-harness" if role == "collect" else None

    root = _topic(tmp_path / "default")
    tasks = [{"id": "t1", "query": "q1"}]
    collect = {"q1": {"sources": _src(1), "claims": [_claim("fact1")]}}
    ad = HarnessWithRoleDefault(_scripted(tasks, collect))
    assert Orchestrator(root, adapter=ad, wiki=FakeWiki(),
                        config=RunConfig(model="common")).run() == EXIT_DONE
    got = {}
    for role, _, kw in ad.calls:
        got.setdefault(role, []).append(kw["model"])
    # the implicit common default is weaker than the role
    assert set(got["collect"]) == {"light-from-harness"}
    assert got["plan"] == got["synth"] == ["common"]       # the thinking phases are untouched

    root2 = _topic(tmp_path / "common-explicit")
    ad2 = HarnessWithRoleDefault(_scripted(tasks, collect))
    assert Orchestrator(root2, adapter=ad2, wiki=FakeWiki(),
                        config=RunConfig(model="common", model_explicit=True)).run() == EXIT_DONE
    assert set(kw["model"] for r, _, kw in ad2.calls if r == "collect") == {"common"}

    root3 = _topic(tmp_path / "role-flag")
    ad3 = HarnessWithRoleDefault(_scripted(tasks, collect))
    assert Orchestrator(root3, adapter=ad3, wiki=FakeWiki(),
                        config=RunConfig(model="common", model_explicit=True,
                                         model_collect="explicit")).run() == EXIT_DONE
    assert set(kw["model"] for r, _, kw in ad3.calls if r == "collect") == {"explicit"}


def test_for_role_rejects_unknown_role(tmp_path):
    """codex review nit-1: an unknown role is no longer treated as THINK - a typo in the role
    name used to route the call silently to the expensive model."""
    from researcher.orchestrator import ROLE_COLLECT, ROLE_THINK
    cfg = RunConfig(model="strong", model_collect="cheap")
    assert cfg.for_role(ROLE_THINK)[0] == "strong"
    assert cfg.for_role(ROLE_COLLECT)[0] == "cheap"
    with pytest.raises(ValueError, match="unknown engine role"):
        cfg.for_role("colect")  # a typo


def test_retry_policy_reaches_adapter(tmp_path):
    """The retry policy lives in the run config and the mechanics live in base.run: the
    orchestrator must pass it to EVERY call, otherwise the retry exists only on paper."""
    root = _topic(tmp_path)
    collect = {"q1": {"sources": _src(1), "claims": [_claim("fact1")]}}
    ad = FakeHarness(_scripted([{"id": "t1", "query": "q1"}], collect))
    Orchestrator(root, adapter=ad, wiki=FakeWiki(),
                 config=RunConfig(retries=4, retry_backoff=1.5)).run()
    assert ad.calls, "no phase was called"
    for role, _, kw in ad.calls:
        assert (kw["retries"], kw["retry_backoff"]) == (4, 1.5), role
    assert _RunConfig().retries == 2  # the production default, not the test wrapper without pauses


def test_effort_reaches_adapter_at_all(tmp_path):
    """The heart of the effort fix: before this wave effort was not passed to the adapter AT ALL
    (it had to be pinned with a PATH shim because the engine never sent the flag). Now every
    call carries it."""
    calls = _routed_calls(tmp_path, RunConfig(model="m", effort="xhigh"))
    assert all(e == "xhigh" for vals in calls.values() for _, e in vals)


# --- units: stop gates ------------------------------------------------------

def test_approved_map_is_drained_before_judge_can_stop(tmp_path):
    root = _topic(tmp_path)
    tasks = [{"id": f"t{i}", "query": f"q{i}"} for i in range(1, 6)]
    collect = {"q1": {"sources": _src(1), "claims": [_claim("fact1")]}}
    for i in range(2, 6):
        collect[f"q{i}"] = {"sources": [], "claims": []}
    ad = FakeHarness(_scripted(tasks, collect))
    Orchestrator(root, adapter=ad, wiki=FakeWiki(),
                 config=RunConfig(max_cycles=10, saturation_rounds=2, pool_size=1)).run()

    st = checkpoint.load(root)
    assert st["rounds_without_claim"] == 0 and st["cycles"] == 1
    assert st["done"][:5] == ["t1", "t2", "t3", "t4", "t5"]
    assert not st["queue"]
    assert st["search_map_progress"]["completed"] == 5
    assert st["stop"]["reason"] == "enough"


def test_judge_enough_stops_without_saturation_probe(tmp_path, capsys):
    root = _topic(tmp_path)
    collected = []

    def responder(role, prompt):
        if role == "plan":
            return json.dumps({"plan_md": "# plan", "tasks": [{"id": "t1", "query": "q0"}]})
        if role == "collect":
            query = prompt.split('Query: "', 1)[1].split('"', 1)[0]
            collected.append(query)
            if query == "q0":
                sources = _src(1) + [{"kind": "paper", "title": "Another domain",
                                      "url": "https://other.example/source", "tool": "arxiv",
                                      "published_at": "2026-08-11",
                                      "published_at_reason": None},
                                     {"kind": "repo", "title": "Third kind",
                                      "url": "https://repo.example/source", "tool": "github",
                                      "published_at": "2026-08-11",
                                      "published_at_reason": None}]
                return json.dumps({
                    "sources": sources,
                    "claims": [_claim("first fact"), _claim("second fact"),
                               _claim("third fact", "https://other.example/source")],
                })
            return json.dumps({"sources": [], "claims": []})
        if role == "verify":
            return json.dumps({
                "verdicts": [{"index": i, "keep": True, "confidence": "high"}
                              for i in range(20)],
                "gaps": [],
            })
        if role == "synth":
            return json.dumps({"final": [{"index": i} for i in range(20)],
                               "pages": [{"title": "Saturation concept",
                                          "claim_indexes": list(range(20)),
                                          "body_md": _page_body()}],
                               "summary_md": "summary", "gaps": []})
        raise AssertionError(role)

    assert Orchestrator(root, adapter=FakeHarness(responder), wiki=FakeWiki(),
                        config=RunConfig(max_cycles=10, saturation_rounds=2,
                                         pool_size=1)).run() == EXIT_DONE
    state = checkpoint.load(root)
    assert collected == ["q0"]
    assert [round_["new_claims"] for round_ in state["rounds"]] == [3]
    assert state["gaps"] == []
    assert state["stop"]["reason"] == "enough"
    assert "Saturation control" not in capsys.readouterr().err


def test_judge_enough_can_stop_before_max_cycles(tmp_path):
    root = _topic(tmp_path)
    collect_calls = 0

    def responder(role, prompt):
        nonlocal collect_calls
        if role == "plan":
            return json.dumps({"plan_md": "# plan", "tasks": [{"id": "t1", "query": "q0"}]})
        if role == "collect":
            collect_calls += 1
            kind, tool = [("web", "web"), ("paper", "arxiv"),
                          ("repo", "github")][collect_calls - 1]
            source = {"kind": kind, "title": f"source {collect_calls}",
                      "url": f"https://d{collect_calls}.example/x", "tool": tool,
                      "published_at": "2026-08-11", "published_at_reason": None}
            return json.dumps({"sources": [source],
                               "claims": [_claim(f"new fact {collect_calls}", source["url"])]})
        if role == "verify":
            return json.dumps({
                "verdicts": [{"index": i, "keep": True, "confidence": "high"}
                              for i in range(20)],
                "gaps": [],
            })
        if role == "synth":
            return json.dumps({"final": [{"index": i} for i in range(20)],
                               "pages": [{"title": "Max-cycles concept",
                                          "claim_indexes": list(range(20)),
                                          "body_md": _page_body()}],
                               "summary_md": "summary", "gaps": []})
        raise AssertionError(role)

    assert Orchestrator(root, adapter=FakeHarness(responder), wiki=FakeWiki(),
                        config=RunConfig(max_cycles=3, saturation_rounds=2,
                                         pool_size=1)).run() == EXIT_DONE
    state = checkpoint.load(root)
    assert collect_calls == 1
    assert state["rounds_without_claim"] == 0
    assert state["stop"]["reason"] == "enough"


def test_critic_gap_does_not_block_judge_stop_after_empty_task(tmp_path):
    root = _topic(tmp_path)
    collected = []
    verify_calls = 0

    def responder(role, prompt):
        nonlocal verify_calls
        if role == "plan":
            return json.dumps({"plan_md": "# plan", "tasks": [
                {"id": "t1", "query": "q1"}, {"id": "t2", "query": "q2"}]})
        if role == "collect":
            query = prompt.split('Query: "', 1)[1].split('"', 1)[0]
            collected.append(query)
            if query == "q1":
                return json.dumps({"sources": _src(1),
                                   "claims": [_claim("first fact"),
                                              _claim("second fact")]})
            if query == "q-gap":
                source = {"kind": "paper", "title": "follow-up", "url": "https://gap.example/x",
                          "tool": "arxiv", "published_at": "2026-08-11",
                          "published_at_reason": None}
                repo = {"kind": "repo", "title": "third kind",
                        "url": "https://repo.example/x", "tool": "github",
                        "published_at": "2026-08-11", "published_at_reason": None}
                return json.dumps({"sources": [source, repo],
                                   "claims": [_claim("follow-up fact", source["url"])]})
            return json.dumps({"sources": [], "claims": []})
        if role == "verify":
            verify_calls += 1
            gaps = ([{"query": "q-gap", "reason": "no data on X", "priority": "high"}]
                    if verify_calls == 1 else [])
            return json.dumps({
                "verdicts": [{"index": i, "keep": True, "confidence": "high"}
                              for i in range(20)],
                "gaps": gaps,
            })
        if role == "synth":
            return json.dumps({"final": [{"index": i} for i in range(20)],
                               "pages": [{"title": "Open-gap concept",
                                          "claim_indexes": list(range(20)),
                                          "body_md": _page_body()}],
                               "summary_md": "summary", "gaps": []})
        raise AssertionError(role)

    assert Orchestrator(root, adapter=FakeHarness(responder), wiki=FakeWiki(),
                        config=RunConfig(max_cycles=4, saturation_rounds=1,
                                         pool_size=1)).run() == EXIT_DONE
    assert "q-gap" not in collected
    assert checkpoint.load(root)["stop"]["reason"] == "enough"


def test_max_cycles_gate(tmp_path):
    root = _topic(tmp_path)
    tasks = [{"id": f"t{i}", "query": f"q{i}"} for i in range(1, 4)]
    collect = {f"q{i}": {"sources": _src(1), "claims": [_claim(f"fact{i}")]} for i in range(1, 4)}
    ad = FakeHarness(_scripted(tasks, collect))
    Orchestrator(root, adapter=ad, wiki=FakeWiki(),
                 config=RunConfig(max_cycles=1, saturation_rounds=99, pool_size=1)).run()

    st = checkpoint.load(root)
    assert st["cycles"] == 1 and st["done"] == ["t1", "t2", "t3"]
    assert not st["queue"]
    assert st["search_map_progress"]["completed"] == 3


def test_pool_clamped_to_three():
    assert RunConfig(pool_size=10).pool() == 3
    assert RunConfig(pool_size=0).pool() == 1
    assert RunConfig(pool_size=2).pool() == 2


# --- units: stops, exit codes and resume ------------------------------------

@pytest.mark.parametrize("stop,code", [("quota", EXIT_QUOTA), ("transient", EXIT_TRANSIENT)])
def test_stop_maps_to_exit_and_keeps_checkpoint(tmp_path, stop, code):
    root = _topic(tmp_path)
    ad = FakeHarness(lambda role, p: RunResult(False, "limit", 1, stop=stop))
    assert Orchestrator(root, adapter=ad, wiki=FakeWiki()).run() == code
    assert checkpoint.load(root)["phase"] == "planned"  # a stop in plan -> checkpoint is valid


def test_unknown_phase_in_checkpoint_raises(tmp_path):
    """The phase-machine guard in the orchestrator: a phase outside PHASES (a hand edit, a
    foreign format, a checkpoint from a future version) raises ValueError instead of running
    silently and doing nothing. checkpoint.save would not let such a phase through
    (test_phase_gate), so the file is written directly."""
    root = _topic(tmp_path)
    path = root / "work" / "checkpoint.json"
    st = json.loads(path.read_text("utf-8"))
    st["phase"] = "chilling"
    path.write_text(json.dumps(st, ensure_ascii=False), "utf-8")
    ad = FakeHarness(
        lambda role, p: pytest.fail("the harness must not be touched with a broken phase"))
    with pytest.raises(ValueError, match="unknown checkpoint phase"):
        Orchestrator(root, adapter=ad, wiki=FakeWiki()).run()


def test_plan_garbage_is_fatal(tmp_path):
    """Junk from the lead is the same fail-closed path as for any other role: a clear error (the
    CLI turns it into stderr plus exit 1), the checkpoint stays planned and the raw answer
    stays on disk."""
    root = _topic(tmp_path)
    ad = FakeHarness(lambda role, p: "there is no json here at all")
    with pytest.raises(ValueError, match="planner"):
        Orchestrator(root, adapter=ad, wiki=FakeWiki(), config=RunConfig()).run()
    assert checkpoint.load(root)["phase"] == "planned"
    assert (root / "work" / "bad-answers" / "plan-attempt-1.txt").exists()


def test_resume_after_quota_in_collect(tmp_path):
    root = _topic(tmp_path)
    tasks = [{"id": "t1", "query": "q1"}, {"id": "t2", "query": "q2"}]
    collect = {
        "q1": {"sources": _src(1),
               "claims": [_claim("fact1"), _claim("fact1-addendum")]},
        "q2": {"sources": [{"kind": "paper", "title": "Another domain",
                              "url": "https://other.example/q2", "tool": "arxiv",
                              "published_at": "2026-08-11", "published_at_reason": None},
                            {"kind": "repo", "title": "Third kind",
                             "url": "https://repo.example/q2", "tool": "github",
                             "published_at": "2026-08-11", "published_at_reason": None}],
               "claims": [_claim("fact2", "https://other.example/q2")]},
    }

    def responder_quota(role, prompt):
        base = _scripted(tasks, collect, page_ready=False)
        if role == "collect" and 'Query: "q2"' in prompt:
            return RunResult(False, "usage limit reached", 1, stop="quota")
        return base(role, prompt)

    ad1 = FakeHarness(responder_quota)
    # Exactly the last allowed cycle, pool 2: t1 is fine, t2 hits the quota.
    cfg = RunConfig(max_cycles=1, saturation_rounds=99, pool_size=2)
    assert Orchestrator(root, adapter=ad1, wiki=FakeWiki(),
                        config=cfg).run() == EXIT_QUOTA
    st = checkpoint.load(root)
    assert st["phase"] == "collecting" and st["done"] == ["t1"]
    assert [q["id"] for q in st["queue"]] == ["t2"]  # t2 is not lost
    assert st["cycles"] == 0 and st["rounds"] == []
    assert st["collecting_round"] == {
        "cycle": 1, "tasks": ["t1", "t2"], "completed": ["t1"], "new_claims": 2,
    }

    # Resume finishes t2 in the SAME cycle. Only after that is max_cycles=1 spent.
    ad2 = FakeHarness(_scripted(tasks, collect, page_ready=False))
    wiki2 = FakeWiki()
    assert Orchestrator(root, adapter=ad2, wiki=wiki2, config=cfg).run() == EXIT_DONE
    done = checkpoint.load(root)
    assert done["phase"] == "done" and done["cycles"] == 1
    assert "collecting_round" not in done
    assert done["rounds"] == [
        {"cycle": 1, "tasks": ["t1", "t2"], "new_claims": 3, "found": 3,
         "accepted_claims": 3}
    ]
    collect_calls = [prompt for role, prompt, _ in ad2.calls if role == "collect"]
    assert len(collect_calls) == 1 and 'Query: "q2"' in collect_calls[0]
    assert {c["text"] for c in wiki2.claims["final"]} == {
        "fact1", "fact1-addendum", "fact2"}


def test_resume_after_crash_between_found_and_checkpoint_is_idempotent(tmp_path, monkeypatch):
    """A hard kill after the found write but before the task checkpoint does not duplicate the
    history and does not turn a genuinely new fact into new_claims=0 on resume."""
    task = {"id": "t1", "query": "q1"}
    root = _topic(tmp_path, phase="collecting", queue=[task], cycles=0,
                  rounds_without_claim=0, run_id="run_test")
    collect = {
        "q1": {"sources": _src(1) + [{"kind": "paper", "title": "Another domain",
                                        "url": "https://other.example/q1", "tool": "arxiv",
                                        "published_at": "2026-08-11",
                                        "published_at_reason": None},
                                       {"kind": "repo", "title": "Third kind",
                                        "url": "https://repo.example/q1", "tool": "github",
                                        "published_at": "2026-08-11",
                                        "published_at_reason": None}],
               "claims": [_claim("fact1"), _claim("fact2"),
                          _claim("fact3", "https://other.example/q1")]},
    }
    responder = _scripted([task], collect, page_ready=False)
    wiki = FakeWiki()
    cfg = RunConfig(max_cycles=1, saturation_rounds=99, pool_size=1)
    real_save = checkpoint.save
    killed = False

    def hard_kill_after_found(topic_dir, state):
        nonlocal killed
        active = state.get("collecting_round") or {}
        if not killed and active.get("completed") == ["t1"]:
            killed = True
            raise RuntimeError("hard kill after found, before the checkpoint")
        return real_save(topic_dir, state)

    monkeypatch.setattr(checkpoint, "save", hard_kill_after_found)
    with pytest.raises(RuntimeError, match="hard kill"):
        Orchestrator(root, adapter=FakeHarness(responder), wiki=wiki, config=cfg).run()

    torn = checkpoint.load(root)
    assert torn["queue"] == [task] and torn["done"] == []
    assert torn["collecting_round"]["completed"] == []
    found_path = root / "work" / "found" / "t1.jsonl"
    assert len(found_path.read_text("utf-8").splitlines()) == 3

    monkeypatch.setattr(checkpoint, "save", real_save)
    resumed = FakeHarness(responder)
    assert Orchestrator(root, adapter=resumed, wiki=wiki, config=cfg).run() == EXIT_DONE
    assert len(found_path.read_text("utf-8").splitlines()) == 3
    done = checkpoint.load(root)
    assert done["rounds"] == [
        {"cycle": 1, "tasks": ["t1"], "new_claims": 3, "found": 3,
         "accepted_claims": 3}
    ]
    collect_calls = [prompt for role, prompt, _ in resumed.calls if role == "collect"]
    assert len(collect_calls) == 1 and 'Query: "q1"' in collect_calls[0]
    assert [c["text"] for c in wiki._read(root, "final")] == ["fact1", "fact2", "fact3"]


def test_resume_after_crash_after_last_task_checkpoint_finishes_round(tmp_path, monkeypatch):
    """A hard kill after the checkpoint of the last task does not lose the round finalization.

    On disk the queue is already empty and the whole batch is completed, but cycles/rounds and
    the removal of collecting_round are not saved yet. Resume must finish that round without
    calling the collector again, instead of moving to verifying with a stale collecting_round.
    """
    task = {"id": "t1", "query": "q1"}
    root = _topic(tmp_path, phase="collecting", queue=[task], cycles=0,
                  rounds_without_claim=0, run_id="run_test")
    collect = {
        "q1": {"sources": _src(1) + [{"kind": "paper", "title": "Another domain",
                                        "url": "https://other.example/q1", "tool": "arxiv",
                                        "published_at": "2026-08-11",
                                        "published_at_reason": None},
                                       {"kind": "repo", "title": "Third kind",
                                        "url": "https://repo.example/q1", "tool": "github",
                                        "published_at": "2026-08-11",
                                        "published_at_reason": None}],
               "claims": [_claim("fact1"), _claim("fact2"),
                          _claim("fact3", "https://other.example/q1")]},
    }
    responder = _scripted([task], collect, page_ready=False)
    wiki = FakeWiki()
    cfg = RunConfig(max_cycles=1, saturation_rounds=99, pool_size=1)
    real_save = checkpoint.save
    killed = False

    def hard_kill_after_last_task_checkpoint(topic_dir, state):
        nonlocal killed
        real_save(topic_dir, state)
        active = state.get("collecting_round") or {}
        if (not killed and state.get("queue") == []
                and active.get("completed") == active.get("tasks") == ["t1"]):
            killed = True
            raise RuntimeError("hard kill after the checkpoint of the last task")

    monkeypatch.setattr(checkpoint, "save", hard_kill_after_last_task_checkpoint)
    with pytest.raises(RuntimeError, match="after the checkpoint of the last task"):
        Orchestrator(root, adapter=FakeHarness(responder), wiki=wiki, config=cfg).run()

    torn = checkpoint.load(root)
    assert torn["queue"] == [] and torn["done"] == ["t1"]
    assert torn["cycles"] == 0 and torn["rounds"] == []
    assert torn["collecting_round"] == {
        "cycle": 1, "tasks": ["t1"], "completed": ["t1"], "new_claims": 3,
    }

    monkeypatch.setattr(checkpoint, "save", real_save)
    resumed = FakeHarness(responder)
    assert Orchestrator(root, adapter=resumed, wiki=wiki, config=cfg).run() == EXIT_DONE
    done = checkpoint.load(root)
    assert done["cycles"] == 1 and "collecting_round" not in done
    assert done["rounds"] == [
        {"cycle": 1, "tasks": ["t1"], "new_claims": 3, "found": 3,
         "accepted_claims": 3}
    ]
    assert not [call for call in resumed.calls if call[0] == "collect"]
    assert [c["text"] for c in wiki._read(root, "final")] == ["fact1", "fact2", "fact3"]


# --- integration with the real llm-wiki socket ------------------------------

@pytest.mark.skipif(not (_LLMWIKI / "llmwiki").exists(),
                    reason="../tool-llm-wiki is missing - real-socket integration skipped")
def test_integration_real_llmwiki(tmp_path):
    sys.path.insert(0, str(_LLMWIKI))
    import llmwiki  # noqa: E402

    root = llmwiki.init_topic(tmp_path, "why pytest fails")
    checkpoint.save(root, {"phase": "planned", "topic": "why pytest fails",
                           "queue": [], "done": [], "sessions": {}})
    collect = {
        "q1": {
            "sources": [{"kind": "web", "title": "repo", "url": "https://e.com/0",
                         "tool": "github", "published_at": "2026-08-01",
                         "published_at_reason": None}],
            "claims": [_claim("pytest fails because of conftest", "https://e.com/0")],
        },
        "q2": {
            "sources": [{"kind": "web", "title": "thread", "url": "https://e.com/1",
                         "tool": "hn", "published_at": None,
                         "published_at_reason": "not found"}],
            "claims": [_claim("fixtures affect collection", "https://e.com/1")],
        },
        "q3": {
            "sources": [{"kind": "web", "title": "docs",
                         "url": "https://docs.example/pytest", "tool": "web",
                         "published_at": "2026-08-03", "published_at_reason": None}],
            "claims": [
                _claim("fixture isolation reduces failures", "https://docs.example/pytest")],
        },
    }
    ad = FakeHarness(_scripted(
        [{"id": "t1", "query": "q1"}, {"id": "t2", "query": "q2"},
         {"id": "t3", "query": "q3"}], collect, page_ready=False))

    code = Orchestrator(root, adapter=ad, wiki=llmwiki, config=RunConfig()).run()
    assert code == EXIT_DONE
    # the real socket would have rejected junk on write; the topic must be clean
    # (validate_topic also catches dangling FKs of synthesis pages - see W1)
    assert llmwiki.validate_topic(root) == []
    sources = [json.loads(line) for line in
               (root / "sources" / "sources.jsonl").read_text("utf-8").splitlines()]
    assert sorted(source["kind"] for source in sources) == ["other", "repo", "web"]
    publication = {source["title"]: source["meta"] for source in sources}
    assert publication == {
        "repo": {"published_at": "2026-08-01", "published_at_reason": None},
        "thread": {"published_at": None, "published_at_reason": "not found"},
        "docs": {"published_at": "2026-08-03", "published_at_reason": None},
    }
    final = (root / "final" / "claims.jsonl").read_text("utf-8").splitlines()
    assert len(final) == 3
    # the final set is filled by promote (a move) - the status stays verified, not "final"
    assert json.loads(final[0])["status"] == "verified"
    # staging is empty: promote moves the row instead of copying it
    staging = root / "staging" / "claims.jsonl"
    assert not staging.exists() or not staging.read_text("utf-8").strip()
    # the synthesis concept page is written into final/wiki/ (W1)
    pages = list((root / "final" / "wiki").glob("*.md"))
    assert len(pages) == 1
    page = pages[0].read_text("utf-8")
    assert "## Sources and freshness" in page
    assert "publication date: 2026-08-01" in page
    assert "publication date: not found" in page
    breadth = checkpoint.load(root)["breadth"]
    assert breadth["topic"]["distinct_domains"] == 2
    assert breadth["topic"]["distinct_kinds"] == 3
    assert breadth["pages"]["Combined concept"]["distinct_domains"] == 2
    assert breadth["pages"]["Combined concept"]["distinct_kinds"] == 3


# --- the map versus gaps, and a typed gate refusal --------------------------

def test_round_prohodit_vsyu_kartu_do_gapov_pri_lyubom_pool():
    """Nine approved clusters do not starve behind a queue of gaps that keeps refilling.

    Before the fix the batch was queue[:pool] while gaps jumped to the head EVERY cycle, so
    after the first `pool` clusters the map never advanced again (measured in production:
    3 clusters out of 11 and out of 9 on two runs).
    """
    from researcher.orchestrator import _round_tasks_with_map_priority

    gaps = [{"id": f"gap_{i}", "query": f"g{i}", "gap_id": f"gap_{i}"} for i in range(72)]
    maps = [{"id": f"c{i}", "query": f"m{i}"} for i in range(9)]
    queue = gaps + maps
    map_ids = {task["id"] for task in maps}

    for pool in (1, 3):
        round_tasks = _round_tasks_with_map_priority(queue, pool, map_ids)
        assert [task["id"] for task in round_tasks] == [f"c{i}" for i in range(9)]
    assert [t["id"] for t in _round_tasks_with_map_priority(gaps, 3, set())] == [
        "gap_0", "gap_1", "gap_2"
    ]


@pytest.mark.parametrize("pool", [1, 3])
def test_9_klasterov_durable_zaversheny_v_pervom_cikle_s_gapami_v_golove(
    tmp_path, pool
):
    """The production scenario: 9 clusters, gaps holding the head, max_cycles=24."""
    maps = [{"id": f"c{i}", "query": f"map-{i}"} for i in range(9)]
    gaps = [
        {"id": f"gap_{i}", "query": f"gap-{i}", "gap_id": f"gap_{i}"}
        for i in range(72)
    ]
    root = _topic(
        tmp_path,
        phase="collecting",
        queue=gaps + maps,
        cycles=0,
        rounds_without_claim=0,
        run_id="run_map_fairness",
    )
    (root / "work" / "search-map.json").write_text(
        json.dumps({"version": 1, "topic": "why pytest fails", "clusters": maps}),
        "utf-8",
    )

    def responder(role, _prompt):
        assert role == "collect"
        return json.dumps({"sources": [], "claims": []})

    orch = Orchestrator(
        root,
        adapter=FakeHarness(responder),
        wiki=FakeWiki(),
        config=RunConfig(max_cycles=24, saturation_rounds=100, pool_size=pool),
    )
    assert orch._phase_collect() == "done"

    state = checkpoint.load(root)
    progress = state["search_map_progress"]
    assert state["cycles"] == 1
    assert progress["completed"] == progress["total"] == 9
    assert progress["completed_ids"] == [f"c{i}" for i in range(9)]
    assert all(f"c{i}" in state["done"] for i in range(9))
    assert state["queue"] == gaps


def test_gate_refusal_is_a_separate_type_with_a_durable_marker(tmp_path):
    """Exit 77 is left only for an honest "nothing to publish"."""
    from researcher.orchestrator import GateRefused

    root = _topic(tmp_path, phase="done", cycles=1, max_cycles=1)

    with pytest.raises(GateRefused) as exc:
        Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=FakeWiki(),
                     config=RunConfig(max_cycles=1)).run()
    assert "nothing to publish" in str(exc.value)

    st = checkpoint.load(root)
    assert st["stop_kind"] == "gate_refused"
    assert st["stop_detail"]["cycles"] == 1 and st["stop_detail"]["max_cycles"] == 1
    assert st["stop_detail"]["reasons"], "the refusal reasons must be durable"


# --- wave: a bounded critic, the judge and a meaning-based stop -------------

def _write_found(root, rows, task="t1"):
    path = root / "work" / "found" / f"{task}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), "utf-8")


def _candidate(text, *, confidence="medium", task="t1"):
    return {
        "text": text, "confidence": confidence, "task": task,
        "evidence": [{"source_id": "src_000000000001", "quote": "quote",
                      "stance": "supports"}],
    }


def _judge_json(verdict="enough", *, why="the answer is covered", gaps=None, clusters=None,
                coverage="## Not closed\n\n- one more check", questions=None):
    return json.dumps({
        "verdict": verdict, "why": why, "coverage_md": coverage,
        "gaps": gaps or [], "new_clusters": clusters or [],
        "open_questions": questions or [],
    }, ensure_ascii=False)


def _verifying_topic(tmp_path, candidates, *, cycles=1, rounds=None, **extra):
    rounds = rounds if rounds is not None else [
        {"cycle": cycles, "tasks": ["t1"], "new_claims": len(candidates),
         "found": len(candidates)}
    ]
    root = _topic(
        tmp_path, phase="verifying", cycles=cycles, rounds=rounds,
        rounds_without_claim=extra.pop("rounds_without_claim", 0),
        run_id="run_test", **extra,
    )
    _write_found(root, candidates)
    return root


def test_verdicts_only_new_candidates_are_sent_to_critic(tmp_path):
    root = _verifying_topic(tmp_path, [_candidate("A"), _candidate("B")])
    calls = []

    def responder(role, prompt):
        assert role == "verify"
        calls.append(prompt)
        return json.dumps({"verdicts": [
            {"index": i, "keep": True, "confidence": "medium"}
            for i in range(10)
        ], "gaps": [{"query": "ignored", "reason": "not a source", "priority": "high"}]})

    adapter = FakeHarness(responder)
    orch = Orchestrator(root, adapter=adapter, wiki=FakeWiki(), config=RunConfig())
    assert orch._phase_verify() == "done"
    state = checkpoint.load(root)
    state.update(phase="verifying", cycles=2)
    state["rounds"].append({"cycle": 2, "tasks": ["t2"], "new_claims": 1, "found": 3})
    checkpoint.save(root, state)
    _write_found(root, [_candidate("C", task="t2")], task="t2")

    assert orch._phase_verify() == "done"
    assert len(calls) == 2
    assert '"text": "A"' in calls[0] and '"text": "B"' in calls[0]
    assert '"text": "C"' in calls[1] and '"text": "A"' not in calls[1]
    assert len((root / "work" / "verdicts.jsonl").read_text("utf-8").splitlines()) == 3


def test_same_text_unions_evidence_and_new_source_rechecks_previous_drop(tmp_path):
    first = _candidate("identical fact", task="t1")
    first["evidence"][0]["source_id"] = "src_one"
    root = _verifying_topic(tmp_path, [first])
    prompts = []

    def critic(role, prompt):
        assert role == "verify"
        prompts.append(prompt)
        return json.dumps({
            "verdicts": [{"index": 0, "keep": False, "confidence": "low"}],
        })

    orch = Orchestrator(
        root, adapter=FakeHarness(critic), wiki=FakeWiki(), config=RunConfig(),
    )
    assert orch._phase_verify() == "done"
    state = checkpoint.load(root)
    state.update(phase="verifying", cycles=2, judge=None)
    checkpoint.save(root, state)
    second = _candidate("identical fact", task="t2")
    second["evidence"][0]["source_id"] = "src_two"
    _write_found(root, [second], task="t2")

    assert orch._phase_verify() == "done"

    assert len(prompts) == 2
    assert '"source_id": "src_one"' in prompts[1]
    assert '"source_id": "src_two"' in prompts[1]
    verdict = json.loads(
        (root / "work" / "verdicts.jsonl").read_text("utf-8").splitlines()[0]
    )
    assert verdict["verdict"] == "drop"
    assert verdict["source_ids"] == ["src_one", "src_two"]


def _stored_claim_with_new_source(tmp_path, zone, wiki):
    from researcher.orchestrator import _claim_verdict_key

    root = _topic(tmp_path, phase="verifying", run_id="run_test")
    wiki.add_claim(
        root, zone, text="already stored fact", confidence="medium", status="verified",
        evidence=[{"source_id": "src_one", "quote": "first", "stance": "supports"}],
    )
    candidate = _candidate("already stored fact")
    candidate["evidence"] = [
        {"source_id": "src_one", "quote": "first", "stance": "supports"},
        {"source_id": "src_two", "quote": "second", "stance": "supports"},
    ]
    key = _claim_verdict_key(candidate["text"])
    verdicts = {key: {"verdict": "keep", "confidence": "high"}}
    orch = Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=wiki)
    return root, candidate, verdicts, orch


def _events_named(root, name):
    return [
        json.loads(line) for line in (root / "work" / "events.jsonl").read_text().splitlines()
        if json.loads(line).get("event") == name
    ]


@pytest.mark.parametrize("zone", ["staging", "final"])
def test_new_evidence_for_stored_claim_is_merged_through_wiki(tmp_path, zone):
    wiki = FakeWiki()
    root, candidate, verdicts, orch = _stored_claim_with_new_source(tmp_path, zone, wiki)

    orch._reconcile_kept_candidates(checkpoint.load(root), [candidate], verdicts)
    orch._reconcile_kept_candidates(checkpoint.load(root), [candidate], verdicts)

    stored = wiki._read(root, zone)
    assert len(stored) == 1
    assert [row["source_id"] for row in stored[0]["evidence"]] == ["src_one", "src_two"]
    if zone == "final":
        assert wiki._read(root, "staging") == [], "a final claim must not be duplicated in staging"
    # the second pass sees src_two already in the zone and does not call the socket:
    # exactly one merge
    assert wiki.merges == [(zone, wiki._cid(candidate["text"]),
                            [{"source_id": "src_two", "quote": "second", "stance": "supports"}])]
    events = _events_named(root, "evidence_merged")
    assert len(events) == 1 and events[0]["zone"] == zone
    assert events[0]["source_ids"] == ["src_two"]
    assert _events_named(root, "evidence_merged_pending") == []


def test_stored_claim_in_both_zones_grows_final(tmp_path):
    wiki = FakeWiki()
    root, candidate, verdicts, orch = _stored_claim_with_new_source(tmp_path, "final", wiki)
    # a crash-window promote: the same row also stayed in staging
    wiki._write(root, "staging", wiki._read(root, "final"))
    orch._reconcile_kept_candidates(checkpoint.load(root), [candidate], verdicts)
    assert [m[0] for m in wiki.merges] == ["final"]
    assert [r["source_id"] for r in wiki._read(root, "staging")[0]["evidence"]] == ["src_one"]


def test_merge_refusal_is_event_not_crash(tmp_path):
    wiki = FakeWiki()
    full_error = "FK: source_id ['src_two'] is not in sources.jsonl: " + "x" * 300
    wiki.merge_error = ValueError(full_error)
    root, candidate, verdicts, orch = _stored_claim_with_new_source(tmp_path, "staging", wiki)

    orch._reconcile_kept_candidates(checkpoint.load(root), [candidate], verdicts)

    stored = wiki._read(root, "staging")
    assert len(stored) == 1 and [r["source_id"] for r in stored[0]["evidence"]] == ["src_one"]
    failed = _events_named(root, "evidence_merge_failed")
    assert len(failed) == 1 and failed[0]["source_ids"] == ["src_two"]
    assert failed[0]["error"] == full_error
    assert failed[0]["evidence"] == {
        "source_id": "src_two", "quote": "second", "stance": "supports",
    }
    assert _events_named(root, "evidence_merged") == []


@pytest.mark.parametrize("zone", ["staging", "final"])
def test_wiki_without_merge_api_keeps_durable_pending_event(tmp_path, zone):
    class OldWiki(FakeWiki):
        merge_claim_evidence = None  # a socket from before merge_claim_evidence existed

    wiki = OldWiki()
    root, candidate, verdicts, orch = _stored_claim_with_new_source(tmp_path, zone, wiki)

    orch._reconcile_kept_candidates(checkpoint.load(root), [candidate], verdicts)
    orch._reconcile_kept_candidates(checkpoint.load(root), [candidate], verdicts)

    stored = wiki._read(root, zone)
    assert len(stored) == 1 and [row["source_id"] for row in stored[0]["evidence"]] == ["src_one"]
    if zone == "final":
        assert wiki._read(root, "staging") == [], "a final claim must not be duplicated in staging"
    events = _events_named(root, "evidence_merged_pending")
    assert len(events) == 1
    assert events[0]["id"] == wiki._cid(candidate["text"])
    assert events[0]["source_id"] == "src_two"
    assert events[0]["evidence"]["quote"] == "second"


def test_empty_unverified_set_does_not_call_critic(tmp_path):
    cand = _candidate("already verified")
    root = _verifying_topic(tmp_path, [cand])
    orch = Orchestrator(root, adapter=FakeHarness(lambda role, _: pytest.fail(role)),
                        wiki=FakeWiki(), config=RunConfig())
    orch._append_verdict_rows([{
        "key": hashlib.sha256("already verified".encode()).hexdigest(),
        "verdict": "drop", "confidence": "low", "cycle": 0,
        "source_ids": ["src_000000000001"],
        "run_id": "run_test", "at": "2026-08-23T00:00:00+00:00",
    }])

    assert orch._phase_verify() == "done"
    assert [role for role, _, _ in orch.adapter.calls] == ["judge"]


def test_legacy_staging_candidate_is_seeded_keep_and_not_rechecked(tmp_path):
    seeded, fresh = _candidate("old staging"), _candidate("new candidate")
    root = _verifying_topic(tmp_path, [seeded, fresh])
    wiki = FakeWiki()
    wiki.add_claim(root, "staging", text=seeded["text"], confidence="medium",
                   status="verified", evidence=seeded["evidence"], run_id="run_test")
    prompts = []
    adapter = FakeHarness(lambda role, prompt: (
        prompts.append(prompt) or json.dumps({"verdicts": [
            {"index": 0, "keep": False, "confidence": "low"}
        ]})))

    Orchestrator(root, adapter=adapter, wiki=wiki, config=RunConfig())._phase_verify()

    rows = [json.loads(line) for line in
            (root / "work" / "verdicts.jsonl").read_text("utf-8").splitlines()]
    assert len(prompts) == 1 and "new candidate" in prompts[0] and "old staging" not in prompts[0]
    assert next(row for row in rows if row.get("seeded"))["verdict"] == "keep"


def test_critic_batches_more_than_150_and_persists_each_batch(tmp_path):
    candidates = [_candidate(f"fact {i}") for i in range(151)]
    root = _verifying_topic(tmp_path, candidates)
    adapter = FakeHarness(lambda role, _: json.dumps({"verdicts": [
        {"index": i, "keep": True, "confidence": "medium"} for i in range(150)
    ]}))

    Orchestrator(root, adapter=adapter, wiki=FakeWiki(), config=RunConfig())._phase_verify()

    assert [role for role, _, _ in adapter.calls].count("verify") == 2
    assert len((root / "work" / "verdicts.jsonl").read_text("utf-8").splitlines()) == 151


def test_critic_returned_gaps_are_ignored_without_gap_event(tmp_path):
    root = _verifying_topic(tmp_path, [_candidate("fact")])
    adapter = FakeHarness(lambda role, _: json.dumps({
        "verdicts": [{"index": 0, "keep": True, "confidence": "medium"}],
        "gaps": [{"query": "do not keep", "reason": "critic", "priority": "high"}],
    }))

    Orchestrator(root, adapter=adapter, wiki=FakeWiki(), config=RunConfig())._phase_verify()

    state = checkpoint.load(root)
    assert not state.get("gaps")
    events = [json.loads(line) for line in (root / "work" / "events.jsonl").read_text().splitlines()]
    assert any(row["event"] == "verification_funnel" and "gaps" not in row for row in events)


@pytest.mark.parametrize("verdict", ["enough", "stuck"])
def test_judge_terminal_verdicts_move_to_synthesis(tmp_path, verdict):
    root = _verifying_topic(tmp_path, [])
    adapter = FakeHarness(lambda *_: pytest.fail("the critic is not needed"),
                          judge_responder=lambda _: _judge_json(verdict, why=f"why {verdict}"))

    Orchestrator(root, adapter=adapter, wiki=FakeWiki(), config=RunConfig())._phase_verify()

    state = checkpoint.load(root)
    assert state["phase"] == "synthesizing"
    assert state["stop"] == {"reason": verdict, "why": f"why {verdict}", "cycles": 1,
                              "rounds_without_claim": 1}


def test_judge_continue_queues_gap_at_head(tmp_path):
    root = _verifying_topic(tmp_path, [], queue=[{"id": "old", "query": "old"}])
    adapter = FakeHarness(lambda *_: pytest.fail("critic"), judge_responder=lambda _: _judge_json(
        "continue", gaps=[{"query": "new gap", "reason": "not closed", "priority": "high"}]))

    Orchestrator(root, adapter=adapter, wiki=FakeWiki(), config=RunConfig())._phase_verify()

    state = checkpoint.load(root)
    assert state["phase"] == "collecting"
    assert state["queue"][0]["query"] == "new gap"
    assert state["gaps"][0]["source"] == "judge" and state["gaps"][0]["status"] == "open"


def test_judge_continue_keeps_already_queued_open_gap_and_collector_executes_it(tmp_path):
    task = {"id": "gap_repeat", "query": "already queued", "gap_id": "gap_repeat"}
    gap = {
        "id": "gap_repeat", "query": task["query"], "reason": "not closed",
        "priority": "high", "source": "judge", "status": "open", "attempts": 1,
        "task_ids": [task["id"]], "completed_task_ids": [], "claims_brought": 0,
        "empty_attempts": 0,
    }
    root = _verifying_topic(
        tmp_path, [], queue=[task], gap_history_version=1, gaps=[gap],
    )
    collected = []

    def responder(role, prompt):
        assert role == "collect"
        collected.append(prompt)
        return json.dumps({"sources": [], "claims": []})

    adapter = FakeHarness(
        responder,
        judge_responder=lambda _: _judge_json(
            "continue",
            gaps=[{"query": "  ALREADY QUEUED ", "reason": "duplicate", "priority": "high"}],
        ),
    )
    orch = Orchestrator(
        root, adapter=adapter, wiki=FakeWiki(),
        config=RunConfig(max_cycles=5, saturation_rounds=2),
    )

    assert orch._phase_verify() == "done"
    state = checkpoint.load(root)
    assert state["phase"] == "collecting" and state["queue"] == [task]

    assert orch._phase_collect() == "done"
    state = checkpoint.load(root)
    assert collected and task["id"] in state["done"] and state["queue"] == []


def test_judge_continue_extends_search_map_and_queue(tmp_path):
    root = _verifying_topic(tmp_path, [])
    (root / "work" / "search-map.json").write_text(json.dumps({
        "version": 1, "topic": "why pytest fails", "clusters": [
            {"id": "c1", "query": "old"}
        ]}), "utf-8")
    adapter = FakeHarness(lambda *_: pytest.fail("critic"), judge_responder=lambda _: _judge_json(
        "continue", clusters=[{"query": "new cluster"}]))

    Orchestrator(root, adapter=adapter, wiki=FakeWiki(), config=RunConfig())._phase_verify()

    search_map = json.loads((root / "work" / "search-map.json").read_text("utf-8"))
    assert [row["query"] for row in search_map["clusters"]] == ["old", "new cluster"]
    assert checkpoint.load(root)["queue"][0]["query"] == "new cluster"


def test_judge_exhausted_gap_is_rejected_and_continue_becomes_enough(tmp_path):
    root = _verifying_topic(tmp_path, [], gap_history_version=1, gaps=[{
        "id": "gap_x", "query": "exhausted", "reason": "empty twice",
        "priority": "high", "source": "judge", "status": "exhausted",
        "attempts": 2, "task_ids": ["gap_x", "gap_x_a2"], "claims_brought": 0,
        "completed_task_ids": ["gap_x", "gap_x_a2"], "empty_attempts": 2,
    }])
    adapter = FakeHarness(lambda *_: pytest.fail("critic"), judge_responder=lambda _: _judge_json(
        "continue", gaps=[{"query": "  EXHAUSTED ", "reason": "one more time",
                            "priority": "high"}]))

    Orchestrator(root, adapter=adapter, wiki=FakeWiki(), config=RunConfig())._phase_verify()

    state = checkpoint.load(root)
    assert state["stop"]["reason"] == "enough"
    assert state["stop"]["why"] == "the judge gave no new directions"
    events = [json.loads(line) for line in (root / "work" / "events.jsonl").read_text().splitlines()]
    assert any(row["event"] == "gap_rejected_exhausted" for row in events)


def test_saturated_overrules_judge_continue_after_two_empty_accepted_rounds(tmp_path):
    root = _verifying_topic(
        tmp_path, [], cycles=2,
        rounds=[{"cycle": 1, "accepted_claims": 0}, {"cycle": 2, "new_claims": 0}],
        rounds_without_claim=1,
    )
    adapter = FakeHarness(lambda *_: pytest.fail("critic"), judge_responder=lambda _: _judge_json(
        "continue", gaps=[{"query": "more", "reason": "want", "priority": "high"}]))

    Orchestrator(root, adapter=adapter, wiki=FakeWiki(), config=RunConfig(max_cycles=9))._phase_verify()

    state = checkpoint.load(root)
    assert state["stop"]["reason"] == "saturated" and state["phase"] == "synthesizing"
    events = [json.loads(line) for line in (root / "work" / "events.jsonl").read_text().splitlines()]
    assert any(row["event"] == "judge_overruled" and row["reason"] == "saturated"
               for row in events)


def test_max_cycles_overrules_judge_and_publishes_path(tmp_path):
    root = _verifying_topic(tmp_path, [], cycles=2)
    adapter = FakeHarness(lambda *_: pytest.fail("critic"), judge_responder=lambda _: _judge_json(
        "continue", gaps=[{"query": "more", "reason": "want", "priority": "high"}]))

    Orchestrator(root, adapter=adapter, wiki=FakeWiki(),
                 config=RunConfig(max_cycles=2, saturation_rounds=9))._phase_verify()

    assert checkpoint.load(root)["stop"]["reason"] == "max_cycles"


def test_gap_becomes_exhausted_after_two_zero_claim_attempts(tmp_path):
    root = _topic(tmp_path, phase="verifying", cycles=0, rounds_without_claim=0,
                  gap_history_version=1, gaps=[])
    orch = Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=FakeWiki(),
                        config=RunConfig(max_cycles=5))
    state = checkpoint.load(root)
    raw = [{"query": "hard gap", "reason": "no data", "priority": "high"}]
    assert orch._queue_gaps(state, raw, source="judge")
    first = state["queue"][0]
    orch._complete_gap_task(state, first, 0)
    assert state["gaps"][0]["status"] == "done"
    state["queue"] = []
    assert orch._queue_gaps(state, raw, source="judge")
    second = state["queue"][0]
    orch._complete_gap_task(state, second, 1)
    assert state["gaps"][0]["status"] == "done"
    state["queue"] = []
    assert orch._queue_gaps(state, raw, source="judge")
    third = state["queue"][0]
    orch._complete_gap_task(state, third, 0)
    assert state["gaps"][0]["status"] == "done"
    state["queue"] = []
    assert orch._queue_gaps(state, raw, source="judge")
    fourth = state["queue"][0]
    orch._complete_gap_task(state, fourth, 0)
    assert state["gaps"][0]["status"] == "exhausted"
    assert state["gaps"][0]["attempts"] == 4
    assert state["gaps"][0]["empty_attempts"] == 2


def test_gap_queue_and_completion_are_idempotent_per_task(tmp_path):
    root = _topic(
        tmp_path, phase="verifying", gap_history_version=1, gaps=[], queue=[],
    )
    orch = Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=FakeWiki())
    state = checkpoint.load(root)
    raw = [
        {"query": " One gap ", "reason": "first one", "priority": "high"},
        {"query": "one GAP", "reason": "duplicate", "priority": "medium"},
    ]

    assert orch._queue_gaps(state, raw, source="judge") is True
    assert len(state["queue"]) == 1
    task = state["queue"][0]
    orch._complete_gap_task(state, task, 0)
    orch._complete_gap_task(state, task, 0)

    gap = state["gaps"][0]
    assert gap["empty_attempts"] == 1
    assert gap["completed_task_ids"] == [task["id"]]


def test_max_cycles_wins_when_saturation_threshold_is_also_reached(tmp_path):
    root = _topic(tmp_path, phase="verifying", cycles=2, rounds_without_claim=2)
    orch = Orchestrator(
        root, adapter=FakeHarness(lambda *_: ""), wiki=FakeWiki(),
        config=RunConfig(max_cycles=2, saturation_rounds=2),
    )

    assert orch._collection_stop_reason(checkpoint.load(root)) == "max_cycles"


def test_judge_validator_normalizes_all_optional_fields_without_rejecting():
    data = {
        "verdict": "continue", "why": "x" * 800,
        "coverage_md": " ".join("word" for _ in range(1600)),
        "gaps": [{"query": str(index)} for index in range(8)] + ["bad"],
        "new_clusters": [{"query": str(index)} for index in range(5)],
        "open_questions": ["q" * 250 for _ in range(8)] + [3],
    }

    Orchestrator._validate_judge(data)

    assert len(data["why"]) == 600
    assert len(data["coverage_md"].split()) == 1500
    assert len(data["gaps"]) == 3 and len(data["new_clusters"]) == 2
    assert len(data["open_questions"]) == 5
    assert all(len(question) == 200 for question in data["open_questions"])


@pytest.mark.parametrize("data", [{}, {"verdict": "unknown"}, {"verdict": 3}])
def test_judge_validator_only_rejects_missing_or_invalid_verdict(data):
    with pytest.raises(ValueError, match="verdict"):
        Orchestrator._validate_judge(data)


@pytest.mark.parametrize("data", [
    {"verdict": "enough"},
    {"verdict": "stuck", "why": None, "gaps": "bad", "open_questions": {}},
    {"verdict": "continue", "coverage_md": 3, "new_clusters": [None]},
])
def test_judge_validator_accepts_missing_or_malformed_optional_fields(data):
    Orchestrator._validate_judge(data)

    assert data["why"] == ""
    assert isinstance(data["coverage_md"], str)
    assert isinstance(data["gaps"], list)
    assert isinstance(data["new_clusters"], list)
    assert isinstance(data["open_questions"], list)


def test_legacy_coverage_input_is_capped_at_1500_words(tmp_path):
    root = _topic(tmp_path, phase="verifying")
    path = root / "work" / "coverage.md"
    path.write_text(" ".join(f"word{i}" for i in range(1600)), "utf-8")
    orch = Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=FakeWiki())

    assert len(orch._coverage_text().split()) == 1500


def test_cycle_done_notice_is_not_repeated_on_same_cycle_resume(tmp_path):
    class Notifier:
        def __init__(self):
            self.events = []

        def emit(self, event, text, **notice):
            self.events.append((event, text, notice))

    root = _verifying_topic(tmp_path, [])
    notifier = Notifier()
    orch = Orchestrator(
        root, adapter=FakeHarness(lambda *_: pytest.fail("critic")), wiki=FakeWiki(),
        config=RunConfig(), notifier=notifier,
    )
    assert orch._phase_verify() == "done"
    state = checkpoint.load(root)
    state["phase"] = "verifying"
    checkpoint.save(root, state)
    assert orch._phase_verify() == "done"

    cycle_notices = [row for row in notifier.events if row[0] == "cycle_done"]
    assert len(cycle_notices) == 1
    notice = cycle_notices[0][2]
    assert notice["kind"] == "progress" and notice["title"] == "cycle 1/3"
    assert notice["lines"][0].startswith(
        "staging 0 (+0 this cycle), sources 0. Judge: enough - "
    )
    assert notice["lines"][1].startswith("Rough pace: ~")


def test_phase_changed_notices_name_verification_and_synthesis(tmp_path):
    class Notifier:
        def __init__(self):
            self.events = []

        def emit(self, event, text, **notice):
            self.events.append((event, text, notice))

    root = _topic(tmp_path, phase="collecting", cycles=1)
    notifier = Notifier()
    orch = Orchestrator(
        root, adapter=FakeHarness(lambda *_: pytest.fail("harness")), wiki=FakeWiki(),
        notifier=notifier,
    )

    assert orch._phase_collect() == "done"
    orch._transition_to_synthesis(checkpoint.load(root), "enough", "there is enough data")

    phase_notices = [row[2] for row in notifier.events if row[0] == "phase_changed"]
    assert [notice["title"] for notice in phase_notices] == ["verify", "synth"]
    assert all(notice["kind"] == "progress" for notice in phase_notices)
    assert "Reason: the judge decided" in phase_notices[1]["lines"][0]


def test_cycle_notice_caps_judge_reason_at_120_characters(tmp_path, monkeypatch):
    class Notifier:
        def __init__(self):
            self.events = []

        def emit(self, event, text, **notice):
            self.events.append((event, notice))

    root = _verifying_topic(tmp_path, [])
    notifier = Notifier()
    orch = Orchestrator(
        root, adapter=FakeHarness(lambda *_: pytest.fail("harness")), wiki=FakeWiki(),
        notifier=notifier,
    )
    monkeypatch.setattr(
        orch, "_judge",
        lambda *args: {"verdict": "enough", "why": "x" * 300, "gaps": []},
    )
    monkeypatch.setattr(orch, "_apply_judge", lambda *args: None)

    assert orch._phase_verify() == "done"

    line = next(notice for event, notice in notifier.events if event == "cycle_done")[
        "lines"
    ][0]
    assert len(line.rsplit(" - ", 1)[1]) == 120


def test_cycle_notice_average_uses_elapsed_time_of_completed_cycles(
    tmp_path, monkeypatch
):
    root = _topic(tmp_path)
    orch = Orchestrator(
        root, adapter=FakeHarness(lambda *_: pytest.fail("harness")), wiki=FakeWiki(),
    )
    orch._notice_started_at = 0
    monkeypatch.setattr("researcher.orchestrator.time.monotonic", iter([600, 1200]).__next__)

    assert orch._cycle_minutes() == 10
    assert orch._cycle_minutes() == 10


def test_done_notice_has_counts_breadth_and_folder(tmp_path, monkeypatch):
    class Notifier:
        def __init__(self):
            self.events = []

        def emit(self, event, text, **notice):
            self.events.append((event, text, notice))

    root = _topic(tmp_path, phase="done", cycles=4, terminal_notices={})
    notifier = Notifier()
    orch = Orchestrator(
        root, adapter=FakeHarness(lambda *_: pytest.fail("harness")), wiki=FakeWiki(),
        notifier=notifier,
    )
    monkeypatch.setattr(orch, "_finish_done_pending", lambda: None)
    monkeypatch.setattr(orch, "_gate_published", lambda: None)
    monkeypatch.setattr(orch, "_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        orch, "_topic_breadth",
        lambda: {"distinct_domains": 5, "distinct_kinds": 2},
    )
    monkeypatch.setattr(
        "researcher.orchestrator.topic_counts",
        lambda _root: {"final": 9, "pages": 3, "cycles": 4},
    )

    assert orch._run_locked() == EXIT_DONE

    started = next(row[2] for row in notifier.events if row[0] == "run_started")
    assert started["kind"] == "info" and started["title"] == "start"
    notice = next(row[2] for row in notifier.events if row[0] == "done")
    assert notice["kind"] == "done" and notice["title"] == "done"
    assert notice["lines"] == [
        "9 claims, 3 pages, 4 cycles. Coverage: domains 5, "
        "source types 2 of 3.",
        f"Folder: {root}",
    ]
def test_judge_context_caps_new_claims_at_60_and_adds_more_note(tmp_path):
    candidates = [_candidate(f"fact {i}", confidence="high") for i in range(65)]
    root = _verifying_topic(tmp_path, candidates)
    orch = Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=FakeWiki())
    verdicts = {
        hashlib.sha256(cand["text"].lower().encode()).hexdigest(): {
            "verdict": "keep", "confidence": "high", "cycle": 1,
        } for cand in candidates
    }
    accepted = orch._accepted_claims_for_cycle(candidates, verdicts, 1)

    context = orch._judge_context(checkpoint.load(root), candidates, verdicts, accepted)

    assert len(context["new_accepted_claims"]) == 60
    assert context["new_accepted_note"] == "and 5 more"


def test_judge_context_bounds_and_orders_four_hundred_gaps(tmp_path):
    gaps = []
    for index in range(400):
        status = "open" if index < 30 else "exhausted" if index < 50 else "done"
        gaps.append({
            "id": f"gap_{index}",
            "query": f"query {index} " + "x" * 300,
            "reason": "reason " + "y" * 300,
            "priority": "high" if index % 3 == 0 else "medium",
            "source": "judge", "status": status, "claims_brought": index,
            "attempts": 1, "task_ids": [f"gap_{index}"],
            "completed_task_ids": [], "empty_attempts": 0,
        })
    root = _verifying_topic(
        tmp_path, [], gap_history_version=1, gaps=gaps,
    )
    orch = Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=FakeWiki())

    context = orch._judge_context(checkpoint.load(root), [], {}, [])
    prompt = _prompt_judge(context)

    assert len(context["gaps"]["open"]) == 20
    priorities = [row["priority"] for row in context["gaps"]["open"]]
    assert priorities == sorted(priorities, key={"high": 0, "medium": 1}.get)
    high_queries = [row["query"] for row in context["gaps"]["open"]
                    if row["priority"] == "high"]
    assert high_queries[0].startswith("query 27")
    assert len(context["gaps"]["exhausted"]) == 10
    assert all(set(row) == {"query"} and len(row["query"]) <= 160
               for row in context["gaps"]["exhausted"])
    assert context["gaps"]["rest"].startswith("350 closed, they brought ")
    assert len(prompt.encode("utf-8")) < 40_000


def test_judge_prompt_has_one_utf8_budget_for_maximal_cyrillic_sections():
    context = {
        "topic": "т" * 500,
        "search_map": [
            {"id": "к" * 80, "query": "к" * 160, "status": "done", "accepted_claims": 999}
            for _ in range(40)
        ],
        "breadth": {"sources": 999, "domains": 999, "source_types": 999},
        "gaps": {
            "open": [
                {"query": "о" * 240, "source": "judge", "priority": "high",
                 "claims_brought": 999}
                for _ in range(20)
            ],
            "exhausted": [{"query": "и" * 160} for _ in range(10)],
            "rest": "закрыто 999, принесли 999 клеймов",
        },
        "new_accepted_claims": ["ф" * 240 for _ in range(60)],
        "new_accepted_note": "и ещё 999",
        "coverage_md": "п" * 8000,
        "counters": {
            "cycle": 999, "max_cycles": 999,
            "rounds_without_new_claims": 999, "staging": 999,
        },
    }

    prompt = _prompt_judge(context)

    assert len(prompt.encode("utf-8")) < 40_000
    serialized = json.loads(prompt.partition("Bounded input:\n")[2])
    assert serialized["gaps"]["exhausted"] == []
    assert len(serialized["gaps"]["open"]) <= 10
    assert len(serialized["new_accepted_claims"]) <= 30


def test_judge_persists_bounded_structured_open_questions(tmp_path):
    root = _verifying_topic(tmp_path, [])
    questions = [f"question {index} " + "x" * 300 for index in range(8)]
    adapter = FakeHarness(
        lambda *_: pytest.fail("critic"),
        judge_responder=lambda _: _judge_json(questions=questions),
    )

    Orchestrator(root, adapter=adapter, wiki=FakeWiki(), config=RunConfig())._phase_verify()

    state = checkpoint.load(root)
    assert state["coverage"]["open_questions"] == state["judge"]["open_questions"]
    assert len(state["coverage"]["open_questions"]) == 5
    assert all(len(question) == 200 for question in state["coverage"]["open_questions"])


def test_empty_judge_questions_have_explicit_coverage_text(tmp_path):
    root = _topic(tmp_path, phase="synthesizing", coverage={"open_questions": []})
    orch = Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=FakeWiki())

    section = orch._coverage_section(checkpoint.load(root))

    assert "the judge named no open questions" in section


def test_coverage_demotes_model_headings_and_rewrite_stays_single(tmp_path):
    root = _topic(
        tmp_path, phase="synthesizing",
        coverage={"open_questions": ["what is left?"]},
    )
    (root / "work" / "coverage.md").write_text(
        "# Top\n\n## Nested\n\n### Deep\n", "utf-8",
    )
    orch = Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=FakeWiki())
    body = "# Page\n\ntext"
    for _ in range(3):
        body = _append_system_section(
            _without_system_section(body, "Coverage"),
            orch._coverage_section(checkpoint.load(root)),
        )

    assert body.count("## Coverage") == 1
    coverage_tail = body.split("## Coverage", 1)[1]
    assert not any(line.lstrip().startswith("#") for line in coverage_tail.splitlines())
    assert "**Top**" in coverage_tail and "**Deep**" in coverage_tail


def test_coverage_flattens_model_lines_and_resume_does_not_duplicate_section(tmp_path):
    root = _topic(
        tmp_path, phase="synthesizing",
        stop={"reason": "enough", "why": "# first\n## why substitution"},
        coverage={"open_questions": ["question\n## Heading"]},
        gaps=[{
            "source": "synth", "status": "unqueued",
            "query": "# query\n## query substitution",
            "reason": "# reason\n## reason substitution",
        }],
    )
    orch = Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=FakeWiki())
    body = "# Page\n\ntext"

    for _ in range(3):
        body = _append_system_section(
            _without_system_section(body, "Coverage"),
            orch._coverage_section(checkpoint.load(root)),
        )

    assert body.count("## Coverage") == 1
    coverage_tail = body.split("## Coverage", 1)[1]
    assert not any(line.lstrip().startswith("#") for line in coverage_tail.splitlines())
    assert "question Heading" in coverage_tail
    assert "first why substitution" in coverage_tail


def test_coverage_lists_only_first_fifteen_domains(tmp_path, monkeypatch):
    root = _topic(tmp_path, phase="synthesizing")
    orch = Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=FakeWiki())
    breadth = {
        "sources": 20, "distinct_domains": 20,
        "domains": [f"d{index}.example" for index in range(20)],
        "distinct_kinds": 1, "kinds": ["web"],
        "domain_gate": True, "kind_gate": False,
    }
    monkeypatch.setattr(orch, "_topic_breadth", lambda: breadth)

    section = orch._coverage_section(checkpoint.load(root))

    assert "d14.example, and 5 more" in section
    assert "d15.example" not in section


def test_missing_or_broken_search_map_extension_is_event_not_fatal(tmp_path):
    root = _topic(tmp_path, phase="verifying")
    orch = Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=FakeWiki())
    state = checkpoint.load(root)

    assert orch._extend_search_map(state, [{"query": "new"}]) == []
    (root / "work" / "search-map.json").write_text("[]", "utf-8")
    assert orch._extend_search_map(state, [{"query": "more"}]) == []

    events = [json.loads(line) for line in
              (root / "work" / "events.jsonl").read_text("utf-8").splitlines()]
    assert [row["event"] for row in events].count("search_map_extend_failed") == 2


def test_judge_reads_previous_coverage_and_updates_it_atomically(tmp_path):
    root = _verifying_topic(tmp_path, [])
    (root / "work" / "coverage.md").write_text("previous coverage", "utf-8")
    seen = []
    adapter = FakeHarness(lambda *_: pytest.fail("critic"), judge_responder=lambda prompt: (
        seen.append(prompt) or _judge_json(coverage="updated coverage")))

    Orchestrator(root, adapter=adapter, wiki=FakeWiki(), config=RunConfig())._phase_verify()

    assert "previous coverage" in seen[0]
    assert (root / "work" / "coverage.md").read_text("utf-8") == "updated coverage\n"
    assert not list((root / "work").glob("coverage.md.*.tmp"))


def test_judge_heartbeat_role_is_closed_and_visible(tmp_path):
    root = _topic(tmp_path, phase="verifying", cycles=4)
    orch = Orchestrator(root, adapter=FakeHarness(lambda *_: "{}"), wiki=FakeWiki())

    orch._call("judge", role="think", tools="", allowed_tools="", heartbeat_role="judge")

    heartbeat = read_heartbeat(root)
    assert heartbeat is not None and heartbeat["role"] == "judge" and heartbeat["cycle"] == 4


def test_verdict_key_uses_dedup_normalization():
    from researcher.orchestrator import _claim_verdict_key

    assert _claim_verdict_key("  Fact\nwith  spaces ") == _claim_verdict_key("fact with spaces")


def test_breadth_shortfall_is_coverage_note_not_refusal(tmp_path):
    root = _topic(tmp_path)
    adapter = FakeHarness(_scripted(
        [{"id": "t1", "query": "q"}],
        {"q": {"sources": _src(1), "claims": [_claim("one fact")] }},
        page_ready=False,
    ))

    assert Orchestrator(
        root, adapter=adapter, wiki=FakeWiki(),
        config=RunConfig(max_cycles=1, pool_size=1),
    ).run() == EXIT_DONE

    state = checkpoint.load(root)
    assert state["phase"] == "done" and state["stop"]["reason"] == "max_cycles"
    synthesis = (root / "work" / "synthesis.md").read_text("utf-8")
    assert "## Coverage" in synthesis
    assert "1 of 3 source types" in synthesis
    assert state.get("stop_kind") != "gate_refused"


def test_coverage_section_is_written_to_every_created_page(tmp_path):
    root = _topic(tmp_path)
    wiki = FakeWiki()
    adapter = FakeHarness(_scripted(
        [{"id": "t1", "query": "q"}],
        {"q": {"sources": _src(1), "claims": [_claim("one fact")] }},
    ))

    assert Orchestrator(root, adapter=adapter, wiki=wiki,
                        config=RunConfig(max_cycles=1, pool_size=1)).run() == EXIT_DONE

    assert wiki.pages
    assert all("## Coverage" in page["body"] for page in wiki.pages.values())
    events = [json.loads(line) for line in (root / "work" / "events.jsonl").read_text().splitlines()]
    assert sum(row["event"] == "coverage_note" for row in events) == 1


def test_page_under_three_facts_is_not_created_but_claim_stays_final(tmp_path):
    root = _topic(tmp_path)
    wiki = FakeWiki()
    adapter = FakeHarness(_scripted(
        [{"id": "t1", "query": "q"}],
        {"q": {"sources": _src(1), "claims": [_claim("one fact")] }},
        page_ready=False,
    ))

    assert Orchestrator(root, adapter=adapter, wiki=wiki,
                        config=RunConfig(max_cycles=1, pool_size=1)).run() == EXIT_DONE

    assert len(wiki.claims["final"]) == 1 and wiki.pages == {}
    synthesis = (root / "work" / "synthesis.md").read_text("utf-8")
    assert "## Unpublished pages" in synthesis
    assert "1 fact(s), at least 3 are required" in synthesis


def test_final_claims_without_concept_page_pass_done_gate(tmp_path):
    root = _topic(tmp_path, phase="done")
    wiki = FakeWiki()
    wiki.add_claim(root, "final", text="fact", confidence="medium", status="verified",
                   evidence=[{"source_id": "src_1", "quote": "q", "stance": "supports"}])

    Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=wiki)._gate_published()


def test_resume_old_synthesizing_checkpoint_with_pending_without_migration(tmp_path):
    root = _topic(tmp_path, phase="synthesizing", cycles=24,
                  stop={"reason": "max_cycles", "why": "the cap"})
    wiki = FakeWiki()
    ids = []
    for i in range(3):
        ids.append(wiki.add_claim(
            root, "staging", text=f"fact {i}", confidence="medium", status="verified",
            evidence=[{"source_id": "src_one", "quote": "q", "stance": "supports"}],
        ))
    pending = {
        "claims": [{"index": i, "id": cid} for i, cid in enumerate(ids)],
        "pages": [{"title": "legacy", "claim_indexes": [0, 1, 2],
                   "body_md": _page_body()}],
        "summary_md": "legacy summary",
    }
    (root / "work" / "synthesis-pending.json").write_text(json.dumps(pending), "utf-8")
    adapter = FakeHarness(lambda role, _: pytest.fail(f"role called: {role}"))

    assert Orchestrator(root, adapter=adapter, wiki=wiki,
                        config=RunConfig(max_cycles=24)).run() == EXIT_DONE

    assert checkpoint.load(root)["phase"] == "done"
    assert "## Coverage" in (root / "work" / "synthesis.md").read_text("utf-8")
    assert not [call for call in adapter.calls if call[0] == "collect"]


def test_resume_old_collecting_checkpoint_with_collecting_round_without_migration(tmp_path):
    """A trimmed live collecting checkpoint, from before verdict/judge/gap history."""
    root = _topic(
        tmp_path, phase="collecting", cycles=0, rounds_without_claim=0,
        queue=[{"id": "t2", "query": "q2"}], done=["t1"], run_id="run_legacy",
        collecting_round={
            "cycle": 1, "tasks": ["t1", "t2"], "completed": ["t1"],
            "new_claims": 1,
        },
    )
    _write_found(root, [_candidate("legacy t1", task="t1")], task="t1")
    adapter = FakeHarness(_scripted(
        [{"id": "t2", "query": "q2"}],
        {"q2": {"sources": _src(1), "claims": [_claim("legacy t2")]}},
        page_ready=False,
    ))

    assert Orchestrator(
        root, adapter=adapter, wiki=FakeWiki(),
        config=RunConfig(max_cycles=1, pool_size=1),
    ).run() == EXIT_DONE

    state = checkpoint.load(root)
    assert state["phase"] == "done" and "collecting_round" not in state
    assert state["rounds"][0]["tasks"] == ["t1", "t2"]
    assert len((root / "work" / "verdicts.jsonl").read_text("utf-8").splitlines()) == 2


def test_resume_old_verifying_checkpoint_without_verdicts_or_judge(tmp_path):
    """A trimmed structure of a live verifying checkpoint is checked once and moves on."""
    candidates = [_candidate("legacy A"), _candidate("legacy B")]
    root = _verifying_topic(
        tmp_path, candidates,
        rounds=[{"cycle": 1, "tasks": ["t1"], "new_claims": 2, "found": 2}],
    )
    adapter = FakeHarness(_scripted([], {}, page_ready=False))

    assert Orchestrator(
        root, adapter=adapter, wiki=FakeWiki(),
        config=RunConfig(max_cycles=1),
    ).run() == EXIT_DONE

    state = checkpoint.load(root)
    assert state["phase"] == "done" and state["judge"]["cycle"] == 1
    assert state["rounds"][0]["accepted_claims"] == 2
    assert len((root / "work" / "verdicts.jsonl").read_text("utf-8").splitlines()) == 2


def test_coverage_event_contains_measured_breadth(tmp_path):
    root = _topic(tmp_path)
    adapter = FakeHarness(_scripted(
        [{"id": "t1", "query": "q"}],
        {"q": {"sources": _src(1), "claims": [_claim("fact")] }},
        page_ready=False,
    ))
    Orchestrator(root, adapter=adapter, wiki=FakeWiki(),
                 config=RunConfig(max_cycles=1, pool_size=1)).run()

    events = [json.loads(line) for line in (root / "work" / "events.jsonl").read_text().splitlines()]
    note = next(row for row in events if row["event"] == "coverage_note")
    assert note["breadth"]["distinct_domains"] == 1
    assert note["breadth"]["distinct_kinds"] == 1


def test_cli_returns_77_on_gate_refusal(tmp_path, monkeypatch, capsys):
    """The exit-code contract: a gate refusal is 77, not 1. The watchdog lives on that
    difference."""
    from researcher import cli, orchestrator

    root = _topic(tmp_path)

    def boom(self):
        raise orchestrator.GateRefused(
            "the compiler gap is not closed and the collection safety bound is spent: x")

    monkeypatch.setattr(orchestrator.Orchestrator, "run", boom)
    code = cli.main(["resume", str(root), "--adapter", "claude"])
    assert code == cli.EXIT_GATE_REFUSED == 77
    assert "publication gate refused" in capsys.readouterr().err


def test_new_evidence_merge_against_real_llmwiki(tmp_path):
    """The socket contract is checked against the real llmwiki, not the fake: FK, schema, dedup."""
    from researcher.orchestrator import _claim_verdict_key

    llmwiki_dir = Path(__file__).resolve().parents[2] / "tool-llm-wiki"
    if not (llmwiki_dir / "llmwiki").is_dir():
        pytest.skip("the neighboring tool-llm-wiki repo is missing")
    if str(llmwiki_dir) not in sys.path:
        sys.path.insert(0, str(llmwiki_dir))
    import llmwiki

    root = llmwiki.init_topic(tmp_path, "merge topic", slug="tm")
    (root / "work").mkdir(exist_ok=True)
    checkpoint.save(root, {"phase": "verifying", "topic": "merge topic", "queue": [],
                           "done": [], "sessions": {}, "run_id": "run_test"})
    src_one = llmwiki.add_source(root, kind="web", title="A", url="https://a.example/1")
    src_two = llmwiki.add_source(root, kind="web", title="B", url="https://b.example/2")
    cid = llmwiki.add_claim(
        root, "staging", text="real fact", confidence="medium", status="verified",
        evidence=[{"source_id": src_one, "quote": "first", "stance": "supports"}],
    )
    candidate = _candidate("real fact")
    candidate["evidence"] = [
        {"source_id": src_one, "quote": "first", "stance": "supports"},
        {"source_id": src_two, "quote": "second", "stance": "contradicts"},
        {"source_id": "src_000000000009", "quote": "into nowhere", "stance": "supports"},
    ]
    verdicts = {_claim_verdict_key("real fact"): {"verdict": "keep", "confidence": "high"}}
    orch = Orchestrator(root, adapter=FakeHarness(lambda *_: ""), wiki=llmwiki)

    # an FK-broken entry does not roll back a valid one: each merge is independent and the
    # failure is durable in full
    orch._reconcile_kept_candidates(checkpoint.load(root), [candidate], verdicts)
    rows = list(llmwiki.read_claims(root, zone="staging"))
    assert len(rows) == 1 and [e["source_id"] for e in rows[0]["evidence"]] == [
        src_one, src_two,
    ]
    failed = _events_named(root, "evidence_merge_failed")
    assert len(failed) == 1 and "FK" in failed[0]["error"]
    assert failed[0]["evidence"] == {
        "source_id": "src_000000000009", "quote": "into nowhere", "stance": "supports",
    }
    assert _events_named(root, "evidence_merged")[0]["source_ids"] == [src_two]

    candidate["evidence"].pop()
    orch._reconcile_kept_candidates(checkpoint.load(root), [candidate], verdicts)
    rows = list(llmwiki.read_claims(root, zone="staging"))
    assert len(rows) == 1 and rows[0]["id"] == cid
    assert [e["source_id"] for e in rows[0]["evidence"]] == [src_one, src_two]
    assert llmwiki.validate_topic(root) == []
    assert len(_events_named(root, "evidence_merged")) == 1
