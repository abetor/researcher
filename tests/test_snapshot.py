"""The `snapshot` fragment for `gateway snapshot` (canon 25.08 item 6) in the status of two
flows: `course-queue --dry-run --json` (coursedump) and `status --base --json` (researcher),
plus reading the coursedump catalog.tsv as data and the `--name` flag."""
from __future__ import annotations

import datetime
import fcntl
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from researcher import checkpoint, cli, course_queue
from researcher.cli import EXIT_OK, main


def _iso(timestamp: float) -> str:
    return datetime.datetime.fromtimestamp(
        timestamp, datetime.timezone.utc
    ).isoformat(timespec="seconds")


def _dump(dumps: Path, slug: str) -> None:
    (dumps / slug).mkdir(parents=True, exist_ok=True)
    (dumps / slug / "INDEX.md").write_text("# transcript\n", "utf-8")


def _course(base: Path, slug: str, *, phase: str, done: list[str], started: float, **extra) -> Path:
    root = base / slug
    (root / "work").mkdir(parents=True, exist_ok=True)
    documents = [
        {"nnn": "001", "document_id": "d1", "title": "a"},
        {"nnn": "002", "document_id": "d2", "title": "b"},
        {"nnn": "003", "document_id": "d3", "title": "c"},
        {"nnn": "004", "document_id": "d4", "title": "d"},
    ]
    checkpoint.save(root, {
        "phase": phase, "run_mode": "import", "topic": f"Course: {slug}", "slug": slug,
        "dump": f"/dumps/{slug}", "documents": documents,
        "decisions": {"001": "KEEP", "002": "KEEP", "003": "DROP", "004": "SKIM"},
        "done": done, "started_at": _iso(started), **extra,
    })
    (root / "work" / "import-plan.json").write_text(json.dumps({
        "documents": [
            {"document_id": "d1", "window_count": 3},
            {"document_id": "d2", "window_count": 2},
            {"document_id": "d3", "window_count": 7},
            {"document_id": "d4", "window_count": 5},
        ],
    }), "utf-8")
    return root


def _touch(path: Path, when: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", "utf-8")
    os.utime(path, (when, when))


# --- catalog.tsv --------------------------------------------------------------

def test_read_catalog_missing_or_none_is_empty(tmp_path):
    assert course_queue.read_catalog(None) == {}
    assert course_queue.read_catalog(tmp_path / "missing.tsv") == {}
    assert course_queue.default_catalog(tmp_path / "out") is None


def test_read_catalog_uses_header_columns_and_skips_comments(tmp_path):
    path = tmp_path / "catalog.tsv"
    path.write_text(
        "# coursedump course registry: one line = one course\n"
        "# title/url/note are filled in by a human\n"
        "\n"
        "slug\ttitle\turl\tmirror\tacquired\n"
        "ab-tappliedaimc\tThe Applied AI Masterclass\thttps://a\t-\t2026-06-16\n"
        "bc-empty\t-\thttps://b\t-\t2026-06-16\n"
        "short-row\n",
        "utf-8",
    )
    titles = course_queue.read_catalog(path)
    assert titles == {"ab-tappliedaimc": "The Applied AI Masterclass"}
    assert course_queue.default_catalog(tmp_path / "out") == path


def test_read_catalog_without_header_takes_first_two_columns(tmp_path):
    path = tmp_path / "plain.tsv"
    path.write_text("x-course\tCourse name\tother\n", "utf-8")
    assert course_queue.read_catalog(path) == {"x-course": "Course name"}


# --- coursedump snapshot ------------------------------------------------------

def test_course_queue_dry_run_json_carries_snapshot_schema(tmp_path, capfd):
    now = float(int(time.time()))
    dumps = tmp_path / "coursedump" / "out"
    base = tmp_path / "topics"
    for slug in ("maven-x", "next-x", "bad-x"):
        _dump(dumps, slug)
    (dumps.parent / "catalog.tsv").write_text(
        "# catalog\nslug\ttitle\nmaven-x\tEnd-to-End Bootcamp\nbad-x\tBad course\n", "utf-8"
    )
    queue = dumps.parent / "queue.txt"
    queue.write_text("maven-x\nnext-x\nbad-x\n", "utf-8")
    running = _course(
        base, "maven-x", phase="import:verify",
        done=["extract:001", "extract:002", "extract:004", "verify:001"], started=now - 7200,
    )
    (running / "work" / "heartbeat.json").write_text(
        json.dumps({"pid": 4242, "role": "verify", "phase": "import:verify"}), "utf-8"
    )
    for nnn in ("001", "002", "004"):
        _touch(running / "work" / "extracted" / f"{nnn}.json", now - 600)
    (running / "staging").mkdir()
    (running / "staging" / "claims.jsonl").write_text('{"id":1}\n{"id":2}\n{"id":3}\n', "utf-8")
    _course(base, "bad-x", phase="import:extract", done=[], started=now - 100, stop_kind="fatal")

    args = ["course-queue", "--queue", str(queue), "--dumps", str(dumps), "--base", str(base)]
    with (running / "work" / "resume.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert main(args + ["--dry-run", "--json"]) == EXIT_OK
    payload = json.loads(capfd.readouterr().out)

    snapshot = payload["snapshot"]
    assert snapshot["flow"] == "coursedump"
    assert set(snapshot) == {"flow", "workers", "queue"}
    [worker] = snapshot["workers"]
    assert set(worker) == {
        "item", "name", "pid", "job", "stage", "done", "total", "started_at",
        "eta_seconds", "eta_total_seconds", "state", "note", "metrics",
    }
    assert worker["item"] == "maven-x" and worker["name"] == "End-to-End Bootcamp"
    assert worker["pid"] == 4242 and worker["job"] is None
    assert worker["stage"] == {"i": 2, "n": 3, "name": "verify"}
    # verify:001 out of the three selected documents (003 is DROP).
    assert (worker["done"], worker["total"]) == (1, 3)
    assert worker["started_at"] == _iso(now - 7200)
    # Stage speed: 600 s for one lesson -> two more lessons ~1200 s; then pages by the
    # model of 72 s per window x 10 windows (3+2+5, the dropped document does not count).
    assert 1150 <= worker["eta_seconds"] <= 1250
    assert 1150 + 720 <= worker["eta_total_seconds"] <= 1250 + 720
    assert worker["state"] == "running" and worker["note"] is None
    assert worker["metrics"] == ["windows 10", "claims 3"]
    assert snapshot["queue"] == [
        {"item": "next-x", "name": None, "state": "waiting"},
        {"item": "bad-x", "name": "Bad course", "state": "fail"},
    ]
    # The old part of the summary is unchanged.
    assert payload["schema_version"] == 1 and payload["remaining"] == 3


def test_course_snapshot_pages_stage_counts_plan_pages_by_receipt_speed(tmp_path):
    now = float(int(time.time()))
    base = tmp_path / "topics"
    root = _course(base, "pages-x", phase="import:pages", done=[], started=now - 3600)
    (root / "work" / "pages-plan.json").write_text(json.dumps([
        {"page": "a", "status": "done"}, {"page": "b", "status": "done"}, {"page": "c", "status": "todo"},
    ]), "utf-8")
    _touch(root / "work" / "import-receipt.json", now - 300)
    (root / "work" / "resume.lock").write_text("pid=77\n", "utf-8")

    snapshot = course_queue.snapshot(
        [{"slug": "pages-x", "state": "running"}], base, None, now=now,
    )
    [worker] = snapshot["workers"]
    assert worker["name"] is None and worker["pid"] == 77
    assert worker["stage"] == {"i": 3, "n": 3, "name": "pages"}
    assert (worker["done"], worker["total"]) == (2, 3)
    assert worker["eta_seconds"] == 150 and worker["eta_total_seconds"] == 150


def test_course_snapshot_falls_back_to_window_model_without_stage_speed(tmp_path):
    now = float(int(time.time()))
    base = tmp_path / "topics"
    _course(base, "fresh-x", phase="import:extract", done=[], started=now - 60)
    snapshot = course_queue.snapshot([{"slug": "fresh-x", "state": "running"}], base, None, now=now)
    [worker] = snapshot["workers"]
    assert worker["stage"] == {"i": 1, "n": 3, "name": "extract"} and worker["pid"] is None
    assert (worker["done"], worker["total"]) == (0, 3)
    # 10 windows x 120 s minus the 60 s elapsed; the extract stage is 18 s per window.
    assert worker["eta_total_seconds"] == 10 * 120 - 60
    assert worker["eta_seconds"] == 10 * 18

    root = _course(base, "noplan-x", phase="import:pages", done=[], started=now - 60)
    snapshot = course_queue.snapshot([{"slug": "noplan-x", "state": "running"}], base, None, now=now)
    [worker] = snapshot["workers"]
    assert (worker["done"], worker["total"]) == (0, None)
    assert worker["eta_seconds"] == 10 * 72 and worker["eta_total_seconds"] == 10 * 120 - 60
    assert not (root / "work" / "pages-plan.json").exists()


def test_course_snapshot_without_checkpoint_is_a_bare_running_worker(tmp_path):
    # The reservation is taken but the topic is not created yet: a worker, no stage.
    base = tmp_path / "topics"
    snapshot = course_queue.snapshot([{"slug": "new-x", "state": "running"}], base, None)
    assert snapshot["workers"] == [{
        "item": "new-x", "name": None, "pid": None, "job": None, "stage": None,
        "done": None, "total": None, "started_at": None, "eta_seconds": None,
        "eta_total_seconds": None, "state": "running", "note": None, "metrics": [],
    }]


# --- researcher snapshot ------------------------------------------------------

def _search_topic(base: Path, folder: str, topic: str, **state) -> Path:
    root = base / folder
    (root / "work").mkdir(parents=True)
    (root / "topic.json").write_text(json.dumps({"topic": topic}), "utf-8")
    checkpoint.save(root, {"topic": topic, **state})
    return root


def test_status_base_json_carries_researcher_snapshot(tmp_path, monkeypatch, capfd):
    base = tmp_path / "topics"
    long_topic = "Senior Go engineer interview meta 2026: formats, what is actually asked"
    moving = _search_topic(
        base, "moving", long_topic, phase="collecting", cycles=3, max_cycles=24,
        run_id="run_20260819T151313660936", name="sobes-meta", done=["c1"],
        judge={"verdict": "continue", "cycle": 3, "why": "w" * 150},
    )
    (moving / "work" / "search-map.json").write_text(
        json.dumps({"clusters": [{"id": "c1"}, {"id": "c2"}]}), "utf-8"
    )
    _search_topic(base, "noname", long_topic, phase="synthesizing", cycles=5, max_cycles=5)
    _search_topic(base, "idle", "standing with no process", phase="collecting", cycles=1)
    _search_topic(
        base, "course", "Course: bc-x", phase="import:extract", run_mode="import", slug="bc-x",
    )
    live = {"moving": 4321, "noname": 4322, "course": 4323}
    monkeypatch.setattr(cli, "_live_lock_pid", lambda topic: live.get(Path(topic).name))
    monkeypatch.setattr(cli, "_process_age_minutes", lambda pid: 1.0)
    wiki = SimpleNamespace(stats=lambda topic: {"claims_staging": 5, "claims_final": 7, "sources": 9})
    monkeypatch.setattr(cli, "_llmwiki", lambda machine=False: wiki)

    rows = cli._status_base_rows(base, stale_minutes=90, wiki=wiki)
    snapshot = cli._status_snapshot(base, rows)

    assert snapshot["flow"] == "researcher" and snapshot["queue"] == []
    workers = {worker["item"]: worker for worker in snapshot["workers"]}
    # An import topic belongs to the coursedump flow, course-queue reports it; a standing
    # topic is not a worker.
    assert set(workers) == {"sobes-meta", "Senior Go engineer interview meta 2026:"}
    worker = workers["sobes-meta"]
    assert set(worker) == {
        "item", "name", "pid", "job", "stage", "done", "total", "started_at",
        "eta_seconds", "eta_total_seconds", "state", "note", "metrics",
    }
    assert worker["name"] == long_topic and worker["pid"] == 4321 and worker["job"] is None
    assert worker["stage"] == {"i": 1, "n": 3, "name": "cycles"}
    assert (worker["done"], worker["total"]) == (3, 24)
    assert worker["started_at"] == "2026-08-19T15:13:13+00:00"
    assert worker["eta_seconds"] is None and worker["eta_total_seconds"] is None
    assert worker["state"] == "running"
    assert worker["note"] == "judge: continue - " + "w" * 100
    assert worker["metrics"] == ["claims 12", "sources 9", "map 1/2"]
    other = workers["Senior Go engineer interview meta 2026:"]
    assert other["stage"] == {"i": 2, "n": 3, "name": "synth"}
    # Without run_id the run start is unknown - the checkpoint's updated_at is used.
    assert other["started_at"] == checkpoint.load(base / "noname")["updated_at"]
    assert other["note"] is None
    assert other["metrics"] == ["claims 12", "sources 9"]

    # The same snapshot appears in the machine output of status --base --json.
    assert main(["status", "--base", str(base), "--json"]) == EXIT_OK
    payload = json.loads(capfd.readouterr().out)
    assert payload["snapshot"] == snapshot


# --- --name ---------------------------------------------------------------------

def test_name_flag_is_stored_in_checkpoint_on_resume_and_start(tmp_path, monkeypatch):
    root = _search_topic(tmp_path / "topics", "t", "a long topic", phase="collecting", cycles=1)
    monkeypatch.setattr(cli, "_resume_once", lambda topic_dir, args, machine: EXIT_OK)
    assert main(["resume", str(root), "--name", "kratko"]) == EXIT_OK
    assert checkpoint.load(root)["name"] == "kratko"
    # Without the flag the name is left alone.
    assert main(["resume", str(root)]) == EXIT_OK
    assert checkpoint.load(root)["name"] == "kratko"

    created = tmp_path / "topics" / "novaya"

    def init_topic(base, topic, slug=None):
        created.mkdir(parents=True)
        (created / "topic.json").write_text(json.dumps({"topic": topic}), "utf-8")
        return created

    monkeypatch.setattr(cli, "_llmwiki", lambda machine=False: SimpleNamespace(init_topic=init_topic))
    assert main([
        "start", "a new topic", "--base", str(tmp_path / "topics"), "--no-run", "--name", "nov",
    ]) == EXIT_OK
    assert checkpoint.load(created)["name"] == "nov"
