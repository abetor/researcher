import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from researcher import checkpoint  # noqa: E402


def test_roundtrip(tmp_path):
    assert checkpoint.load(tmp_path) is None
    checkpoint.save(tmp_path, {"phase": "collecting", "queue": ["q1"], "sessions": {"claude": "s1"}})
    state = checkpoint.load(tmp_path)
    assert state["phase"] == "collecting" and state["sessions"]["claude"] == "s1"
    assert "updated_at" in state
    assert (tmp_path / "work" / "checkpoint.json").exists()


def test_phase_gate(tmp_path):
    with pytest.raises(ValueError):
        checkpoint.save(tmp_path, {"phase": "chilling"})


def test_replace_failure_keeps_old_checkpoint(tmp_path, monkeypatch):
    """Atomic swap: a failure exactly at the swap (os.replace) must not damage the
    existing checkpoint - the core invariant 'any stop leaves a valid checkpoint' is
    enforced, not merely promised. The mutation "write straight into checkpoint.json
    instead of tmp+os.replace" fails this test (without tmp+replace there is no
    exception at all, and the file is already overwritten)."""
    checkpoint.save(tmp_path, {"phase": "collecting", "topic": "t", "queue": [{"id": "t1"}]})
    old = checkpoint.load(tmp_path)

    def boom(src, dst):
        raise OSError("disk died exactly at the swap")

    monkeypatch.setattr(checkpoint.os, "replace", boom)
    with pytest.raises(OSError):
        checkpoint.save(tmp_path, {"phase": "done", "topic": "t", "queue": []})

    assert checkpoint.load(tmp_path) == old  # the old content is intact and reads back in full
    tmps = list((tmp_path / "work").glob("checkpoint.json.*.tmp"))
    assert len(tmps) == 1 and '"done"' in tmps[0].read_text("utf-8")  # the new state landed in tmp...
    assert checkpoint.load(tmp_path)["phase"] == "collecting"   # ...and is NOT treated as the checkpoint


def test_checkpoint_temp_names_are_unique(tmp_path, monkeypatch):
    """Two concurrent saves must not write into one fixed checkpoint.tmp."""
    sources = []

    def boom(src, dst):
        sources.append(Path(src))
        raise OSError("stop before replace")

    monkeypatch.setattr(checkpoint.os, "replace", boom)
    for phase in ("collecting", "done"):
        with pytest.raises(OSError):
            checkpoint.save(tmp_path, {"phase": phase})
    assert len(set(sources)) == 2 and all(p.exists() for p in sources)
