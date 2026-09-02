"""Isolate tests from the user's disk and credentials: HOME/cwd -> tmp, env -> clean.

A test that accidentally reaches into the real $HOME (configs, state dirs) or
litters the cwd silently damages the owner's machine - we cut that off by
construction, not by discipline. autouse fixtures: every test is hermetic by design.
"""
import re
import sys
from pathlib import Path

import pytest

# Import the package without installing it: tests run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Credential stripping: env vars declared in .env.sample (including the
# commented-out examples) are invisible to tests. Motivating incident: the owner's
# real token in the process env leaks into a "hermetic" test - the test is green
# locally, hits the live API, and is red on a clean machine. Conservative approach:
# we strip ONLY the names listed in .env.sample, no heuristics over all of
# os.environ - wrongly stripping a foreign (PATH-like) variable is worse than a miss.
_ENV_VAR_RE = re.compile(r"([A-Z][A-Z0-9_]*)=")
_CREDENTIAL_ENV_NAMES = ["GITHUB_TOKEN", "EXA_API_KEY", "JINA_API_KEY"]


def _env_sample_vars(sample: Path | None = None) -> list[str]:
    """Variable names from .env.sample; line format: [# ]NAME=   # why."""
    if sample is None:
        return list(_CREDENTIAL_ENV_NAMES)
    if not sample.is_file():
        return []
    names = []
    for line in sample.read_text("utf-8").splitlines():
        m = _ENV_VAR_RE.match(line.lstrip("# ").strip())
        if m:
            names.append(m.group(1))
    return names


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    # The underscore in the name is deliberate: tmp_path/"home" collided with child
    # repo tests creating their own "home" dir in tmp_path (FileExistsError in the pilot).
    home = tmp_path / "_isolated_home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("TOOLS_DATA", raising=False)
    monkeypatch.chdir(tmp_path)
    for name in _env_sample_vars():
        monkeypatch.delenv(name, raising=False)
