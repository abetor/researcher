"""File primitives for atomic writes and a self-ignoring state directory.

This is vendored support code owned by this repository.
"""
from __future__ import annotations

import os
from pathlib import Path

# Self-ignore: git treats the directory as invisible in ANY repo without depending on
# that repo's root .gitignore - state will not slip into `git add .` even in someone
# else's project.
GITIGNORE_BODY = "# tool state directory - not part of the repo\n*\n"


def atomic_write(path: str | Path, data: str | bytes, encoding: str = "utf-8") -> None:
    """Write through a sibling temporary file and an atomic os.replace.

    Readers never observe a partial file. A crashed process can leave a temporary
    fragment, but cannot corrupt the previous value.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(data, str):
        tmp.write_text(data, encoding=encoding)
    else:
        tmp.write_bytes(data)
    os.replace(tmp, path)


def ensure_state_dir(path: str | Path) -> Path:
    """Create a state directory with a self-ignoring .gitignore and return it.

    The operation is idempotent and never overwrites an existing .gitignore.
    Exclusive creation also prevents clobbering during concurrent startup.
    """
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    gi = d / ".gitignore"
    if not gi.exists():
        try:
            with open(gi, "x", encoding="utf-8") as f:
                f.write(GITIGNORE_BODY)
        except FileExistsError:
            pass  # created between exists() and open() - the file is there, which is what we want
    return d
