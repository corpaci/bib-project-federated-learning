"""Locate experiments/ so the PaDiM apps can import padim.py.

Flower COPIES the app to ~/.flwr/apps/<fab-id>/ before running it, so a path
relative to this file lands inside that copy, where experiments/ does not exist.
Find the real repo instead, and fail with an actionable message if we cannot.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _candidates(hint: str | None) -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("AITDM_REPO_ROOT")
    if env:                                   # explicit override wins
        out.append(Path(env) / "experiments")
    if hint:
        # partitions-dir is an absolute repo path (data/partitions/<name>), so
        # the repo root is one of its parents. Walk up and look for experiments/.
        # Walk the path AS GIVEN as well as the resolved one: on the cluster
        # data/ can be a symlink out of the repo, so resolving
        # first walks ~/AITDM_Project/... and never passes the repo root.
        for p in dict.fromkeys([Path(hint).absolute(), Path(hint).resolve()]):
            out.extend(parent / "experiments" for parent in [p, *p.parents])
    # Last resort: an editable/source checkout next to this file.
    out.append(Path(__file__).resolve().parents[2] / "experiments")
    return out


def add_experiments_to_path(partitions_dir: str | None = None) -> Path:
    """Put the repo's experiments/ on sys.path and return it."""
    for cand in _candidates(partitions_dir):
        if (cand / "padim.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    raise ModuleNotFoundError(
        "cannot find the repository's experiments/padim.py. Flower runs the app "
        "from a COPY in ~/.flwr/apps, so it is located from 'partitions-dir' "
        f"(given: {partitions_dir!r}). Pass an absolute partitions-dir inside the "
        "repository, or set AITDM_REPO_ROOT=/path/to/Industrial-Anomaly-Detection.")
