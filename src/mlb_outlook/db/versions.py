"""Data and model version identifiers stored on every prediction row."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

# The code the design freeze pins. The model version is the last commit that touched any of
# it, so commits to the app, the comps, or the docs do not mint a new model version.
FROZEN_PATHS = (
    "src/mlb_outlook/models",
    "src/mlb_outlook/features",
    "src/mlb_outlook/evaluation",
    "src/mlb_outlook/config.py",
    "configs/folds.yaml",
)
MODEL_NAME = "ridge_rel+resid_q_pa"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_code_commit(repo: str | Path = ".") -> str | None:
    """Full hash of the last commit touching the frozen paths, or None outside git."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *FROZEN_PATHS],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = result.stdout.strip()
    return commit or None


def model_version(repo: str | Path = ".") -> str:
    """`ridge_rel+resid_q_pa@<12-char commit>`, or `@unversioned` outside git."""
    commit = frozen_code_commit(repo)
    return f"{MODEL_NAME}@{commit[:12] if commit else 'unversioned'}"
