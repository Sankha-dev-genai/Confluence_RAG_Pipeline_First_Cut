"""
Run an ingestion as a subprocess and stream its log lines back to the caller.

Used by the in-app "Add a knowledge base" wizard so a long ingest (download ->
clean -> chunk -> embed -> index) shows live progress without freezing the app,
and so credentials stay in the child process's environment (loaded from .env by
config) and are never handled in the UI.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def stream_ingest(collection: str, steps: list[str] | None = None) -> Iterator[str]:
    """
    Launch `python main.py --collection <collection> [--steps ...]` and yield
    each output line as it arrives. Raises RuntimeError on non-zero exit.
    """
    cmd = [sys.executable, "main.py", "--collection", collection]
    if steps:
        cmd += ["--steps", *steps]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # so log lines arrive promptly

    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line.rstrip("\n")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Ingestion exited with code {proc.returncode}")


def run_ingest_collect(collection: str, steps: list[str] | None = None) -> list[str]:
    """Blocking convenience wrapper: run and return all log lines."""
    return list(stream_ingest(collection, steps))