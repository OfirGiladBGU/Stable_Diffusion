#!/usr/bin/env python3
"""
Recursively find .npz files under given folders, extract the 'voxel' array,
and save it as .npy into an output directory created inside the npz's folder:

    <npz_parent>/<npz_parent_name>_output/<npz_stem>_voxel.npy

Rules:
- Input is a hardcoded list: INPUT_DIRS. If empty, script does nothing.
- Search recursively for *.npz in each input dir.
- Each .npz must contain a key 'voxel'. If missing -> fail the run.
- Save array as-is (no reshaping). Overwrite always.
- Parallelized (max workers capped).

Run:
    python extraction_tools/extract_npz.py
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


class MissingVoxelKeyError(RuntimeError):
    pass


@dataclass
class TaskResult:
    src: Path
    dst: Optional[Path]
    ok: bool
    error: Optional[str] = None


def iter_npz_files(root: Path) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return root.rglob("*.npz")


def output_dir_for_npz(npz_path: Path) -> Path:
        """
        Create output dir path inside the npz's parent directory, named:
            <npz_parent>/<npz_parent_name>_output

        Note: This matches the requirement "inside the original npz file dir".
        """
        parent = npz_path.parent.parent
        return parent / f"{parent.name}_output"


def process_one(npz_path: Path, voxel_key: str) -> TaskResult:
    """
    Load 'voxel' from npz and save to parent/<parent>_output/<stem>_voxel.npy
    Overwrite always.
    """
    try:
        out_dir = output_dir_for_npz(npz_path)
        os.makedirs(str(out_dir), exist_ok=True)
        out_path = out_dir / f"{npz_path.stem}_{voxel_key}.npy"

        # Load .npz safely (no pickle)
        with np.load(npz_path, allow_pickle=False) as data:
            if voxel_key not in data.files:
                raise MissingVoxelKeyError(f"Missing key '{voxel_key}' in {npz_path}")
            arr = data[voxel_key]

        # Save as binary .npy, overwrite directly (no temp file)
        np.save(out_path, arr)

        return TaskResult(src=npz_path, dst=out_path, ok=True)

    except MissingVoxelKeyError as e:
        return TaskResult(src=npz_path, dst=None, ok=False, error=str(e))
    except Exception as e:
        # Capture full traceback for debugging
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        return TaskResult(src=npz_path, dst=None, ok=False, error=f"{tb}")


def collect_targets(dirs: List[Path]) -> List[Path]:
    targets: List[Path] = []
    for d in dirs:
        targets.extend(list(iter_npz_files(d)))
    # Deduplicate while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in targets:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def main() -> None:
    # ----------------- CONFIG (edit here) -----------------
    # Provide the list of folders to scan (recursively). Empty => do nothing.
    INPUT_DIRS: List[str] = [
        "/home/ofirgila/PycharmProjects/Stable_Diffusion/downloads/PSL",
        "/home/ofirgila/PycharmProjects/Stable_Diffusion/downloads/TPMS",
        "/home/ofirgila/PycharmProjects/Stable_Diffusion/downloads/Truss",
    ]

    # Cap workers to avoid saturating CPU. None => auto = min(8, os.cpu_count() or 1)
    MAX_WORKERS: Optional[int] = None

    # Key to extract from each .npz
    VOXEL_KEY: str = "voxel"
    # ----------------- END CONFIG -----------------

    # If input list empty, do nothing
    if not INPUT_DIRS:
        print("No input directories provided. Nothing to do.")
        return

    # Resolve and validate inputs
    roots = [Path(p).resolve() for p in INPUT_DIRS]
    for r in roots:
        if not r.exists() or not r.is_dir():
            print(f"Warning: Skipping non-directory or missing path: {r}")

    targets = collect_targets(roots)
    if not targets:
        print("No .npz files found under provided directories.")
        return

    print(f"Found {len(targets)} .npz files. Starting extraction...")

    # Determine workers
    if MAX_WORKERS is None:
        max_workers = max(1, min(8, os.cpu_count() or 1))
    else:
        max_workers = max(1, int(MAX_WORKERS))

    # Run in parallel
    failures: List[TaskResult] = []
    processed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one, p, VOXEL_KEY): p for p in targets}
        for fut in as_completed(futures):
            res: TaskResult = fut.result()
            processed += 1
            if res.ok:
                print(f"[{processed}/{len(targets)}] ✓ {res.src.name} -> {res.dst}")
            else:
                print(f"[{processed}/{len(targets)}] ✗ {res.src} :: {res.error}")
                failures.append(res)
                # Fail on missing key
                if isinstance(res.error, str) and f"Missing key '{VOXEL_KEY}'" in res.error:
                    print("Fatal: 'voxel' key missing. Aborting.")
                    # Best-effort early abort: break loop (running tasks may still complete)
                    break

    # If we aborted early due to missing key, signal failure
    if failures and any(
        res.error and f"Missing key '{VOXEL_KEY}'" in res.error for res in failures
    ):
        sys.exit(1)

    # Summarize
    if failures:
        print(f"Completed with {len(failures)} failures out of {len(targets)} files.")
        # Non-missing-key failures are non-fatal; exit 0
    else:
        print(f"All {len(targets)} files processed successfully.")


if __name__ == "__main__":
    main()