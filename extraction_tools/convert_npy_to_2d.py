#!/usr/bin/env python3
"""
Recursively find .npy voxel files under given folders, run 2D projections,
and save images into a sibling directory named "<parent>_2d" at the same level
as used by extract_npz.py (i.e., sibling of "<parent>_output").

Example:
  Given: /.../TPMS_output/00314_voxel.npy
  Images -> /.../TPMS_2d/00314_voxel_front.png (and others)

Rules:
- Input is a hardcoded list: INPUT_DIRS. If empty, script does nothing.
- Search recursively for *.npy in each input dir.
- Assumes each .npy is a 3D array (voxel grid). Saves images for selected views.
- Overwrite always.
- Parallelized (max workers capped).

Run:
  python extraction_tools/convert_npy_to_2d.py
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Dict

import numpy as np

# Ensure this script's folder (extraction_tools) is on sys.path for reliable local imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Local import (same folder)
from projection_2d_utils import project_3d_to_2d


# Config lives inside main() for easier tweaking at runtime.


@dataclass
class TaskResult:
    src: Path
    ok: bool
    error: Optional[str] = None


def iter_npy_files(root: Path) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return root.rglob("*.npy")


def output_dir_for_npy(npy_path: Path) -> Path:
    """
    Compute images output dir as a sibling of the source parent folder, named:
      <base>/<base_name>_2d

    If the npy resides inside a folder named '*_output', we go one level up.
    Otherwise, use the current folder as the base.
    """
    parent = npy_path.parent
    if parent.name.endswith("_output") and parent.parent.exists():
        base = parent.parent
    else:
        base = parent
    return base / f"{base.name}_2d"


def _try_save_image(out_path: Path, img: np.ndarray, image_format: str) -> None:
    """
    Save a 2D uint8 image. Prefer PNG via imageio or PIL; fallback to PGM.
    Overwrites existing files.
    """
    # Ensure parent exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure dtype and 2D
    if img.ndim != 2:
        raise ValueError(f"Image must be 2D, got shape {img.shape}")
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    saved = False

    # Try imageio (v3)
    if not saved and image_format.lower() == "png":
        try:
            import imageio.v3 as iio  # type: ignore

            iio.imwrite(out_path.with_suffix(".png"), img)
            saved = True
        except Exception:
            saved = False

    # Try PIL
    if not saved and image_format.lower() == "png":
        try:
            from PIL import Image  # type: ignore

            Image.fromarray(img).save(out_path.with_suffix(".png"))
            saved = True
        except Exception:
            saved = False

    # Fallback: save as binary PGM
    if not saved:
        pgm_path = out_path.with_suffix(".pgm")
        height, width = img.shape
        with open(pgm_path, "wb") as f:
            header = f"P5\n{width} {height}\n255\n".encode("ascii")
            f.write(header)
            f.write(img.tobytes())


def process_one(npy_path: Path, projection_options: Dict[str, bool], image_format: str) -> TaskResult:
    try:
        # Load array
        arr = np.load(npy_path, allow_pickle=False)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {arr.shape} for {npy_path}")

        # Projections
        proj = project_3d_to_2d(
            data_3d=arr,
            projection_options=projection_options,
            source_data_filepath=None,
            component_3d=None,
        )

        # Determine output dir
        out_dir = output_dir_for_npy(npy_path)

        # Save images for available views
        base_stem = npy_path.stem
        parent_stem = str(npy_path.parent.stem).replace("_output", "")
        view_key_to_suffix = {
            "front_image": "front",
            "back_image": "back",
            "top_image": "top",
            "bottom_image": "bottom",
            "left_image": "left",
            "right_image": "right",
        }

        any_saved = False
        for k, suffix in view_key_to_suffix.items():
            img = proj.get(k)
            if img is None:
                continue
            # out_path = out_dir / f"{base_stem}_{suffix}"
            out_path = out_dir / f"{parent_stem}_{base_stem}_{suffix}"
            _try_save_image(out_path, img, image_format)
            any_saved = True

        if not any_saved:
            raise RuntimeError("No projection images were produced.")

        return TaskResult(src=npy_path, ok=True)

    except Exception as e:
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        return TaskResult(src=npy_path, ok=False, error=tb)


def collect_targets(dirs: List[Path]) -> List[Path]:
    targets: List[Path] = []
    for d in dirs:
        targets.extend(list(iter_npy_files(d)))
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

    # Views to generate
    PROJECTION_OPTIONS: Dict[str, bool] = {
        "front": True,
        "back": True,
        "top": True,
        "bottom": True,
        "left": True,
        "right": True,
    }

    # Cap workers to avoid saturating CPU. None => auto = min(8, os.cpu_count() or 1)
    MAX_WORKERS: Optional[int] = None

    # Image format preference: png if possible, otherwise fallback to PGM
    IMAGE_FORMAT: str = "png"
    # ----------------- END CONFIG -----------------

    if not INPUT_DIRS:
        print("No input directories provided. Nothing to do.")
        return

    roots = [Path(p).resolve() for p in INPUT_DIRS]
    for r in roots:
        if not r.exists() or not r.is_dir():
            print(f"Warning: Skipping non-directory or missing path: {r}")

    targets = collect_targets(roots)
    if not targets:
        print("No .npy files found under provided directories.")
        return

    print(f"Found {len(targets)} .npy files. Starting 2D projections...")

    if MAX_WORKERS is None:
        max_workers = max(1, min(8, os.cpu_count() or 1))
    else:
        max_workers = max(1, int(MAX_WORKERS))

    failures: List[TaskResult] = []
    processed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(process_one, p, PROJECTION_OPTIONS, IMAGE_FORMAT): p
            for p in targets
        }
        for fut in as_completed(futures):
            res: TaskResult = fut.result()
            processed += 1
            if res.ok:
                print(f"[{processed}/{len(targets)}] ✓ {res.src.name}")
            else:
                print(f"[{processed}/{len(targets)}] ✗ {res.src} :: {res.error}")
                failures.append(res)

    if failures:
        print(f"Completed with {len(failures)} failures out of {len(targets)} files.")
    else:
        print(f"All {len(targets)} files processed successfully.")


if __name__ == "__main__":
    main()
