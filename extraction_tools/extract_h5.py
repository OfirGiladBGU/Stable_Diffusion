#!/usr/bin/env python3
"""
Open and inspect .h5/.hdf5 files in a folder.

Usage:
  python open_h5.py --dir .            # scan current repo for .h5 files and print structure
  python open_h5.py path/to/file.h5    # inspect a single file
  python open_h5.py --dir . --extract --dataset /group/subds --outdir out

Features:
- Recursively finds .h5/.hdf5 files under a directory
- Prints a readable tree of groups and datasets with shapes and dtypes
- Optionally extracts a dataset to a .npy file

Requires: h5py, numpy
Install: conda activate sd; pip install h5py numpy
"""

import os
import h5py
import numpy as np
from typing import Optional


def find_h5_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(('.h5', '.hdf5', '.hf5')):
                yield os.path.join(dirpath, fn)


def print_h5_tree(path: str, max_elems_preview: int = 8):
    print(f"\n== {path} ==")
    try:
        with h5py.File(path, 'r') as f:
            def visit(name, obj):
                indent = '  ' * (name.count('/') if name else 0)
                if isinstance(obj, h5py.Dataset):
                    shape = obj.shape
                    dtype = obj.dtype
                    size = obj.size
                    preview = ''
                    try:
                        # show a tiny preview for small-ish datasets
                        if size > 0 and size <= max_elems_preview * 16:
                            # load small datasets fully
                            data = obj[()]
                            preview = f' preview={np.array(data).ravel()[:max_elems_preview].tolist()}'
                        elif size > 0 and obj.ndim == 1:
                            preview = f' preview={obj[:max_elems_preview].tolist()}'
                    except Exception as e:
                        preview = f' (preview failed: {e})'

                    print(f"{indent}D: /{name}  shape={shape} dtype={dtype} size={size}{preview}")
                elif isinstance(obj, h5py.Group):
                    print(f"{indent}G: /{name}")

            # top-level groups
            for name, obj in f.items():
                visit(name, obj)
                if isinstance(obj, h5py.Group):
                    obj.visititems(lambda n, o: visit(f"{name}/{n}", o))
    except Exception as e:
        print(f"Failed to open {path}: {e}")


def extract_dataset(h5_path: str, dataset_path: str, out_dir: str) -> Optional[str]:
    """Extract a dataset (by absolute path inside HDF5) and save as .npy files in out_dir.
    If a dataset is large (estimated bytes > SIZE_THRESHOLD_BYTES) and has a sample dimension
    (first axis > 1), the function will save each sample separately into a subdirectory to
    avoid loading the whole dataset into memory. Returns path(s) saved or None on error.
    """
    SIZE_THRESHOLD_BYTES = 200 * 1024 * 1024  # 200 MB - above this we stream per-sample
    saved_paths = []
    try:
        with h5py.File(h5_path, 'r') as f:
            dp = dataset_path.lstrip('/')
            if dp not in f:
                print(f"Dataset {dataset_path} not found in {h5_path}")
                return None
            ds = f[dp]
            if not isinstance(ds, h5py.Dataset):
                print(f"Path {dataset_path} is not a dataset")
                return None

            # estimate total bytes
            dtype = ds.dtype
            shape = ds.shape
            elem_size = dtype.itemsize if hasattr(dtype, 'itemsize') else None
            total_bytes = None
            if elem_size is not None:
                total_elems = 1
                for s in shape:
                    total_elems *= s
                total_bytes = total_elems * int(elem_size)

            base = os.path.basename(h5_path)
            name = dp.replace('/', '_').strip('_')
            os.makedirs(out_dir, exist_ok=True)

            # If dataset is big and has a leading sample axis, save per-sample files
            if total_bytes is not None and total_bytes > SIZE_THRESHOLD_BYTES and len(shape) >= 1 and shape[0] > 1:
                sample_dir = os.path.join(out_dir, f"{base}__{name}_samples")
                os.makedirs(sample_dir, exist_ok=True)
                n = shape[0]
                for i in range(n):
                    try:
                        item = ds[i]
                        out_path = os.path.join(sample_dir, f"sample_{i:05d}.npy")
                        np.save(out_path, np.array(item))
                        saved_paths.append(out_path)
                    except Exception as e:
                        print(f"Failed to save sample {i} of {dp} from {h5_path}: {e}")
                return sample_dir
            else:
                # read whole dataset into memory (ok for small/moderate datasets)
                arr = ds[()]
                out_path = os.path.join(out_dir, f"{base}__{name}.npy")
                # Convert object dtype to numpy-friendly format
                if arr.dtype == object:
                    try:
                        arr = np.array(arr.tolist())
                    except Exception:
                        # fallback: save with numpy.save allowing pickled objects
                        np.save(out_path, arr, allow_pickle=True)
                        return out_path
                np.save(out_path, np.array(arr))
                return out_path
    except Exception as e:
        print(f"Failed to extract {dataset_path} from {h5_path}: {e}")
        return None


def list_datasets(h5_path: str):
    """Return a list of dataset paths (absolute) inside the HDF5 file."""
    paths = []
    try:
        with h5py.File(h5_path, 'r') as f:
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    paths.append('/' + name)
            f.visititems(visitor)
    except Exception as e:
        print(f"Failed to list datasets in {h5_path}: {e}")
    return paths


def main():
    # ----------------- CONFIG (edit here) -----------------
    # Set these variables directly to control behavior. Do NOT use CLI.
    # Examples:
    #   CFG_DIR = '.'
    #   CFG_FILE = 'data/myfile.h5'
    #   CFG_EXTRACT = True
    #   CFG_DATASET = '/group/dataset'
    #   CFG_OUTDIR = 'extracted'
    CFG_DIR = r'C:\Users\User\PycharmProjects\Stable_Diffusion\h5_data'                # directory to scan for .h5 files (set to None to skip scanning)
    CFG_FILE = None              # path to a single .h5 file to inspect (overrides CFG_DIR)
    CFG_EXTRACT = True           # whether to extract a dataset (now enabled by default)
    CFG_DATASET = None           # dataset path inside h5 to extract, e.g. '/group/subds' (None => extract all)
    CFG_OUTDIR = 'h5_export'     # where to save extracted .npy files (changed to repo output folder)
    # ----------------- END CONFIG -----------------

    targets = []
    if CFG_FILE:
        if not os.path.exists(CFG_FILE):
            print(f"File not found: {CFG_FILE}")
            return
        targets = [CFG_FILE]
    else:
        if CFG_DIR is None:
            print("No CFG_FILE or CFG_DIR set. Edit the CFG_* variables in this script.")
            return
        root = CFG_DIR
        targets = list(find_h5_files(root))
        if not targets:
            print(f"No .h5 files found under {root}")
            return

    for t in targets:
        print_h5_tree(t)
        if CFG_EXTRACT:
            if CFG_DATASET:
                out = extract_dataset(t, CFG_DATASET, CFG_OUTDIR)
                if out:
                    print(f"Saved dataset to {out}")
            else:
                # extract all datasets
                ds_list = list_datasets(t)
                if not ds_list:
                    print(f"No datasets found to extract in {t}")
                for ds_path in ds_list:
                    out = extract_dataset(t, ds_path, CFG_OUTDIR)
                    if out:
                        print(f"Saved dataset to {out}")


if __name__ == '__main__':
    main()
