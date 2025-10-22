#!/usr/bin/env python3
"""
Extract image data from .h5/.hdf5 files and save as PNG images.

Usage:
  python extract_h5.py

Features:
- Processes a list of .h5/.hdf5 files (specified in INPUT_FILES)
- Prints a readable tree of groups and datasets with shapes and dtypes
- Extracts datasets that can be visualized as images and saves them as PNG files
- Creates a folder for each h5 file (using its stem name) inside the h5 file's parent directory
- Preserves the internal h5 structure in the output folder hierarchy

Rules:
- Input is a hardcoded list: INPUT_FILES. If empty, script does nothing.
- For each h5 file, creates <h5_parent>/<h5_stem>_output/ folder
- Saves all extractable images as PNG files inside the output folder
- Preserves the h5 internal group structure as subdirectories

Requires: h5py, numpy, pillow (PIL)
Install: conda activate sd; pip install h5py numpy pillow
"""

import os
import h5py
import numpy as np
from typing import List
from pathlib import Path
from PIL import Image


def output_dir_for_h5(h5_path: Path) -> Path:
    """
    Create output dir path inside the h5's parent directory, named:
        <h5_parent>/<h5_stem>_output

    Note: This matches the pattern used in extract_npz.py
    """
    parent = h5_path.parent
    return parent / f"{h5_path.stem}_output"


def can_be_image(shape, dtype):
    """Check if a dataset can be visualized as an image."""
    if len(shape) < 2:
        return False
    
    # 2D grayscale: (H, W)
    if len(shape) == 2:
        return True
    
    # 3D with channels: (H, W, C) where C is 1, 3, or 4
    if len(shape) == 3 and shape[2] in [1, 3, 4]:
        return True
    
    # 3D grayscale stack: (N, H, W) - can save as multiple images
    if len(shape) == 3:
        return True
    
    # 4D with channels: (N, H, W, C) where C is 1, 3, or 4
    if len(shape) == 4 and shape[3] in [1, 3, 4]:
        return True
    
    return False


def normalize_array_for_image(arr):
    """Normalize array to 0-255 uint8 range for image saving."""
    arr = np.array(arr)
    
    # Handle boolean arrays
    if arr.dtype == bool:
        return (arr * 255).astype(np.uint8)
    
    # Handle integer types
    if np.issubdtype(arr.dtype, np.integer):
        # If already in a reasonable range, just convert
        if arr.min() >= 0 and arr.max() <= 255:
            return arr.astype(np.uint8)
        # Otherwise normalize
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
        return arr
    
    # Handle floating point
    if np.issubdtype(arr.dtype, np.floating):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
        return arr
    
    # Default: try to convert directly
    return arr.astype(np.uint8)


def extract_images_from_dataset(h5_path: str, dataset_path: str, base_output_dir: str):
    """Extract dataset as PNG image(s) if it can be visualized as an image."""
    try:
        with h5py.File(h5_path, 'r') as f:
            dp = dataset_path.lstrip('/')
            if dp not in f:
                return
            ds = f[dp]
            if not isinstance(ds, h5py.Dataset):
                return
            
            shape = ds.shape
            dtype = ds.dtype
            
            if not can_be_image(shape, dtype):
                return
            
            # Create output directory structure mirroring h5 internal structure
            dataset_parts = dp.split('/')
            dataset_name = dataset_parts[-1]
            group_path = '/'.join(dataset_parts[:-1]) if len(dataset_parts) > 1 else ''
            
            out_dir = os.path.join(base_output_dir, group_path) if group_path else base_output_dir
            os.makedirs(out_dir, exist_ok=True)
            
            # Load the data
            data = ds[()]
            
            # Handle different dimensionalities
            if len(shape) == 2:
                # 2D grayscale image
                img_data = normalize_array_for_image(data)
                img = Image.fromarray(img_data, mode='L')
                out_path = os.path.join(out_dir, f"{dataset_name}.png")
                img.save(out_path)
                print(f"  → Saved 2D image: {out_path}")
                
            elif len(shape) == 3 and shape[2] in [1, 3, 4]:
                # Single image with channels (H, W, C)
                img_data = normalize_array_for_image(data)
                if shape[2] == 1:
                    img = Image.fromarray(img_data[:, :, 0], mode='L')
                elif shape[2] == 3:
                    img = Image.fromarray(img_data, mode='RGB')
                elif shape[2] == 4:
                    img = Image.fromarray(img_data, mode='RGBA')
                out_path = os.path.join(out_dir, f"{dataset_name}.png")
                img.save(out_path)
                print(f"  → Saved {shape[2]}-channel image: {out_path}")
                
            elif len(shape) == 3:
                # Stack of 2D grayscale images (N, H, W)
                n_images = shape[0]
                for i in range(n_images):
                    img_data = normalize_array_for_image(data[i])
                    img = Image.fromarray(img_data, mode='L')
                    out_path = os.path.join(out_dir, f"{dataset_name}_{i:04d}.png")
                    img.save(out_path)
                print(f"  → Saved {n_images} images from stack: {out_dir}/{dataset_name}_*.png")
                
            elif len(shape) == 4 and shape[3] in [1, 3, 4]:
                # Stack of images with channels (N, H, W, C)
                n_images = shape[0]
                for i in range(n_images):
                    img_data = normalize_array_for_image(data[i])
                    if shape[3] == 1:
                        img = Image.fromarray(img_data[:, :, 0], mode='L')
                    elif shape[3] == 3:
                        img = Image.fromarray(img_data, mode='RGB')
                    elif shape[3] == 4:
                        img = Image.fromarray(img_data, mode='RGBA')
                    out_path = os.path.join(out_dir, f"{dataset_name}_{i:04d}.png")
                    img.save(out_path)
                print(f"  → Saved {n_images} {shape[3]}-channel images from stack: {out_dir}/{dataset_name}_*.png")
                
    except Exception as e:
        print(f"  ✗ Failed to extract images from {dataset_path}: {e}")



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
                    can_img = can_be_image(shape, dtype)
                    img_marker = ' [IMAGE]' if can_img else ''
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

                    print(f"{indent}D: /{name}  shape={shape} dtype={dtype} size={size}{img_marker}{preview}")
                elif isinstance(obj, h5py.Group):
                    print(f"{indent}G: /{name}")

            # top-level groups
            for name, obj in f.items():
                visit(name, obj)
                if isinstance(obj, h5py.Group):
                    obj.visititems(lambda n, o: visit(f"{name}/{n}", o))
    except Exception as e:
        print(f"Failed to open {path}: {e}")


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
    # Provide the list of .h5 files to process. Empty => do nothing.
    INPUT_FILES: List[str] = [
        "D:/AllProjects/PycharmProjects/Stable_Diffusion/downloads/MICRO2D_homogenized.h5",
        "D:/AllProjects/PycharmProjects/Stable_Diffusion/downloads/MICRO2D_localized_subsample_32bit.h5",
    ]
    # ----------------- END CONFIG -----------------

    # If input list empty, do nothing
    if not INPUT_FILES:
        print("No input files provided. Nothing to do.")
        return

    # Resolve and validate inputs
    targets = [Path(p).resolve() for p in INPUT_FILES]
    valid_targets = []
    for t in targets:
        if not t.exists():
            print(f"Warning: Skipping non-existent file: {t}")
        elif not t.is_file():
            print(f"Warning: Skipping non-file path: {t}")
        elif not t.suffix.lower() in ['.h5', '.hdf5', '.hf5']:
            print(f"Warning: Skipping non-h5 file: {t}")
        else:
            valid_targets.append(t)

    if not valid_targets:
        print("No valid .h5 files found in the input list.")
        return

    print(f"Found {len(valid_targets)} H5 file(s) to process\n")
    
    for t in valid_targets:
        # Print the structure
        print_h5_tree(str(t))
        
        # Create output folder for this h5 file using its stem name in its parent dir
        h5_output_dir = output_dir_for_h5(t)
        os.makedirs(str(h5_output_dir), exist_ok=True)
        
        print(f"\nExtracting images from {t.name} to {h5_output_dir}...")
        
        # Get all datasets and try to extract them as images
        ds_list = list_datasets(str(t))
        if not ds_list:
            print(f"  No datasets found in {t}")
            continue
        
        for ds_path in ds_list:
            extract_images_from_dataset(str(t), ds_path, str(h5_output_dir))
        
        print(f"Completed processing {t.name}\n")
        print("=" * 80)


if __name__ == '__main__':
    main()
