#!/usr/bin/env python3
"""
Extract .tar.gz files to subdirectories named after each archive.

Usage:
  python extract_tar_gz.py

Features:
- Processes a list of .tar.gz files (specified in INPUT_FILES)
- Extracts each archive to a subdirectory with the archive's stem name inside the archive's parent directory
- Shows progress and error handling

Rules:
- Input is a hardcoded list: INPUT_FILES. If empty, script does nothing.
- For each tar.gz file, creates <tar_parent>/<tar_stem>/ folder
- Extracts contents into that folder

Requires: tarfile (built-in), pathlib (built-in)
"""

import tarfile
from pathlib import Path
from typing import List, Optional


def output_dir_for_tar(tar_path: Path) -> Path:
    """
    Create output dir path inside the tar's parent directory, named:
        <tar_parent>/<tar_stem>

    Note: This matches the pattern used in other extraction scripts.
    """
    parent = tar_path.parent
    # Get base name without .tar.gz extension
    stem = tar_path.name.replace('.tar.gz', '').replace('.tgz', '')
    return parent / stem


def extract_archive(tar_path: Path, extract_to: Path) -> bool:
    """Extract a tar.gz archive to the specified directory.
    
    Args:
        tar_path: Path to the .tar.gz file
        extract_to: Directory to extract contents into
        
    Returns:
        True if extraction succeeded, False otherwise
    """
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        
        print(f"✓ Successfully extracted: {tar_path.name} -> {extract_to.name}/")
        return True
        
    except Exception as e:
        print(f"✗ Failed to extract {tar_path.name}: {e}")
        return False


def main():
    # ----------------- CONFIG (edit here) -----------------
    # Provide the list of .tar.gz files to process. Empty => do nothing.
    INPUT_FILES: List[str] = [
        "/home/ofirgila/PycharmProjects/Stable_Diffusion/downloads/PSL.tar.gz",
        "/home/ofirgila/PycharmProjects/Stable_Diffusion/downloads/TPMS.tar.gz",
        "/home/ofirgila/PycharmProjects/Stable_Diffusion/downloads/Truss.tar.gz",
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
        elif not (t.suffix == '.gz' and '.tar' in t.name):
            print(f"Warning: Skipping non-tar.gz file: {t}")
        else:
            valid_targets.append(t)
    
    if not valid_targets:
        print("No valid .tar.gz files found in the input list.")
        return
    
    print(f"Found {len(valid_targets)} .tar.gz file(s) to extract\n")
    
    # Extract each archive
    success_count = 0
    for tar_path in valid_targets:
        extract_to = output_dir_for_tar(tar_path)
        
        print(f"Extracting {tar_path.name} to {extract_to}...")
        if extract_archive(tar_path, extract_to):
            success_count += 1
    
    print(f"\nExtraction complete! Successfully extracted {success_count}/{len(valid_targets)} archives")


if __name__ == '__main__':
    main()
