#!/usr/bin/env python3
"""
Extract all .tar.gz files in the downloads folder to subdirectories named after each archive.

Usage:
  python extract_tar_gz.py

Features:
- Finds all .tar.gz files in the downloads folder
- Extracts each archive to a subdirectory with the archive's base name
- Shows progress and error handling

Requires: tarfile (built-in), pathlib (built-in)
"""

import tarfile
from pathlib import Path
from typing import List


def find_tar_gz_files(downloads_dir: Path) -> List[Path]:
    """Find all .tar.gz files in the downloads directory."""
    return list(downloads_dir.glob("*.tar.gz"))


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
    # Set the path to your downloads folder relative to project root
    CFG_DOWNLOADS_DIR = Path(__file__).parent.parent / "downloads"
    # ----------------- END CONFIG -----------------
    
    # Check if downloads directory exists
    if not CFG_DOWNLOADS_DIR.exists():
        print(f"Error: downloads directory not found at {CFG_DOWNLOADS_DIR}")
        return
    
    if not CFG_DOWNLOADS_DIR.is_dir():
        print(f"Error: {CFG_DOWNLOADS_DIR} is not a directory")
        return
    
    print(f"Scanning for .tar.gz files in: {CFG_DOWNLOADS_DIR}")
    
    # Find all tar.gz files
    tar_files = find_tar_gz_files(CFG_DOWNLOADS_DIR)
    
    if not tar_files:
        print("No .tar.gz files found in downloads/")
        return
    
    print(f"Found {len(tar_files)} .tar.gz file(s)\n")
    
    # Extract each archive
    success_count = 0
    for tar_path in tar_files:
        # Get base name without .tar.gz extension
        base_name = tar_path.name.replace('.tar.gz', '')
        extract_to = CFG_DOWNLOADS_DIR / base_name
        
        print(f"Extracting {tar_path.name}...")
        if extract_archive(tar_path, extract_to):
            success_count += 1
    
    print(f"\nExtraction complete! Successfully extracted {success_count}/{len(tar_files)} archives")


if __name__ == '__main__':
    main()
