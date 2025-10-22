#!/usr/bin/env python3
"""
Move a directory with all its contents to a different location.

Usage:
  python extraction_tools/move_dir.py

Edit the SOURCE_DIR and DEST_DIR variables in main() to specify:
- SOURCE_DIR: The directory to move (must exist)
- DEST_DIR: The destination parent directory (must exist)

The source directory will be moved into the destination as:
  DEST_DIR/<source_dir_name>/

Example:
  SOURCE_DIR = "/path/to/my_folder"
  DEST_DIR = "/path/to/target"
  Result: /path/to/target/my_folder/
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def move_directory(source: Path, destination: Path) -> None:
    """
    Move a directory and all its contents to a destination.
    
    Args:
        source: Path to the source directory (must exist)
        destination: Path to the destination parent directory (must exist)
        
    The source directory will be moved into destination as:
      destination/<source_name>/
    """
    # Validate source
    if not source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {source}")
    
    # Validate destination
    if not destination.exists():
        raise FileNotFoundError(f"Destination directory does not exist: {destination}")
    if not destination.is_dir():
        raise NotADirectoryError(f"Destination is not a directory: {destination}")
    
    # Compute final target path
    target = destination / source.name
    
    # Check if target already exists
    if target.exists():
        raise FileExistsError(
            f"Target already exists: {target}\n"
            f"Please remove or rename it before moving."
        )
    
    # Move the directory
    print(f"Moving: {source}")
    print(f"To:     {target}")
    
    shutil.move(str(source), str(target))
    
    print(f"✓ Successfully moved to: {target}")


def main() -> None:
    # ----------------- CONFIG (edit here) -----------------
    # Source directory to move (full path)
    SOURCE_DIR: str = "/home/ofirgila/PycharmProjects/ControlNet"
    
    # Destination parent directory (full path)
    # The source will be moved INTO this directory
    DEST_DIR: str = "/groups/asharf_group/ofirgila"
    # ----------------- END CONFIG -----------------
    
    source = Path(SOURCE_DIR).resolve()
    dest = Path(DEST_DIR).resolve()
    
    try:
        move_directory(source, dest)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
