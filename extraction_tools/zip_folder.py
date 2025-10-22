#!/usr/bin/env python3
"""
Script to zip a folder's contents.
Usage: python zip_folder.py <folder_path> [output_zip_name]
"""

import os
import sys
import zipfile
from pathlib import Path


def zip_folder(folder_path, output_zip=None, exclude_patterns=None):
    """
    Zip the contents of a folder.
    
    Args:
        folder_path: Path to the folder to zip
        output_zip: Optional output zip file name (defaults to folder_name.zip)
        exclude_patterns: Optional list of patterns to exclude (e.g., ['*.pyc', '__pycache__'])
    
    Returns:
        Path to the created zip file
    """
    folder_path = Path(folder_path).resolve()
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Determine output zip name
    if output_zip is None:
        output_zip = f"{folder_path.name}.zip"
    
    output_zip = Path(output_zip)
    if not output_zip.suffix:
        output_zip = output_zip.with_suffix('.zip')
    
    # Default exclusions
    if exclude_patterns is None:
        exclude_patterns = []
    
    print(f"Zipping folder: {folder_path}")
    print(f"Output file: {output_zip}")
    
    # Create zip file
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through all files and subdirectories
        for root, dirs, files in os.walk(folder_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(
                Path(d).match(pattern) for pattern in exclude_patterns
            )]
            
            for file in files:
                # Skip excluded files
                if any(Path(file).match(pattern) for pattern in exclude_patterns):
                    continue
                
                file_path = Path(root) / file
                # Create archive name relative to the folder being zipped
                arcname = file_path.relative_to(folder_path.parent)
                
                print(f"  Adding: {arcname}")
                zipf.write(file_path, arcname)
    
    file_size = output_zip.stat().st_size
    print(f"\nZip file created successfully!")
    print(f"Size: {file_size / (1024**2):.2f} MB")
    print(f"Location: {output_zip.resolve()}")
    
    return output_zip


def main():
    # Hard-coded paths
    folder_path = "/groups/asharf_group/ofirgila/Stable_Diffusion/data_lattice"
    output_zip = None  # Will default to folder_name.zip
    
    # Optional: exclude common patterns (uncomment to use)
    # exclude_patterns = ['*.pyc', '__pycache__', '.git', '*.log', '.DS_Store']
    exclude_patterns = []
    
    try:
        zip_folder(folder_path, output_zip, exclude_patterns)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
