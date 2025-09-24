#!/bin/bash

# Recursive Image Copy Script
# Copies all images from a source directory (recursively) to a target directory
# without maintaining the folder structure (flattens all images into one directory)

# Function to display usage
usage() {
    echo "Usage: $0 <source_directory> <target_directory>"
    echo ""
    echo "Example: $0 /path/to/source /path/to/target"
    echo ""
    echo "This script will:"
    echo "  - Search recursively for all image files in the source directory"
    echo "  - Copy them to the target directory (flattened structure)"
    echo "  - Handle filename conflicts by adding numbers to duplicate names"
    echo "  - Support common image formats: png, jpg, jpeg, gif, bmp, tiff, webp"
}

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "ERROR: Incorrect number of arguments"
    usage
    exit 1
fi

SRC="$1"
TRG="$2"

echo "Recursive Image Copy Script"
echo "=========================="
echo "Source directory: $SRC"
echo "Target directory: $TRG"
echo ""

# Check if source directory exists
if [ ! -d "$SRC" ]; then
    echo "ERROR: Source directory does not exist: $SRC"
    exit 1
fi

# Create target directory if it doesn't exist
mkdir -p "$TRG"

# Count files before copy
BEFORE=$(find "$TRG" -type f 2>/dev/null | wc -l)
echo "Files in target directory before copy: $BEFORE"

# Find all image files recursively
echo "Searching for image files..."
IMAGE_FILES=$(find "$SRC" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.gif" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.tif" -o -iname "*.webp" \) 2>/dev/null)

# Count total images found
TOTAL_IMAGES=$(echo "$IMAGE_FILES" | grep -c .)
if [ $TOTAL_IMAGES -eq 0 ]; then
    echo "No image files found in source directory!"
    exit 0
fi

echo "Found $TOTAL_IMAGES image files to copy"
echo "Starting copy process..."

# Initialize counters
COPIED=0
SKIPPED=0

# Copy each image file
while IFS= read -r file; do
    if [ -n "$file" ]; then
        # Get just the filename (without path)
        filename=$(basename "$file")
        
        # Target file path
        target_file="$TRG/$filename"
        
        # Handle filename conflicts by adding a number
        if [ -f "$target_file" ]; then
            # Get file extension and base name
            extension="${filename##*.}"
            basename="${filename%.*}"
            
            # Find a unique filename
            counter=1
            while [ -f "$TRG/${basename}_${counter}.${extension}" ]; do
                counter=$((counter + 1))
            done
            target_file="$TRG/${basename}_${counter}.${extension}"
            
            echo "  Conflict resolved: $filename -> ${basename}_${counter}.${extension}"
        fi
        
        # Copy the file
        if cp "$file" "$target_file" 2>/dev/null; then
            COPIED=$((COPIED + 1))
            
            # Show progress every 100 files
            if [ $((COPIED % 100)) -eq 0 ]; then
                echo "  Progress: $COPIED / $TOTAL_IMAGES files copied"
            fi
        else
            echo "  ERROR: Failed to copy $file"
            SKIPPED=$((SKIPPED + 1))
        fi
    fi
done <<< "$IMAGE_FILES"

echo ""
echo "Copy process completed!"
echo "======================"

# Count files after copy
AFTER=$(find "$TRG" -type f 2>/dev/null | wc -l)
ACTUALLY_ADDED=$((AFTER - BEFORE))

echo "Summary:"
echo "  Images found: $TOTAL_IMAGES"
echo "  Successfully copied: $COPIED"
echo "  Failed/skipped: $SKIPPED"
echo "  Files in target before: $BEFORE"
echo "  Files in target after: $AFTER"
echo "  Net files added: $ACTUALLY_ADDED"

echo ""
echo "Sample files in target directory:"
ls -la "$TRG" | head -10