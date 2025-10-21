#!/bin/bash

# Image copy script - handles large numbers of files
# Copies all images from source to target directory

SRC="/home/ofirgila/PycharmProjects/ControlNet/training/fill50k/target"
TRG="/home/ofirgila/PycharmProjects/Stable_Diffusion/data/original"

echo "Copying images from:"
echo "  Source: $SRC"
echo "  Target: $TRG"

# Check if source directory exists
if [ ! -d "$SRC" ]; then
    echo "ERROR: Source directory does not exist: $SRC"
    exit 1
fi

# Create target directory if it doesn't exist
mkdir -p "$TRG"

# Count files before copy
BEFORE=$(find "$TRG" -type f 2>/dev/null | wc -l)
echo "Files in target before copy: $BEFORE"

# Use find and xargs to handle large number of files
echo "Starting copy (this may take a while for 50,000 files)..."

find "$SRC" -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.bmp" -o -name "*.tiff" | \
xargs -I {} cp {} "$TRG"/

echo "Copy completed!"

# Count files after copy
AFTER=$(find "$TRG" -type f 2>/dev/null | wc -l)
ACTUALLY_COPIED=$((AFTER - BEFORE))

echo "  Files before: $BEFORE"
echo "  Files after: $AFTER"
echo "  Actually copied: $ACTUALLY_COPIED files"

echo "Sample files in target directory:"
ls -la "$TRG" | head -10
