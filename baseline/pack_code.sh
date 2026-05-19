#!/bin/bash

# Script to pack workspace into zip file while respecting .gitignore rules
# Usage: ./pack_workspace.sh [output_filename.zip]

# Set default output filename
OUTPUT_FILE="${1:-code_$(date +%Y%m%d_%H%M%S).zip}"

# Ensure output filename has .zip extension
if [[ ! "$OUTPUT_FILE" =~ \.zip$ ]]; then
    OUTPUT_FILE="${OUTPUT_FILE}.zip"
fi

echo "Creating archive: $OUTPUT_FILE"
echo "Respecting .gitignore rules..."

# Check if .gitignore exists
if [ ! -f ".gitignore" ]; then
    echo "Warning: .gitignore not found, packing all files"
    zip -r "$OUTPUT_FILE" . -x "*.git*" "$OUTPUT_FILE"
    exit 0
fi

# Create temporary file list
TEMP_FILE=$(mktemp)

# Use git ls-files if in a git repository
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Using git to list tracked files..."
    git ls-files > "$TEMP_FILE"
    
    # Add this script itself if not tracked by git
    SCRIPT_NAME=$(basename "$0")
    if ! grep -q "^${SCRIPT_NAME}$" "$TEMP_FILE"; then
        echo "$SCRIPT_NAME" >> "$TEMP_FILE"
    fi
else
    echo "Not a git repository, using find with .gitignore patterns..."
    
    # Find all files and filter using .gitignore patterns
    find . -type f -not -path "*/\.git/*" | sed 's|^\./||' > "$TEMP_FILE.all"
    
    # Filter out files matching .gitignore patterns
    while IFS= read -r pattern; do
        # Skip empty lines and comments
        [[ -z "$pattern" || "$pattern" =~ ^# ]] && continue
        
        # Remove leading/trailing whitespace
        pattern=$(echo "$pattern" | xargs)
        
        # Use grep to filter out matching patterns
        grep -v "^${pattern//\*/.*}" "$TEMP_FILE.all" > "$TEMP_FILE.tmp" 2>/dev/null || true
        mv "$TEMP_FILE.tmp" "$TEMP_FILE.all"
    done < .gitignore
    
    mv "$TEMP_FILE.all" "$TEMP_FILE"
    
    # Add this script itself
    SCRIPT_NAME=$(basename "$0")
    if ! grep -q "^${SCRIPT_NAME}$" "$TEMP_FILE"; then
        echo "$SCRIPT_NAME" >> "$TEMP_FILE"
    fi
fi

# Count files to be archived
FILE_COUNT=$(wc -l < "$TEMP_FILE")
echo "Found $FILE_COUNT files to archive"

# Create zip archive from file list
if [ -s "$TEMP_FILE" ]; then
    zip -@ "$OUTPUT_FILE" < "$TEMP_FILE"
    ZIP_EXIT_CODE=$?
    
    if [ $ZIP_EXIT_CODE -eq 0 ]; then
        ZIP_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo ""
        echo "✓ Archive created successfully: $OUTPUT_FILE"
        echo "  Size: $ZIP_SIZE"
        echo "  Files: $FILE_COUNT"
    else
        echo "✗ Error creating archive (exit code: $ZIP_EXIT_CODE)"
        rm -f "$TEMP_FILE"
        exit 1
    fi
else
    echo "✗ No files to archive"
    rm -f "$TEMP_FILE"
    exit 1
fi

# Cleanup
rm -f "$TEMP_FILE"

echo "Done!"
