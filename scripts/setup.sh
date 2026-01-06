#!/bin/bash
# Setup Script for NegAlign
# This script prepares the NegAlign environment with necessary data

set -e

echo "=========================================="
echo "NegAlign Setup"
echo "=========================================="

# Create data directory
mkdir -p ./data

# Check if negative labels already exist
if [ -f "./data/neg_labels_noun.txt" ] && [ -f "./data/neg_labels_adj.txt" ]; then
    echo "Negative labels already exist in ./data/"
    echo "Skipping copy step."
else
    # Default source path (can be overridden)
    NEGREFINE_PATH="${NEGREFINE_PATH:-../negrefine_analysis/NegRefine/output/imagenet/seed_0}"

    echo ""
    echo "Looking for negative labels in: $NEGREFINE_PATH"

    if [ -f "$NEGREFINE_PATH/neg_labels_noun.txt" ]; then
        echo "Found negative labels! Copying to ./data/"
        cp "$NEGREFINE_PATH/neg_labels_noun.txt" ./data/
        cp "$NEGREFINE_PATH/neg_labels_adj.txt" ./data/
        echo "✓ Copied negative labels successfully"
    else
        echo ""
        echo "ERROR: Negative labels not found at default location."
        echo ""
        echo "Please specify the path to your NegRefine output:"
        echo "  export NEGREFINE_PATH=/path/to/NegRefine/output/imagenet/seed_0"
        echo "  bash setup.sh"
        echo ""
        echo "Or manually copy the files:"
        echo "  cp /path/to/neg_labels_noun.txt ./data/"
        echo "  cp /path/to/neg_labels_adj.txt ./data/"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run quickstart: bash quickstart.sh"
echo "  2. Or run individual scripts as needed"
echo ""
