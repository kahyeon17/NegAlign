#!/bin/bash
# Quick Start Script for NegAlign

set -e  # Exit on error

echo "=========================================="
echo "NegAlign Quick Start"
echo "=========================================="

# Check environment
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

if ! python -c "import torch" &> /dev/null; then
    echo "Error: PyTorch not installed"
    exit 1
fi

echo "Environment check: OK"

# Step 1: Split negatives (if not already done)
NEG_NOUN="./data/neg_labels_noun.txt"
NEG_ADJ="./data/neg_labels_adj.txt"
SPLIT_DIR="./data/negatives_split"

if [ ! -f "$SPLIT_DIR/neg_word_scores.csv" ]; then
    echo ""
    echo "=========================================="
    echo "Step 1: Splitting negative labels..."
    echo "=========================================="

    if [ ! -f "$NEG_NOUN" ]; then
        echo "Error: $NEG_NOUN not found"
        echo "Please ensure negative labels exist in ./data/"
        echo "Run: mkdir -p ./data && cp /path/to/neg_labels_*.txt ./data/"
        exit 1
    fi

    python split_negatives_by_clip.py \
        --noun_file "$NEG_NOUN" \
        --adj_file "$NEG_ADJ" \
        --output_dir "$SPLIT_DIR" \
        --tau 0.02 \
        --device cuda:1

    echo "Split completed!"
else
    echo ""
    echo "Negative labels already split (found $SPLIT_DIR/neg_word_scores.csv)"
    echo "Use --recompute to re-split"
fi

# Step 2: Test CAM extraction
echo ""
echo "=========================================="
echo "Step 2: Testing CAM extraction..."
echo "=========================================="

python cam_bg.py

echo "CAM test completed!"

# Step 3: Run sanity check
echo ""
echo "=========================================="
echo "Step 3: Running sanity check..."
echo "=========================================="

python clip_negalign.py

echo "Sanity check completed!"

# Step 4: Instructions for WaterBirds evaluation
echo ""
echo "=========================================="
echo "Step 4: Next Steps"
echo "=========================================="
echo ""
echo "To evaluate on WaterBirds, run:"
echo ""
echo "  python test_waterbirds_negalign.py \\"
echo "    --waterbirds_root /path/to/waterbirds \\"
echo "    --placesbg_root /path/to/placesbg \\"
echo "    --output_dir ./results/waterbirds_negalign \\"
echo "    --use_neglabel_star \\"
echo "    --use_role_aware \\"
echo "    --use_p_align \\"
echo "    --lambda_values 0.0 0.5 1.0 2.0 5.0 10.0"
echo ""
echo "Results will be saved to: ./results/waterbirds_negalign/results.json"
echo ""
echo "=========================================="
echo "Quick start completed successfully!"
echo "=========================================="
