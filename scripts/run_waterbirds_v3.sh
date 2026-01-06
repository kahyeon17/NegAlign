#!/bin/bash
# Test NegAlign V3 (Soft Weighting) on WaterBirds vs PlacesBG
# Run on GPU1 to avoid interfering with ongoing ImageNet experiment on GPU0

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/waterbirds_v3_${TIMESTAMP}.log"

echo "Starting WaterBirds V3 test at $(date)"
echo "Log file: $LOG_FILE"

# Test with v2_sign_flip variant (best from previous experiments)
nohup python tests/test_waterbirds_v3.py \
  --waterbirds_root /home/kahyeon/datasets/waterbird_complete95_forest2water2 \
  --placesbg_root /home/kahyeon/datasets/placesbg \
  --output_dir ./results/waterbirds_v3 \
  --device cuda:1 \
  --num_samples 2000 \
  --use_soft_weighting \
  --soft_weight_scale 20.0 \
  --use_neglabel_star \
  --topk_k 10 \
  --use_p_align \
  --p_align_variant v2_sign_flip \
  --lambda_values 0.0 0.5 1.0 2.0 3.0 5.0 10.0 \
  --cam_fg_percentile 80 \
  --seed 42 \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "Monitor progress with: tail -f $LOG_FILE"
echo "Check GPU with: nvidia-smi"
echo ""
echo "To check process status:"
echo "  ps aux | grep test_waterbirds_v3"
echo ""
echo "Experiment details:"
echo "  - Dataset: WaterBirds vs PlacesBG"
echo "  - Samples: 2000 per dataset"
echo "  - Device: cuda:1 (GPU1)"
echo "  - V3 Feature: Soft weighting (all 6370 negatives)"
echo "  - Scale: 20.0"
echo "  - Variant: v2_sign_flip"
echo "  - Lambda sweep: [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]"
echo ""
echo "Expected time: ~30-40 minutes"
