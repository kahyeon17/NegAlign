#!/bin/bash

cd /home/kahyeon/research/NegAlign

# Use percentile-based split (top/bottom 30%)
CUDA_VISIBLE_DEVICES=0 python core/split_negatives_by_clip.py \
  --noun_file /home/kahyeon/research/negrefine_analysis/NegRefine/output_imagenet/neg_labels_noun.txt \
  --adj_file /home/kahyeon/research/negrefine_analysis/NegRefine/output_imagenet/neg_labels_adj.txt \
  --output_dir output_imagenet/negatives_split_negrefine_8428_pct30 \
  --percentile 30 \
  --device cuda:0 \
  --recompute

echo "Split complete!"
echo "Files saved to: output_imagenet/negatives_split_negrefine_8428/"
