# NegAlign: Background-Bias-Aware Zero-Shot OOD Detection

NegAlign is a minimal modification of NegRefine that addresses background bias in zero-shot OOD detection using:
- **S_NegLabel*** (modified base score with configurable enhancements)
- **P_align** (CAM-based background alignment, NO SAM)

**Final score:** `S_final = S_NegLabel* + λ × P_align`

---

## Project Structure

```
NegAlign/
├── utils/                       # Utilities from NegRefine (class_names, create_negs, etc.)
├── data/                        # Negative labels and split results (created by setup.sh)
├── split_negatives_by_clip.py  # Word-level obj/bg classification using CLIP
├── cam_bg.py                    # Grad-CAM background embedding extraction
├── clip_negalign.py             # Main NegAlign model (integrates everything)
├── test_waterbirds_negalign.py  # Evaluation script for WaterBirds
├── example_usage.py             # Example usage scenarios
├── setup.sh                     # Setup script to copy negative labels
├── quickstart.sh                # Quick start script
└── README.md                    # This file
```

---

## Key Features

### 1. S_NegLabel* (Modified Base Score)

Built on NegRefine's softmax-based grouping score with optional modifications:

- **(M1) TopK Aggregation**: Replace max with TopKMean (configurable k, default 10)
- **(M2) Role-Aware Negatives**: Use object-leaning negatives (N_obj) only for base scoring
- **(M3) Scale Stabilization**: Z-score normalization (optional, off by default)

**All modifications are TOGGLEABLE** to reproduce S_NegLabel_plain exactly.

### 2. Negative Vocabulary Splitting

**Word-level SOFT classification** into object vs background using CLIP embeddings:

```
Object templates:     "a photo of a {w}", "a close-up photo of a {w}"
Background templates: "a scene of {w}", "a background of {w}", "a texture of {w}"

r(w) = cos(e(w), E_obj(w)) - cos(e(w), E_bg(w))

Classification (tau=0.05):
  - if r(w) > +tau: object-leaning
  - if r(w) < -tau: background-leaning
  - else: ambiguous
```

**Outputs:**
- `neg_word_scores.csv`: All words with r-scores and categories
- `neg_object.txt`: Object-leaning words (used for N_obj)
- `neg_background.txt`: Background-leaning words (used for N_bg)
- `neg_ambiguous.txt`: Ambiguous words

### 3. P_align (Background Calibration)

**CAM-based background alignment (NO SAM):**

```
Step 1: Extract background embedding I_bg using Grad-CAM
Step 2: S_bg_pos = TopKMean(cos(I_bg, T_pos[c_hat]), k=10)
Step 3: S_bg_neg = TopKMean(cos(I_bg, E(N_bg)), k=5)

P_align = S_bg_pos - S_bg_neg
```

**Uses:**
- Grad-CAM on CLIP ViT transformer blocks
- Percentile-based foreground/background masking (default: 80th percentile)
- Morphological dilation to reduce object-edge leakage (default: 1 pixel)

---

## Installation

```bash
cd /home/kahyeon/research/NegAlign

# Ensure dependencies are installed
pip install torch torchvision clip scipy pandas tqdm
```

---

## Quick Start

### Step 0: Setup Data

First, run the setup script to copy negative labels from NegRefine:

```bash
bash setup.sh
```

Or manually copy the files:

```bash
mkdir -p ./data
cp /path/to/NegRefine/output/imagenet/seed_0/neg_labels_noun.txt ./data/
cp /path/to/NegRefine/output/imagenet/seed_0/neg_labels_adj.txt ./data/
```

### Step 1: Run Quick Start

The quickstart script will automatically split negatives and run sanity checks:

```bash
bash quickstart.sh
```

Or run individual steps manually:

```bash
# Split negative labels
python split_negatives_by_clip.py \
  --noun_file ./data/neg_labels_noun.txt \
  --adj_file ./data/neg_labels_adj.txt \
  --output_dir ./data/negatives_split \
  --tau 0.05 \
  --device cuda:0
```

**Output:**
```
Split Summary:
  Object-leaning:      2847 (44.7%)
  Background-leaning:  1892 (29.7%)
  Ambiguous:           1631 (25.6%)
  Total:               6370
```

### 2. Test CAM Extraction

```bash
python cam_bg.py
```

### 3. Run Sanity Check

```bash
python clip_negalign.py
```

**Expected output:**
```
SANITY CHECK RESULTS
==================================================
Predicted class idx: 123
Predicted class:     cardigan

S_NegLabel_plain:    0.8523
S_NegLabel*:         0.8523

P_align:             0.0234
CAM valid:           True

S_final:             0.8757
  = S_NegLabel* + 1.0 * P_align
  = 0.8523 + 1.0 * 0.0234
==================================================
```

### 4. Evaluate on WaterBirds

```bash
python test_waterbirds_negalign.py \
  --waterbirds_root /path/to/waterbirds \
  --placesbg_root /path/to/placesbg \
  --output_dir ./results/waterbirds_negalign \
  --device cuda:0 \
  --use_neglabel_star \
  --use_role_aware \
  --use_p_align \
  --lambda_values 0.0 0.5 1.0 2.0 5.0 10.0
```

**Output:** `results.json` with lambda sweep and best configuration

---

## Configuration Options

### Negative Split

- `--neg_split_tau`: Classification threshold (default: 0.05)
- `--neg_use_ambiguous_in_obj`: Include ambiguous words in N_obj (default: False)
- `--neg_split_recompute`: Recompute split even if files exist (default: False)

### S_NegLabel* Modifications

- `--use_neglabel_star`: Enable S_NegLabel* modifications (default: False)
- `--topk_aggregation_k`: TopK aggregation parameter (default: 10)
- `--use_role_aware_negatives`: Use N_obj only for base scoring (default: False)
- `--use_scale_stabilization`: Z-score normalization (default: False)

### P_align

- `--use_p_align`: Enable P_align term (default: False)
- `--lambda_bg`: Lambda weighting for P_align (default: 1.0)
- `--pos_topk`: TopK for positive alignment (default: 10)
- `--neg_topk`: TopK for negative alignment (default: 5)

### CAM

- `--cam_fg_percentile`: Foreground percentile threshold (default: 80)
- `--cam_dilate_px`: Dilation pixels (default: 1)
- `--cam_block`: Transformer block index (default: -1, last block)

---

## Implementation Details

### Minimal Disruption Philosophy

NegAlign **reuses** NegRefine infrastructure:
- CLIP model loading and preprocessing
- CSP prompt templates
- Positive class encoding
- Softmax-based grouping logic
- Evaluation metrics (AUROC, FPR95, etc.)

NegAlign **modifies only**:
- Negative label vocabulary (split into obj/bg)
- Base score computation (optional enhancements)
- Background embedding extraction (CAM instead of SAM)

### Key Differences from NegRefine

| Component | NegRefine | NegAlign |
|-----------|-----------|----------|
| Negative Filter | LLM-based (Qwen2.5-14B) | **Disabled** (use initial labels directly) |
| Multi-Matching Score | Enabled | **Disabled** |
| Background Extraction | SAM segmentation | **Grad-CAM attention** |
| Negative Vocabulary | Combined noun+adj | **Split into N_obj + N_bg** |
| Base Score | S_NegLabel (fixed) | **S_NegLabel*** (configurable) |
| Final Score | S_NegLabel + λ·P_align | **S_NegLabel* + λ·P_align** |

### Reproducibility

- Deterministic behavior (respects `seed` parameter)
- All modifications are toggleable
- Setting all flags to False reproduces baseline NegLabel behavior

---

## Citation

If you use NegAlign in your research, please cite:

```bibtex
@inproceedings{negalign2024,
  title={NegAlign: Background-Bias-Aware Zero-Shot OOD Detection},
  author={Your Name},
  booktitle={Conference},
  year={2024}
}
```

---

## Troubleshooting

### CAM Extraction Fails

**Symptom:** `cam_valid=False` or `Warning: CAM extraction failed`

**Solutions:**
- Check CUDA memory (CAM requires backward pass)
- Try reducing `cam_dilate_px` to 0
- Try different `cam_block` (e.g., -2, -3)

### No Background Negatives Found

**Symptom:** `N_bg=0` after splitting

**Solutions:**
- Lower `neg_split_tau` (try 0.03 or 0.02)
- Check that `neg_labels_adj.txt` contains scene/texture words

### Poor OOD Performance

**Symptom:** AUROC < 90%

**Solutions:**
- Enable `--use_neglabel_star` and `--use_role_aware`
- Increase `lambda_bg` (try 2.0 or 5.0)
- Adjust `cam_fg_percentile` (try 70 or 90)

---

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
