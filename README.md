# NegAlign: Background-Bias-Aware Zero-Shot OOD Detection

NegAlign is a minimal modification of NegRefine that addresses background bias in zero-shot OOD detection using:
- **S_NegLabel*** (modified base score with configurable enhancements)
- **P_align** (CAM-based background alignment, NO SAM)

**Final score:** `S_final = S_NegLabel* + λ × P_align`

---

## 📁 Project Structure

```
NegAlign/
├── core/                           # 핵심 모델 파일
│   ├── clip_negalign.py           # Main NegAlign model (original)
│   ├── clip_negalign_v2.py        # Enhanced with 8 P_align variants
│   ├── cam_bg.py                  # Grad-CAM background extraction
│   └── split_negatives_by_clip.py # Negative vocabulary splitting
│
├── tests/                          # 평가 스크립트
│   ├── test_imagenet_validation.py        # ImageNet vs OOD (4 methods)
│   ├── test_imagenet_p_align_variants.py  # P_align variant testing
│   ├── test_waterbirds_negalign.py        # WaterBirds evaluation
│   └── test_waterbirds_plain.py           # Baseline comparison
│
├── scripts/                        # 실행 스크립트
│   ├── setup.sh                   # Initial setup
│   ├── quickstart.sh              # Quick start
│   └── run_imagenet_p_align_test.sh  # P_align variant test runner
│
├── examples/                       # 예제 코드
│   ├── example_usage.py           # Basic usage examples
│   └── split_negatives_semantic.py  # Semantic splitting example
│
├── docs/                          # 문서
│   ├── README.md                  # This file (main)
│   ├── SETUP_GUIDE.md            # Installation guide
│   ├── IMPLEMENTATION_SUMMARY.md  # Technical details
│   ├── PALIGN_RESCUE_PLAN.md     # P_align improvement strategies
│   ├── IMAGENET_TEST_GUIDE.md    # ImageNet testing guide
│   └── DELIVERABLES.md           # Project deliverables
│
├── logs/                          # 실험 로그
│   ├── imagenet_val_4methods.log
│   ├── waterbirds_comparison_v3.log
│   └── imagenet_p_align_test_*.log
│
├── data/                          # 데이터 파일
│   ├── neg_labels_noun.pkl       # Negative labels (nouns)
│   ├── neg_labels_adj.pkl        # Negative labels (adjectives)
│   └── negatives_split/          # Split negative vocabularies
│
├── results/                       # 실험 결과
│   ├── imagenet_validation/
│   ├── imagenet_p_align_variants/
│   └── waterbirds_2class/
│
└── utils/                         # 유틸리티 함수
    ├── class_names.py
    ├── create_negs.py
    └── ood_evaluate.py
```

---

## 🚀 Quick Start

### 1. Setup

```bash
cd /home/kahyeon/research/NegAlign
bash scripts/setup.sh
```

### 2. Run Quick Test

```bash
bash scripts/quickstart.sh
```

### 3. Test P_align Variants (ImageNet 5000 samples)

```bash
bash scripts/run_imagenet_p_align_test.sh
```

**실시간 로그 확인:**
```bash
tail -f logs/imagenet_p_align_test_*.log
```

---

## 💡 Key Features

### 1. S_NegLabel* (Modified Base Score)

Built on NegRefine's softmax-based grouping score with optional modifications:

- **(M1) TopK Aggregation**: Replace max with TopKMean (configurable k, default 10)
- **(M2) Role-Aware Negatives**: Use object-leaning negatives (N_obj) only for base scoring
- **(M3) Scale Stabilization**: Z-score normalization (optional, off by default)

**All modifications are TOGGLEABLE** to reproduce S_NegLabel_plain exactly.

### 2. Negative Vocabulary Splitting

**Word-level SOFT classification** into object vs background using CLIP embeddings:

```python
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

```python
Step 1: Extract background embedding I_bg using Grad-CAM
Step 2: S_bg_pos = TopKMean(cos(I_bg, T_pos[c_hat]), k=10)
Step 3: S_bg_neg = TopKMean(cos(I_bg, E(N_bg)), k=5)

P_align = S_bg_pos - S_bg_neg
```

**Enhanced V2 with 8 variants:**
- v1_original: `S_bg_pos - S_bg_neg`
- v2_sign_flip: `-(S_bg_pos - S_bg_neg)` ⭐
- v3_ratio: `S_bg_pos / S_bg_neg` ⭐
- v4_log_ratio: `log(S_bg_pos / S_bg_neg)`
- v5_neg_only: `-S_bg_neg`
- v6_pos_only: `S_bg_pos`
- v7_normalized: Normalized difference
- v8_squared_diff: Squared difference

---

## 📊 Usage Examples

### Basic Usage

```python
from core import CLIPNegAlign

# Initialize model
model = CLIPNegAlign(
    train_dataset='imagenet',
    arch='ViT-B/16',
    device='cuda:0',
    output_folder='./data/',
    load_saved_labels=True,
    use_neglabel_star=True,
    use_role_aware_negatives=True,
    use_p_align=True,
    lambda_bg=2.0
)

# Process image
from PIL import Image
img = Image.open('test.jpg').convert('RGB')
img_tensor = model.clip_preprocess(img).unsqueeze(0).to('cuda:0')

# Get score
details = model.detection_score(img_tensor, orig_image=img, return_details=True)
print(f"S_final: {details['s_final']:.4f}")
print(f"P_align: {details['p_align']:.4f}")
```

### Testing P_align Variants

```python
from core import CLIPNegAlignV2

# Test with sign-flip variant
model = CLIPNegAlignV2(
    train_dataset='imagenet',
    device='cuda:0',
    use_p_align=True,
    p_align_variant='v2_sign_flip',  # Try different variants
    lambda_bg=5.0
)

# Get scores
details = model.detection_score(img_tensor, return_details=True)
```

---

## 📈 Experimental Results

### ImageNet (5000 samples) vs OOD

**Current baseline (λ=0):**

| OOD Dataset | Plain (6370) | Star (2866) | Best Method |
|-------------|--------------|-------------|-------------|
| iNaturalist | 0.9971 | **0.9977** | Star |
| NINCO | 0.7934 | **0.8041** | Star |
| Texture | **0.9382** | 0.8936 | Plain |

**Issue:** P_align shows minimal improvement with original formulation (best λ=0.0)

**Solution:** Testing 8 P_align variants to find better formulation

---

## 🔧 Configuration Options

### Negative Split Parameters

```python
neg_split_tau=0.05              # Classification threshold
neg_use_ambiguous_in_obj=False  # Include ambiguous in N_obj
neg_split_recompute=False       # Force recomputation
```

### S_NegLabel* Modifications

```python
use_neglabel_star=False         # Enable modifications
topk_aggregation_k=10           # TopK parameter
use_role_aware_negatives=False  # Use N_obj only
use_scale_stabilization=False   # Z-score normalization
```

### P_align Parameters

```python
use_p_align=False               # Enable P_align
lambda_bg=1.0                   # P_align weight
pos_topk=10                     # TopK for positive alignment
neg_topk=5                      # TopK for negative alignment
p_align_variant='v1_original'   # Variant selection (V2 only)
```

### CAM Parameters

```python
cam_fg_percentile=80            # Foreground threshold
cam_dilate_px=1                 # Dilation iterations
cam_block=-1                    # Transformer block index
```

---

## 📚 Documentation

- **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)**: Detailed installation instructions
- **[IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)**: Technical implementation details
- **[PALIGN_RESCUE_PLAN.md](docs/PALIGN_RESCUE_PLAN.md)**: Strategies to improve P_align
- **[IMAGENET_TEST_GUIDE.md](docs/IMAGENET_TEST_GUIDE.md)**: ImageNet testing guide

---

## 🎯 Current Focus: P_align Improvement

**Problem:** Original P_align formulation shows minimal ID-OOD separation

**Approach:** Testing 8 different P_align variants to find better formulation

**Running Experiment:**
```bash
# Monitor progress
tail -f logs/imagenet_p_align_test_*.log

# Check results (after 6-8 hours)
cat results/imagenet_p_align_variants/imagenet_variant_comparison.json
```

---

## 🐛 Troubleshooting

### Import Errors

After reorganization, update import paths:
```python
# Old
from clip_negalign import CLIPNegAlign

# New
from core import CLIPNegAlign
```

### CAM Extraction Fails

```python
# Try different settings
cam_fg_percentile=70  # Lower threshold
cam_dilate_px=0       # No dilation
cam_block=-2          # Different block
```

### OOM Errors

```python
# Reduce batch size or samples
samples_per_class=3   # Instead of 5
max_ood_samples=3000  # Instead of 5000
```

---

## 📝 Citation

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

## 🤝 Contributing

This is a research project. For questions or issues, please refer to the documentation in `docs/`.

---

## 📧 Contact

For technical questions, check:
1. `docs/IMPLEMENTATION_SUMMARY.md` - Technical details
2. `docs/PALIGN_RESCUE_PLAN.md` - P_align improvement strategies
3. Log files in `logs/` - Experimental results
