# NegAlign V3: Soft Weighting Strategy

**Created**: 2026-01-05
**Status**: Implementation Complete, Testing Pending

---

## Problem: Ambiguous Negatives Discarded

In V1/V2, negative words are split using a hard threshold (tau=0.05):

```
r(w) = cos(e(w), E_obj(w)) - cos(e(w), E_bg(w))

if r > +0.05:  N_obj (object-leaning)
if r < -0.05:  N_bg (background-leaning)
else:          N_amb (ambiguous) ← DISCARDED!
```

**Statistics:**
- N_obj: 2866 words
- N_bg: 2866 words
- N_amb: 638 words ← **Wasted!**

---

## Solution: Soft Weighting

Use **all 6370 negatives** with r-score-based weights:

```python
# Soft weights using sigmoid
weight_obj(w) = sigmoid(r(w) * scale)   # High when r > 0 (object-leaning)
weight_bg(w) = sigmoid(-r(w) * scale)   # High when r < 0 (background-leaning)

# scale = 20.0 (default)
# Makes sigmoid steep: ~0.9 for |r| > 0.1, ~0.5 for r ≈ 0
```

### Weighted Scoring

**S_NegLabel* with soft weighting:**
```python
sim_neg = image @ neg_features_all.T  # (1, 6370)
sim_neg_weighted = sim_neg * weights_obj  # Element-wise multiplication
s_neg = TopKMean(sim_neg_weighted, k=10)
```

**P_align with soft weighting:**
```python
sim_bg_neg = bg_embedding @ neg_features_all.T  # (1, 6370)
sim_bg_weighted = sim_bg_neg * weights_bg  # Element-wise multiplication
s_bg_neg = TopKMean(sim_bg_weighted, k=5)
```

---

## Weight Distribution (scale=20.0)

### Overall Statistics

| Metric | Object Weights | Background Weights |
|--------|---------------|-------------------|
| Mean | 0.406 | 0.595 |
| Std | 0.096 | 0.096 |
| >0.9 | 6 words | 20 words |
| >0.7 | 36 words | 766 words |
| <0.3 | 766 words | 36 words |

**Ambiguous (both 0.3-0.7)**: 5568 words

### By Original Category

| Category | Count | Avg Obj Weight | Avg Bg Weight | r-score Range |
|----------|-------|---------------|---------------|---------------|
| object | 2866 | 0.485 | 0.515 | [-0.017, +0.228] |
| background | 2866 | 0.326 | 0.674 | [-0.281, -0.021] |
| ambiguous | 638 | 0.408 | 0.592 | [-0.021, -0.017] |

---

## Example Words

### High Object Weight (obj > 0.9)

| Word | r-score | obj | bg | Category |
|------|---------|-----|----|----|
| wild angelica | +0.228 | 0.990 | 0.010 | object |
| wild calla | +0.183 | 0.975 | 0.025 | object |
| wild spurge | +0.162 | 0.962 | 0.038 | object |
| wild teasel | +0.132 | 0.934 | 0.066 | object |
| wild hollyhock | +0.128 | 0.928 | 0.072 | object |
| wild ginger | +0.123 | 0.922 | 0.078 | object |

### High Background Weight (bg > 0.9)

| Word | r-score | obj | bg | Category |
|------|---------|-----|----|----|
| vanda | -0.281 | 0.004 | 0.996 | background |
| armeria | -0.258 | 0.006 | 0.994 | background |
| carina | -0.240 | 0.008 | 0.992 | background |
| disa | -0.232 | 0.010 | 0.990 | background |
| salal | -0.191 | 0.022 | 0.978 | background |
| salix | -0.175 | 0.030 | 0.970 | background |

### Ambiguous (r ≈ 0, both weights ≈ 0.5)

| Word | r-score | obj | bg | Category |
|------|---------|-----|----|----|
| acute pyelonephritis | 0.000 | 0.500 | 0.500 | object |
| hematocrit | -0.000 | 0.500 | 0.500 | object |
| peter pan collar | -0.000 | 0.500 | 0.500 | object |
| common vetchling | +0.000 | 0.500 | 0.500 | object |

---

## Hard vs Soft Comparison

### Hard Threshold (tau=0.05)

| Category | Count | Usage |
|----------|-------|-------|
| N_obj | 2866 | Used for S_NegLabel* |
| N_bg | 2866 | Used for P_align |
| N_amb | 638 | **DISCARDED** |
| **Total** | **5732 / 6370** | **90% utilization** |

### Soft Weighting (weight > 0.5)

| Category | Count | Usage |
|----------|-------|-------|
| High obj weight | 844 | Strong contribution to S_NegLabel* |
| High bg weight | 5526 | Strong contribution to P_align |
| Both moderate | 5568 | Contribute to both with medium weights |
| **Total** | **6370 / 6370** | **100% utilization ✅** |

---

## Key Benefits

### 1. Information Preservation
- **No words discarded**: All 6370 negatives contribute
- **Ambiguous words useful**: Contribute to both obj and bg appropriately
- **+10% more data**: 638 additional words now utilized

### 2. Smooth Gradients
- **No hard cutoff**: Avoids arbitrary threshold artifacts
- **Continuous weighting**: Gradual transition from obj to bg
- **Differentiable**: Could enable future gradient-based optimization

### 3. Semantic Consistency
- **Words with r ≈ 0**: Naturally get weight ≈ 0.5 for both
- **Extreme r-scores**: Get near 0 or 1, similar to hard threshold
- **Middle range**: Smooth interpolation based on semantic similarity

---

## Implementation

### File: [core/clip_negalign_v3.py](../core/clip_negalign_v3.py)

**Key parameters:**
```python
CLIPNegAlignV3(
    use_soft_weighting=True,      # Enable soft weighting
    soft_weight_scale=20.0,        # Sigmoid sharpness
    # ... other params same as V2 ...
)
```

**Usage:**
```python
from core.clip_negalign_v3 import CLIPNegAlignV3

model = CLIPNegAlignV3(
    train_dataset='imagenet',
    device='cuda:0',
    use_soft_weighting=True,
    soft_weight_scale=20.0,
    use_neglabel_star=True,
    topk_aggregation_k=10,
    use_p_align=True,
    p_align_variant='v2_sign_flip',  # Compatible with V2 variants
    lambda_bg=2.0
)

# Use same as V1/V2
score = model.detection_score(img_tensor, orig_image=img)
```

---

## Testing Plan

### Phase 1: Quick Validation (GPU1, 30 min)

Test V3 on small ImageNet subset to verify:
- Weights load correctly
- Scoring works as expected
- No runtime errors

```bash
# Test script: examples/quick_test_v3.py (already run ✅)
python examples/quick_test_v3.py
```

### Phase 2: ImageNet Comparison (GPU1, 2-3 hours)

Compare V3 vs V2 on ImageNet 1000 samples:
- V3 soft weighting (scale=20)
- V2 hard threshold (current best)
- Baseline (λ=0)

**Expected outcome**: V3 should match or beat V2 due to better negative utilization

### Phase 3: Full Test (if Phase 2 shows improvement)

Run full 5000 sample test with:
- Multiple scale values: [10, 20, 30, 50]
- Multiple variants: v2_sign_flip, v3_ratio, v5_neg_only
- All OOD datasets: iNaturalist, NINCO, Texture

---

## Hyperparameter: soft_weight_scale

Controls sigmoid sharpness:

| Scale | sigmoid(0.1 * scale) | sigmoid(0.05 * scale) | Behavior |
|-------|---------------------|----------------------|----------|
| 10 | 0.731 | 0.622 | Soft, many ambiguous |
| 20 | 0.881 | 0.731 | **Balanced (default)** |
| 30 | 0.953 | 0.818 | Sharp, fewer ambiguous |
| 50 | 0.993 | 0.924 | Very sharp, ~hard threshold |

**Recommendation**: Start with 20.0, tune if needed

---

## Next Steps

1. **Import path fixes**: Resolve module import issues for full model testing
2. **Small-scale test**: 1000 samples on GPU1 to verify functionality
3. **V2 vs V3 comparison**: Same experimental setup, compare AUROC
4. **Scale tuning**: If V3 works, try scales [10, 20, 30] to optimize
5. **Combine with V2 variants**: Test soft_weighting + v2_sign_flip

---

## Status

- ✅ Implementation complete
- ✅ Weight analysis verified
- ⏳ Module import issues (need to fix paths)
- ⏳ Full model testing pending
- ⏳ ImageNet evaluation pending

---

## Files

- **Model**: [core/clip_negalign_v3.py](../core/clip_negalign_v3.py)
- **Analysis script**: [examples/quick_test_v3.py](../examples/quick_test_v3.py)
- **This doc**: [docs/V3_SOFT_WEIGHTING.md](V3_SOFT_WEIGHTING.md)

---

**Last updated**: 2026-01-05
