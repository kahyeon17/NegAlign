# P_align 살리기 작전 🚀

## 📊 현재 상황 분석

### 문제점 진단

```
ID (ImageNet):     P_align = -0.1193
OOD (iNaturalist): P_align = -0.1286
Gap:               0.0093 ← TOO SMALL! ❌

OOD (NINCO):       P_align = -0.1235
Gap:               0.0042 ← EVEN SMALLER! ❌

OOD (Texture):     P_align = -0.1416
Gap:               0.0223 ← SLIGHTLY BETTER, BUT WRONG DIRECTION ⚠️
```

**핵심 문제:**
1. **P_align이 모두 음수** → Background가 predicted class보다 generic negatives와 더 유사
2. **ID-OOD gap이 너무 작음** (0.004~0.022) → 판별력 없음
3. **Texture에서 gap이 역방향** → OOD가 더 음수 → 잘못된 신호

---

## 🎯 P_align 개선 전략 (우선순위 순)

### Strategy 1: **부호 반전** (가장 간단, 즉시 테스트 가능) ⭐⭐⭐

**가설:** 현재 P_align의 부호가 잘못됨

```python
# 현재 (v1_original)
P_align = S_bg_pos - S_bg_neg  # OOD가 더 음수 → 잘못된 방향

# 제안 (v2_sign_flip)
P_align = S_bg_neg - S_bg_pos  # = -(S_bg_pos - S_bg_neg)
```

**기대 효과:**
- Texture: gap = -0.0223 → +0.0223 (올바른 방향)
- iNaturalist: gap = -0.0093 → +0.0093
- Lambda > 0일 때 성능 향상 가능

**테스트 방법:**
```bash
python test_p_align_variants.py \
  --ood_dataset texture \
  --n_samples 500 \
  --variants v2_sign_flip \
  --device cuda:0
```

---

### Strategy 2: **Ratio-based Scoring** (Gap 증폭) ⭐⭐⭐

**가설:** 차이(difference)보다 비율(ratio)이 더 큰 gap을 만듦

```python
# v3_ratio
P_align = S_bg_pos / (S_bg_neg + 1e-8)

# Example:
# ID:  S_bg_pos=-0.10, S_bg_neg=-0.12 → ratio = 0.833
# OOD: S_bg_pos=-0.11, S_bg_neg=-0.14 → ratio = 0.786
# Gap: 0.047 (원래 0.01에서 4배 증폭!)
```

**장점:**
- Difference의 작은 gap을 ratio로 증폭
- 음수 값이어도 비율로 판별 가능

**테스트:**
```bash
python test_p_align_variants.py \
  --ood_dataset texture \
  --variants v3_ratio v4_log_ratio
```

---

### Strategy 3: **Negative-Only Scoring** (Simplification) ⭐⭐

**가설:** S_bg_pos가 noise일 수 있음. Background는 negatives로만 판별

```python
# v5_neg_only
P_align = -S_bg_neg

# 직관: OOD background는 generic negatives와 더 유사 (S_bg_neg가 덜 음수)
# Example:
# ID:  S_bg_neg=-0.12 → P_align=0.12
# OOD: S_bg_neg=-0.14 → P_align=0.14
# Gap: 0.02 (틀림! 역방향) ❌

# 수정: S_bg_neg 자체를 사용
P_align = S_bg_neg  # 더 음수일수록 OOD-like
```

**테스트:**
```bash
python test_p_align_variants.py \
  --variants v5_neg_only v6_pos_only
```

---

### Strategy 4: **Foreground-Background Contrast** (근본적 재설계) ⭐⭐⭐⭐

**가설:** Background만 보지 말고, FG와 BG의 **대비**를 봐야 함

```python
# 현재: Background embedding만 추출
bg_embedding = extract_bg_embedding(image, bg_mask)
P_align = S_bg_pos - S_bg_neg

# 제안: Foreground embedding도 추출
fg_embedding = extract_fg_embedding(image, fg_mask)

# Foreground와 Background의 class-alignment 차이
S_fg_pos = fg_embedding @ text_feature_c_hat
S_bg_pos = bg_embedding @ text_feature_c_hat

P_align = S_fg_pos - S_bg_pos  # FG-BG separation
```

**직관:**
- **ID:** Foreground가 class와 강하게 align → S_fg_pos >> S_bg_pos → **높은 P_align**
- **OOD:** Background bias로 FG-BG 구분 약함 → S_fg_pos ≈ S_bg_pos → **낮은 P_align**

**장점:**
- Background noise에 덜 민감
- FG-BG separation이 background bias의 직접적 지표

**구현 필요:**
```python
def extract_fg_embedding(clip_model, image_tensor, fg_mask):
    """Extract foreground embedding using CAM mask."""
    # Get patch embeddings (before projection)
    patch_tokens = clip_model.visual.forward_patches(image_tensor)

    # Apply foreground mask
    fg_embedding = masked_pooling(patch_tokens, fg_mask, mode='mean')

    # Project to CLIP space
    fg_embedding = fg_embedding @ clip_model.visual.proj
    fg_embedding = fg_embedding / fg_embedding.norm()

    return fg_embedding
```

---

### Strategy 5: **Multi-Scale CAM** (Background 추출 개선) ⭐⭐

**가설:** Single-block CAM이 부정확함 → Multi-scale averaging 필요

```python
# 현재: Last block only
cam = compute_cam(image, block_idx=-1)

# 제안: Multiple blocks
cams = []
for block_idx in [-1, -2, -3]:  # Last 3 blocks
    cam_i = compute_cam(image, block_idx)
    cams.append(cam_i)

# Weighted average (later blocks have more weight)
weights = [0.5, 0.3, 0.2]
multi_scale_cam = sum(w * c for w, c in zip(weights, cams))
```

**기대 효과:**
- 더 정확한 foreground/background 분리
- Edge leakage 감소

---

### Strategy 6: **Adaptive Threshold** (Per-image calibration) ⭐⭐

**가설:** Fixed percentile (80th)이 모든 이미지에 적합하지 않음

```python
# 현재: Fixed 80th percentile
threshold = np.percentile(cam, 80)

# 제안: Adaptive threshold (Otsu's method)
from skimage.filters import threshold_otsu
threshold = threshold_otsu(cam)

# 또는: Entropy-based
def entropy_threshold(cam):
    """Find threshold that maximizes entropy difference."""
    best_threshold = 0
    best_entropy_diff = 0

    for percentile in range(50, 95, 5):
        t = np.percentile(cam, percentile)
        fg = cam >= t
        bg = cam < t

        # Compute entropy difference
        fg_entropy = -np.sum(cam[fg] * np.log(cam[fg] + 1e-8))
        bg_entropy = -np.sum(cam[bg] * np.log(cam[bg] + 1e-8))
        entropy_diff = abs(fg_entropy - bg_entropy)

        if entropy_diff > best_entropy_diff:
            best_entropy_diff = entropy_diff
            best_threshold = t

    return best_threshold
```

---

### Strategy 7: **Learned Lambda Weighting** (Data-driven) ⭐⭐⭐⭐

**가설:** Fixed λ는 suboptimal → Learn from data

```python
# Option A: Per-dataset optimal lambda (validation set)
validation_results = {}
for lambda_val in [0, 0.5, 1.0, 2.0, 5.0, 10.0]:
    auroc = evaluate_with_lambda(lambda_val, val_id, val_ood)
    validation_results[lambda_val] = auroc

best_lambda = max(validation_results, key=validation_results.get)

# Option B: Adaptive lambda based on P_align confidence
def adaptive_lambda(p_align, cam_valid):
    """Higher lambda when P_align is more confident."""
    if not cam_valid:
        return 0.0

    # Use P_align magnitude as confidence
    confidence = abs(p_align)

    # Scale lambda inversely with confidence
    # Low confidence → low weight
    lambda_val = min(10.0, confidence * 50.0)

    return lambda_val

# Option C: Learn linear combination
# S_final = α * S_star + β * P_align
# Learn α, β from validation set using logistic regression
```

---

## 🧪 실험 프로토콜

### Phase 1: Quick Variant Testing (1-2시간)

```bash
# Test all variants on small sample
python test_p_align_variants.py \
  --ood_dataset texture \
  --n_samples 500 \
  --device cuda:0

# Expected output: variant ranking with improvement scores
```

**Decision Point:**
- If **any variant shows >1% improvement** → Proceed to Phase 2
- If **no improvement** → Move to Strategy 4 (FG-BG contrast)

---

### Phase 2: Best Variant Full Test (2-3시간)

```bash
# Test best variant on full dataset
python test_imagenet_validation.py \
  --p_align_variant v2_sign_flip \  # Or best from Phase 1
  --samples_per_class 5 \
  --max_ood_samples 5000
```

**Success Criteria:**
- AUROC improvement > 0.5% on at least 1 OOD dataset
- Best λ > 0 (P_align actually used)

---

### Phase 3: FG-BG Contrast (if Phase 1/2 fail) (1일)

**Implement new method:**

```python
# In clip_negalign_v2.py, add:

def _compute_p_align_fg_bg(self, image_tensor, predicted_class_idx):
    """Foreground-Background contrast version."""
    text_feature_c_hat = self.pos_features[predicted_class_idx]

    # Get CAM
    cam = self.cam_generator.compute_cam(image_tensor, text_feature_c_hat)

    # Get FG and BG masks
    fg_mask, bg_mask = cam_to_masks(cam, self.cam_fg_percentile, self.cam_dilate_px)

    # Extract both embeddings
    fg_embedding = extract_fg_embedding(self.clip_model, image_tensor, fg_mask)
    bg_embedding = extract_bg_embedding(self.clip_model, image_tensor, bg_mask)

    # Compute class alignment for each
    s_fg_pos = float((fg_embedding @ text_feature_c_hat).item())
    s_bg_pos = float((bg_embedding @ text_feature_c_hat).item())

    # P_align = FG-BG separation
    p_align = s_fg_pos - s_bg_pos

    return p_align, True
```

**Expected Improvement:**
- Gap should be **5-10x larger** (현재 0.01 → 0.05~0.1)
- More interpretable (FG-BG separation is what we want)

---

### Phase 4: Multi-Scale + Adaptive (최종 단계) (2일)

Combine best strategies:
1. FG-BG contrast
2. Multi-scale CAM
3. Adaptive threshold
4. Learned lambda

---

## 📈 예상 성능 개선

### Conservative Estimate (Strategy 1-3만)

```
현재 (Best λ=0.0):
  iNaturalist: 0.9977
  NINCO:       0.8041
  Texture:     0.9382

Strategy 2 (v2_sign_flip) 성공 시:
  iNaturalist: 0.9977 → 0.9980 (+0.03%)
  NINCO:       0.8041 → 0.8090 (+0.49%) ✅
  Texture:     0.9382 → 0.9420 (+0.38%) ✅
```

### Optimistic Estimate (Strategy 4 FG-BG contrast)

```
FG-BG contrast 성공 시:
  iNaturalist: 0.9977 → 0.9985 (+0.08%)
  NINCO:       0.8041 → 0.8200 (+1.59%) 🔥
  Texture:     0.9382 → 0.9500 (+1.18%) 🔥
```

### Best Case (All strategies combined)

```
Multi-scale + FG-BG + Learned λ:
  iNaturalist: 0.9977 → 0.9990 (+0.13%)
  NINCO:       0.8041 → 0.8300 (+2.59%) 🚀
  Texture:     0.9382 → 0.9550 (+1.68%) 🚀
```

---

## ✅ 실행 체크리스트

### Immediate Actions (오늘)

- [ ] **Run test_p_align_variants.py** on Texture dataset
  ```bash
  cd /home/kahyeon/research/NegAlign
  python test_p_align_variants.py \
    --ood_dataset texture \
    --n_samples 500 \
    --device cuda:0
  ```

- [ ] **Analyze results** and pick best variant
- [ ] **If successful:** Test on full dataset
- [ ] **If failed:** Start implementing FG-BG contrast

### Short-term (이번 주)

- [ ] Implement `extract_fg_embedding()` function
- [ ] Add `v9_fg_bg_contrast` variant
- [ ] Test FG-BG contrast on all 3 OOD datasets
- [ ] Compare with baseline

### Medium-term (다음 주)

- [ ] Implement multi-scale CAM
- [ ] Add adaptive threshold
- [ ] Validation-based lambda tuning
- [ ] Write ablation study section for paper

---

## 🎓 논문 작성 전략 (P_align 성공 시)

### Main Story

**Title:** "Foreground-Background Contrast for Background-Bias-Aware OOD Detection"

**Key Contributions:**
1. Role-aware negative vocabulary (N_obj vs N_bg)
2. **FG-BG contrast metric** for background bias detection
3. Multi-scale CAM for accurate region separation
4. Adaptive weighting for dataset-specific calibration

**Positioning:**
- First to use FG-BG contrast for OOD detection
- Interpretable metric (FG-BG separation)
- Strong empirical results on background-biased OOD

---

## 🎓 논문 작성 전략 (P_align 실패 시)

### Honest Analysis Paper

**Title:** "What Makes Background Calibration Effective? An Empirical Study"

**Key Contributions:**
1. Comprehensive analysis of P_align variants (8 variants tested)
2. **Negative result:** Simple background alignment insufficient
3. **Finding:** FG-BG contrast necessary but not sufficient
4. Dataset-specific recommendations (when to use what)

**Value:**
- Important negative results for community
- Deep analysis of failure modes
- Future research directions

---

## 📞 다음 단계

**즉시 실행:**
```bash
cd /home/kahyeon/research/NegAlign
chmod +x test_p_align_variants.py
python test_p_align_variants.py --ood_dataset texture --n_samples 500
```

**결과 확인 후:**
1. **개선 있음 (>1%):** Full dataset test 진행
2. **개선 미미 (<1%):** FG-BG contrast 구현 시작
3. **성능 악화:** 원인 분석 및 대안 검토

**코드 파일:**
- `clip_negalign_v2.py`: 8가지 P_align variants 구현 완료 ✅
- `test_p_align_variants.py`: Variant testing script 완료 ✅
- 다음: `cam_bg_v2.py` (FG-BG contrast용 fg_embedding 추출)

**예상 소요 시간:**
- Phase 1 (variant test): 1-2시간
- Phase 2 (full test): 2-3시간
- Phase 3 (FG-BG): 1일
- Phase 4 (final): 2일

**Total:** 3-4일이면 P_align 완전 해결 가능! 🚀
