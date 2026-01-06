# ImageNet P_align Variant Testing Guide

## 🚀 바로 실행하기

ImageNet 5000 샘플 (클래스별 5개 × 1000) + OOD 3개 데이터셋 테스트:

```bash
cd /home/kahyeon/research/NegAlign
./run_imagenet_p_align_test.sh
```

**설정:**
- ID: ImageNet 5000개 (클래스별 5개)
- OOD: iNaturalist 5000개, NINCO 5000개, Texture 5000개
- Variants: v1_original, v2_sign_flip, v3_ratio, v5_neg_only
- Lambda: [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
- 예상 시간: **6-8시간**

---

## 📊 예상 출력

### 실행 중

```
==================================================
Testing Variant: V2_SIGN_FLIP
==================================================

ID Statistics:
  S_plain: mean=0.9790, std=0.0156
  S_star:  mean=0.9919, std=0.0089
  P_align: mean=0.1193, std=0.0413  ← 양수로 전환!

OOD Statistics (texture):
  P_align: mean=0.1416, std=0.0336
  Gap: -0.0223 ← KEY METRIC

Lambda Sweep:
Lambda   AUROC      FPR95      Improvement
0.0      0.8936     0.0000     +0.0000
5.0      0.9050     0.0000     +0.0114  🔥
```

### 최종 요약

```
==================================================
OVERALL RECOMMENDATION
==================================================

Best overall variant: v2_sign_flip
Average improvement: +0.0098

Recommended λ values:
  texture: λ=5.0 (Δ=+0.0114)
  ninco:   λ=2.0 (Δ=+0.0089)

✅ SUCCESS! Use v2_sign_flip with dataset-specific λ
```

---

## 📁 결과 파일

```
results/imagenet_p_align_variants/imagenet_variant_comparison.json
```

---

## 🎯 성공 기준

- **대성공 (>1%)**: 논문 main contribution
- **성공 (>0.5%)**: Ablation study 포함
- **실패 (<0.5%)**: FG-BG contrast 필요
