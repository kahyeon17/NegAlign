# Tests

평가 및 테스트 스크립트 모음

## Main Test Scripts

### ImageNet Tests

- **test_imagenet_validation.py**: ImageNet vs multiple OOD datasets
  - Class-balanced sampling
  - 4 methods comparison (Plain, Star, Plain+P_align, Star+P_align)
  - Lambda sweep

- **test_imagenet_p_align_variants.py**: P_align variant testing ⭐ Current
  - Tests 8 different P_align formulations
  - Comprehensive evaluation on 3 OOD datasets
  - Automatic best variant selection

### WaterBirds Tests

- **test_waterbirds_negalign.py**: WaterBirds dataset evaluation
  - Background-biased dataset
  - Lambda sweep for P_align

- **test_waterbirds_plain.py**: Baseline comparison on WaterBirds

## Usage

### Run P_align Variant Test

```bash
python tests/test_imagenet_p_align_variants.py \
  --imagenet_root /path/to/imagenet \
  --ood_root /path/to/ood \
  --ood_datasets inaturalist ninco texture \
  --samples_per_class 5 \
  --variants v1_original v2_sign_flip v3_ratio v5_neg_only \
  --device cuda:0
```

### Run ImageNet Validation

```bash
python tests/test_imagenet_validation.py \
  --imagenet_root /path/to/imagenet \
  --ood_root /path/to/ood \
  --samples_per_class 5 \
  --device cuda:0
```

## Import from Core

All test scripts import from `core/`:

```python
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core import CLIPNegAlign, CLIPNegAlignV2
```
