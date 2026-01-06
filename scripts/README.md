# Scripts

실행 가능한 스크립트 모음

## Setup Scripts

- **setup.sh**: Initial setup (copy negative labels from NegRefine)
- **quickstart.sh**: Quick start test

## Experiment Scripts

- **run_imagenet_p_align_test.sh**: ImageNet P_align variant testing (6-8 hours)
  - 5000 ImageNet samples (5 per class)
  - 3 OOD datasets (iNaturalist, NINCO, Texture)
  - 4 P_align variants tested

## Usage

```bash
# Setup
bash scripts/setup.sh

# Quick test
bash scripts/quickstart.sh

# Full P_align experiment
bash scripts/run_imagenet_p_align_test.sh
```
