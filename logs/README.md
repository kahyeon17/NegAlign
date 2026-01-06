# Logs

실험 로그 파일 모음

## ImageNet Experiments

- **imagenet_val_4methods.log**: 4가지 방법 비교 (Plain, Star, +P_align)
- **imagenet_validation_test.log**: Validation test 로그
- **imagenet_p_align_test_*.log**: P_align variant testing (CURRENT)

## WaterBirds Experiments

- **waterbirds_2class_experiment.log**: 2-class WaterBirds test
- **waterbirds_comparison*.log**: Multiple comparison runs

## Lambda Search

- **lambda_search.log**: Lambda hyperparameter tuning

## Current Running Experiment

Check progress:
```bash
tail -f logs/imagenet_p_align_test_*.log
```

Monitor GPU:
```bash
watch -n 1 nvidia-smi
```
