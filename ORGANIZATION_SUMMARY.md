# NegAlign 폴더 정리 완료 ✅

**정리 완료 시각:** 2026-01-05

---

## 📁 새로운 폴더 구조

```
NegAlign/
├── core/              ← 핵심 모델 코드 (4 files)
├── tests/             ← 테스트 스크립트 (4 files)
├── scripts/           ← 실행 스크립트 (3 files)
├── examples/          ← 예제 코드 (2 files)
├── docs/              ← 문서 (6 files)
├── logs/              ← 실험 로그 (8 files)
├── results/           ← 실험 결과 (JSON)
├── data/              ← 데이터 파일
├── utils/             ← 유틸리티
├── README.md          ← 메인 README
└── INDEX.md           ← 빠른 파일 탐색 가이드
```

---

## 📝 각 폴더별 내용

### core/ (핵심 모델)
- `clip_negalign.py` - 메인 NegAlign 모델
- `clip_negalign_v2.py` - 8가지 P_align variants
- `cam_bg.py` - CAM 기반 background 추출
- `split_negatives_by_clip.py` - Negative vocabulary 분리
- `__init__.py` - 모듈 초기화

### tests/ (테스트 스크립트)
- `test_imagenet_p_align_variants.py` ⭐ **현재 실행 중**
- `test_imagenet_validation.py` - ImageNet vs OOD (4 methods)
- `test_waterbirds_negalign.py` - WaterBirds 평가
- `test_waterbirds_plain.py` - Baseline 비교
- `README.md` - 테스트 스크립트 설명

### scripts/ (실행 스크립트)
- `run_imagenet_p_align_test.sh` ⭐ **현재 사용**
- `setup.sh` - 초기 setup
- `quickstart.sh` - Quick start
- `README.md` - 스크립트 설명

### examples/ (예제)
- `example_usage.py` - 기본 사용 예제
- `split_negatives_semantic.py` - Semantic splitting 예제

### docs/ (문서)
- `README.md` - 프로젝트 메인 문서
- `SETUP_GUIDE.md` - 설치 가이드
- `IMPLEMENTATION_SUMMARY.md` - 구현 상세
- `PALIGN_RESCUE_PLAN.md` - P_align 개선 전략
- `IMAGENET_TEST_GUIDE.md` - ImageNet 테스트 가이드
- `DELIVERABLES.md` - 프로젝트 산출물

### logs/ (실험 로그)
- `imagenet_p_align_test_20260105_063455.log` ⭐ **현재 실행 중**
- `imagenet_val_4methods.log` - 4가지 방법 비교
- `waterbirds_comparison_v3.log` - WaterBirds 최종
- 기타 과거 실험 로그들
- `README.md` - 로그 파일 설명

### results/ (실험 결과)
- `imagenet_p_align_variants/` - **6-8시간 후 생성 예정**
- `imagenet_validation/` - 기존 ImageNet 결과
- `waterbirds_2class/` - WaterBirds 결과

---

## 🔄 주요 변경사항

### Import 경로 변경

**Before (정리 전):**
```python
from clip_negalign import CLIPNegAlign
from cam_bg import ClipViTGradCAM
```

**After (정리 후):**
```python
from core import CLIPNegAlign, CLIPNegAlignV2
from core.cam_bg import ClipViTGradCAM
```

**Utils는 변경 없음:**
```python
from utils import ood_evaluate  # 그대로
```

---

## 🚀 현재 실행 중인 실험

**PID:** 693732
**Command:** `test_imagenet_p_align_variants.py`
**Started:** 06:35
**Status:** ✅ Running (약 15% 완료, ID 처리 중)
**Progress:** 740/5000 samples processed
**ETA:** 약 5-6시간 남음

**로그 확인:**
```bash
tail -f logs/imagenet_p_align_test_*.log
```

**GPU 모니터링:**
```bash
nvidia-smi
```

---

## 📚 새로 추가된 파일

1. **INDEX.md** - 빠른 파일 탐색 가이드
2. **core/__init__.py** - Core 모듈 초기화
3. **tests/README.md** - 테스트 스크립트 설명
4. **scripts/README.md** - 실행 스크립트 설명
5. **logs/README.md** - 로그 파일 설명
6. **README.md** - 업데이트된 메인 README

---

## ⚡ 빠른 액세스

### 자주 찾는 파일

```bash
# 메인 README
cat README.md

# 파일 탐색 가이드
cat INDEX.md

# P_align 개선 전략
cat docs/PALIGN_RESCUE_PLAN.md

# 현재 실험 로그
tail -f logs/imagenet_p_align_test_*.log

# 실험 코드
vim core/clip_negalign_v2.py
vim tests/test_imagenet_p_align_variants.py
```

### 현재 실험 상태 확인

```bash
# 프로세스 확인
ps aux | grep test_imagenet_p_align

# 진행 상황
tail -30 logs/imagenet_p_align_test_*.log

# GPU 사용률
nvidia-smi
```

---

## 🎯 다음 단계 (실험 완료 후)

1. **결과 확인**
```bash
cat results/imagenet_p_align_variants/imagenet_variant_comparison.json | jq
```

2. **Best variant 분석**
```bash
# Best variant 추출
python3 << 'EOF'
import json
with open('results/imagenet_p_align_variants/imagenet_variant_comparison.json', 'r') as f:
    data = json.load(f)

for variant, result in data['results'].items():
    avg_improvement = sum(
        ood['improvement'] for ood in result['ood_results'].values()
    ) / len(result['ood_results'])
    print(f"{variant}: avg Δ = {avg_improvement:+.4f}")
EOF
```

3. **의사결정**
   - **개선 있음 (>0.5%)**: 논문 작성, Full test 진행
   - **개선 미미 (<0.5%)**: FG-BG contrast 구현 (Phase 3)

---

## 📖 참고 문서

- **시작하기**: `README.md`
- **파일 찾기**: `INDEX.md`
- **P_align 개선**: `docs/PALIGN_RESCUE_PLAN.md`
- **실험 가이드**: `docs/IMAGENET_TEST_GUIDE.md`

---

## ✅ 정리 완료 체크리스트

- [x] 로그 파일 → `logs/`
- [x] 실행 스크립트 → `scripts/`
- [x] 테스트 스크립트 → `tests/`
- [x] 핵심 모델 → `core/`
- [x] 문서 파일 → `docs/`
- [x] 예제 코드 → `examples/`
- [x] Import 경로 정리
- [x] README 파일들 생성
- [x] INDEX.md 생성
- [x] 폴더별 README 생성

---

**정리 완료! 실험이 끝나면 `results/` 폴더를 확인하세요.** 🎉
