# NegAlign Project Index 📑

빠른 파일 탐색을 위한 인덱스

---

## 🎯 지금 바로 필요한 파일

### 실험 실행
- **현재 실행 중:** `scripts/run_imagenet_p_align_test.sh`
- **로그 확인:** `tail -f logs/imagenet_p_align_test_*.log`
- **결과 확인:** `results/imagenet_p_align_variants/imagenet_variant_comparison.json`

### 코드 수정
- **메인 모델:** `core/clip_negalign.py`
- **P_align variants:** `core/clip_negalign_v2.py` (8가지 variant)
- **Background 추출:** `core/cam_bg.py`

---

## 📁 디렉토리 구조

```
NegAlign/
├── core/           → 핵심 모델 코드
├── tests/          → 평가 스크립트
├── scripts/        → 실행 스크립트
├── examples/       → 예제 코드
├── docs/           → 문서
├── logs/           → 실험 로그
├── results/        → 실험 결과
├── data/           → 데이터 파일
└── utils/          → 유틸리티
```

---

## 🔬 핵심 파일 (Core)

| 파일 | 설명 | 주요 클래스/함수 |
|------|------|------------------|
| `clip_negalign.py` | 메인 NegAlign 모델 | `CLIPNegAlign` |
| `clip_negalign_v2.py` | 8가지 P_align variants | `CLIPNegAlignV2` |
| `cam_bg.py` | CAM 기반 background 추출 | `ClipViTGradCAM`, `extract_bg_embedding` |
| `split_negatives_by_clip.py` | Negative vocabulary 분리 | `split_negatives` |

**사용 예시:**
```python
from core import CLIPNegAlign, CLIPNegAlignV2
```

---

## 🧪 테스트 스크립트 (Tests)

| 파일 | 용도 | 실행 시간 |
|------|------|-----------|
| `test_imagenet_p_align_variants.py` ⭐ | **P_align variant 테스트 (현재 실행 중)** | 6-8시간 |
| `test_imagenet_validation.py` | ImageNet vs OOD (4 methods) | 3-4시간 |
| `test_waterbirds_negalign.py` | WaterBirds 평가 | 1-2시간 |
| `test_waterbirds_plain.py` | WaterBirds baseline | 1-2시간 |

**사용 예시:**
```bash
python tests/test_imagenet_p_align_variants.py --device cuda:0
```

---

## 📜 실행 스크립트 (Scripts)

| 파일 | 설명 |
|------|------|
| `setup.sh` | 초기 setup (negative labels 복사) |
| `quickstart.sh` | Quick start test |
| `run_imagenet_p_align_test.sh` | **P_align variant test (현재 사용)** |

**사용 예시:**
```bash
bash scripts/run_imagenet_p_align_test.sh
```

---

## 📚 문서 (Docs)

| 파일 | 내용 |
|------|------|
| `README.md` | 프로젝트 전체 개요 |
| `SETUP_GUIDE.md` | 설치 가이드 |
| `IMPLEMENTATION_SUMMARY.md` | 기술 구현 상세 |
| `PALIGN_RESCUE_PLAN.md` | **P_align 개선 전략** ⭐ |
| `IMAGENET_TEST_GUIDE.md` | ImageNet 테스트 가이드 |
| `DELIVERABLES.md` | 프로젝트 산출물 |

**바로 읽기:**
- P_align이 왜 안되는지: `docs/PALIGN_RESCUE_PLAN.md`
- 실험 결과 분석: `docs/IMPLEMENTATION_SUMMARY.md`

---

## 📊 실험 로그 (Logs)

### 현재 실행 중
- `imagenet_p_align_test_20260105_063455.log` ⭐ **← 지금 확인**

### 과거 실험
- `imagenet_val_4methods.log` - 4가지 방법 비교 (Plain/Star/+P_align)
- `waterbirds_comparison_v3.log` - WaterBirds 최종 비교
- `lambda_search.log` - Lambda tuning

**로그 확인:**
```bash
# 실시간
tail -f logs/imagenet_p_align_test_*.log

# 전체 보기
less logs/imagenet_val_4methods.log
```

---

## 📈 실험 결과 (Results)

```
results/
├── imagenet_p_align_variants/  ← 현재 실험 (6-8시간 후 생성됨)
│   └── imagenet_variant_comparison.json
├── imagenet_validation/
│   └── validation_results.json
└── waterbirds_2class/
    └── comparison_results.json
```

**결과 확인:**
```bash
# P_align variant 결과 (실험 완료 후)
cat results/imagenet_p_align_variants/imagenet_variant_comparison.json | jq '.results | keys'

# 과거 ImageNet 결과
cat results/imagenet_validation/validation_results.json | jq '.ood_results'
```

---

## 💡 예제 코드 (Examples)

| 파일 | 설명 |
|------|------|
| `example_usage.py` | 기본 사용 예제 |
| `split_negatives_semantic.py` | Semantic splitting 예제 |

---

## 🔧 유틸리티 (Utils)

| 파일 | 설명 |
|------|------|
| `class_names.py` | ImageNet class names, templates |
| `create_negs.py` | CSP negative label 생성 |
| `ood_evaluate.py` | OOD 평가 metrics (AUROC, FPR95) |

---

## 🎯 Quick Actions

### 현재 실험 확인
```bash
# 실시간 로그
tail -f logs/imagenet_p_align_test_*.log

# GPU 사용률
nvidia-smi

# 프로세스 확인
ps aux | grep test_imagenet_p_align
```

### 실험 완료 후
```bash
# 결과 확인
cat results/imagenet_p_align_variants/imagenet_variant_comparison.json

# Best variant 추출
python3 << 'EOF'
import json
with open('results/imagenet_p_align_variants/imagenet_variant_comparison.json', 'r') as f:
    data = json.load(f)

# Find best variant
for variant, result in data['results'].items():
    avg_improvement = sum(
        ood['improvement'] for ood in result['ood_results'].values()
    ) / len(result['ood_results'])
    print(f"{variant}: avg Δ = {avg_improvement:+.4f}")
EOF
```

### 코드 수정
```bash
# P_align 새 variant 추가
vim core/clip_negalign_v2.py

# Test script 수정
vim tests/test_imagenet_p_align_variants.py
```

---

## 📖 읽는 순서 추천

### 처음 시작하는 경우
1. `README.md` - 프로젝트 개요
2. `docs/SETUP_GUIDE.md` - 설치
3. `examples/example_usage.py` - 사용법
4. `docs/IMPLEMENTATION_SUMMARY.md` - 구현 상세

### P_align 개선하고 싶은 경우
1. `docs/PALIGN_RESCUE_PLAN.md` - 개선 전략 ⭐
2. `core/clip_negalign_v2.py` - Variant 구현
3. `logs/imagenet_val_4methods.log` - 현재 문제 확인
4. `tests/test_imagenet_p_align_variants.py` - 테스트 코드

### 실험 결과 분석하고 싶은 경우
1. `logs/imagenet_val_4methods.log` - 기존 결과
2. `results/imagenet_validation/validation_results.json` - JSON 결과
3. `docs/IMPLEMENTATION_SUMMARY.md` - 분석 방법

---

## 🔍 파일 찾기

### 특정 기능 찾기
```bash
# P_align 관련
grep -r "p_align" core/ tests/

# CAM 관련
grep -r "ClipViTGradCAM" core/

# Evaluation 관련
grep -r "evaluate_all" tests/
```

### 최근 수정된 파일
```bash
# 최근 1일 이내
find . -type f -name "*.py" -mtime -1 -ls

# 최근 수정된 로그
ls -lt logs/ | head -5
```

---

## ⚡ 자주 사용하는 명령어

```bash
# 실험 실행
bash scripts/run_imagenet_p_align_test.sh

# 로그 실시간 확인
tail -f logs/imagenet_p_align_test_*.log

# GPU 모니터링
watch -n 1 nvidia-smi

# 결과 확인 (실험 완료 후)
cat results/imagenet_p_align_variants/imagenet_variant_comparison.json | jq

# 프로세스 확인
ps aux | grep python | grep test_imagenet

# 디스크 사용량
du -sh results/ logs/
```

---

## 📞 문제 해결

### Import 오류
```python
# 정리 후 새 import 경로
from core import CLIPNegAlign, CLIPNegAlignV2
from core.cam_bg import ClipViTGradCAM

# utils는 그대로
from utils import ood_evaluate
```

### 경로 오류
```bash
# 항상 NegAlign/ 루트에서 실행
cd /home/kahyeon/research/NegAlign
python tests/test_imagenet_p_align_variants.py
```

### 실험 중단됐는지 확인
```bash
# 프로세스 확인
ps aux | grep test_imagenet_p_align

# 없으면 재시작
bash scripts/run_imagenet_p_align_test.sh
```

---

**이 파일은 프로젝트 탐색을 위한 인덱스입니다. 자주 찾는 파일을 여기서 빠르게 찾으세요!**
