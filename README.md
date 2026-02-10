# Hyperbolic Hypergraph Model

하이퍼볼릭 하이퍼그래프 모델 관련 파일들을 관리하는 디렉토리입니다.

## 폴더 구조

```
hyperbolic_model/
├── model/                          # 모델 코드
│   ├── hierarchical_hyperbolic_hypergraph.py  # 메인 모델
│   ├── hyperbolic_losses.py       # Loss 함수들
│   └── hyperbolic_data_loader.py  # 데이터 로더
│
├── scripts/                        # 분석/평가 스크립트
│   ├── train_hyperbolic_hypergraph.py  # 학습 스크립트 (상위 디렉토리)
│   ├── evaluate_checkpoint.py     # 체크포인트 평가
│   ├── visualize_training_results.py  # 학습 결과 시각화
│   ├── visualize_poincare_embeddings.py  # Poincaré Disk 시각화
│   ├── ablation_study_margin.py   # Ablation Study
│   ├── experiment_margin_sweep.py # Margin 실험
│   ├── analyze_initial_loss.py   # 초기 Loss 분석
│   └── analyze_training_results.py  # 학습 결과 분석
│
└── results/                        # 생성되는 결과물
    └── checkpoints/                # 모델 체크포인트, 학습 히스토리, 그래프
        ├── *.pt                    # 모델 체크포인트
        ├── training_history.json   # 학습 히스토리
        ├── *.png                   # 시각화 그래프
        └── *.csv                   # 분석 결과
```

## 사용 방법

### 학습 실행

**방법 1: 쉘 스크립트 사용 (권장, OpenMP 에러 자동 해결)**

```bash
cd hyperbolic_model
./run_training.sh
```

**방법 2: 직접 실행**

```bash
cd hyperbolic_model
export KMP_DUPLICATE_LIB_OK=TRUE  # OpenMP 에러 해결
python train_hyperbolic_hypergraph.py \
    --data_path ../cleaned_data/cleaned_complete_data.json \
    --vocab_path ../feature_encoding/vocabularies.json \
    --output_dir ./results/checkpoints
```

**⚠️ 주의**: `model/` 폴더의 파일들은 모듈이므로 직접 실행할 수 없습니다. 반드시 `train_hyperbolic_hypergraph.py`를 사용하세요.

### 체크포인트 평가

```bash
cd hyperbolic_model/scripts
python evaluate_checkpoint.py \
    --checkpoint ../results/checkpoints/final_model.pt \
    --vocab_path ../../feature_encoding/vocabularies.json \
    --data_path ../../cleaned_data/cleaned_complete_data.json
```

### 시각화 생성

```bash
cd hyperbolic_model/scripts
python visualize_training_results.py \
    ../results/checkpoints/training_history.json \
    ../results/checkpoints
```

### Poincaré Disk 시각화

```bash
cd hyperbolic_model/scripts
python visualize_poincare_embeddings.py \
    --model_path ../results/checkpoints/final_model.pt \
    --vocab_path ../../feature_encoding/vocabularies.json \
    --data_path ../../cleaned_data/cleaned_complete_data.json \
    --output_path ../results/checkpoints/poincare_visualization.png
```

### 조향사 혼동 패턴 분석

모델이 어떤 조향사를 가장 헷갈려 하는지 분석합니다:

```bash
cd hyperbolic_model/scripts
KMP_DUPLICATE_LIB_OK=TRUE python analyze_blender_confusion.py \
    --checkpoint ../results/checkpoints/final_model.pt \
    --data_path ../../cleaned_data/cleaned_complete_data.json \
    --vocab_path ../../feature_encoding/vocabularies.json \
    --k 10 \
    --num_samples 5000
```

**출력 결과:**
- 가장 자주 혼동되는 조향사 Top-10
- 가장 자주 혼동되는 조향사 쌍 Top-10
- Confusion Matrix 시각화 (`blender_confusion_matrix.png`)
- 상세 분석 결과 JSON (`blender_confusion_analysis.json`)

## 주요 파일 설명

### 모델 코드 (`model/`)
- `hierarchical_hyperbolic_hypergraph.py`: 계층적 하이퍼볼릭 하이퍼그래프 모델
- `hyperbolic_losses.py`: 하이퍼볼릭 공간에서의 Loss 함수들
- `hyperbolic_data_loader.py`: 하이퍼볼릭 데이터 로더

### 스크립트 (`scripts/`)
- `train_hyperbolic_hypergraph.py`: 메인 학습 스크립트
- `evaluate_checkpoint.py`: 체크포인트 평가 (Top-1, Top-5, Top-10)
- `visualize_training_results.py`: 학습 결과 시각화
- `visualize_poincare_embeddings.py`: Poincaré Disk 시각화
- `ablation_study_margin.py`: Margin Ablation Study
- `experiment_margin_sweep.py`: Margin 값 실험

### 결과물 (`results/`)
- `checkpoints/`: 모든 학습 결과물 저장
