# 데이터셋 일치 절차 (하이퍼볼릭 ↔ 베이스라인)

하이퍼볼릭 모델과 RecBole 베이스라인(LightGCN, HGCF, CML)이 **동일한 train/val/test 분할**을 쓰도록 하는 방법입니다.

---

## 원칙: 한 곳이 “진짜” 데이터셋

- **진짜 데이터셋**은 하이퍼볼릭 학습이 저장하는 `output_dir/datasets/` 폴더입니다.
- 베이스라인은 이 폴더를 `--dataset_dir`로 지정하면, 그 안의 `train_combinations.json`, `val_combinations.json`, `test_combinations.json`(및 `test_combinations_original.json`)을 RecBole 형식으로 변환해 사용합니다.
- 따라서 **같은 폴더를 가리키기만 하면** 데이터셋이 자동으로 일치합니다.

---

## 절차 1: 하이퍼볼릭 먼저 한 번 돌리기 (데이터 생성)

```bash
cd newcode2/hyperbolic_model
python train_hyperbolic_hypergraph.py
# 기본 출력: results/checkpoints/datasets/
#   train_combinations.json, val_combinations.json, test_combinations.json,
#   test_combinations_original.json, split_info.json
```

학습을 끝까지 하지 않아도, 데이터 분할이 끝난 시점에 위 파일들이 저장됩니다.  
이미 이전에 학습을 돌렸다면 `results/checkpoints/datasets/` (또는 `--output_dir`로 준 경로의 `datasets/`)가 이미 있을 수 있습니다.

---

## 절차 2: 베이스라인은 “그 폴더”를 지정해서 실행

**같은 데이터셋**을 쓰려면, 위에서 쓰는 **datasets 폴더 경로**를 그대로 넘깁니다.

```bash
cd newcode2/recbole_baselines

# 방법 A: checkpoints까지 주면 스크립트가 자동으로 .../datasets 찾음
python run_baselines.py --dataset_dir ../hyperbolic_model/results/checkpoints

# 방법 B: datasets 폴더를 직접 지정
python run_baselines.py --dataset_dir ../hyperbolic_model/results/checkpoints/datasets
```

- `run_baselines.py`는 `--dataset_dir`에 `train_combinations.json`이 없고 `datasets/train_combinations.json`이 있으면 자동으로 `datasets` 쪽을 사용합니다.
- `prepare_recbole_data.py`가 같은 디렉터리의 JSON을 읽어 RecBole용 `.inter`를 만들고, `split_info.json`이 있으면 “[데이터셋 일치] …” 메시지로 동일 분할 사용을 출력합니다.

---

## 절차 3: (선택) 스크립트로 한 번에

```bash
cd newcode2/recbole_baselines
./run_same_as_hyperbolic.sh
# 또는 하이퍼볼릭 output 디렉터리 지정
./run_same_as_hyperbolic.sh /path/to/hyperbolic_model/results/checkpoints
```

`run_same_as_hyperbolic.sh`는 기본값으로 `../hyperbolic_model/results/checkpoints`를 쓰고, 그 안에 `datasets/train_combinations.json`이 있으면 `datasets`를 `--dataset_dir`로 넘깁니다.

---

## 확인 사항

| 항목 | 설명 |
|------|------|
| **같은 경로** | 베이스라인 `--dataset_dir` = 하이퍼볼릭 `output_dir/datasets` (또는 checkpoints면 자동으로 datasets 사용) |
| **test_combinations_original.json** | 있으면 RecBole 변환 시 test로 이 파일을 사용 (순수 원본 테스트, 공정 비교용) |
| **split_info.json** | 있으면 `prepare_recbole_data.py` 실행 시 “[데이터셋 일치] …” 로 seed·train/val/test 크기 출력 |

---

## 요약

1. 하이퍼볼릭 학습을 한 번 돌려 `results/checkpoints/datasets/` (또는 사용한 output의 `datasets/`)를 만든다.
2. 베이스라인 실행 시 `--dataset_dir`에 **그 경로**를 준다.  
   - `../hyperbolic_model/results/checkpoints` 또는  
   - `../hyperbolic_model/results/checkpoints/datasets`
3. 별도로 베이스라인용 데이터를 따로 만들지 않으면, **데이터셋이 일치한 상태**로 비교할 수 있다.
