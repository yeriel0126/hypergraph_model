# RecBole 베이스라인 (LightGCN, HGCF, CML)

하이퍼볼릭 모델과 **동일한 ID 매핑·분할·평가 지표**로 베이스라인을 돌려 공정 비교합니다.

**데이터셋 일치:** 하이퍼볼릭과 같은 train/val/test를 쓰려면 **DATASET_ALIGNMENT.md** 참고.

**공정 실험 원칙:** 베이스라인과 제안 모델은 **모델만 다르고**, 데이터셋·분할·**입력(분자+노트)**·평가 지표·시드를 동일하게 맞춥니다. **공정 비교는 Content\* (ContentLightGCN, ContentBPR, ContentCML, ContentHGCF)만 해당**합니다. ID 기반(LightGCN, BPR, HGCF, CML)은 분자·노트를 쓰지 않고 user_id/item_id만 사용하므로 **비교실험이 아닙니다.** 기본 실행은 Content\* 4개만 돌립니다.

## 한 번에 3개 모델 실행

**기본 동작: 한 번 실행하면 LightGCN → HGCF → CML 순서로 모두 실행됩니다.**

```bash
python run_baselines.py --dataset_dir ../hyperbolic_model/results/checkpoints
```

- 별도 옵션 없으면 위 3개 모델이 **같은 명령 한 번**에 순서대로 학습·평가됩니다.
- 개별 실행이 필요하면: `--models LightGCN` / `--models HGCF` / `--models CML` 로 나눠 실행하면 됩니다.
- 상세 검증 항목은 **BASELINE_CHECKLIST.md** 참고.

---

## Step 1: 데이터 ID 매핑 (공통)

모든 베이스라인이 **같은 숫자 ID**를 쓰도록 한 번만 생성합니다.

- **Perfumer ID**: 0, 1, 2, …, 5672 (블렌더/조향사 쪽)
- **Ingredient ID**: 0, 1, 2, …, N (성분/분자)

```bash
python build_id_mapping_and_edges.py
# 기본 경로: --data_path ../cleaned_data/cleaned_complete_data.json, --vocab_path ../feature_encoding/vocabularies.json
```

생성 파일:
- `data/id_mapping.json` — 매핑 테이블 (perfumer_id_to_name, ingredient_id_to_key)
- `data/edges_train.json`, `edges_valid.json`, `edges_test.json` — (perfumer_id, ingredient_id) 엣지 리스트
- `data/adjacency_info.json` — 행 크기 등
- `data/recbole/odor/odor.part1.inter`, `part2`, `part3` — RecBole benchmark 형식

---

## Step 2: 상호작용 행렬 / 인접 행렬 / CML 샘플링

- **LightGCN / HGCF**: 위 엣지 리스트로 **인접 행렬 A** 구성. `load_adjacency_and_pairs.py`에서 `build_adjacency_from_edges()` 사용.
- **CML**: Positive pair(실제 사용 성분) + Negative pair(미사용 성분) 샘플링은 `load_adjacency_and_pairs.py`의 `get_positive_negative_pairs()` 사용. RecBole CML은 동일 `.inter`만 주면 내부에서 negative 샘플링.

```bash
python load_adjacency_and_pairs.py   # data/ 있는지 확인 후 인접 행렬·쌍 샘플 확인
```

---

## Step 3: HGCF 곡률(c) 설정

HGCF는 하이퍼볼릭 곡률 **c**가 중요합니다.

- **공정 비교**: 우리 쪽 하이퍼볼릭 모델과 동일한 **c = 0.4** 사용 (`train_hyperbolic_hypergraph.py` 기본값).
- 설정 참고: `config/hgcf_curvature.yaml` (curvature: 0.4).  
  HGCF를 별도 레포로 돌릴 때 위 값으로 맞추거나, 논문 기본값 실험 후 c=0.4 실험을 추가로 돌리면 됩니다.

---

## 지표·데이터셋을 하이퍼볼릭과 동일하게

- **동일 데이터셋**: 하이퍼볼릭이 쓰는 `train_combinations.json`, `val_combinations.json`, `test_combinations_original.json`을 그대로 사용 (레시피→블렌더 추천).
- **동일 지표 키**: 결과 JSON에 하이퍼볼릭 `val_metrics`와 같은 키 사용 → `hit_rate@1`, `hit_rate@5`, `hit_rate@k`, `mrr`, `ndcg@k` (숫자 직접 비교 가능).

```bash
# 하이퍼볼릭 체크포인트 경로(또는 그 아래 datasets) 지정 → 동일 데이터로 베이스라인만 실행
python run_baselines.py --dataset_dir ../hyperbolic_model/results/checkpoints
# 또는 한 번에
./run_same_as_hyperbolic.sh
./run_same_as_hyperbolic.sh /path/to/hyperbolic/output_dir
```

## 기존 요약

- **데이터 (통일 ID 모드)**: Perfumer–Ingredient 엣지, 동일 시드로 train/valid/test 분할
- **지표**: HR@1, HR@5, HR@10, MRR, NDCG@10 (full ranking). 출력 키는 위와 동일.
- **분할**: `benchmark_filename`: part1=train, part2=valid, part3=test

## 설치

```bash
pip install -r requirements_recbole.txt
```

## 사용법

### 0) 통일 ID 기반 실행 (권장 — Step 1 실행 후)

```bash
python build_id_mapping_and_edges.py
python run_baselines.py --use_unified_id
```

### 1) 하이퍼볼릭 학습에서 쓴 레시피→블렌더 데이터셋으로 실행

하이퍼볼릭 학습을 한 번이라도 돌렸다면 `output_dir/datasets` 안에 `train_combinations.json`, `val_combinations.json`, `test_combinations.json`이 있습니다. 이 경로를 주면 자동으로 RecBole용 .inter로 변환한 뒤 학습합니다.

```bash
cd newcode2/recbole_baselines

# 예: 체크포인트/데이터셋 경로
python run_baselines.py --dataset_dir ../hyperbolic_model/results/checkpoints/datasets
```

데이터 변환 결과는 `{dataset_dir}/recbole_data/odor/` 아래에 `odor.part1.inter`, `odor.part2.inter`, `odor.part3.inter`로 저장됩니다.

### 2) 이미 RecBole .inter가 있을 때

```bash
python run_baselines.py --recbole_data_path /path/to/recbole_data --dataset_name odor
```

이때 RecBole이 읽는 경로는 `{recbole_data_path}/{dataset_name}/` 이며, 그 안에 `{dataset_name}.part1.inter`, `part2`, `part3`가 있어야 합니다.

### 3) 수동으로 데이터만 변환

```bash
python prepare_recbole_data.py --dataset_dir /path/to/datasets
# 출력 기본: /path/to/datasets/recbole_data/odor/
```

### 4) 모델: LightGCN, HGCF, CML (기본 모두 실행)

기본으로 **LightGCN**, **HGCF**, **CML** 세 모델을 모두 돌리며, 동일 지표(HR@1, HR@5, HR@10, Recall@1/5/10, MRR, NDCG@10)가 나옵니다. HGCF는 공식 [HGCF](https://github.com/layer6ai-labs/HGCF) (`run_official_hgcf.py`)를 사용합니다.

```bash
python run_baselines.py --dataset_dir ...
# 또는 특정 모델만
python run_baselines.py --dataset_dir ... --models LightGCN CML
```

## 결과

- `results/recbole_{모델명}.json`: 모델별 테스트 결과 (HR@1/5/10, MRR, NDCG@10 포함)
- `results/recbole_baseline_summary.json`: 모든 모델 요약

이 지표를 하이퍼볼릭 모델의 검증/테스트 HR@10, NDCG@10, MRR과 비교하면 됩니다.

## 참고

- RecBole 문서: https://recbole.io/
- 데이터 형식: RecBole [Atomic Files](https://recbole.io/docs/user_guide/data/atomic_files.html) – `user_id:token` = Perfumer ID, `item_id:token` = Ingredient ID (통일 ID 사용 시)
- 분할: [benchmark_filename](https://recbole.io/docs/user_guide/data/data_settings.html)으로 미리 나눈 part1/part2/part3 사용
- HGCF 곡률: `config/hgcf_curvature.yaml` 참고 (c=0.4 권장)
- **HGCF**: **공식 레포** [layer6ai-labs/HGCF](https://github.com/layer6ai-labs/HGCF) 사용. `run_baselines.py`에서 HGCF 선택 시 자동으로 클론(최초 1회)·패치·실행 후 동일 지표로 저장. **`--dataset_dir` 또는 `--recbole_data_path` 사용 시** LightGCN/CML과 **동일한** RecBole part1/part3 데이터로 학습·평가(공정 비교). 그렇지 않을 때만 `data/odor_hgcf`(build_id_mapping 기반) 사용. 상세: **HGCF_OFFICIAL_SETUP.md**
