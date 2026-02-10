# 베이스라인 구성 체크리스트

## 0. 데이터셋 맞추기 (필수 — 하이퍼볼릭과 동일 데이터)

- 베이스라인은 **하이퍼볼릭과 동일한 파일**을 씁니다:
  - **Train** → `train_combinations.json`
  - **Val** → `val_combinations.json`
  - **Test(원본)** → `test_combinations_original.json` (있으면 이걸로, 없으면 `test_combinations.json`)
- `prepare_recbole_data.py`가 위 파일을 읽어 part1/part2/part3 .inter로 변환하므로, 예: Train=43,111 / Val=10,633 / Test(원본)=11,248 인 split이 그대로 베이스라인에도 적용됩니다.
- **기본 경로**: 모든 모델이 `hyperbolic_model/results/checkpoints/datasets` 폴더의 train/val/test(원본) JSON을 사용합니다. 인자 없이 실행하면 이 경로가 기본값입니다.
- **구비된 데이터가 이미 다른 폴더에 있는 경우**: `python run_baselines.py --dataset_dir <train_combinations.json 이 있는 폴더 경로>` 로 지정하세요.
- **주의**: `--recbole_data_path` / `--use_unified_id` 는 하이퍼볼릭과 다른 데이터일 수 있으므로 공정 비교 시 **`--dataset_dir`** 로 동일 데이터 경로를 쓰세요.

### 멀티 인풋 (같은 데이터셋 = 같은 입력 형태)

- **기본 실행되는 모델(Content\*)** 은 하이퍼볼릭과 **동일한 멀티 인풋**을 씁니다:
  - **레시피** = (분자 SMILES → GNN 또는 지문) + **노트(notes)** + 가변 길이 분자 수.
  - `HyperbolicRecipeDataset` + `train/val/test_combinations.json` → 각 샘플이 `molecules`(smiles, notes) + `target_blenders` 로 구성.
- 즉, **같은 데이터셋**을 쓰면 **멀티 인풋(분자+노트)** 도 동일하게 적용됩니다.

### ⚠️ ID 기반 베이스라인은 비교실험이 아님

- **LightGCN, BPR, HGCF, CML** (Content 없이)은 **분자·노트 입력을 전혀 쓰지 않습니다.**
- 입력: **user_id**(레시피 인덱스) + **item_id**(블렌더 ID) 상호작용만 사용 → 레시피 **내용**(무슨 분자, 무슨 노트)은 사용하지 않음.
- 따라서 제안 모델(분자+노트 → 하이퍼엣지)과 **조건이 다릅니다. 비교실험으로 사용하면 안 됩니다.**
- **공정 비교**는 **Content\*** (ContentLightGCN, ContentBPR, ContentCML, ContentHGCF)만 해당합니다. 기본 실행도 Content\*만 돌리도록 되어 있습니다.
- ID 기반은 참고용으로만 돌리려면 `--models LightGCN` 등으로 따로 지정할 수 있으며, 실행 시 경고가 출력됩니다.

---

## 1. 공정 실험 조건 (필수)

**베이스라인과 제안(하이퍼볼릭) 모델은 “모델만 다르고, 나머지 조건은 동일”해야 합니다.**

| 조건 | 설명 | 적용 여부 |
|------|------|-----------|
| **데이터셋** | 동일한 train/val/test 분할 (레시피→블렌더 조합) | ✓ Content\*: 동일 JSON·동일 분자+노트 입력 |
| **입력 조건** | 레시피 = 분자+노트 (동일 인풋) | ✓ Content\*만 해당. ID 기반(LightGCN/BPR/HGCF/CML)은 분자 입력 없음 → **비교실험 아님** |
| **곡률(c)** | HGCF 곡률 = 제안 하이퍼볼릭 모델과 동일 (기본 c=0.4) | ✓ `config/hgcf_curvature.yaml` / train_hyperbolic_hypergraph.py 기본값 |
| **평가 지표** | HR(Hit Rate) + Recall @1/5/10, MRR, NDCG@10 동일 키로 출력 | ✓ |
| **시드** | 동일 시드(기본 2024) 사용 | ✓ `--seed` |
| **파인튜닝/학습 설정** | 가능한 범위에서 동일(에폭 등) | ✓ 설정 파일·기본값 통일 권장 |

`--dataset_dir` 로 실행할 때 **HGCF도** 해당 경로에서 생성한 RecBole part1/part3를 사용하므로, 제안 모델과 **같은 데이터**로 학습·평가됩니다.

---

## 2. 모델: 한 번에 돌리는가?

**예. 한 번에 순서대로 실행됩니다.**

- `python run_baselines.py --dataset_dir ...` 한 번 실행하면
- **기본(공정 비교)** 으로 **ContentLightGCN → ContentBPR → ContentCML → ContentHGCF** 4개가 순서대로 학습·평가됩니다.
- 각 모델이 끝날 때마다 `results/recbole_ContentLightGCN.json` 등이 저장되고,
- 끝나면 `results/recbole_baseline_summary.json` 에 결과가 합쳐져 저장됩니다.
- ID 기반(LightGCN, BPR, HGCF, CML)은 **비교실험이 아니므로** 기본 목록에 없으며, `--models LightGCN` 등으로 따로 지정 시 경고가 출력됩니다.

모델을 따로 돌리고 싶으면:

```bash
python run_baselines.py --dataset_dir ... --models LightGCN
python run_baselines.py --dataset_dir ... --models HGCF
python run_baselines.py --dataset_dir ... --models CML
```

---

## 3. 구성 검증 요약

| 항목 | 설명 | 확인 |
|------|------|------|
| **데이터 소스** | `--dataset_dir`(조합 JSON) / `--recbole_data_path`(.inter) / `--use_unified_id` 중 하나 필수 | ✓ |
| **데이터 변환** | `dataset_dir` 사용 시 `prepare_recbole_data.py`로 part1/part2/part3 .inter 자동 생성 | ✓ |
| **LightGCN** | RecBole `LightGCN` 사용, 동일 .inter·동일 지표 | ✓ |
| **HGCF** | **공식** [layer6ai-labs/HGCF](https://github.com/layer6ai-labs/HGCF) 사용; `--dataset_dir` 시 LightGCN/CML과 **동일** part1/part3 .inter 사용 (동일 데이터·동일 지표) | ✓ |
| **CML** | RecBole `CML` 사용 (있으면), 동일 .inter·동일 지표 | ✓ |
| **지표 키** | `hit_rate@1/5/k`, `recall@1/5/k`, `mrr`, `ndcg@k` (하이퍼볼릭 val_metrics와 동일) | ✓ |
| **분할** | benchmark_filename part1=train, part2=valid, part3=test | ✓ |

---

## 4. 실행 전 확인

1. **RecBole 설치**: `pip install -r requirements_recbole.txt`
2. **데이터 존재**:
   - `--dataset_dir` 쓸 때: 해당 경로에 `train_combinations.json`, `val_combinations.json`, `test_combinations.json` (또는 `test_combinations_original.json`) 존재
   - `--use_unified_id` 쓸 때: `python build_id_mapping_and_edges.py` 를 먼저 실행해 둔 상태
3. **결과 경로**: `results/` (기본) 또는 `--results_dir` 에 JSON 쓰기 가능

---

## 5. 실행 후 확인

- `results/recbole_LightGCN.json`, `recbole_HGCF.json`, `recbole_CML.json` 생성 여부
- `results/recbole_baseline_summary.json` 에 3개 모델 엔트리 존재
- 터미널 끝에 "베이스라인 지표 요약" 블록으로 HR@1/5/10, Recall@1/5/10, MRR, NDCG@10 출력

위가 모두 만족하면 베이스라인 구성이 의도대로 된 것으로 보면 됩니다.
