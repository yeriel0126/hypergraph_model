# 실험 세팅 정리 (하이퍼볼릭 vs 베이스라인 공정 비교)

하이퍼볼릭 모델과 RecBole 베이스라인(LightGCN, CML)을 **구조만 다르고 나머지는 동일**하게 맞춘 내용입니다.

---

## 1. Positive Set 처리

**하이퍼볼릭:** 복수 정답(positive set)을 **전부** 사용합니다.

- `target_blenders` / `target_blender`: 레시피당 여러 조향사가 정답일 수 있음.
- Loss: `_target_blenders_to_positive_set_tensor`로 `[batch_size, max_positives]` 패딩 텐서 생성 → loss 내부에서 **모든 유효한 정답**에 대해 거리/마진 계산 (예: positive 거리 평균).
- 평가: `true_blenders = set(target_blenders[i])` 로 **모든 정답**을 모아, Top-K 예측이 그 중 하나라도 포함하면 Hit로 처리.

**베이스라인(RecBole):**

- `prepare_recbole_data.py`의 `combinations_to_inter_rows`에서  
  `target_blenders`(또는 `target_blender`)를 리스트로 받아 **각 정답마다** `(user_id, item_id)` 행을 추가합니다.  
  → 한 레시피에 정답 조향사가 여러 명이면 **전부** .inter에 들어갑니다.
- RecBole은 (user, item) 쌍 기준 학습/평가하므로, “한 레시피–여러 정답”이 **여러 행**으로 들어간 것과 동일하게, **전부 positive**로 처리됩니다.

**정리:** 하이퍼볼릭은 “복수 정답 전부 사용”, 베이스라인은 “복수 정답을 여러 (user, item) 행으로 저장해 전부 사용” → **동일한 Positive Set 처리**입니다.

---

## 2. 임베딩 차원(Embedding Dimension) 통일

**하이퍼볼릭:**

- 기본 `embed_dim = 128` (`train_hyperbolic_hypergraph.py`).
- `gnn_output_dim`, `note_hyperbolic_dim`, `blender_dim`, `channel1_output_dim`, `channel2_output_dim` 등이 이 값으로 통일됨 (기본 128).

**베이스라인:**

- `config/odor.yaml`에 **`embed_size: 128`** 설정.
- LightGCN, CML 등 RecBole 모델이 이 값을 사용해 **하이퍼볼릭과 동일한 128차원**으로 학습/평가됩니다.

**의도:**  
하이퍼볼릭의 강점이 “낮은 차원에서도 계층 구조를 잘 표현한다”는 점이므로, **같은 저차원(128)**에서 비교해야 공정합니다. 유클리드 모델이 보통 고차원(256 등)에서 유리하므로, 128로 맞춰 두었을 때 성능 차이가 나면 “하이퍼볼릭이 이 데이터의 계층을 더 잘 잡는다”는 해석이 가능합니다.

---

## 3. 네거티브 샘플링 / 평가 방식 (Full Ranking)

**하이퍼볼릭:**

- **평가 시:** `model.blender_anchors()`로 **전체 조향사** 임베딩를 가져온 뒤,  
  레시피 임베딩과 **전체 조향사**와의 거리를 계산하고, 거리 기준으로 정렬해 Top-K를 뽑습니다.  
  → **Full ranking** (후보 = 전체 조향사, 샘플링 없음).

**베이스라인(RecBole):**

- `config/odor.yaml`의 `eval_args`:
  - `mode: full` → **전체 아이템(조향사) 대상** 랭킹.
  - `group_by: user` → 유저(레시피) 단위로 그룹핑.
- 따라서 평가 시에도 **전체 조향사**를 후보로 두고 랭킹합니다 (full ranking).

**정리:**  
하이퍼볼릭은 “전체 조향사와 거리 비교 → Top-K”, 베이스라인은 “full 모드로 전체 조향사 랭킹 → Top-K”로, **동일한 평가 방식(전체 후보, 샘플링 없음)**입니다.  
학습 시 negative 샘플링 전략은 모델마다 다를 수 있으나, **최종 지표(HR@K, MRR, NDCG@10)**는 둘 다 **전체 후보 기준 full ranking**으로 계산됩니다.

---

## 요약 표

| 항목 | 하이퍼볼릭 | 베이스라인 (RecBole) |
|------|------------|----------------------|
| **Positive Set** | 복수 정답 전부 사용 (target_blenders 전체) | 동일 (복수 정답 → 여러 (user, item) 행) |
| **임베딩 차원** | 128 (embed_dim) | 128 (embed_size in odor.yaml) |
| **평가 방식** | Full ranking (전체 조향사 후보) | Full ranking (eval_args.mode: full) |
| **시드** | 42 (RANDOM_SEED) | 42 (seed in odor.yaml / run_baselines) |
| **데이터** | results/checkpoints/datasets | 동일 (--dataset_dir → 같은 train/val/test) |
| **지표** | HR@1/5/10, MRR, NDCG@10 | Recall@1/5/10, MRR, NDCG@10 (동일 의미) |

이 설정으로 **구조만 다르고 실험 세팅은 동일**하게 맞춰 두었습니다.

---

## 유클리드 CML (거리 기반, 단독 구현)

RecBole에 CML이 없어 **유클리드 거리 기반 CML**을 `run_cml_euclidean.py`로 따로 구현해 두었습니다.

- **모델:** `cml_euclidean_model.py` — Recipe(유저)·Blender(아이템) 임베딩, **유클리드(L2) 거리**, pairwise margin loss.
- **데이터:** 하이퍼볼릭과 동일 — `train/val/test_combinations.json` (또는 `test_combinations_original.json`) 사용.
- **Positive Set:** 복수 정답 전부 — `target_blenders` 각각 (user, item) 쌍으로 학습.
- **임베딩 차원:** 128 (기본, `--embed_dim`로 변경 가능).
- **시드:** 42 (기본).
- **평가:** Full ranking (전체 조향사 후보), HR@1/5/10, MRR, NDCG@10.

실행:

```bash
cd newcode2/recbole_baselines
python run_cml_euclidean.py
# 또는 데이터 경로 지정
python run_cml_euclidean.py --dataset_dir ../hyperbolic_model/results/checkpoints/datasets
```

결과는 `results/recbole_CML_Euclidean.json`에 저장되며, 하이퍼볼릭·LightGCN·BPR과 같은 키로 비교할 수 있습니다.

---

## 하이퍼볼릭이 베이스라인보다 성능이 좋아야 하는 이유

**가설:** 제안 모델(하이퍼볼릭 + 계층 + 하이퍼그래프 + 분자 구조)은 **레시피(분자 조합) 내용**을 직접 보고, **계층적 조향사 공간**을 사용하므로, “레시피–조향사”만 보는 CF 베이스라인보다 성능이 좋을 수 있다.

| 구분 | 베이스라인 (LightGCN, BPR, HGCF, CML) | 하이퍼볼릭 제안 모델 |
|------|--------------------------------------|----------------------|
| **입력** | (user=레시피 ID, item=조향사 ID) 상호작용만 | 레시피 **내용**(분자·노트·SMILES) + 상호작용 |
| **구조** | 유클리드(또는 Poincaré) 임베딩, GCN 없음/단순 | 하이퍼볼릭 + GNN(분자 그래프) + 계층(그룹) |
| **정보량** | 동일 분할·동일 positive set이면 **동일** | 레시피 구조·분자 정보 **추가** |

**공정 비교 조건 (이미 맞춤):**

1. **동일 데이터 분할**  
   하이퍼볼릭이 저장한 `train/val/test_combinations.json` → `prepare_recbole_data.py`로 .inter 생성 → 베이스라인이 **같은 train/val/test** 사용.
2. **동일 지표**  
   HR@1/5/10, Recall@1/5/10, MRR, NDCG@10 — 정의·키 모두 통일됨.
3. **동일 시드(42)**  
   하이퍼볼릭·베이스라인 모두 42 사용.

**실험 순서 권장:**

1. 하이퍼볼릭 학습 1회 → `results/checkpoints/datasets/`에 `train/combinations.json`, `val_combinations.json`, `test_combinations.json` 생성.
2. `python prepare_recbole_data.py --dataset_dir ../hyperbolic_model/results/checkpoints/datasets`  
   → 같은 분할로 RecBole .inter 생성.
3. `python run_baselines.py --dataset_dir ../hyperbolic_model/results/checkpoints/datasets --models LightGCN BPR HGCF CML`  
   → 베이스라인 결과 저장.
4. 하이퍼볼릭 **검증/테스트** 지표(HR@10, Recall@10, MRR, NDCG@10)와 베이스라인 **테스트** 지표를 같은 키로 비교.

**성능이 기대만큼 안 나올 때 점검:**

- 하이퍼볼릭: `--group_loss_weight`, 학습률, early stopping, 에폭 수, curvature(c).
- 베이스라인: 동일 `--dataset_dir`·동일 `.inter` 사용 여부, `split_info.json`으로 분할 일치 확인.

---

## 베이스라인이 하이퍼볼릭보다 잘 나오는 이유 (요약)

| 원인 | 설명 |
|------|------|
| **1. Val vs Test** | 학습 로그의 HR@10은 **검증(Val)** 기준. 베이스라인 수치는 **테스트(Test)** 기준인 경우가 많음. Val은 보통 Test보다 낮게 나옴. → **Test vs Test**로 맞춰 비교해야 함 (학습 끝 하이퍼볼릭 Test HR@10 vs 베이스라인 Test HR@10). |
| **2. 태스크 난이도 (분자 단위 분할)** | 하이퍼볼릭은 **분자 단위** 분할(75/12.5/12.5)을 씀. 테스트 레시피 = **학습에 한 번도 안 나온 분자**로만 구성 → “새 레시피(새 분자 조합)에 대한 블렌더 예측”이라 **일반화 난이도가 높음**. 베이스라인은 같은 JSON을 쓰면 **같은 레시피 집합**으로 train/val/test를 쓰지만, 모델 구조가 다름(아래 3번). |
| **3. 입력 정보량** | 베이스라인(LightGCN 등): (user_id, item_id) 상호작용만 사용. 하이퍼볼릭: 레시피 **내용**(분자·노트·SMILES)을 입력으로 사용. 분자 단위 분할에서는 테스트 레시피가 “완전히 새로운 조합”이라 내용 기반 모델이 **처음 보는 입력**에 대해 맞추기 어려울 수 있음. 반면 CF는 같은 split이면 “테스트 유저 = 학습에 안 나온 user_id”라 베이스라인도 cold-user 문제를 겪지만, **아이템(조향사) 공간**만 잘 배우면 상대적으로 유리할 수 있음. |
| **4. Val Loss 상승** | 로그에서 Val Loss가 6.3 → 6.9로 **증가**하면 과적합 또는 학습률/스케줄 이슈. HR@10이 8% 전후에서 정체되면 학습이 목표를 잘 못 찾고 있는 상태. → `--learning_rate` 5e-5, `--scheduler_metric val_loss`, 에폭·early stopping 점검. |

**공정 비교 체크리스트**

1. **동일 데이터**: `run_baselines.py --dataset_dir .../hyperbolic_model/results/checkpoints/datasets` 로 하이퍼볼릭이 쓴 train/val/test와 동일하게 사용.
2. **동일 지표·동일 세트**: 하이퍼볼릭 **Test** HR@10/MRR/NDCG vs 베이스라인 **Test** HR@10/MRR/NDCG (학습 끝에 출력되는 하이퍼볼릭 Test 지표 사용).
3. **학습 안정화**: Val loss 발산 시 `--learning_rate 5e-5`, `--scheduler_metric val_loss` 시도; `best_model.pt`로 Test 평가했는지 확인.

---

## 하이퍼볼릭 8% vs 베이스라인 47% — 원인·체크리스트 (상세)

**가능한 원인**

1. **Val vs Test 비교**  
   학습 중 출력되는 HR@10은 **검증(Val)** 세트 기준입니다. 베이스라인 47%는 보통 **테스트(Test)** 세트 기준입니다.  
   - **조치**: 학습 종료 후 하이퍼볼릭이 **Test 세트**에서 Best 모델로 HR@10을 한 번 더 계산해 출력합니다. 이 **Test HR@10**과 베이스라인 Test HR@10을 비교하세요.

2. **데이터/분할 불일치**  
   - 하이퍼볼릭을 `--load_dataset` 없이 돌리면 **분자 단위** 분할(75/12.5/12.5)로 train/val/test가 만들어집니다. 테스트 레시피는 **학습에 안 나온 분자**만 사용해 난이도가 높습니다.  
   - 베이스라인은 `prepare_recbole_data.py`로 만든 **같은** train/val/test JSON을 쓰면 **같은 분할**입니다.  
   - **조치**:  
     - 1) 하이퍼볼릭 1회 학습 → `results/checkpoints/datasets/`에 JSON 생성  
     - 2) `prepare_recbole_data.py --dataset_dir .../datasets`  
     - 3) `run_baselines.py --dataset_dir .../datasets`  
     이렇게 하면 **동일 split**으로 비교됩니다.

3. **모델/학습 설정**  
   Val/Test 모두 낮으면 학습이 부족했을 수 있습니다.  
   - 에폭 수 늘리기, `--num_epochs`  
   - 학습률·스케줄: `--learning_rate`, `--lr_scheduler`  
   - loss: BPR / margin 등  
   - `best_model.pt`로 평가했는지 확인 (학습 끝에 출력되는 Test HR@10이 best 모델 기준입니다).
