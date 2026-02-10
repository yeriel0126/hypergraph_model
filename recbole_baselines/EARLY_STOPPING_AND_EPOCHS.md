# 모든 모델 Early Stopping / 에폭 설정 요약

| 모델 | 최대 에폭 | Early Stopping | 기준 지표 | 비고 |
|------|-----------|----------------|-----------|------|
| **제안 모델** (train_hyperbolic_hypergraph.py) | 30 | ✅ patience=5 | Val HR@10 | `--num_epochs`, `--early_stopping_patience` |
| **ContentLightGCN** (train_euclidean_content_baseline.py) | 30 | ✅ patience=10 | Val HR@10 | `--epochs`, `--early_stopping_patience` |
| **ContentBPR** | 30 | ✅ patience=10 | Val HR@10 | 동일 스크립트 (--model_name ContentBPR) |
| **ContentCML** (train_euclidean_content_cml.py) | 80 | ✅ patience=10 | Val HR@10 | `--epochs`, `--early_stopping_patience` |
| **ContentHGCF** (train_hyperbolic_content_hgcf.py) | 50 | ✅ patience=10 | Val HR@10 | `--epochs`, `--early_stopping_patience` |
| **RecBole LightGCN/BPR** (config/odor.yaml) | 200 | ✅ stopping_step=15 | NDCG@10 | RecBole 기본 조기 종료 |
| **CML 유클리드** (run_cml_euclidean.py) | 200 | ✅ stopping_step=15 | Val HR@10 | `--epochs`, `--stopping_step` |
| **공식 HGCF** (run_official_hgcf.py) | 200 | ❌ 없음 | — | `--epochs` 만큼 전부 학습 |

---

## 상세 (스크립트/설정 위치)

### 제안 모델
- **파일**: `hyperbolic_model/train_hyperbolic_hypergraph.py`
- `--num_epochs` default=**30**
- `--early_stopping_patience` default=**5** (Val HR@10 기준)
- ReduceLROnPlateau: `--plateau_patience` default=4

### Content* (분자 GNN 입력)
- **ContentLightGCN / ContentBPR**: `train_euclidean_content_baseline.py`  
  - `--epochs` **30**, `--early_stopping_patience` **10**
- **ContentCML**: `train_euclidean_content_cml.py`  
  - `--epochs` **80**, `--early_stopping_patience` **10**
- **ContentHGCF**: `train_hyperbolic_content_hgcf.py`  
  - `--epochs` **50**, `--early_stopping_patience` **10**

### RecBole ID 기반 (LightGCN, BPR)
- **파일**: `recbole_baselines/config/odor.yaml`
- `epochs`: **200**
- `stopping_step`: **15** (valid_metric=NDCG@10 기준)

### CML 유클리드 (run_cml_euclidean.py)
- `--epochs` **200**, `--stopping_step` **15** (Val HR@10 연속 15 에폭 개선 없으면 종료)

### 공식 HGCF (run_official_hgcf.py)
- `--epochs` **200**, **Early stopping 없음** (200 에폭 전부 수행)
