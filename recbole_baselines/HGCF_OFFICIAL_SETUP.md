# 공식 HGCF 레포를 우리 실험에 맞게 조정하기

**공식 구현**: [layer6ai-labs/HGCF](https://github.com/layer6ai-labs/HGCF) (WWW'21) — 코드는 GitHub 그대로 사용.  
**데이터·실험세팅**: 제안모델·다른 베이스라인과 **동일** (RecBole part1=train, part3=test, 동일 split / seed=42, 지표 HR@1/5/10, Recall, MRR, NDCG@10).

---

## 0. 실행 방법 (데이터·실험세팅 = 제안모델과 동일)

- **`run_baselines.py`로 실행**: `python run_baselines.py --dataset_dir ... --models HGCF` → RecBole part1/part3 경로를 HGCF에 넘기므로 **동일 데이터·동일 split** 사용.
- **단독 실행**: `python run_official_hgcf.py` → RecBole 데이터를 `data/recbole/odor`, `../hyperbolic_model/.../recbole_data/odor` 등에서 **자동 탐색**. 직접 지정 시 `--recbole_inter_dir /path/to/odor`.

---

## 1. 필요한 변경 요약

| 항목 | 공식 HGCF | 우리 실험에 맞게 |
|------|-----------|------------------|
| 데이터 | `user_item_list.pkl` → 내부에서 train/test 분할 | **미리 분할된** `train.pkl`, `test.pkl` 사용 (우리 split 유지) |
| 데이터셋 이름 | Amazon-CD, Amazon-Book, yelp | **`odor`** 추가 |
| 곡률 `c` | 기본 1 | 우리 하이퍼볼릭과 맞추려면 **0.4** 권장 (config에서 `c` 변경) |
| 지표 | Recall@5,10,20,50 / NDCG@5,10,20,50 | 우리는 **HR@1,5,10, MRR, NDCG@10** 사용 → HGCF 출력에서 5,10 활용 + MRR 별도 계산 가능 |

---

## 2. 단계별 설정

### Step A. 우리 쪽에서 HGCF용 데이터 생성

```bash
cd newcode2/recbole_baselines
# 이미 build_id_mapping_and_edges.py 로 data/edges_*.json 이 있다고 가정
python prepare_hgcf_data.py
# 출력: data/odor_hgcf/train.pkl, data/odor_hgcf/test.pkl
```

### Step B. HGCF 레포 클론 및 데이터 복사

```bash
cd newcode2/recbole_baselines
git clone https://github.com/layer6ai-labs/HGCF.git hgcf_repo
mkdir -p hgcf_repo/data/odor
cp data/odor_hgcf/train.pkl hgcf_repo/data/odor/
cp data/odor_hgcf/test.pkl  hgcf_repo/data/odor/
```

### Step C. HGCF `utils/data_generator.py` 에 `odor` 데이터셋 추가

`hgcf_repo/utils/data_generator.py` 의 `Data` 클래스 `__init__` 안에서, `elif dataset.split('-')[0] in ['Amazon', 'yelp']:` 블록 **바로 앞**에 아래 블록을 추가합니다.

```python
elif dataset == 'odor':
    # 우리 실험: 미리 분할된 train/test (동일 split으로 공정 비교)
    pkl_path = os.path.join('./data/', dataset)
    self.pkl_path = pkl_path
    self.dataset = dataset
    with open(os.path.join(pkl_path, 'train.pkl'), 'rb') as f:
        self.train_dict = pkl.load(f)
    with open(os.path.join(pkl_path, 'test.pkl'), 'rb') as f:
        self.test_dict = pkl.load(f)
    self.num_users = max(self.train_dict.keys()) + 1
    all_items = set()
    for items in list(self.train_dict.values()) + list(self.test_dict.values()):
        all_items.update(items)
    self.num_items = max(all_items) + 1 if all_items else 0
    self.adj_train, _ = self.generate_adj()
    if eval(norm_adj):
        self.adj_train_norm = normalize(self.adj_train + sp.eye(self.adj_train.shape[0]))
        self.adj_train_norm = sparse_mx_to_torch_sparse_tensor(self.adj_train_norm)
    print('num_users %d, num_items %d' % (self.num_users, self.num_items))
    print('adjacency matrix shape: ', self.adj_train.shape)
    self.user_item_csr = self.generate_rating_matrix([*self.train_dict.values()], self.num_users, self.num_items)
```

(기존 `generate_adj`, `normalize`, `sparse_mx_to_torch_sparse_tensor`, `generate_rating_matrix` 는 그대로 사용됩니다.)

### Step D. config에서 `odor` 및 곡률 설정

`hgcf_repo/config.py` 의 `config_args['data_config']['dataset']` 기본값을 `'odor'`로 바꾸거나, 실행 시 인자로 넘깁니다. 곡률은 `model_config` 의 `'c'`를 **0.4**로 두면 우리 하이퍼볼릭 모델과 비슷한 공간입니다.

```python
# config.py 예시
'c': (0.4, 'hyperbolic radius (0.4 = our hyperbolic model 동일 비교용)'),
'dataset': ('odor', 'which dataset to use'),
```

실행 예:

```bash
cd hgcf_repo
python run.py --dataset odor --c 0.4
```

### Step E. 지표 매핑

- HGCF 출력: Recall@5, Recall@10, NDCG@5, NDCG@10 등.
- 우리 지표와의 대응:
  - **hit_rate@5** ← Recall@5  
  - **hit_rate@10** ← Recall@10  
  - **ndcg@k** / **ndcg@10** ← NDCG@10  
- MRR은 공식 `run.py`/`eval_metrics.py`에 없으면, `pred_list`로 랭크 구한 뒤 `1/rank` 평균으로 계산해 넣으면 됩니다.

---

## 3. 요약

- **데이터**: `prepare_hgcf_data.py`로 생성한 `train.pkl` / `test.pkl`를 HGCF의 `data/odor/`에 넣고, **Data 클래스에 `dataset == 'odor'` 분기만 추가**하면 됩니다.
- **곡률**: `c=0.4`로 맞추면 우리 실험과 동일 조건에 가깝습니다.
- **지표**: Recall@5/10, NDCG@10은 그대로 쓰고, 필요하면 MRR만 예측 리스트 기준으로 추가 계산하면 됩니다.

이렇게 조정하면 공식 HGCF를 **우리 실험(동일 데이터·동일 지표)에 맞게** 쓸 수 있습니다.
