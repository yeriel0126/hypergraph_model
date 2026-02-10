# 프리컴퓨팅 vs GNN end-to-end 학습

## 성능이 에폭마다 똑같이 나올 때

**원인 후보**: 프리컴퓨팅 사용 시 분자 쪽이 **고정 랜덤 투영**이라 레시피 구분력이 부족해 `z_recipe`가 붕괴할 수 있음.

- 프리컴퓨팅 = 랜덤 초기화 GNN으로 한 번만 임베딩 계산 → 학습 시 GNN **미업데이트**
- 노트 + 퓨전만 학습되므로, 분자 정보가 “의미 없는” 고정 벡터면 점수가 거의 안 바뀜

## 비교 방법: `--no_precomputed`

프리컴퓨팅을 끄고 **GNN을 end-to-end로 학습**해 보면 원인 확인에 도움이 됩니다.

```bash
# run_baselines (Content* 4개 모두)
python run_baselines.py --no_precomputed

# ContentLightGCN만 직접 실행
python train_euclidean_content_baseline.py --load_dataset results/checkpoints/datasets --no_precomputed
```

- `--no_precomputed` 사용 시: GNN이 학습되므로 분자 임베딩이 데이터에 맞게 바뀌고, Val HR@10이 에폭에 따라 변할 가능성이 큼.
- 이렇게 하면 “성능 고정”이 **프리컴퓨팅(고정 분자 벡터)** 때문인지, 다른 요인(데이터/손실/LR 등) 때문인지 구분할 수 있음.

## 정리

| 모드 | 분자 임베딩 | GNN 학습 | 용도 |
|------|-------------|----------|------|
| 기본 (프리컴퓨팅 O) | .pt에서 로드, 고정 | 안 함 | 속도 우선, 랜덤 투영 |
| `--no_precomputed` | 매 배치 GNN forward | 함 | 성능/구분력 우선, 원인 비교 |

학습된 GNN으로 프리컴퓨팅을 다시 만들고 싶다면, 먼저 `--no_precomputed`로 학습한 뒤  
`precompute_molecule_embeddings.py --encoder_checkpoint path/to/best.pt` 로 .pt를 다시 생성하면 됩니다.
