# 하이퍼볼릭 공간 + 하이퍼그래프 구조 검증

## 의도한 구조 요약

1. **하이퍼볼릭 공간**: Poincaré Ball (geoopt), 곡률 c (학습 가능)
2. **입력**: **분자(SMILES GNN) + 노트(FastText→하이퍼볼릭)** 만 사용. **블렌더는 입력에 넣지 않음** (정답 유출 방지).
3. **하이퍼그래프**:
   - **하단**: 분자 인코더, 노트 인코더 (블렌더 앵커는 **랭킹/손실**에서만 사용)
   - **Channel 1 (원자 하이퍼엣지)**: [1분자 + N노트] → z_i (하이퍼볼릭). 블렌더 미사용.
   - **Channel 2 (레시피 하이퍼엣지)**: 가변 개수 z_i → **Mobius 평균** → z_recipe (하이퍼볼릭)
   - **랭킹**: z_recipe와 **블렌더 앵커** 간 Poincaré 거리 → Top-K (블렌더는 여기서만 사용)

---

## 검증 결과

### ✅ 하이퍼볼릭 공간

| 항목 | 구현 위치 | 상태 |
|------|-----------|------|
| Poincaré Ball | `geoopt.PoincareBall(c=c)` | ✅ 전 모듈 공통 |
| 곡률 c (학습 가능) | `HierarchicalFragranceHypergraph.c_raw` (softplus) | ✅ |
| 지수 사상 expmap0 | 노트, Channel1, Channel2, BlenderAnchor, Loss | ✅ |
| 로그 사상 logmap0 | Channel2 Mobius mean (접공간 가중평균) | ✅ |
| Poincaré 거리 | `manifold.dist()` (랭킹·손실) | ✅ |

**수정 반영**: learnable curvature 사용 시 **forward() 시작 시** 현재 c로 manifold를 만들고, `note_encoder`, `channel1`, `channel2`, `blender_anchors`에 **동일 manifold 참조**를 넣어서 한 forward/loss 계산 안에서 곡률이 일치하도록 했음.

---

### ✅ 하이퍼그래프 계층

| 계층 | 역할 | 구현 | 상태 |
|------|------|------|------|
| **Bottom** | SMILES → GNN | `SMILESGNNEncoder` (GCN/GAT, global_mean_pool) | ✅ 유클리드 출력 |
| **Bottom** | 노트 → 하이퍼볼릭 | `HyperbolicNoteEmbedding` (Embedding → MLP → expmap0) | ✅ |
| **Bottom** | 블렌더 앵커 | `BlenderAnchorEmbedding` (랭킹·손실용만, Channel 1 입력 아님) | ✅ |
| **Channel 1** | 원자 하이퍼엣지 | **분자+노트만** (블렌더 입력 없음, 정답 유출 방지) → projection → attention → fusion → expmap0 → z_i | ✅ z_i ∈ Poincaré |
| **Channel 2** | 레시피 하이퍼엣지 | z_1,…,z_n → **Mobius mean** (logmap0 → 가중평균 → expmap0) → z_recipe | ✅ 하이퍼그래프 집계 |
| **Ranking** | 거리 기반 랭킹 | `manifold.dist(z_recipe, blender_anchors)` → Top-K | ✅ |

- **Channel 2 = 하이퍼엣지 집계**: “가변 개수 분자(노드)” → “하나의 레시피(상위 하이퍼엣지)”로 모으는 구조가 **Mobius mean**으로 구현되어 있음. ✅

---

### ⚠️ 참고 사항 (설계 선택)

1. **Channel 1의 attention**:  
   **분자(하이퍼볼릭) + 노트(Linear 투영)** 두 개만 쌓아서 유클리드 attention → fusion → expmap0으로 z_i를 Poincaré에 넣음.  
   블렌더는 입력에 넣지 않음. 중간은 유클리드 연산, **최종 z_i만** Poincaré.

2. **Channel 2 입력**:  
   Channel 1 출력 z_i는 이미 expmap0으로 Poincaré에 있음. Channel 2의 `transform`은 **유클리드 Linear**인데, forward 내부에서 다시 `expmap0(molecule_embs)`를 적용해 집계 전에 Poincaré로 맞춤.  
   따라서 **Mobius mean** 단계는 하이퍼볼릭 위에서의 집계로 일관됨. ✅

3. **손실**:  
   `HyperbolicTripletMarginLoss` 등에서 **Poincaré 거리** 기반으로 학습. ✅

---

## 수정한 버그

- **곡률 동기화**:  
  learnable curvature일 때, 서브모듈이 예전 manifold를 쓰던 문제.  
  → **forward() 맨 앞**에서 현재 c로 manifold를 만들고, `note_encoder`, `channel1`, `channel2`, `blender_anchors`에 **동일 manifold** 참조를 넣어 한 forward 내 곡률 일치.

- **블렌더 입력 제거**:  
  Channel 1에 블렌더를 넣으면 정답(타깃 블렌더) 정보가 유출됨.  
  → Channel 1은 **분자+노트만** 사용. 블렌더 앵커는 **랭킹(거리 계산)·손실(positive/negative)** 에서만 사용.

---

## 요약

- **하이퍼볼릭 공간**: Poincaré Ball + expmap0/logmap0/dist 일관 사용. 곡률 c 학습 가능, forward 시 서브모듈과 동기화.
- **입력**: **분자 + 노트만**. 블렌더는 **입력에 넣지 않음** (정답 유출 방지).
- **하이퍼그래프**:  
  - Channel 1 = [1분자 + N노트] → 원자 하이퍼엣지 z_i (하이퍼볼릭).  
  - Channel 2 = 여러 z_i → **Mobius mean** → 레시피 z_recipe (하이퍼볼릭).  
  - 랭킹 = z_recipe와 **블렌더 앵커** 간 Poincaré 거리 → Top-K (블렌더는 여기서만 사용).

의도한 “하이퍼볼릭 공간 위의 하이퍼그래프 + 블렌더 미입력” 구조가 코드와 일치함.
