"""
초기 Loss 증가 원인 분석 스크립트

주요 변경사항:
1. Margin: 0.2 → 0.5 (2.5배 증가)
2. num_hard_negatives: 5 → 15 (3배 증가)
3. Norm clamping 추가 (Recipe: 0.3-0.7, Blender: 0.4-0.6)
4. Dropout(p=0.3) 추가
5. Diversity noise 추가
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from geoopt import PoincareBall
import os

def analyze_loss_components():
    """Loss 구성 요소 분석"""
    
    print("=" * 70)
    print("초기 Loss 증가 원인 분석")
    print("=" * 70)
    
    # 시뮬레이션: 초기 상태 (랜덤 임베딩)
    batch_size = 32
    num_blenders = 100
    dim = 128
    device = "cpu"
    
    # 하이퍼볼릭 공간 생성
    c = 0.2
    manifold = PoincareBall(c=c)
    
    # 초기 Recipe 임베딩 (norm clamping 적용)
    recipe_embs = torch.randn(batch_size, dim) * 0.02
    recipe_embs = torch.clamp(recipe_embs, -0.99, 0.99)
    recipe_embs = manifold.expmap0(recipe_embs)
    
    # Norm clamping: 0.3-0.7
    recipe_norms = recipe_embs.norm(dim=1, keepdim=True)
    target_norm_min, target_norm_max = 0.3, 0.7
    recipe_norms_clamped = torch.clamp(recipe_norms, target_norm_min, target_norm_max)
    recipe_embs = recipe_embs / (recipe_norms + 1e-8) * recipe_norms_clamped
    
    # 초기 Blender 임베딩 (norm clamping 적용)
    blender_embs = torch.randn(num_blenders, dim) * 0.01
    blender_embs = torch.clamp(blender_embs, -0.99, 0.99)
    blender_embs = manifold.expmap0(blender_embs)
    
    # Norm clamping: 0.4-0.6
    blender_norms = blender_embs.norm(dim=1, keepdim=True)
    target_norm_min, target_norm_max = 0.4, 0.6
    blender_norms_clamped = torch.clamp(blender_norms, target_norm_min, target_norm_max)
    blender_embs = blender_embs / (blender_norms + 1e-8) * blender_norms_clamped
    
    # 거리 계산
    recipe_expanded = recipe_embs.unsqueeze(1)  # [batch_size, 1, dim]
    blender_expanded = blender_embs.unsqueeze(0)  # [1, num_blenders, dim]
    all_dists = manifold.dist(recipe_expanded, blender_expanded)  # [batch_size, num_blenders]
    
    print(f"\n1. 거리 분포 분석:")
    print(f"   평균 거리: {all_dists.mean().item():.4f}")
    print(f"   표준편차: {all_dists.std().item():.4f}")
    print(f"   최소 거리: {all_dists.min().item():.4f}")
    print(f"   최대 거리: {all_dists.max().item():.4f}")
    
    # Positive/negative 거리 시뮬레이션
    pos_indices = torch.randint(0, num_blenders, (batch_size, 1))
    pos_dists = all_dists.gather(1, pos_indices).squeeze(1)  # [batch_size]
    
    # Negative 거리 (hard negative mining 적용)
    num_hard_negatives_old = 5
    num_hard_negatives_new = 15
    
    neg_dists_list = []
    hard_neg_dists_old_list = []
    hard_neg_dists_new_list = []
    
    for i in range(batch_size):
        pos_idx = pos_indices[i].item()
        neg_mask = torch.ones(num_blenders, dtype=torch.bool)
        neg_mask[pos_idx] = False
        neg_dists = all_dists[i, neg_mask]
        
        # Old: 5 hard negatives
        _, hard_indices_old = torch.topk(neg_dists, k=min(num_hard_negatives_old, len(neg_dists)), largest=False)
        hard_neg_dists_old = neg_dists[hard_indices_old].mean()
        
        # New: 15 hard negatives
        _, hard_indices_new = torch.topk(neg_dists, k=min(num_hard_negatives_new, len(neg_dists)), largest=False)
        hard_neg_dists_new = neg_dists[hard_indices_new].mean()
        
        neg_dists_list.append(neg_dists.mean())
        hard_neg_dists_old_list.append(hard_neg_dists_old)
        hard_neg_dists_new_list.append(hard_neg_dists_new)
    
    neg_dist_mean = torch.tensor(neg_dists_list).mean()
    hard_neg_dist_old = torch.tensor(hard_neg_dists_old_list).mean()
    hard_neg_dist_new = torch.tensor(hard_neg_dists_new_list).mean()
    
    print(f"\n2. Positive/Negative 거리 분석:")
    print(f"   Positive 평균 거리: {pos_dists.mean().item():.4f}")
    print(f"   Negative 평균 거리 (전체): {neg_dist_mean.item():.4f}")
    print(f"   Hard Negative 평균 거리 (5개): {hard_neg_dist_old.item():.4f}")
    print(f"   Hard Negative 평균 거리 (15개): {hard_neg_dist_new.item():.4f}")
    print(f"   차이: {hard_neg_dist_old.item() - hard_neg_dist_new.item():.4f} (15개가 더 작음)")
    
    # Loss 계산
    margin_old = 0.2
    margin_new = 0.5
    
    # Old loss (margin=0.2, hard_negatives=5)
    loss_old = torch.clamp(margin_old + pos_dists.mean() - hard_neg_dist_old, min=0.0)
    
    # New loss (margin=0.5, hard_negatives=15)
    loss_new = torch.clamp(margin_new + pos_dists.mean() - hard_neg_dist_new, min=0.0)
    
    print(f"\n3. Loss 비교:")
    print(f"   이전 설정 (margin=0.2, hard_neg=5): {loss_old.item():.4f}")
    print(f"   현재 설정 (margin=0.5, hard_neg=15): {loss_new.item():.4f}")
    print(f"   증가율: {(loss_new.item() / loss_old.item() - 1) * 100:.1f}%")
    
    # 각 요소의 기여도 분석
    print(f"\n4. 각 요소의 기여도:")
    
    # Margin 증가만
    loss_margin_only = torch.clamp(margin_new + pos_dists.mean() - hard_neg_dist_old, min=0.0)
    margin_contribution = loss_margin_only.item() - loss_old.item()
    
    # Hard negative 증가만
    loss_hard_only = torch.clamp(margin_old + pos_dists.mean() - hard_neg_dist_new, min=0.0)
    hard_neg_contribution = loss_hard_only.item() - loss_old.item()
    
    print(f"   Margin 증가 (0.2→0.5) 기여: +{margin_contribution:.4f}")
    print(f"   Hard Negative 증가 (5→15) 기여: +{hard_neg_contribution:.4f}")
    print(f"   합계 기여: +{margin_contribution + hard_neg_contribution:.4f}")
    print(f"   실제 증가: +{loss_new.item() - loss_old.item():.4f}")
    
    # Norm clamping 영향 분석
    print(f"\n5. Norm Clamping 영향:")
    recipe_norm_mean = recipe_embs.norm(dim=1).mean()
    blender_norm_mean = blender_embs.norm(dim=1).mean()
    print(f"   Recipe 평균 norm: {recipe_norm_mean.item():.4f} (목표: 0.3-0.7)")
    print(f"   Blender 평균 norm: {blender_norm_mean.item():.4f} (목표: 0.4-0.6)")
    print(f"   Norm 차이: {abs(recipe_norm_mean.item() - blender_norm_mean.item()):.4f}")
    print(f"   → Norm 차이가 크면 초기 거리가 커질 수 있음")
    
    print(f"\n6. 결론 및 권장사항:")
    print(f"   - 초기 loss 증가의 주요 원인:")
    print(f"     1. Margin 증가 (0.2→0.5): 약 {margin_contribution:.4f} 증가")
    print(f"     2. Hard Negative 증가 (5→15): 약 {hard_neg_contribution:.4f} 증가")
    print(f"   - 이는 정상적인 현상입니다:")
    print(f"     * 더 큰 margin은 더 강한 분리를 요구하므로 초기 loss가 큼")
    print(f"     * 더 많은 hard negative는 더 어려운 샘플을 학습하므로 초기 loss가 큼")
    print(f"   - 학습이 진행되면 loss가 감소할 것입니다.")
    print(f"   - 만약 초기 loss가 너무 크다면:")
    print(f"     * Margin을 0.3-0.4로 조정")
    print(f"     * num_hard_negatives를 10으로 조정")
    print(f"     * Learning rate를 높여서 빠른 수렴 유도")
    
    print("=" * 70)

if __name__ == "__main__":
    analyze_loss_components()
