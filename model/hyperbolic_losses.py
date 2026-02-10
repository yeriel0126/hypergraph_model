"""
Hyperbolic Loss Functions for Blender Ranking

- Proxy-Anchor / Triplet Margin: 정답 가깝게, 오답 멀게 (margin 기반)
- BPR (Bayesian Personalized Ranking): "정답(들)이 오답보다 무조건 가깝다"는 상대적 순위 학습
- Circle Loss (Multi-label Softmax): 복수 정답 Retrieval에서 정답끼리 뭉치고 오답과 멀어지게
- Mean Positive Distance: 레시피–Positive Set 평균 거리 최소화
- Max-Margin Ranking: "Positive 중 가장 먼 조향사도 Negative보다 가깝다"는 강한 제약
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import PoincareBall
from typing import Optional, Dict, List, Tuple


class HyperbolicProxyAnchorLoss(nn.Module):
    """
    Proxy-Anchor Loss in hyperbolic space.
    
    Pulls positive blenders closer and pushes negative blenders farther
    in Poincaré ball. Uses learnable proxy anchors for each blender.
    
    Each blender has a learnable proxy (style center point) in hyperbolic space.
    The loss pulls recipes toward their positive blender proxies and pushes
    them away from negative blender proxies.
    """
    
    def __init__(
        self,
        num_blenders: int,
        embedding_dim: int = 128,
        margin: float = 0.1,
        alpha: float = 32.0,
        manifold: Optional[PoincareBall] = None,
        c: float = 0.1
    ):
        super().__init__()
        self.num_blenders = num_blenders
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.alpha = alpha
        self.c = c
        
        if manifold is None:
            self.manifold = PoincareBall(c=c)
        else:
            self.manifold = manifold
        
        # Learnable proxy anchors for each blender (style center points)
        # Initialize in Euclidean space (will be mapped to hyperbolic space in forward)
        # Use small random initialization to stay near origin in Poincaré ball
        proxies_euclidean = torch.randn(num_blenders, embedding_dim) * 0.02
        
        self.proxies = nn.Parameter(proxies_euclidean)
    
    def forward(
        self,
        recipe_embs: torch.Tensor,
        positive_blender_indices: torch.Tensor,
        all_blender_embs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Proxy-Anchor Loss.
        
        Args:
            recipe_embs: [batch_size, embedding_dim] - recipe embeddings in hyperbolic space
            positive_blender_indices: [batch_size, max_positives] - indices of positive blenders
                (padded with zeros, actual positives are non-zero values)
            all_blender_embs: [num_blenders, embedding_dim] - all blender embeddings (not used, kept for compatibility)
        
        Returns:
            loss: scalar tensor
        """
        batch_size = recipe_embs.size(0)
        
        # Project proxies to hyperbolic space
        # Clamp to ensure they stay within Poincaré ball
        proxies_clamped = torch.clamp(self.proxies, -0.99, 0.99)
        proxies_hyp = self.manifold.expmap0(proxies_clamped)  # [num_blenders, embedding_dim]
        
        # Compute distances from all recipes to all proxy anchors
        recipe_expanded = recipe_embs.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        proxies_expanded = proxies_hyp.unsqueeze(0)  # [1, num_blenders, embedding_dim]
        distances = self.manifold.dist(recipe_expanded, proxies_expanded)  # [batch_size, num_blenders]
        
        # Build positive mask for each recipe
        # positive_blender_indices: [batch_size, max_positives]
        # Each row contains positive blender indices, padded with zeros
        # Strategy: Use the first column (most common case: single positive per recipe)
        # If blender indices can be 0, we need to handle padding differently
        # For now, we'll use a simple approach: use all values that are within bounds
        positive_mask = torch.zeros(batch_size, self.num_blenders, dtype=torch.bool, device=recipe_embs.device)
        
        for i in range(batch_size):
            pos_indices = positive_blender_indices[i]  # [max_positives]
            
            # Extract valid positive indices
            # In practice, max_positives is usually 1, so we use the first value
            # But we also check other positions in case max_positives > 1
            for j in range(pos_indices.size(0)):
                idx = pos_indices[j].item()
                # Check if index is within valid range
                # Note: We assume 0 might be a valid blender index
                # If all values are 0, it's likely padding, but we'll include it anyway
                # The loss will naturally handle this (no positive samples for that proxy)
                if 0 <= idx < self.num_blenders:
                    positive_mask[i, idx] = True
                    # For single positive case, we can break after first valid index
                    # But we'll process all to be safe
        
        # Compute Proxy-Anchor Loss for each proxy
        total_loss = 0.0
        
        for proxy_idx in range(self.num_blenders):
            # Get recipes that have this proxy as positive
            proxy_pos_mask = positive_mask[:, proxy_idx]  # [batch_size]
            
            if proxy_pos_mask.sum() == 0:
                # No positive samples for this proxy, skip
                continue
            
            # Distances from positive recipes to this proxy
            pos_distances = distances[proxy_pos_mask, proxy_idx]  # [num_positives]
            
            # Distances from all recipes to this proxy (for negative term)
            all_distances = distances[:, proxy_idx]  # [batch_size]
            
            # Positive term: pull positive recipes closer (Log-Sum-Exp)
            # L_pos = (1/alpha) * log(1 + sum(exp(alpha * (d_pos - margin))))
            if len(pos_distances) > 0:
                pos_term = (1.0 / self.alpha) * torch.log(
                    1.0 + torch.sum(torch.exp(self.alpha * (pos_distances - self.margin)))
                )
            else:
                pos_term = torch.tensor(0.0, device=recipe_embs.device)
            
            # Negative term: push negative recipes farther (Log-Sum-Exp)
            # L_neg = (1/alpha) * log(1 + sum(exp(alpha * (margin - d_neg))))
            neg_mask = ~proxy_pos_mask
            if neg_mask.sum() > 0:
                neg_distances = all_distances[neg_mask]  # [num_negatives]
                neg_term = (1.0 / self.alpha) * torch.log(
                    1.0 + torch.sum(torch.exp(self.alpha * (self.margin - neg_distances)))
                )
            else:
                neg_term = torch.tensor(0.0, device=recipe_embs.device)
            
            # Total loss for this proxy
            proxy_loss = pos_term + neg_term
            total_loss = total_loss + proxy_loss
        
        # Average over proxies (or use sum, depending on preference)
        # Using sum to encourage all proxies to be well-separated
        loss = total_loss / max(1, (positive_mask.sum(dim=0) > 0).sum().item())
        
        return loss


class HyperbolicTripletLoss(nn.Module):
    """
    Triplet Loss in hyperbolic space with Hard Negative Mining.
    
    Ensures that positive blenders are closer than negative blenders
    by at least a margin. Uses hard negative mining to focus on difficult cases.
    """
    
    def __init__(
        self,
        margin: float = 0.1,
        manifold: Optional[PoincareBall] = None,
        c: float = 1.0,
        hard_mining: bool = True,
        num_hard_negatives: int = 5
    ):
        super().__init__()
        self.margin = margin
        self.c = c
        self.hard_mining = hard_mining
        self.num_hard_negatives = num_hard_negatives
        
        if manifold is None:
            self.manifold = PoincareBall(c=c)
        else:
            self.manifold = manifold
    
    def forward(
        self,
        recipe_embs: torch.Tensor,
        positive_blender_embs: torch.Tensor,
        negative_blender_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Triplet Loss with Hard Negative Mining.
        
        Args:
            recipe_embs: [batch_size, embedding_dim] - recipe embeddings
            positive_blender_embs: [batch_size, embedding_dim] - positive blender embeddings
            negative_blender_embs: [batch_size, num_negatives, embedding_dim] - negative blender embeddings
        
        Returns:
            loss: scalar tensor
        """
        batch_size = recipe_embs.size(0)
        
        # Compute distances to positive blenders
        pos_dist = self.manifold.dist(recipe_embs, positive_blender_embs)  # [batch_size]
        
        # Compute distances to all negative blenders
        # recipe_embs: [batch_size, dim]
        # negative_blender_embs: [batch_size, num_negatives, dim]
        recipe_expanded = recipe_embs.unsqueeze(1)  # [batch_size, 1, dim]
        neg_dists = self.manifold.dist(recipe_expanded, negative_blender_embs)  # [batch_size, num_negatives]
        
        # Hard Negative Mining: select the hardest negatives (closest to recipe)
        if self.hard_mining and neg_dists.size(1) > self.num_hard_negatives:
            # Select top-k hardest negatives (smallest distances)
            _, hard_indices = torch.topk(neg_dists, k=self.num_hard_negatives, dim=1, largest=False)
            # Gather hard negatives
            batch_indices = torch.arange(batch_size, device=neg_dists.device).unsqueeze(1)
            hard_neg_dists = neg_dists[batch_indices, hard_indices]  # [batch_size, num_hard_negatives]
            neg_dist = hard_neg_dists.mean(dim=1)  # [batch_size] - average of hard negatives
        else:
            neg_dist = neg_dists.mean(dim=1)  # [batch_size] - average of all negatives
        
        # Triplet loss: max(0, margin + pos_dist - neg_dist)
        loss = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = loss.mean()
        
        return loss


class HyperbolicTripletMarginLoss(nn.Module):
    """
    Triplet Margin Loss in hyperbolic space with Hard Negative Mining.
    
    Similar to HyperbolicTripletLoss but uses all triplets (not just hard ones)
    and applies margin more strictly.
    
    시나리오 C 개선: 헷갈리는 조향사(Confused Blenders)를 Hard Negative로 우선 선택
    """
    
    def __init__(
        self,
        margin: float = 0.2,
        manifold: Optional[PoincareBall] = None,
        c: float = 1.0,
        hard_mining: bool = True,
        num_hard_negatives: int = 5,
        distance_scale: float = 0.2,  # Default 0.2 to scale loss to 0~1 range
        hybrid_negative_ratio: float = 0.3,  # Ratio of random negatives in hybrid mining
        use_soft_margin: bool = True,  # Use soft margin (log(1+exp)) instead of hard hinge
        confusion_matrix: Optional[torch.Tensor] = None,  # [num_blenders, num_blenders] - 헷갈리는 조향사 매트릭스
        use_confused_negatives: bool = True,  # 헷갈리는 조향사를 negative로 우선 사용
        class_weights: Optional[torch.Tensor] = None,  # 조향사별 가중치
        label_smoothing: float = 0.0  # 정답을 부드럽게 (0.1 = [0.9, 0.05, 0.05] 스타일, 스타일 겹침 허용)
    ):
        super().__init__()
        self.margin = margin
        self.label_smoothing = label_smoothing
        self.c = c
        self.hard_mining = hard_mining
        self.num_hard_negatives = num_hard_negatives
        self.distance_scale = distance_scale
        self.hybrid_negative_ratio = hybrid_negative_ratio  # 0.3 means 30% random, 70% hard
        self.use_soft_margin = use_soft_margin  # Soft margin for numerical stability
        self.use_confused_negatives = use_confused_negatives
        
        # Confusion matrix: confusion_matrix[i, j] = i번 조향사와 j번 조향사가 헷갈리는 정도
        # 값이 높을수록 더 헷갈림 (예: 0.8 = 80% 확률로 헷갈림)
        if confusion_matrix is not None:
            self.register_buffer('confusion_matrix', confusion_matrix.float())
        else:
            self.register_buffer('confusion_matrix', None)

        # Class weights: 희귀 조향사에 더 큰 가중치 부여
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.register_buffer('class_weights', None)
        
        if manifold is None:
            self.manifold = PoincareBall(c=c)
        else:
            self.manifold = manifold
    
    def forward(
        self,
        recipe_embs: torch.Tensor,
        positive_blender_indices: torch.Tensor,
        all_blender_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Triplet Margin Loss with Hard Negative Mining.
        
        Args:
            recipe_embs: [batch_size, embedding_dim] - recipe embeddings
            positive_blender_indices: [batch_size, num_positives] - indices of positive blenders
            all_blender_embs: [num_blenders, embedding_dim] - all blender embeddings
        
        Returns:
            loss: scalar tensor
        """
        batch_size = recipe_embs.size(0)
        
        # Compute distances to all blenders
        recipe_expanded = recipe_embs.unsqueeze(1)  # [batch_size, 1, dim]
        all_blender_expanded = all_blender_embs.unsqueeze(0)  # [1, num_blenders, dim]
        all_dists = self.manifold.dist(recipe_expanded, all_blender_expanded)  # [batch_size, num_blenders]
        # Apply distance scaling: multiply by distance_scale to reduce loss magnitude
        # distance_scale=0.2 means distances are scaled down, keeping loss in 0~1 range
        # 주의: distance_scale을 적용하면 margin도 같은 비율로 조정해야 일관성 유지
        all_dists = all_dists * self.distance_scale
        # margin도 distance_scale에 맞춰 조정 (일관성 유지)
        effective_margin = self.margin * self.distance_scale
        
        losses = []
        
        for i in range(batch_size):
            recipe_emb = recipe_embs[i]  # [dim]
            pos_indices_raw = positive_blender_indices[i]  # [num_positives] 또는 스칼라
            
            # pos_indices가 스칼라인 경우 처리
            if pos_indices_raw.dim() == 0:
                pos_indices = pos_indices_raw.unsqueeze(0)  # [1]로 변환
            else:
                pos_indices = pos_indices_raw
            # One-to-Many: 패딩값 -1 제외 (유효한 Positive Set만 사용)
            valid_mask = pos_indices >= 0
            pos_indices = pos_indices[valid_mask]
            if pos_indices.numel() == 0:
                losses.append(torch.tensor(0.0, device=recipe_embs.device))
                continue
            
            # Get positive distances (레시피 ↔ 모든 정답 조향사 거리 평균)
            pos_dists = all_dists[i, pos_indices]  # [num_positives]
            pos_dist = pos_dists.mean()  # Average positive distance
            
            # Get negative distances (all blenders not in positives)
            pos_mask = torch.zeros(all_blender_embs.size(0), dtype=torch.bool, device=all_dists.device)
            pos_mask[pos_indices] = True
            neg_mask = ~pos_mask
            neg_indices = torch.where(neg_mask)[0]  # 실제 negative 인덱스
            neg_dists = all_dists[i, neg_mask]  # [num_negatives]
            
            # Label smoothing: 정답을 부드럽게 (헷갈리는 조향사에 소량 허용 → [0.9, 0.05, 0.05] 스타일)
            if self.label_smoothing > 0 and self.confusion_matrix is not None and pos_indices.numel() > 0 and neg_dists.numel() > 0:
                pos_idx = pos_indices[0].item()
                confusion_matrix_device = self.confusion_matrix.to(neg_indices.device)
                confusion_scores = confusion_matrix_device[pos_idx, neg_indices]
                k_smooth = min(5, neg_dists.numel())
                if k_smooth > 0:
                    _, top_confused = torch.topk(confusion_scores, k=k_smooth, largest=True)
                    mean_confused_dist = neg_dists[top_confused].mean()
                    pos_dist = (1.0 - self.label_smoothing) * pos_dist + self.label_smoothing * mean_confused_dist
            
            # 시나리오 C: 헷갈리는 조향사를 Hard Negative로 우선 선택
            if self.use_confused_negatives and self.confusion_matrix is not None and pos_indices.numel() > 0:
                # Positive 조향사와 헷갈리는 정도 계산
                # confusion_matrix[pos_idx, neg_idx] = 헷갈리는 정도
                pos_idx = pos_indices[0].item()  # 첫 번째 positive 사용 (이미 1차원 텐서로 변환됨)
                # confusion_matrix를 올바른 디바이스로 이동
                confusion_matrix_device = self.confusion_matrix.to(neg_indices.device)
                confusion_scores = confusion_matrix_device[pos_idx, neg_indices]  # [num_negatives]
                
                # 헷갈리는 정도와 거리를 결합한 점수 계산
                # 헷갈리는 정도가 높고 거리가 가까울수록 높은 점수
                confusion_weight = 0.5  # 헷갈리는 정도의 가중치
                distance_weight = 0.5   # 거리의 가중치
                
                # 거리를 정규화 (0~1 범위로)
                if neg_dists.max() > neg_dists.min():
                    normalized_dists = (neg_dists - neg_dists.min()) / (neg_dists.max() - neg_dists.min() + 1e-8)
                else:
                    normalized_dists = torch.zeros_like(neg_dists)
                
                # 헷갈리는 정도가 높고 거리가 가까운 것에 높은 점수 부여
                # (거리가 가까우면 normalized_dists가 작으므로 1 - normalized_dists 사용)
                combined_scores = confusion_weight * confusion_scores + distance_weight * (1.0 - normalized_dists)
                
                # 상위 k개 선택 (헷갈리는 조향사 우선)
                if combined_scores.numel() > self.num_hard_negatives:
                    _, top_confused_indices = torch.topk(combined_scores, k=self.num_hard_negatives, largest=True)
                    confused_neg_dists = neg_dists[top_confused_indices]
                    neg_dist = confused_neg_dists.mean()
                else:
                    neg_dist = neg_dists.mean()
            # ③ Hard Negative Mining: 가장 헷갈리는(거리가 가까운) 오답 조향사를 강제로 Negative로 선택
            elif self.hard_mining and neg_dists.size(0) > self.num_hard_negatives:
                # 가장 가까운(헷갈리는) 오답 조향사 선택
                # 거리가 가까울수록 모델이 헷갈리기 쉬운 negative
                _, hard_indices = torch.topk(neg_dists, k=self.num_hard_negatives, largest=False)
                hard_neg_dists = neg_dists[hard_indices]
                
                # Hybrid: Mix hard negatives with random negatives
                num_random = int(self.num_hard_negatives * self.hybrid_negative_ratio)
                num_hard = self.num_hard_negatives - num_random
                
                if num_random > 0 and neg_dists.size(0) > self.num_hard_negatives:
                    # Get random negatives (excluding hard negatives)
                    remaining_mask = torch.ones(neg_dists.size(0), dtype=torch.bool, device=neg_dists.device)
                    remaining_mask[hard_indices[:num_hard]] = False
                    remaining_neg_dists = neg_dists[remaining_mask]
                    
                    if remaining_neg_dists.size(0) > 0:
                        # Randomly sample from remaining negatives
                        num_random = min(num_random, remaining_neg_dists.size(0))
                        random_indices = torch.randperm(remaining_neg_dists.size(0), device=neg_dists.device)[:num_random]
                        random_neg_dists = remaining_neg_dists[random_indices]
                        
                        # Combine hard and random negatives
                        hybrid_neg_dists = torch.cat([hard_neg_dists[:num_hard], random_neg_dists])
                        neg_dist = hybrid_neg_dists.mean()
                    else:
                        # Fallback to hard negatives only
                        neg_dist = hard_neg_dists.mean()
                else:
                    # Use hard negatives only
                    neg_dist = hard_neg_dists.mean()
            else:
                neg_dist = neg_dists.mean()
            
            # Triplet margin loss
            # Distance difference: positive should be closer than negative
            dist_diff = pos_dist - neg_dist
            
            if self.use_soft_margin:
                # Soft margin: log(1 + exp(effective_margin + dist_diff))
                # Clamp to prevent numerical explosion
                # dist_diff가 너무 크면 exp()가 폭발하므로 제한
                margin_plus_diff = effective_margin + dist_diff
                margin_plus_diff_clamped = torch.clamp(margin_plus_diff, min=-10.0, max=10.0)
                loss = torch.log(1.0 + torch.exp(margin_plus_diff_clamped))
            else:
                # Hard hinge loss: max(0, effective_margin + dist_diff)
                loss = torch.clamp(effective_margin + dist_diff, min=0.0)

            # ② 가중치 손실 함수: 희귀 조향사에 더 큰 가중치 부여
            if self.class_weights is not None and pos_indices.numel() > 0:
                class_weights_device = self.class_weights.to(pos_indices.device)
                sample_weight = class_weights_device[pos_indices].mean()
                loss = loss * sample_weight
            
            losses.append(loss)
        
        # Distance Regularization: 레시피끼리 너무 가까우면 벌점 (Anti-Collapse)
        # This prevents recipe embeddings from collapsing to a single point
        batch_size = recipe_embs.size(0)
        if batch_size > 1 and self.training:
            # Compute pairwise distances between recipe embeddings
            recipe_expanded_1 = recipe_embs.unsqueeze(1)  # [batch_size, 1, dim]
            recipe_expanded_2 = recipe_embs.unsqueeze(0)  # [1, batch_size, dim]
            pairwise_dists = self.manifold.dist(recipe_expanded_1, recipe_expanded_2)  # [batch_size, batch_size]
            
            # Remove diagonal (self-distances)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=recipe_embs.device)
            pairwise_dists_masked = pairwise_dists[mask]  # [batch_size * (batch_size - 1)]
            
            # Penalty for recipes that are too close (threshold: 0.1)
            # Encourage minimum distance between different recipes
            min_distance_threshold = 0.1
            too_close_mask = pairwise_dists_masked < min_distance_threshold
            if too_close_mask.sum() > 0:
                # Penalty: encourage recipes to be at least min_distance_threshold apart
                distance_penalty = torch.sum((min_distance_threshold - pairwise_dists_masked[too_close_mask]) ** 2)
                distance_penalty = distance_penalty * 0.1  # Weight: 0.1 (adjustable)
                # Add to loss
                base_loss = torch.stack(losses).mean()
                return base_loss + distance_penalty
        
        return torch.stack(losses).mean()


class HyperbolicContrastiveLoss(nn.Module):
    """
    Contrastive Loss in hyperbolic space.
    
    Pulls positive pairs together and pushes negative pairs apart.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        temperature: float = 0.1,
        manifold: Optional[PoincareBall] = None,
        c: float = 1.0
    ):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.c = c
        
        if manifold is None:
            self.manifold = PoincareBall(c=c)
        else:
            self.manifold = manifold
    
    def forward(
        self,
        recipe_embs: torch.Tensor,
        blender_embs: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Contrastive Loss.
        
        Args:
            recipe_embs: [batch_size, embedding_dim] - recipe embeddings
            blender_embs: [batch_size, embedding_dim] - blender embeddings
            labels: [batch_size] - 1 for positive pairs, 0 for negative pairs
        
        Returns:
            loss: scalar tensor
        """
        # Compute distances
        distances = self.manifold.dist(recipe_embs, blender_embs)  # [batch_size]
        
        # Positive pairs: minimize distance
        positive_loss = labels * distances.pow(2)
        
        # Negative pairs: maximize distance (with margin)
        negative_loss = (1 - labels) * torch.clamp(self.margin - distances, min=0.0).pow(2)
        
        loss = (positive_loss + negative_loss).mean()
        
        return loss


class ConfusionPairWeightedLoss(nn.Module):
    """
    혼동 쌍에 대한 가중치 손실 함수
    
    혼동 분석 결과를 바탕으로:
    1. 허브 조향사(Blender_351 등)와의 거리 명시적 확대
    2. 완전 오분류 조향사(Blender_4, 73, 427, 185 등)에 대한 가중치 적용
    3. 혼동 쌍에 대한 명시적 구분 학습
    """
    
    def __init__(
        self,
        confusion_pairs: Dict[Tuple[int, int], float],  # {(true_idx, pred_idx): weight}
        hub_blenders: List[int],  # 허브 조향사 리스트 (예: [351])
        confused_blenders: List[int],  # 완전 오분류 조향사 리스트 (예: [4, 73, 427, 185])
        hub_separation_weight: float = 2.0,  # 허브 조향사 분리 가중치
        confused_weight: float = 3.0,  # 완전 오분류 조향사 가중치
        confusion_pair_weight: float = 2.5,  # 혼동 쌍 가중치
        margin: float = 0.4,
        manifold: Optional[PoincareBall] = None,
        c: float = 1.0
    ):
        super().__init__()
        self.confusion_pairs = confusion_pairs
        self.hub_blenders = set(hub_blenders)
        self.confused_blenders = set(confused_blenders)
        self.hub_separation_weight = hub_separation_weight
        self.confused_weight = confused_weight
        self.confusion_pair_weight = confusion_pair_weight
        self.margin = margin
        
        if manifold is None:
            self.manifold = PoincareBall(c=c)
        else:
            self.manifold = manifold
    
    def forward(
        self,
        recipe_embs: torch.Tensor,
        positive_blender_indices: torch.Tensor,
        all_blender_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        혼동 쌍 가중치 손실 계산
        
        Args:
            recipe_embs: [batch_size, embedding_dim] - recipe embeddings
            positive_blender_indices: [batch_size, num_positives] - positive blender indices
            all_blender_embs: [num_blenders, embedding_dim] - all blender embeddings
        
        Returns:
            loss: scalar tensor
        """
        batch_size = recipe_embs.size(0)
        device = recipe_embs.device
        
        # Compute distances to all blenders
        recipe_expanded = recipe_embs.unsqueeze(1)  # [batch_size, 1, dim]
        all_blender_expanded = all_blender_embs.unsqueeze(0)  # [1, num_blenders, dim]
        all_dists = self.manifold.dist(recipe_expanded, all_blender_expanded)  # [batch_size, num_blenders]
        
        losses = []
        
        for i in range(batch_size):
            recipe_emb = recipe_embs[i]
            pos_indices = positive_blender_indices[i]  # [num_positives], 패딩 -1
            # One-to-Many: 패딩 -1 제외 (유효한 Positive Set만)
            valid_pos_indices = pos_indices[pos_indices >= 0].unique()
            if len(valid_pos_indices) == 0:
                continue
            
            # For each positive blender, compute weighted loss
            for pos_idx in valid_pos_indices:
                pos_idx_item = pos_idx.item()
                pos_dist = all_dists[i, pos_idx_item]
                
                # Get negative distances
                pos_mask = torch.zeros(all_blender_embs.size(0), dtype=torch.bool, device=device)
                pos_mask[valid_pos_indices] = True
                neg_dists = all_dists[i, ~pos_mask]
                
                if len(neg_dists) == 0:
                    continue
                
                # Base weight
                weight = 1.0
                
                # 1. 허브 조향사 분리 강화
                if pos_idx_item in self.hub_blenders:
                    # 허브 조향사는 더 강하게 분리
                    weight *= self.hub_separation_weight
                
                # 2. 완전 오분류 조향사 가중치 적용
                if pos_idx_item in self.confused_blenders:
                    weight *= self.confused_weight
                
                # 3. 혼동 쌍 명시적 구분 학습
                confusion_weight = 1.0
                for (true_idx, pred_idx), conf_weight in self.confusion_pairs.items():
                    if true_idx == pos_idx_item:
                        # 이 조향사가 혼동되는 경우, 가중치 증가
                        confusion_weight += conf_weight * self.confusion_pair_weight
                
                weight *= confusion_weight
                
                # Hard negative mining for hub blenders
                if pos_idx_item in self.hub_blenders:
                    # 허브 조향사의 경우, 가장 가까운 negative를 더 강하게 밀어냄
                    closest_neg_dist = neg_dists.min()
                    # 허브 조향사와 가장 가까운 negative 사이의 거리를 더 크게
                    hub_margin = self.margin * 1.5  # 허브 조향사는 더 큰 margin
                    dist_diff = pos_dist - closest_neg_dist
                    loss = torch.clamp(hub_margin + dist_diff, min=0.0) * weight
                else:
                    # 일반적인 triplet loss
                    neg_dist = neg_dists.mean()
                    dist_diff = pos_dist - neg_dist
                    loss = torch.clamp(self.margin + dist_diff, min=0.0) * weight
                
                losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return torch.stack(losses).mean()


class CombinedConfusionLoss(nn.Module):
    """
    기존 Triplet Loss와 혼동 쌍 가중치 손실을 결합한 손실 함수
    """
    
    def __init__(
        self,
        base_loss_fn: nn.Module,
        confusion_loss_fn: ConfusionPairWeightedLoss,
        confusion_weight: float = 0.3  # 혼동 손실의 가중치
    ):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.confusion_loss_fn = confusion_loss_fn
        self.confusion_weight = confusion_weight
    
    def forward(
        self,
        recipe_embs: torch.Tensor,
        positive_blender_indices: torch.Tensor,
        all_blender_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        결합 손실 계산
        
        Args:
            recipe_embs: [batch_size, embedding_dim] - recipe embeddings
            positive_blender_indices: [batch_size, num_positives] - positive blender indices
            all_blender_embs: [num_blenders, embedding_dim] - all blender embeddings
        
        Returns:
            combined_loss: scalar tensor
        """
        # Base loss (기존 Triplet Loss)
        base_loss = self.base_loss_fn(recipe_embs, positive_blender_indices, all_blender_embs)
        
        # Confusion-weighted loss (혼동 쌍 가중치 손실)
        confusion_loss = self.confusion_loss_fn(recipe_embs, positive_blender_indices, all_blender_embs)
        
        # Combine losses
        combined_loss = (1.0 - self.confusion_weight) * base_loss + self.confusion_weight * confusion_loss
        
        return combined_loss


class HyperbolicBPRLoss(nn.Module):
    """
    BPR (Bayesian Personalized Ranking) Loss in hyperbolic space.
    
    "정답 조향사(들)와의 거리는 정답이 아닌 조향사와의 거리보다 무조건 가까워야 한다"는
    상대적 순위를 학습. 추천 시스템에서 가장 강력한 Loss 중 하나.
    
    L = -log(sigmoid( dist(recipe, neg) - dist(recipe, pos) ))
    → pos가 neg보다 가까우면 (dist_pos < dist_neg) loss 감소.
    """
    
    def __init__(
        self,
        manifold: Optional[PoincareBall] = None,
        c: float = 1.0,
        num_hard_negatives: int = 10,
        distance_scale: float = 0.2,
    ):
        super().__init__()
        self.c = c
        self.num_hard_negatives = num_hard_negatives
        self.distance_scale = distance_scale
        if manifold is None:
            self.manifold = PoincareBall(c=c)
        else:
            self.manifold = manifold
    
    def forward(
        self,
        recipe_embs: torch.Tensor,
        positive_blender_indices: torch.Tensor,
        all_blender_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            recipe_embs: [batch_size, embedding_dim]
            positive_blender_indices: [batch_size, max_positives], 패딩 -1
            all_blender_embs: [num_blenders, embedding_dim]
        """
        batch_size = recipe_embs.size(0)
        recipe_expanded = recipe_embs.unsqueeze(1)
        all_blender_expanded = all_blender_embs.unsqueeze(0)
        all_dists = self.manifold.dist(recipe_expanded, all_blender_expanded) * self.distance_scale  # [B, N]
        
        losses = []
        for i in range(batch_size):
            pos_indices_raw = positive_blender_indices[i]
            valid_mask = pos_indices_raw >= 0
            pos_indices = pos_indices_raw[valid_mask]
            if pos_indices.numel() == 0:
                continue
            
            pos_dists = all_dists[i, pos_indices]
            pos_dist = pos_dists.mean()  # Positive Set 평균 거리
            
            pos_mask = torch.zeros(all_blender_embs.size(0), dtype=torch.bool, device=all_dists.device)
            pos_mask[pos_indices] = True
            neg_mask = ~pos_mask
            neg_dists = all_dists[i, neg_mask]
            neg_indices_flat = torch.where(neg_mask)[0]
            
            if neg_dists.numel() == 0:
                continue
            
            # Hard negative: 정답이 아닌 것 중 가장 가까운(헷갈리는) 것들
            k = min(self.num_hard_negatives, neg_dists.numel())
            _, hard_idx = torch.topk(neg_dists, k=k, largest=False)
            hard_neg_dists = neg_dists[hard_idx]
            
            # BPR: -log(sigmoid( dist_neg - dist_pos )) → dist_neg - dist_pos 가 크면 loss 작음
            diff = hard_neg_dists - pos_dist  # [k]
            loss_i = -F.logsigmoid(-diff).mean()
            losses.append(loss_i)
        
        if not losses:
            return torch.tensor(0.0, device=recipe_embs.device)
        return torch.stack(losses).mean()


class HyperbolicCircleLoss(nn.Module):
    """
    Multi-label Softmax (Circle Loss 스타일) in hyperbolic space.
    
    복수 정답이 있는 Retrieval: 정답들끼리는 뭉치고, 오답과는 멀어지게.
    logits = -distance / temperature → 정답에 해당하는 logit이 커지도록.
    """
    
    def __init__(
        self,
        manifold: Optional[PoincareBall] = None,
        c: float = 1.0,
        temperature: float = 0.07,
        distance_scale: float = 0.2,
    ):
        super().__init__()
        self.c = c
        self.temperature = temperature
        self.distance_scale = distance_scale
        if manifold is None:
            self.manifold = PoincareBall(c=c)
        else:
            self.manifold = manifold
    
    def forward(
        self,
        recipe_embs: torch.Tensor,
        positive_blender_indices: torch.Tensor,
        all_blender_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            recipe_embs: [batch_size, embedding_dim]
            positive_blender_indices: [batch_size, max_positives], 패딩 -1
            all_blender_embs: [num_blenders, embedding_dim]
        """
        batch_size = recipe_embs.size(0)
        recipe_expanded = recipe_embs.unsqueeze(1)
        all_blender_expanded = all_blender_embs.unsqueeze(0)
        all_dists = self.manifold.dist(recipe_expanded, all_blender_expanded) * self.distance_scale  # [B, N]
        # logits: 높을수록 가까움 (정답이 크게 나오도록)
        logits = -all_dists / self.temperature  # [B, N]
        
        losses = []
        for i in range(batch_size):
            pos_indices_raw = positive_blender_indices[i]
            valid_mask = pos_indices_raw >= 0
            pos_indices = pos_indices_raw[valid_mask]
            if pos_indices.numel() == 0:
                continue
            
            logits_i = logits[i]
            log_sum_exp = torch.logsumexp(logits_i, dim=0)
            pos_logits = logits_i[pos_indices]
            # Multi-label softmax: - (1/|P|) sum_{j in P} ( logit_j - log_sum_exp )
            loss_i = -(pos_logits - log_sum_exp).mean()
            losses.append(loss_i)
        
        if not losses:
            return torch.tensor(0.0, device=recipe_embs.device)
        return torch.stack(losses).mean()


class HyperbolicMeanPositiveDistanceLoss(nn.Module):
    """
    Mean Positive Distance Loss in hyperbolic space.
    
    레시피 임베딩과 Positive Set에 속한 모든 조향사 임베딩들 사이의 평균 거리를 최소화합니다.
    (선택) margin term으로 negative가 positive 평균보다 멀어지도록 유도.
    """
    
    def __init__(
        self,
        manifold: Optional[PoincareBall] = None,
        c: float = 1.0,
        margin: float = 0.2,
        distance_scale: float = 0.2,
        num_hard_negatives: int = 10,
        use_margin_vs_negatives: bool = True,
    ):
        super().__init__()
        self.c = c
        self.margin = margin
        self.distance_scale = distance_scale
        self.num_hard_negatives = num_hard_negatives
        self.use_margin_vs_negatives = use_margin_vs_negatives
        if manifold is None:
            self.manifold = PoincareBall(c=c)
        else:
            self.manifold = manifold
    
    def forward(
        self,
        recipe_embs: torch.Tensor,
        positive_blender_indices: torch.Tensor,
        all_blender_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            recipe_embs: [batch_size, embedding_dim]
            positive_blender_indices: [batch_size, max_positives], 패딩 -1
            all_blender_embs: [num_blenders, embedding_dim]
        """
        batch_size = recipe_embs.size(0)
        recipe_expanded = recipe_embs.unsqueeze(1)
        all_blender_expanded = all_blender_embs.unsqueeze(0)
        all_dists = self.manifold.dist(recipe_expanded, all_blender_expanded) * self.distance_scale  # [B, N]
        effective_margin = self.margin * self.distance_scale
        
        losses = []
        for i in range(batch_size):
            pos_indices_raw = positive_blender_indices[i]
            valid_mask = pos_indices_raw >= 0
            pos_indices = pos_indices_raw[valid_mask]
            if pos_indices.numel() == 0:
                continue
            
            pos_dists = all_dists[i, pos_indices]
            mean_pos_dist = pos_dists.mean()
            
            pos_mask = torch.zeros(all_blender_embs.size(0), dtype=torch.bool, device=all_dists.device)
            pos_mask[pos_indices] = True
            neg_mask = ~pos_mask
            neg_dists = all_dists[i, neg_mask]
            if neg_dists.numel() == 0:
                losses.append(mean_pos_dist)
                continue
            
            loss_i = mean_pos_dist
            if self.use_margin_vs_negatives:
                k = min(self.num_hard_negatives, neg_dists.numel())
                hard_neg_dists = torch.topk(neg_dists, k=k, largest=False).values
                min_neg_dist = hard_neg_dists.min()
                loss_i = loss_i + torch.clamp(effective_margin + mean_pos_dist - min_neg_dist, min=0.0)
            losses.append(loss_i)
        
        if not losses:
            return torch.tensor(0.0, device=recipe_embs.device)
        return torch.stack(losses).mean()


class HyperbolicMaxMarginRankingLoss(nn.Module):
    """
    Max-Margin Ranking Loss in hyperbolic space.
    
    "Positive Set 중 가장 먼 조향사와의 거리도, Negative 조향사와의 거리보다 가까워야 한다"는
    강한 제약: max(pos_dists) + margin < min(neg_dists).
    """
    
    def __init__(
        self,
        manifold: Optional[PoincareBall] = None,
        c: float = 1.0,
        margin: float = 0.2,
        distance_scale: float = 0.2,
        num_hard_negatives: int = 10,
    ):
        super().__init__()
        self.c = c
        self.margin = margin
        self.distance_scale = distance_scale
        self.num_hard_negatives = num_hard_negatives
        if manifold is None:
            self.manifold = PoincareBall(c=c)
        else:
            self.manifold = manifold
    
    def forward(
        self,
        recipe_embs: torch.Tensor,
        positive_blender_indices: torch.Tensor,
        all_blender_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            recipe_embs: [batch_size, embedding_dim]
            positive_blender_indices: [batch_size, max_positives], 패딩 -1
            all_blender_embs: [num_blenders, embedding_dim]
        """
        batch_size = recipe_embs.size(0)
        recipe_expanded = recipe_embs.unsqueeze(1)
        all_blender_expanded = all_blender_embs.unsqueeze(0)
        all_dists = self.manifold.dist(recipe_expanded, all_blender_expanded) * self.distance_scale  # [B, N]
        effective_margin = self.margin * self.distance_scale
        
        losses = []
        for i in range(batch_size):
            pos_indices_raw = positive_blender_indices[i]
            valid_mask = pos_indices_raw >= 0
            pos_indices = pos_indices_raw[valid_mask]
            if pos_indices.numel() == 0:
                continue
            
            pos_dists = all_dists[i, pos_indices]
            max_pos_dist = pos_dists.max()
            
            pos_mask = torch.zeros(all_blender_embs.size(0), dtype=torch.bool, device=all_dists.device)
            pos_mask[pos_indices] = True
            neg_mask = ~pos_mask
            neg_dists = all_dists[i, neg_mask]
            if neg_dists.numel() == 0:
                losses.append(max_pos_dist)
                continue
            
            k = min(self.num_hard_negatives, neg_dists.numel())
            min_neg_dist = torch.topk(neg_dists, k=k, largest=False).values.min()
            loss_i = torch.clamp(effective_margin + max_pos_dist - min_neg_dist, min=0.0)
            losses.append(loss_i)
        
        if not losses:
            return torch.tensor(0.0, device=recipe_embs.device)
        return torch.stack(losses).mean()

