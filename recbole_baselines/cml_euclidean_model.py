"""
유클리드 거리 기반 CML (Collaborative Metric Learning).
하이퍼볼릭 모델과 동일 실험 세팅: embed_dim=128, full ranking, HR@1/5/10, MRR, NDCG@10.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CML_Euclidean(nn.Module):
    """Recipe(유저) / Blender(아이템) 임베딩, 유클리드 거리 기반. Positive Set 전부 사용."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 128,
        margin: float = 0.5,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.margin = margin
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """유클리드 거리 (L2). user_ids [B], item_ids [B] -> [B]."""
        u = self.user_embed(user_ids)   # [B, D]
        i = self.item_embed(item_ids)   # [B, D]
        return torch.norm(u - i, dim=1)

    def get_user_embeddings(self) -> torch.Tensor:
        return self.user_embed.weight   # [num_users, D]

    def get_item_embeddings(self) -> torch.Tensor:
        return self.item_embed.weight  # [num_items, D]

    def distances(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """[B, D], [N, D] -> [B, N] L2 거리."""
        return torch.cdist(user_emb, item_emb, p=2)


def cml_margin_loss(
    user_ids: torch.Tensor,      # [B]
    pos_item_ids: torch.Tensor,  # [B] (또는 [B, max_pos] 패딩)
    neg_item_ids: torch.Tensor,  # [B]
    model: "CML_Euclidean",
    margin: float | None = None,
) -> torch.Tensor:
    """Pairwise margin loss: d(u,p) - d(u,n) + margin. Positive Set은 배치 내 대표 1개 사용 (동일 세팅 시 복수 정답은 여러 (u,p) 쌍으로 들어옴)."""
    m = margin if margin is not None else model.margin
    d_pos = model(user_ids, pos_item_ids)  # [B]
    d_neg = model(user_ids, neg_item_ids)  # [B]
    return F.relu(d_pos - d_neg + m).mean()
