"""
Content-aware LightGCN baseline: 분자(GNN) + 노트 입력 → 유클리드 공간 랭킹.

하이퍼볼릭 모델과 동일한 입력(분자 SMILES GNN, 노트)을 사용하고,
공간만 유클리드로 두어 공정 비교 (같은 조건, 다른 기하).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch
from typing import Optional, Tuple

from .hierarchical_hyperbolic_hypergraph import SMILESGNNEncoder


class SimpleEuclideanManifold:
    """평가/손실 호환용: manifold.dist(u, v) = L2 거리."""
    def dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # u: [B, D], v: [N, D] -> [B, N] or u: [B, D], v: [B, D] -> [B]
        if u.dim() == 2 and v.dim() == 2:
            if u.size(0) == v.size(0) and u.size(0) > 1 and v.size(0) != u.size(0):
                pass
            return torch.cdist(u, v, p=2)
        return (u - v).norm(dim=-1, p=2)


class EuclideanNoteEncoder(nn.Module):
    """노트 임베딩 (유클리드, 하이퍼볼릭 투영 없음)."""
    def __init__(self, vocab_size: int, embedding_dim: int = 300, output_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
        )
        nn.init.xavier_uniform_(self.embed.weight, gain=0.1)
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, note_indices: torch.Tensor) -> torch.Tensor:
        # note_indices: [B, max_mol, max_notes] -> [B, max_mol, output_dim]
        B, M, N = note_indices.shape
        emb = self.embed(note_indices).mean(dim=2)
        out = self.proj(emb.reshape(-1, self.embedding_dim))
        return out.view(B, M, -1)


class ContentLightGCN(nn.Module):
    """
    분자(GNN) + 노트 → 레시피 임베딩(유클리드), 조향사 임베딩(유클리드), L2 거리 랭킹.
    하이퍼볼릭 모델과 동일한 배치 형식·평가 인터페이스(blender_anchors, rank_blenders, manifold.dist).
    """
    def __init__(
        self,
        node_dim: int = 9,
        edge_dim: int = 3,
        gnn_hidden_dim: int = 128,
        gnn_output_dim: int = 128,
        gnn_num_layers: int = 3,
        gnn_architecture: str = "GCN",
        vocab_size: int = 435,
        note_embedding_dim: int = 300,
        note_output_dim: int = 128,
        num_blenders: int = 100,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.manifold = SimpleEuclideanManifold()

        self.smiles_encoder = SMILESGNNEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_layers=gnn_num_layers,
            architecture=gnn_architecture,
            dropout=dropout,
        )
        self.note_encoder = EuclideanNoteEncoder(
            vocab_size=vocab_size,
            embedding_dim=note_embedding_dim,
            output_dim=note_output_dim,
        )
        # Fusion: Concat + Linear + LayerNorm (단순 + 대신 변별력 유지)
        # mol_proj, note_proj 각각 투영 → concat [2*D] → fusion_proj [D] → (마스크 평균) → recipe_ln
        self.mol_proj = nn.Linear(gnn_output_dim, embedding_dim)
        self.note_proj = nn.Linear(note_output_dim, embedding_dim)
        self.fusion_proj = nn.Linear(embedding_dim * 2, embedding_dim)
        for m in (self.mol_proj, self.note_proj, self.fusion_proj):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            nn.init.zeros_(m.bias)
        self.recipe_ln = nn.LayerNorm(embedding_dim)
        self.recipe_dropout = nn.Dropout(p=dropout)
        self.blender_emb = nn.Embedding(num_blenders, embedding_dim)
        nn.init.normal_(self.blender_emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        smiles_graphs: Batch,
        smiles_batch: torch.Tensor,
        note_indices: torch.Tensor,
        blender_indices: Optional[torch.Tensor] = None,
        molecule_mask: Optional[torch.Tensor] = None,
        precomputed_mol_embs: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ):
        batch_size = note_indices.size(0)
        max_molecules = note_indices.size(1)
        device = note_indices.device

        if precomputed_mol_embs is not None:
            mol_embs = precomputed_mol_embs.to(device)
            if mol_embs.size(1) != max_molecules:
                mol_embs = F.pad(mol_embs, (0, 0, 0, max(0, max_molecules - mol_embs.size(1))))
        else:
            node_batch = smiles_graphs.batch if hasattr(smiles_graphs, 'batch') and smiles_graphs.batch is not None else self._batch_from_smiles_batch(smiles_graphs, smiles_batch, device)
            mol_embs = self.smiles_encoder(
                smiles_graphs.x,
                smiles_graphs.edge_index,
                getattr(smiles_graphs, 'edge_attr', None),
                node_batch,
            )
            num_graphs = mol_embs.size(0)
            if num_graphs != batch_size * max_molecules:
                mol_embs = mol_embs[: batch_size * max_molecules]
            mol_embs = mol_embs.view(batch_size, max_molecules, -1)

        # 노트: [B, max_mol, max_notes] -> [B, max_mol, note_dim]
        note_embs = self.note_encoder(note_indices)
        if note_embs.size(1) != max_molecules:
            note_embs = F.pad(note_embs, (0, 0, 0, max_molecules - note_embs.size(1)))

        # 스케일 균형: 퓨전 전 L2 정규화
        mol_embs = F.normalize(mol_embs, p=2, dim=-1)
        note_embs = F.normalize(note_embs, p=2, dim=-1)

        # Concat + Linear + LayerNorm (단순 합산 아님)
        mol_h = self.mol_proj(mol_embs)
        note_h = self.note_proj(note_embs)
        fused = self.fusion_proj(torch.cat([mol_h, note_h], dim=-1))
        if molecule_mask is not None:
            mask = molecule_mask.unsqueeze(-1).float()
            fused = fused * mask
            cnt = mask.sum(dim=1).clamp(min=1e-8)
            z_recipe = (fused.sum(dim=1) / cnt).squeeze(1)
        else:
            z_recipe = fused.mean(dim=1)
        z_before_ln = z_recipe
        z_recipe = self.recipe_ln(z_recipe)
        z_recipe = self.recipe_dropout(z_recipe)
        if return_debug:
            return (z_recipe, {"mol_embs": mol_embs, "note_embs": note_embs, "fused": fused, "z_before_ln": z_before_ln})
        return z_recipe

    def _batch_from_smiles_batch(self, smiles_graphs, smiles_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
        num_nodes = smiles_graphs.x.size(0)
        node_batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        if hasattr(smiles_graphs, 'ptr') and smiles_graphs.ptr is not None:
            ptr = smiles_graphs.ptr
            for i in range(len(ptr) - 1):
                node_batch[ptr[i] : ptr[i + 1]] = smiles_batch[i].item() if smiles_batch.dim() > 0 else smiles_batch[i]
        return node_batch

    def blender_anchors(self, blender_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if blender_indices is None:
            return self.blender_emb.weight
        return self.blender_emb(blender_indices)

    def rank_blenders(
        self,
        z_recipe: torch.Tensor,
        k: int = 10,
        blender_embs: Optional[torch.Tensor] = None,
        distance_scale: float = 1.0,
        clamp_max: Optional[float] = None,
        use_temperature_scaling: bool = True,
        temperature: float = 0.07,
        normalize_blender_embs: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if blender_embs is None:
            blender_embs = self.blender_emb.weight
        # 노름 큰 소수 블렌더가 항상 상위에 오는 편향 완화: 랭킹 시 방향(각도)만 사용
        if normalize_blender_embs:
            blender_embs = F.normalize(blender_embs, p=2, dim=-1)
        distances = torch.cdist(z_recipe, blender_embs, p=2)
        if distance_scale != 1.0:
            distances = distances * distance_scale
        if clamp_max is not None:
            distances = distances.clamp(max=clamp_max)
        if use_temperature_scaling:
            scores = torch.exp(-distances / temperature)
            top_k_scores, top_k_indices = torch.topk(scores, k, dim=1, largest=True)
            batch_idx = torch.arange(distances.size(0), device=distances.device).unsqueeze(1)
            top_k_distances = distances[batch_idx, top_k_indices]
        else:
            top_k_distances, top_k_indices = torch.topk(distances, k, dim=1, largest=False)
        return top_k_indices, top_k_distances
