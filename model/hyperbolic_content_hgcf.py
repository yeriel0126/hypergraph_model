"""
Content-aware HGCF baseline: 분자(GNN) + 노트 → 포앵카레 공간 랭킹.

하이퍼볼릭 CF(HGCF) 스타일: 동일 입력(분자 SMILES GNN, 노트)을 사용하고
포앵카레 볼에서 거리 기반 랭킹. 공식 HGCF는 (user_id, item_id)만 쓰므로
공정 비교를 위해 콘텐츠(분자+노트) 입력 버전.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Optional, Tuple

from geoopt import PoincareBall
from .hierarchical_hyperbolic_hypergraph import SMILESGNNEncoder
from .euclidean_content_lightgcn import EuclideanNoteEncoder


class ContentHGCF(nn.Module):
    """
    분자(GNN) + 노트 → 레시피 임베딩(포앵카레), 조향사 임베딩(포앵카레), 포앵카레 거리 랭킹.
    evaluate_model / HyperbolicBPRLoss와 호환 (manifold.dist, rank_blenders).
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
        c: float = 0.4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.c = c
        self.manifold = PoincareBall(c=c)

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
        self.mol_note_fusion = nn.Sequential(
            nn.Linear(gnn_output_dim + note_output_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.recipe_dropout = nn.Dropout(p=dropout)
        # 조향사 앵커: 유클리드 파라미터 → expmap0으로 포앵카레
        self.blender_anchors_raw = nn.Parameter(torch.randn(num_blenders, embedding_dim) * 0.02)
        nn.init.normal_(self.blender_anchors_raw, mean=0.0, std=0.02)

    def forward(
        self,
        smiles_graphs: Batch,
        smiles_batch: torch.Tensor,
        note_indices: torch.Tensor,
        blender_indices: Optional[torch.Tensor] = None,
        molecule_mask: Optional[torch.Tensor] = None,
        precomputed_mol_embs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = note_indices.size(0)
        max_molecules = note_indices.size(1)
        device = note_indices.device

        if precomputed_mol_embs is not None:
            mol_embs = precomputed_mol_embs.to(device)
            if mol_embs.size(1) != max_molecules:
                mol_embs = F.pad(mol_embs, (0, 0, 0, max(0, max_molecules - mol_embs.size(1))))
        else:
            node_batch = (
                smiles_graphs.batch
                if hasattr(smiles_graphs, "batch") and smiles_graphs.batch is not None
                else self._batch_from_smiles_batch(smiles_graphs, smiles_batch, device)
            )
            mol_embs = self.smiles_encoder(
                smiles_graphs.x,
                smiles_graphs.edge_index,
                getattr(smiles_graphs, "edge_attr", None),
                node_batch,
            )
            num_graphs = mol_embs.size(0)
            if num_graphs != batch_size * max_molecules:
                mol_embs = mol_embs[: batch_size * max_molecules]
            mol_embs = mol_embs.view(batch_size, max_molecules, -1)

        note_embs = self.note_encoder(note_indices)
        if note_embs.size(1) != max_molecules:
            note_embs = F.pad(note_embs, (0, 0, 0, max_molecules - note_embs.size(1)))

        fused = self.mol_note_fusion(torch.cat([mol_embs, note_embs], dim=-1))
        if molecule_mask is not None:
            mask = molecule_mask.unsqueeze(-1).float()
            fused = fused * mask
            cnt = mask.sum(dim=1).clamp(min=1e-8)
            z_tangent = (fused.sum(dim=1) / cnt).squeeze(1)
        else:
            z_tangent = fused.mean(dim=1)
        z_tangent = self.recipe_dropout(z_tangent)
        # 스케일 다운 후 포앵카레로 투영 (수치 안정)
        z_tangent = z_tangent * 0.5
        z_tangent = torch.clamp(z_tangent, -0.99, 0.99)
        z_recipe = self.manifold.expmap0(z_tangent)
        return z_recipe

    def _batch_from_smiles_batch(self, smiles_graphs, smiles_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
        num_nodes = smiles_graphs.x.size(0)
        node_batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        if hasattr(smiles_graphs, "ptr") and smiles_graphs.ptr is not None:
            ptr = smiles_graphs.ptr
            for i in range(len(ptr) - 1):
                node_batch[ptr[i] : ptr[i + 1]] = smiles_batch[i].item() if smiles_batch.dim() > 0 else smiles_batch[i]
        return node_batch

    def blender_anchors(self, blender_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        anchors = torch.clamp(self.blender_anchors_raw, -0.99, 0.99)
        anchors_hyp = self.manifold.expmap0(anchors)
        if blender_indices is not None:
            return anchors_hyp[blender_indices]
        return anchors_hyp

    def compute_poincare_distance(
        self,
        z_recipe: torch.Tensor,
        blender_embs: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        clamp_max: Optional[float] = 10.0,
    ) -> torch.Tensor:
        if blender_embs is None:
            blender_embs = self.blender_anchors()
        batch_size = z_recipe.size(0)
        num_blenders = blender_embs.size(0)
        z_recipe_expanded = z_recipe.unsqueeze(1).expand(-1, num_blenders, -1)
        blender_expanded = blender_embs.unsqueeze(0).expand(batch_size, -1, -1)
        distances = self.manifold.dist(z_recipe_expanded, blender_expanded) / scale
        if clamp_max is not None:
            distances = distances.clamp(min=0.0, max=clamp_max)
        return distances

    def compute_temperature_scores(self, distances: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        return torch.exp(-distances / temperature)

    def rank_blenders(
        self,
        z_recipe: torch.Tensor,
        k: int = 10,
        blender_embs: Optional[torch.Tensor] = None,
        distance_scale: float = 1.0,
        clamp_max: Optional[float] = 10.0,
        use_temperature_scaling: bool = True,
        temperature: float = 0.07,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        distances = self.compute_poincare_distance(
            z_recipe, blender_embs, scale=distance_scale, clamp_max=clamp_max
        )
        if use_temperature_scaling:
            scores = self.compute_temperature_scores(distances, temperature=temperature)
            top_k_scores, top_k_indices = torch.topk(scores, k, dim=1, largest=True)
            batch_idx = torch.arange(distances.size(0), device=distances.device).unsqueeze(1)
            top_k_distances = distances[batch_idx, top_k_indices]
        else:
            top_k_distances, top_k_indices = torch.topk(distances, k, dim=1, largest=False)
        return top_k_indices, top_k_distances
