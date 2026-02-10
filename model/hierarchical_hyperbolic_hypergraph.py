"""
Hierarchical Hyperbolic Hypergraph for Fragrance Synergy & Blender Ranking

This module implements a geometric deep learning model that processes variable-length
molecule combinations in hyperbolic space to recommend optimal blenders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import List, Dict, Optional, Tuple
import numpy as np
from geoopt import PoincareBall
from geoopt.manifolds.stereographic import math as geo_math

# torch_geometric은 내부적으로 torch_scatter 사용 가능. 없으면 오류 시:
#   pip install torch-scatter (PyTorch/CUDA 버전에 맞게)
#   https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    print("Warning: fasttext not available. Using random initialization for note embeddings.")


class HyperbolicNoteEmbedding(nn.Module):
    """
    FastText-based note embedding with hyperbolic projection.
    
    Uses fasttext-wiki-news-subwords-300 for initial word embeddings,
    then projects to hyperbolic space.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hyperbolic_dim: int = 128,
        fasttext_model_path: Optional[str] = None,
        manifold: Optional[PoincareBall] = None,
        c: float = 1.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hyperbolic_dim = hyperbolic_dim
        self.c = c
        
        # Initialize PoincareBall manifold
        if manifold is None:
            self.manifold = PoincareBall(c=c)
        else:
            self.manifold = manifold
        
        # FastText embedding layer
        self.fasttext_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Load FastText model if available
        self.fasttext_model = None
        if FASTTEXT_AVAILABLE and fasttext_model_path:
            try:
                self.fasttext_model = fasttext.load_model(fasttext_model_path)
                # Initialize embedding weights from FastText
                self._load_fasttext_weights()
            except Exception as e:
                print(f"Warning: Could not load FastText model: {e}")
        
        # Projection to hyperbolic space
        self.euclidean_to_hyperbolic = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, hyperbolic_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _load_fasttext_weights(self):
        """Load pre-trained FastText weights into embedding layer."""
        if self.fasttext_model is None:
            return
        
        # This is a placeholder - actual implementation would map
        # vocabulary indices to FastText word vectors
        # For now, we use random initialization
        pass
    
    def _init_weights(self):
        """Initialize weights with small values for numerical stability."""
        nn.init.xavier_uniform_(self.fasttext_embedding.weight, gain=0.1)
        for layer in self.euclidean_to_hyperbolic:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
    
    def forward(self, note_indices: torch.Tensor) -> torch.Tensor:
        """
        Embed notes and project to hyperbolic space.
        
        Args:
            note_indices: [batch_size, num_notes] - indices of notes
            
        Returns:
            hyperbolic_embeddings: [batch_size, num_notes, hyperbolic_dim] in PoincareBall
        """
        # Get FastText embeddings
        # note_indices: [batch_size, num_notes]
        batch_size, num_notes = note_indices.shape
        
        # Flatten for embedding lookup
        flat_indices = note_indices.reshape(-1)  # [batch_size * num_notes]
        euclidean_emb = self.fasttext_embedding(flat_indices)  # [batch_size * num_notes, embedding_dim]
        
        # Project to hyperbolic space
        hyperbolic_emb = self.euclidean_to_hyperbolic(euclidean_emb)  # [batch_size * num_notes, hyperbolic_dim]
        
        # Project to PoincareBall (exponential map)
        hyperbolic_emb = hyperbolic_emb.reshape(batch_size, num_notes, self.hyperbolic_dim)
        
        # Clip to ensure numerical stability
        hyperbolic_emb = hyperbolic_emb * 0.5  # Scale down to prevent extreme values
        hyperbolic_emb = torch.clamp(hyperbolic_emb, -0.9, 0.9)  # Less aggressive clamping
        
        # Project to PoincareBall using exponential map
        hyperbolic_emb = self.manifold.expmap0(hyperbolic_emb)
        
        # Ensure output has minimum norm to prevent collapse
        hyperbolic_emb_norm = hyperbolic_emb.norm(dim=-1, keepdim=True)
        min_norm = 0.01
        hyperbolic_emb = hyperbolic_emb / torch.clamp(hyperbolic_emb_norm, min=min_norm) * torch.clamp(hyperbolic_emb_norm, min=min_norm)
        
        return hyperbolic_emb


class SMILESGNNEncoder(nn.Module):
    """
    GNN encoder for SMILES molecules.
    Supports both GCN and GAT architectures.
    """
    
    def __init__(
        self,
        node_dim: int = 9,
        edge_dim: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        architecture: str = "GCN",
        dropout: float = 0.1
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.architecture = architecture
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            if architecture == "GCN":
                conv = GCNConv(hidden_dim, hidden_dim)
            elif architecture == "GAT":
                conv = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
            else:
                raise ValueError(f"Unknown architecture: {architecture}")
            self.convs.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """완화된 gain(0.5): GNN이 너무 크면 하이퍼볼릭 투영에서 수치 불안정."""
        gain_gnn = 0.5
        nn.init.xavier_uniform_(self.input_proj.weight, gain=gain_gnn)
        for layer in self.output_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain_gnn)
                if layer.bias is not None:
                    nn.init.normal_(layer.bias, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode molecular graph.
        
        Args:
            x: [num_nodes, node_dim] - node features
            edge_index: [2, num_edges] - edge indices
            edge_attr: [num_edges, edge_dim] - edge features (optional)
            batch: [num_nodes] - batch assignment (optional)
            
        Returns:
            molecule_embedding: [batch_size, output_dim] or [1, output_dim] if batch is None
        """
        # Input projection
        x = self.input_proj(x)
        
        # GNN layers (LayerNorm으로 스케일 폭발 억제, global_mean_pool로 합계 대신 평균)
        for i, conv in enumerate(self.convs):
            if self.architecture == "GCN":
                x_new = conv(x, edge_index)
            else:  # GAT
                x_new = conv(x, edge_index)
            x = x + x_new  # Residual connection
            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling: mean aggregation (sum 대신 평균으로 수치 안정)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class FingerprintEncoder(nn.Module):
    """
    ECFP/Morgan fingerprint → 임베딩. GNN 대신 분자 지문만 쓸 때 사용.
    입력: [batch_size, max_molecules, fp_dim], 출력: [batch_size, max_molecules, output_dim]
    """

    def __init__(
        self,
        fp_dim: int = 2048,
        output_dim: int = 128,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fp_dim = fp_dim
        self.output_dim = output_dim
        hidden_dim = hidden_dim or min(512, fp_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mol_fingerprints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mol_fingerprints: [batch_size, max_molecules, fp_dim]
        Returns:
            [batch_size, max_molecules, output_dim]
        """
        return self.mlp(mol_fingerprints)


class Channel1AtomicHyperedge(nn.Module):
    """
    Channel 1: Atomic Hyperedge
    Combines [1 molecule + N notes] into a single hyperedge (z_i).
    블렌더는 입력에 넣지 않음 (정답 유출 방지). 블렌더 앵커는 랭킹/손실에서만 사용.
    """
    
    def __init__(
        self,
        molecule_dim: int = 128,
        note_dim: int = 128,
        output_dim: int = 128,
        manifold: Optional[PoincareBall] = None,
        c: float = 1.0
    ):
        super().__init__()
        self.molecule_dim = molecule_dim
        self.note_dim = note_dim
        self.output_dim = output_dim
        self.c = c
        
        if manifold is None:
            self.manifold = PoincareBall(c=c)
        else:
            self.manifold = manifold
        
        self.molecule_proj = nn.Linear(molecule_dim, output_dim)
        self.note_proj = nn.Linear(note_dim, output_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # 완화된 gain(0.25): 하이퍼볼릭 경로가 많아서 gain=1.0이면 그래디언트 요동·무작위 예측 수준 유지
        gain_ch1 = 0.25
        for layer in [self.molecule_proj, self.note_proj]:
            nn.init.xavier_uniform_(layer.weight, gain=gain_ch1)
            nn.init.normal_(layer.bias, mean=0.0, std=0.01)
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain_ch1)
                if layer.bias is not None:
                    nn.init.normal_(layer.bias, mean=0.0, std=0.01)
    
    def forward(
        self,
        molecule_emb: torch.Tensor,
        note_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Create atomic hyperedge from molecule + notes only (no blender input).
        """
        batch_size = molecule_emb.size(0)
        
        mol_proj = self.molecule_proj(molecule_emb)
        note_proj = self.note_proj(note_embs.mean(dim=1))
        
        mol_proj = torch.clamp(mol_proj, -0.99, 0.99)
        mol_proj_hyp = self.manifold.expmap0(mol_proj)
        
        components = torch.stack([mol_proj_hyp, note_proj], dim=1)
        attended, _ = self.attention(components, components, components)
        concat = attended.reshape(batch_size, -1)
        z_i = self.fusion(concat)
        
        # Project to hyperbolic space
        z_i = z_i * 0.5  # Scale down to prevent extreme values
        z_i = torch.clamp(z_i, -0.9, 0.9)  # Less aggressive clamping
        z_i = self.manifold.expmap0(z_i)
        
        # Ensure output has minimum norm to prevent collapse
        z_i_norm = z_i.norm(dim=1, keepdim=True)
        min_norm = 0.01
        z_i = z_i / torch.clamp(z_i_norm, min=min_norm) * torch.clamp(z_i_norm, min=min_norm)
        
        return z_i


class Channel2Synergy(nn.Module):
    """
    Channel 2: Global Recipe Synergy
    Combines variable number of molecule nodes (z_1, z_2, ..., z_n)
    into a single upper-level hyperedge (Recipe) using Mobius Mean.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
        manifold: Optional[PoincareBall] = None,
        c: float = 1.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.c = c
        
        if manifold is None:
            self.manifold = PoincareBall(c=c)
        else:
            self.manifold = manifold
        
        # Optional: learnable transformation before aggregation
        self.transform = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, output_dim)
        )
        
        # Dropout and LayerNorm for recipe embedding regularization
        # Applied in Euclidean space before final projection to hyperbolic space
        # 0.1: 과한 dropout은 정보 유실로 학습 저하 → 완화
        self.recipe_dropout = nn.Dropout(p=0.1)
        # LayerNorm은 제거 (collapse 방지)
        # self.recipe_layernorm = nn.LayerNorm(output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """완화된 gain(0.25): Channel2 transform이 강하면 Mobius/정규화 전 그래디언트 요동."""
        gain_ch2 = 0.25
        for layer in self.transform:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain_ch2)
                if layer.bias is not None:
                    nn.init.normal_(layer.bias, mean=0.0, std=0.01)
    
    def mobius_mean(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Mobius mean in PoincareBall.
        
        Args:
            x: [batch_size, num_molecules, dim] - molecule embeddings in hyperbolic space
            weights: [batch_size, num_molecules] - optional weights for each molecule
            
        Returns:
            mean: [batch_size, dim] - Mobius mean in hyperbolic space
        """
        batch_size, num_molecules, dim = x.shape
        
        if weights is None:
            weights = torch.ones(batch_size, num_molecules, device=x.device) / num_molecules
        else:
            # Normalize weights
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Möbius mean (하이퍼볼릭 공간의 올바른 평균): 유클리드 식 sum(x)/n 은 곡률 때문에
        # "가운데"가 아니므로, tangent space에서 평균 후 expmap0 사용.
        # Project to tangent space at origin
        x_tangent = self.manifold.logmap0(x)  # [batch_size, num_molecules, dim]
        
        # Weighted average in tangent space
        weights_expanded = weights.unsqueeze(-1)  # [batch_size, num_molecules, 1]
        mean_tangent = (x_tangent * weights_expanded).sum(dim=1)  # [batch_size, dim]
        
        # Mobius mean 노이즈 제거 (과한 노이즈는 boundary로 밀어냄)
        
        # Ensure mean_tangent is not too small (prevent collapse)
        mean_tangent_norm = mean_tangent.norm(dim=1, keepdim=True)
        min_norm = 0.01  # Minimum norm to prevent collapse
        mean_tangent = mean_tangent / torch.clamp(mean_tangent_norm, min=min_norm) * torch.clamp(mean_tangent_norm, min=min_norm)
        
        # Project back to hyperbolic space
        mean_hyp = self.manifold.expmap0(mean_tangent)
        
        return mean_hyp
    
    def forward(
        self,
        molecule_embs: torch.Tensor,
        molecule_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate variable number of molecules into recipe embedding.
        
        Args:
            molecule_embs: [batch_size, max_molecules, input_dim] - molecule embeddings
            molecule_mask: [batch_size, max_molecules] - mask for valid molecules (1=valid, 0=padding)
            
        Returns:
            z_recipe: [batch_size, output_dim] - recipe embedding in hyperbolic space
        """
        batch_size, max_molecules, input_dim = molecule_embs.shape
        
        # Apply transformation if needed
        if input_dim != self.output_dim:
            molecule_embs = self.transform(molecule_embs.reshape(-1, input_dim))
            molecule_embs = molecule_embs.reshape(batch_size, max_molecules, self.output_dim)
        
        # Ensure embeddings are in hyperbolic space
        # Scale embeddings to have reasonable magnitude before projection
        molecule_embs = molecule_embs * 0.5  # Scale down to prevent extreme values
        molecule_embs = torch.clamp(molecule_embs, -0.9, 0.9)  # Less aggressive clamping
        molecule_embs = self.manifold.expmap0(molecule_embs)
        
        # Compute weights from mask
        if molecule_mask is not None:
            weights = molecule_mask.float()
        else:
            weights = torch.ones(batch_size, max_molecules, device=molecule_embs.device)
        
        # Compute Mobius mean
        z_recipe = self.mobius_mean(molecule_embs, weights)
        
        # Apply regularization: Dropout to prevent collapse
        # Project to tangent space (Euclidean) for regularization
        z_recipe_tangent = self.manifold.logmap0(z_recipe)  # [batch_size, output_dim]
        
        # Apply Dropout only (LayerNorm 제거 - collapse 방지)
        z_recipe_tangent = self.recipe_dropout(z_recipe_tangent)
        
        # diversity_noise=0: 노이즈가 벼랑 끝으로 밀어냄 (완전 비활성)
        
        # Tangent space clipping: 벽으로 튕겨나가는 것 원천 봉쇄
        z_recipe_tangent = torch.clamp(z_recipe_tangent, min=-15.0, max=15.0)
        
        # Project back to hyperbolic space
        z_recipe = self.manifold.expmap0(z_recipe_tangent)
        if hasattr(self.manifold, 'proj'):
            z_recipe = self.manifold.proj(z_recipe)
        # Soft cap: norm > 0.55이면 0.55로 스케일 (경계 쏠림·norm drift 방지, 변별력 유지)
        recipe_norms = z_recipe.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cap = 0.55
        scale = (recipe_norms > cap).float() * (cap / recipe_norms) + (recipe_norms <= cap).float()
        z_recipe = z_recipe * scale
        if hasattr(self.manifold, 'proj'):
            z_recipe = self.manifold.proj(z_recipe)
        return z_recipe


class BlenderAnchorEmbedding(nn.Module):
    """
    Blender Anchor: Fixed coordinates in hyperbolic space (Learnable Parameters).
    Each blender is represented as a learnable point in PoincareBall.
    """
    
    def __init__(
        self,
        num_blenders: int,
        embedding_dim: int = 128,
        manifold: Optional[PoincareBall] = None,
        c: float = 1.0
    ):
        super().__init__()
        self.num_blenders = num_blenders
        self.embedding_dim = embedding_dim
        self.c = c
        
        if manifold is None:
            self.manifold = PoincareBall(c=c)
        else:
            self.manifold = manifold
        
        # Learnable blender anchors in Euclidean space (will be projected)
        self.blender_anchors = nn.Parameter(
            torch.randn(num_blenders, embedding_dim) * 0.03
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights: slightly larger std for better initial direction spread."""
        nn.init.normal_(self.blender_anchors, mean=0.0, std=0.03)
    
    def forward(self, blender_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get blender embeddings.
        
        Args:
            blender_indices: [batch_size, num_blenders] or None for all blenders
            
        Returns:
            blender_embs: [batch_size, num_blenders, embedding_dim] or [num_blenders, embedding_dim]
                in PoincareBall
        """
        # Project to hyperbolic space
        anchors = torch.clamp(self.blender_anchors, -0.99, 0.99)
        anchors_hyp = self.manifold.expmap0(anchors)
        
        # Normalize blender embeddings: allow overlap with recipe norm (0.01~0.5) for discriminative distance
        # Target norm: 0.5~0.85 (이전 0.8~0.9는 레시피와 반경 분리되어 거리 변별력 상실)
        blender_norms = anchors_hyp.norm(dim=1, keepdim=True)  # [num_blenders, 1]
        target_norm_min, target_norm_max = 0.5, 0.85
        
        # Soft clamping: encourage blenders to move toward boundary
        # Use tanh-based soft clamping to allow gradual movement
        norm_center = (target_norm_min + target_norm_max) / 2  # 0.85
        norm_range = (target_norm_max - target_norm_min) / 2  # 0.05
        
        # Soft clamp using tanh: maps to [-1, 1] then scales to [min, max]
        normalized_norms = (blender_norms - norm_center) / (norm_range + 1e-8)
        clamped_normalized = torch.tanh(normalized_norms)  # [-1, 1]
        blender_norms_clamped = clamped_normalized * norm_range + norm_center
        
        # Apply clamped norms
        anchors_hyp = anchors_hyp / (blender_norms + 1e-8) * blender_norms_clamped
        
        if blender_indices is not None:
            # Select specific blenders
            batch_size, num_blenders = blender_indices.shape
            
            # Clamp indices to valid range [0, num_blenders-1]
            # 0 is padding, so we'll handle it separately
            clamped_indices = torch.clamp(blender_indices, 0, self.num_blenders - 1)
            
            # Use advanced indexing: anchors_hyp[clamped_indices]
            # This gives [batch_size, num_blenders, embedding_dim]
            selected = anchors_hyp[clamped_indices]
            
            # Mask out padding (where original indices were 0)
            mask = (blender_indices > 0).unsqueeze(-1)  # [batch_size, num_blenders, 1]
            selected = selected * mask.float()
            
            return selected
        else:
            return anchors_hyp


class GroupAnchorEmbedding(nn.Module):
    """
    Group Anchor: Learnable coordinates in hyperbolic space for odor groups (계층 보조 loss용).
    레시피 임베딩을 해당 그룹 앵커 쪽으로 당기는 보조 loss에 사용.
    - 중심 밀림 방지: Blender 앵커처럼 norm을 경계 쪽(0.5~0.85)으로 유도.
    - 그룹 간 분리: separation_loss()로 서로 다른 그룹 앵커가 겹치지 않도록 margin 유지.
    """

    def __init__(
        self,
        num_groups: int,
        embedding_dim: int = 128,
        manifold: Optional[PoincareBall] = None,
        c: float = 1.0
    ):
        super().__init__()
        self.num_groups = num_groups
        self.embedding_dim = embedding_dim
        self.c = c
        if manifold is None:
            self.manifold = PoincareBall(c=c)
        else:
            self.manifold = manifold
        self.group_anchors = nn.Parameter(torch.randn(num_groups, embedding_dim) * 0.01)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.group_anchors, mean=0.0, std=0.01)

    def forward(self, group_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            group_indices: [batch_size] or None for all groups
        Returns:
            [batch_size, embedding_dim] or [num_groups, embedding_dim] in PoincareBall
        """
        anchors = torch.clamp(self.group_anchors, -0.99, 0.99)
        anchors_hyp = self.manifold.expmap0(anchors)
        # 중심 밀림 방지: 그룹 앵커도 원점에서 밀어내기 (norm 0.5~0.85)
        group_norms = anchors_hyp.norm(dim=1, keepdim=True).clamp(min=1e-8)
        target_min, target_max = 0.5, 0.85
        center, span = (target_min + target_max) / 2, (target_max - target_min) / 2
        normalized = (group_norms - center) / (span + 1e-8)
        clamped_norms = torch.tanh(normalized) * span + center
        anchors_hyp = anchors_hyp / group_norms * clamped_norms
        if group_indices is not None:
            group_indices = group_indices.clamp(0, self.num_groups - 1)
            return anchors_hyp[group_indices]
        return anchors_hyp

    def separation_loss(self, margin: float = 0.3) -> torch.Tensor:
        """
        Inter-group separation: 서로 다른 그룹 앵커 간 거리가 margin 이상이 되도록.
        loss = mean over (i<j) of max(0, margin - dist(anchor_i, anchor_j)).
        """
        if self.num_groups < 2:
            return torch.tensor(0.0, device=self.group_anchors.device)
        all_anchors = self.forward(None)  # [num_groups, dim]
        dists = self.manifold.dist(
            all_anchors.unsqueeze(1),
            all_anchors.unsqueeze(0)
        )  # [G, G]
        # upper triangle (i < j)
        i, j = torch.triu_indices(self.num_groups, self.num_groups, offset=1, device=all_anchors.device)
        pair_dists = dists[i, j]
        violations = (margin - pair_dists).clamp(min=0.0)
        return violations.mean()


class HierarchicalFragranceHypergraph(nn.Module):
    """
    Main model: Hierarchical Hyperbolic Hypergraph for Fragrance Synergy & Blender Ranking.
    
    Architecture:
    1. Bottom Layer: Multi-modal encoding (SMILES GNN, Notes FastText). Blender Anchors는 랭킹/손실에만 사용.
    2. Channel 1: Atomic Hyperedge (molecule + notes만, 블렌더 입력 없음 → 정답 유출 방지)
    3. Channel 2: Global Recipe Synergy (variable molecules → recipe)
    4. Ranking: Poincaré Distance between z_recipe and Blender Anchors → Top-K
    """
    
    def __init__(
        self,
        # SMILES GNN parameters
        node_dim: int = 9,
        edge_dim: int = 3,
        gnn_hidden_dim: int = 128,
        gnn_output_dim: int = 128,
        gnn_num_layers: int = 3,
        gnn_architecture: str = "GCN",
        
        # Notes parameters
        vocab_size: int = 435,  # Number of note types
        note_embedding_dim: int = 300,
        note_hyperbolic_dim: int = 128,
        fasttext_model_path: Optional[str] = None,
        
        # Blender parameters
        num_blenders: int = 100,
        blender_dim: int = 128,
        # Group parameters (계층 보조 loss)
        num_groups: int = 1,
        group_dim: Optional[int] = None,
        
        # Channel dimensions
        channel1_output_dim: int = 128,
        channel2_output_dim: int = 128,
        
        # Hyperbolic space parameters (고정 곡률 권장: 수치 안정성)
        c: float = 0.5,  # Poincaré ball curvature (0.5=완만·유클리드 성질 섞임)
        learnable_curvature: bool = False,  # True면 학습 가능, False면 c 고정
        
        # Other parameters
        dropout: float = 0.1,
        # Fingerprint (ECFP/Morgan) 모드: GNN 대신 분자 지문 사용
        use_fingerprint: bool = False,
        fp_dim: int = 2048,
    ):
        super().__init__()
        
        self.num_blenders = num_blenders
        self.num_groups = max(1, num_groups)
        group_dim = group_dim if group_dim is not None else blender_dim
        self.use_fingerprint = use_fingerprint
        self.gnn_output_dim = gnn_output_dim

        # Learnable curvature parameter (고정 권장: learnable 시 c가 커지면 거리 폭발)
        if learnable_curvature:
            # softplus(c_raw) = c, 초기값 c로 시작. 상한 1.0으로 clamp해 두어 급격한 휨 방지
            raw_c_init = torch.log(torch.exp(torch.tensor([min(c, 0.5)], dtype=torch.float32)) - 1.0 + 1e-8)
            self.c_raw = nn.Parameter(raw_c_init)
            self._c_max = 1.0  # learnable 시 c 상한 (너무 크면 미세 차이도 거리 폭발)
        else:
            self.register_buffer('c_raw', torch.tensor([c], dtype=torch.float32))
        
        # Initialize PoincareBall manifold (will be updated dynamically)
        c_val = torch.nn.functional.softplus(self.c_raw).item() if learnable_curvature else c
        self.manifold = PoincareBall(c=c_val)
        
        # Bottom Layer: Multi-modal encoding (GNN 또는 Fingerprint)
        self.smiles_encoder = SMILESGNNEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_layers=gnn_num_layers,
            architecture=gnn_architecture,
            dropout=dropout
        ) if not use_fingerprint else None
        self.fingerprint_encoder = FingerprintEncoder(
            fp_dim=fp_dim,
            output_dim=gnn_output_dim,
            dropout=dropout
        ) if use_fingerprint else None
        
        self.note_encoder = HyperbolicNoteEmbedding(
            vocab_size=vocab_size,
            embedding_dim=note_embedding_dim,
            hyperbolic_dim=note_hyperbolic_dim,
            fasttext_model_path=fasttext_model_path,
            manifold=self.manifold,
            c=c
        )
        
        self.blender_anchors = BlenderAnchorEmbedding(
            num_blenders=num_blenders,
            embedding_dim=blender_dim,
            manifold=self.manifold,
            c=c
        )
        self.group_anchors = GroupAnchorEmbedding(
            num_groups=self.num_groups,
            embedding_dim=group_dim,
            manifold=self.manifold,
            c=c
        ) if self.num_groups > 1 else None

        # Channel 1: Atomic Hyperedge (분자+노트만, 블렌더 입력 없음)
        self.channel1 = Channel1AtomicHyperedge(
            molecule_dim=gnn_output_dim,
            note_dim=note_hyperbolic_dim,
            output_dim=channel1_output_dim,
            manifold=self.manifold,
            c=c
        )
        
        # Channel 2: Global Recipe Synergy
        self.channel2 = Channel2Synergy(
            input_dim=channel1_output_dim,
            output_dim=channel2_output_dim,
            manifold=self.manifold,
            c=c
        )
        
    def forward(
        self,
        # SMILES inputs
        smiles_graphs: Batch,  # Batched molecular graphs
        smiles_batch: torch.Tensor,  # Batch assignment for graphs [num_graphs]
        
        # Notes inputs
        note_indices: torch.Tensor,  # [batch_size, max_molecules, max_notes]
        note_mask: Optional[torch.Tensor] = None,  # [batch_size, max_molecules, max_notes]
        
        # Blender inputs (for Channel 1)
        blender_indices: Optional[torch.Tensor] = None,  # [batch_size, max_molecules, max_blenders]
        
        # Molecule mask for variable-length recipes
        molecule_mask: Optional[torch.Tensor] = None,  # [batch_size, max_molecules]
        
        # 프리컴퓨팅: 지정 시 GNN 생략, [batch_size, max_molecules, gnn_output_dim]
        precomputed_mol_embs: Optional[torch.Tensor] = None,
        # ECFP/Morgan 지문 모드: [batch_size, max_molecules, fp_dim] → fingerprint_encoder로 임베딩
        mol_fingerprints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the hierarchical hypergraph.
        
        Returns:
            z_recipe: [batch_size, channel2_output_dim] - recipe embedding in hyperbolic space
        """
        batch_size = note_indices.size(0)
        max_molecules = note_indices.size(1)
        
        # Learnable curvature: 모든 서브모듈이 동일한 Poincaré 곡률 사용 (forward/loss 일관성)
        if isinstance(self.c_raw, nn.Parameter):
            c_val = torch.nn.functional.softplus(self.c_raw).clamp(0.01, getattr(self, '_c_max', 1.0)).item()
            manifold = PoincareBall(c=c_val)
            self.manifold = manifold
            self.note_encoder.manifold = manifold
            self.channel1.manifold = manifold
            self.channel2.manifold = manifold
            self.blender_anchors.manifold = manifold
            if self.group_anchors is not None:
                self.group_anchors.manifold = manifold

        # 분자 임베딩: 지문 > 프리컴퓨팅 > GNN
        if mol_fingerprints is not None and self.fingerprint_encoder is not None:
            mol_embs = self.fingerprint_encoder(mol_fingerprints)  # [B, max_mol, gnn_output_dim]
        elif precomputed_mol_embs is not None:
            # [batch_size, max_molecules, gnn_output_dim] (차원 일치 시 그대로 사용)
            mol_embs = precomputed_mol_embs
        else:
            # Process all molecules in batch through GNN (use_fingerprint=False일 때만)
            if self.smiles_encoder is None:
                raise RuntimeError(
                    "Model is in fingerprint mode (use_fingerprint=True) but mol_fingerprints was not provided. "
                    "Ensure the dataloader passes mol_fingerprints when using --use_fingerprint."
                )
            # smiles_graphs is a Batch object containing all graphs
            # smiles_batch indicates which graph belongs to which sample and molecule
            if hasattr(smiles_graphs, 'batch') and smiles_graphs.batch is not None:
                node_batch = smiles_graphs.batch
            else:
                num_nodes = smiles_graphs.x.size(0)
                node_batch = torch.zeros(num_nodes, dtype=torch.long, device=smiles_graphs.x.device)
                if hasattr(smiles_graphs, 'ptr'):
                    ptr = smiles_graphs.ptr
                    for i in range(len(ptr) - 1):
                        start_idx = ptr[i]
                        end_idx = ptr[i + 1]
                        graph_idx = smiles_batch[i] if i < len(smiles_batch) else 0
                        node_batch[start_idx:end_idx] = graph_idx
                else:
                    num_graphs = len(smiles_batch)
                    nodes_per_graph = num_nodes // num_graphs if num_graphs > 0 else 0
                    for i in range(num_graphs):
                        start_idx = i * nodes_per_graph
                        end_idx = (i + 1) * nodes_per_graph if i < num_graphs - 1 else num_nodes
                        node_batch[start_idx:end_idx] = smiles_batch[i]
            unique_values = torch.unique(node_batch, sorted=True)
            if len(unique_values) > 0 and (unique_values[0] != 0 or unique_values[-1] != len(unique_values) - 1):
                batch_mapping = {int(val.item()): idx for idx, val in enumerate(unique_values)}
                continuous_batch = torch.tensor(
                    [batch_mapping[int(val.item())] for val in node_batch],
                    dtype=torch.long,
                    device=node_batch.device
                )
            else:
                continuous_batch = node_batch
            all_mol_embs = self.smiles_encoder(
                smiles_graphs.x,
                smiles_graphs.edge_index,
                smiles_graphs.edge_attr if hasattr(smiles_graphs, 'edge_attr') else None,
                batch=continuous_batch
            )
            mol_embs = torch.zeros(
                batch_size, max_molecules, all_mol_embs.size(1),
                dtype=all_mol_embs.dtype,
                device=all_mol_embs.device
            )
            original_graph_indices = smiles_batch
            for graph_idx in range(min(len(all_mol_embs), len(original_graph_indices))):
                original_batch_idx = original_graph_indices[graph_idx].item()
                sample_idx = int(original_batch_idx) // max_molecules
                mol_idx = int(original_batch_idx) % max_molecules
                if sample_idx < batch_size and mol_idx < max_molecules:
                    mol_embs[sample_idx, mol_idx] = all_mol_embs[graph_idx]
        
        # Process each molecule through Channel 1
        molecule_embs_list = []
        
        for mol_idx in range(max_molecules):
            # Get molecule embedding for this position
            mol_emb = mol_embs[:, mol_idx, :]  # [batch_size, gnn_output_dim]
            
            # Get notes for this molecule
            notes = note_indices[:, mol_idx, :]  # [batch_size, max_notes]
            note_embs = self.note_encoder(notes)  # [batch_size, max_notes, note_hyperbolic_dim]
            
            # Channel 1: Create atomic hyperedge (분자+노트만, 블렌더 입력 없음 → 정답 유출 방지)
            z_i = self.channel1(mol_emb, note_embs)  # [batch_size, channel1_output_dim]
            molecule_embs_list.append(z_i)
        
        # Stack molecule embeddings
        molecule_embs = torch.stack(molecule_embs_list, dim=1)  # [batch_size, max_molecules, channel1_output_dim]
        
        # Channel 2: Global Recipe Synergy
        z_recipe = self.channel2(molecule_embs, molecule_mask)  # [batch_size, channel2_output_dim]
        
        return z_recipe
    
    def compute_poincare_distance(
        self,
        z_recipe: torch.Tensor,
        blender_embs: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        clamp_max: Optional[float] = 10.0
    ) -> torch.Tensor:
        """
        Compute Poincaré distance between recipe and all blenders.
        
        Args:
            z_recipe: [batch_size, dim] - recipe embeddings
            blender_embs: [num_blenders, dim] or None - blender embeddings (if None, use all)
            scale: scaling factor for distance normalization
            clamp_max: maximum distance value (None to disable)
            
        Returns:
            distances: [batch_size, num_blenders] - Poincaré distances (scaled and clamped)
        """
        # Update manifold with current curvature
        # Use softplus to ensure c is always positive
        if isinstance(self.c_raw, nn.Parameter):
            c_val = torch.nn.functional.softplus(self.c_raw).clamp(0.01, getattr(self, '_c_max', 1.0)).item()
            self.manifold = PoincareBall(c=c_val)
        else:
            c_val = self.c_raw.item()
            self.manifold = PoincareBall(c=c_val)
        
        if blender_embs is None:
            blender_embs = self.blender_anchors()  # [num_blenders, dim]
        
        batch_size = z_recipe.size(0)
        num_blenders = blender_embs.size(0)
        
        # Expand for batch computation
        z_recipe_expanded = z_recipe.unsqueeze(1)  # [batch_size, 1, dim]
        blender_embs_expanded = blender_embs.unsqueeze(0)  # [1, num_blenders, dim]
        
        # Compute Poincaré distance
        distances = self.manifold.dist(
            z_recipe_expanded.expand(-1, num_blenders, -1),
            blender_embs_expanded.expand(batch_size, -1, -1)
        )  # [batch_size, num_blenders]
        
        # Scale and normalize for numerical stability
        distances = distances / scale
        
        # Clamp distances to prevent extreme values
        # Increased clamp_max to allow larger distance range (for better separation)
        if clamp_max is not None:
            # Use larger clamp_max to allow distances up to 5.0+ for better blender separation
            distances = torch.clamp(distances, min=0.0, max=clamp_max)
        else:
            # If no clamp_max, allow full distance range (up to ~10.0 in Poincaré ball)
            # This enables better separation when blenders are at boundary (norm 0.8-0.9)
            pass
        
        return distances
    
    def compute_temperature_scores(
        self,
        distances: torch.Tensor,
        temperature: float = 0.07  # Temperature=0.07 for maximum discrimination in top ranks
    ) -> torch.Tensor:
        """
        Convert Poincaré distances to scores using temperature scaling.
        
        Formula: Score = exp(-d(u,v) / τ)
        
        Temperature=0.07 maximizes discrimination in top ranks for better NDCG.
        Lower temperature creates sharper score distribution, prioritizing closer blenders.
        
        Args:
            distances: [batch_size, num_blenders] - Poincaré distances
            temperature: τ (temperature parameter, lower = sharper distribution, default 0.07)
            
        Returns:
            scores: [batch_size, num_blenders] - temperature-scaled scores (higher = better)
        """
        # Temperature scaling: exp(-distance / temperature)
        # Temperature=0.07 maximizes discrimination in top ranks for better NDCG
        scores = torch.exp(-distances / temperature)
        return scores
    
    def rank_blenders(
        self,
        z_recipe: torch.Tensor,
        k: int = 10,
        blender_embs: Optional[torch.Tensor] = None,
        distance_scale: float = 1.0,
        clamp_max: Optional[float] = 10.0,
        use_temperature_scaling: bool = True,
        temperature: float = 0.07  # Temperature=0.07 for maximum discrimination in top ranks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rank blenders by Poincaré distance (with optional temperature scaling) and return Top-K.
        
        Args:
            z_recipe: [batch_size, dim] - recipe embeddings
            k: number of top blenders to return
            blender_embs: [num_blenders, dim] or None - blender embeddings
            distance_scale: scaling factor for distance normalization
            clamp_max: maximum distance value (None to disable)
            use_temperature_scaling: if True, use temperature scaling for ranking (default True)
            temperature: temperature parameter for scaling (lower = sharper, default 0.07)
            
        Returns:
            top_k_indices: [batch_size, k] - indices of top-k blenders
            top_k_distances: [batch_size, k] - distances to top-k blenders
        """
        distances = self.compute_poincare_distance(z_recipe, blender_embs, scale=distance_scale, clamp_max=clamp_max)
        
        if use_temperature_scaling:
            # Convert distances to scores using temperature scaling
            # Score = exp(-distance / temperature)
            # Lower temperature (e.g., 0.1) makes close blenders have much higher scores
            scores = self.compute_temperature_scores(distances, temperature=temperature)
            # Rank by scores (higher = better)
            top_k_scores, top_k_indices = torch.topk(scores, k, dim=1, largest=True)
            # Get corresponding distances
            batch_indices = torch.arange(distances.size(0), device=distances.device).unsqueeze(1)
            top_k_distances = distances[batch_indices, top_k_indices]
        else:
            # Original: rank by distance (lower = better)
            top_k_distances, top_k_indices = torch.topk(distances, k, dim=1, largest=False)
        
        return top_k_indices, top_k_distances
    
    def get_embedding_statistics(
        self,
        z_recipe: torch.Tensor,
        blender_embs: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute embedding statistics for monitoring collapse.
        
        Args:
            z_recipe: [batch_size, dim] - recipe embeddings
            blender_embs: [num_blenders, dim] or None - blender embeddings
            
        Returns:
            stats: dictionary with statistics
        """
        if blender_embs is None:
            blender_embs = self.blender_anchors()
        
        # Compute distances
        distances = self.compute_poincare_distance(z_recipe, blender_embs, scale=1.0, clamp_max=None)
        
        # Recipe embedding statistics
        recipe_std = z_recipe.std(dim=0).mean().item()  # Average std across dimensions
        recipe_mean_norm = z_recipe.norm(dim=1).mean().item()  # Average norm
        
        # Blender embedding statistics
        blender_std = blender_embs.std(dim=0).mean().item()
        blender_mean_norm = blender_embs.norm(dim=1).mean().item()
        
        # Distance statistics
        mean_distance = distances.mean().item()
        std_distance = distances.std().item()
        min_distance = distances.min().item()
        max_distance = distances.max().item()
        
        # Curvature (using softplus to ensure positivity)
        if isinstance(self.c_raw, nn.Parameter):
            curvature = torch.nn.functional.softplus(self.c_raw).item()
        else:
            curvature = self.c_raw.item()
        
        return {
            'recipe_std': recipe_std,
            'recipe_mean_norm': recipe_mean_norm,
            'blender_std': blender_std,
            'blender_mean_norm': blender_mean_norm,
            'mean_distance': mean_distance,
            'std_distance': std_distance,
            'min_distance': min_distance,
            'max_distance': max_distance,
            'curvature': curvature
        }
    
    def recommend(
        self,
        z_recipe: torch.Tensor,
        k: int = 10,
        blender_embs: Optional[torch.Tensor] = None,
        blender_id_to_name: Optional[Dict[int, str]] = None,
        return_distances: bool = True,
        return_names: bool = False
    ) -> List[Dict]:
        """
        레시피 임베딩을 입력받아 여러 개의 블렌더를 랭킹 순서대로 추천합니다.
        
        Args:
            z_recipe: [batch_size, dim] 또는 [dim] - recipe embedding
            k: 추천할 블렌더 개수
            blender_embs: [num_blenders, dim] 또는 None - blender embeddings
            blender_id_to_name: blender ID를 이름으로 매핑하는 딕셔너리 (optional)
            return_distances: 거리 정보를 반환할지 여부
            return_names: 블렌더 이름을 반환할지 여부
        
        Returns:
            recommendations: 추천 블렌더 리스트 (batch_size개), 각 항목은 리스트:
                [
                    {
                        'rank': int,  # 랭킹 (1부터 시작)
                        'blender_id': int,  # 블렌더 ID
                        'blender_name': str (optional),  # 블렌더 이름
                        'distance': float (optional),  # Poincaré 거리
                    },
                    ...
                ]
        """
        # Handle single sample case
        if z_recipe.dim() == 1:
            z_recipe = z_recipe.unsqueeze(0)
        
        batch_size = z_recipe.size(0)
        
        # Rank blenders
        top_k_indices, top_k_distances = self.rank_blenders(
            z_recipe, 
            k=k, 
            blender_embs=blender_embs
        )
        
        # Convert to list of recommendations
        all_recommendations = []
        
        for batch_idx in range(batch_size):
            recommendations = []
            top_k_indices_np = top_k_indices[batch_idx].cpu().numpy()
            top_k_distances_np = top_k_distances[batch_idx].cpu().numpy()
            
            for rank in range(k):
                blender_id = int(top_k_indices_np[rank])
                rec = {
                    'rank': rank + 1,
                    'blender_id': blender_id
                }
                
                if return_names and blender_id_to_name and blender_id in blender_id_to_name:
                    rec['blender_name'] = blender_id_to_name[blender_id]
                
                if return_distances:
                    rec['distance'] = float(top_k_distances_np[rank])
                
                recommendations.append(rec)
            
            all_recommendations.append(recommendations)
        
        # Return single list if batch_size == 1
        if batch_size == 1:
            return all_recommendations[0]
        else:
            return all_recommendations
    
    def get_embedding_statistics(
        self,
        z_recipe: torch.Tensor,
        blender_embs: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute embedding statistics for monitoring collapse.
        
        Args:
            z_recipe: [batch_size, dim] - recipe embeddings
            blender_embs: [num_blenders, dim] or None - blender embeddings
            
        Returns:
            stats: dictionary with statistics
        """
        if blender_embs is None:
            blender_embs = self.blender_anchors()
        
        # Compute distances
        distances = self.compute_poincare_distance(z_recipe, blender_embs, scale=1.0, clamp_max=None)
        
        # Recipe embedding statistics
        recipe_std = z_recipe.std(dim=0).mean().item()  # Average std across dimensions
        recipe_mean_norm = z_recipe.norm(dim=1).mean().item()  # Average norm
        
        # Blender embedding statistics
        blender_std = blender_embs.std(dim=0).mean().item()
        blender_mean_norm = blender_embs.norm(dim=1).mean().item()
        
        # Distance statistics
        mean_distance = distances.mean().item()
        std_distance = distances.std().item()
        min_distance = distances.min().item()
        max_distance = distances.max().item()
        
        # Curvature (using softplus to ensure positivity)
        if isinstance(self.c_raw, nn.Parameter):
            curvature = torch.nn.functional.softplus(self.c_raw).item()
        else:
            curvature = self.c_raw.item()
        
        return {
            'recipe_std': recipe_std,
            'recipe_mean_norm': recipe_mean_norm,
            'blender_std': blender_std,
            'blender_mean_norm': blender_mean_norm,
            'mean_distance': mean_distance,
            'std_distance': std_distance,
            'min_distance': min_distance,
            'max_distance': max_distance,
            'curvature': curvature
        }


if __name__ == "__main__":
    """
    모듈 테스트 코드
    ⚠️  이 파일은 모듈이므로 직접 실행하지 마세요.
    학습은 train_hyperbolic_hypergraph.py를 사용하세요.
    """
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # OpenMP 에러 해결
    
    print("=" * 70)
    print("⚠️  이 파일은 모듈입니다. 직접 실행할 수 없습니다.")
    print("=" * 70)
    print("\n사용 방법:")
    print("  학습 실행:")
    print("    cd ../")
    print("    python train_hyperbolic_hypergraph.py \\")
    print("        --data_path ../cleaned_data/cleaned_complete_data.json \\")
    print("        --vocab_path ../feature_encoding/vocabularies.json")
    print("\n  또는 쉘 스크립트 사용:")
    print("    ./run_training.sh")
    print("\n  모듈로 import:")
    print("    from model import HierarchicalFragranceHypergraph")
    print("=" * 70)
    
    # 간단한 모델 초기화 테스트
    try:
        print("\n모델 초기화 테스트...")
        model = HierarchicalFragranceHypergraph(
            node_dim=9,
            edge_dim=3,
            gnn_hidden_dim=128,
            gnn_output_dim=128,
            gnn_num_layers=3,
            vocab_size=435,
            note_embedding_dim=300,
            note_hyperbolic_dim=128,
            num_blenders=100,
            blender_dim=128,
            channel1_output_dim=128,
            channel2_output_dim=128,
            c=0.1,
            learnable_curvature=False,
            dropout=0.1
        )
        print("✓ 모델 초기화 성공!")
        print(f"  총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ 모델 초기화 실패: {e}")
