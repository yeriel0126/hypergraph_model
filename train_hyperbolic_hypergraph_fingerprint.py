"""
Training script for Hierarchical Hyperbolic Hypergraph with ECFP/Morgan fingerprints (지문 전용).

분자 입력: SMILES → Morgan/ECFP 지문 → FingerprintEncoder (GNN 미사용).
기본 제안 모델(GNN)은 train_hyperbolic_hypergraph.py 를 사용하세요.
"""

# Set environment variable for OpenMP (macOS compatibility) - MUST be before any imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import gc
import inspect
import numpy as np
import json
import pathlib
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent))

try:
    from geoopt.optim import RiemannianAdam
    from geoopt.manifolds import Euclidean
    _RIEMANNIAN_AVAILABLE = True
except ImportError:
    _RIEMANNIAN_AVAILABLE = False

from model.hierarchical_hyperbolic_hypergraph import HierarchicalFragranceHypergraph
from model.hyperbolic_data_loader import HyperbolicRecipeDataset, collate_hyperbolic_recipes
from model.hyperbolic_losses import (
    HyperbolicTripletMarginLoss,
    ConfusionPairWeightedLoss,
    CombinedConfusionLoss,
    HyperbolicBPRLoss,
    HyperbolicCircleLoss,
    HyperbolicMeanPositiveDistanceLoss,
    HyperbolicMaxMarginRankingLoss,
)

# Import utility functions
from utils import (
    compute_confusion_matrix_from_data,
    compute_blender_class_weights,
    downsample_blender_combinations,
    compute_molecule_frequency_weights,
    convert_to_json_serializable,
    export_combination_format,
    filter_common_molecules,
    detect_cheat_key_molecules,
    filter_duplicate_combinations,
    apply_cheat_key_masking,
    add_gaussian_noise_to_recipe,
    run_leakage_diagnostics
)
from data import build_group_vocab_and_blender_to_group, create_recipe_combinations
from config import parse_train_args, get_device, apply_memory_and_device_options

# 고정된 random seed 값 (데이터 무결성 보장, config와 동기화)
RANDOM_SEED = 42
TRAIN_SIZE = 60000
VAL_SIZE = 10000
TEST_SIZE = 11248


def clear_device_memory(device: torch.device) -> None:
    """
    좀비 메모리 정리: 학습을 껐다 켜면 메모리가 조각나서 OOM이 나는 현상 완화.
    gc + CUDA/MPS cache 비우기로 가능한 한 연속된 여유 공간을 확보.
    """
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()


def set_all_seeds(seed: int = RANDOM_SEED, fast_mode: bool = False):
    """
    모든 random seed를 일관되게 설정하여 실험 재현성 보장.
    
    Args:
        seed: Random seed 값 (기본값: 42)
    """
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic behavior for reproducibility (fast_mode면 성능 우선)
    if fast_mode:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(data_path: str, vocab_path: str) -> Tuple[List[Dict], Dict]:
    """
    Load recipe data and vocabulary.
    
    Args:
        data_path: Path to cleaned_complete_data.json
        vocab_path: Path to vocabularies.json
    
    Returns:
        records: List of recipe records
        vocab_data: Vocabulary data (notes, blenders, etc.)
    """
    # 경로 검증
    data_path_obj = Path(data_path)
    vocab_path_obj = Path(vocab_path)
    
    if not data_path_obj.exists():
        print(f"❌ 에러: 데이터 파일을 찾을 수 없습니다: {data_path}")
        print(f"   절대 경로: {data_path_obj.absolute()}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if not vocab_path_obj.exists():
        print(f"❌ 에러: 어휘 파일을 찾을 수 없습니다: {vocab_path}")
        print(f"   절대 경로: {vocab_path_obj.absolute()}")
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    # Extract records - cleaned_complete_data.json은 {"data": [...]} 구조
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # 'data' 키를 먼저 확인 (cleaned_complete_data.json 형식)
        if 'data' in data:
            records = data['data']
        # 'records' 키 확인 (다른 형식)
        elif 'records' in data:
            records = data['records']
        else:
            print(f"⚠️  경고: 데이터 구조를 인식할 수 없습니다. 사용 가능한 키: {list(data.keys())}")
            records = []
    else:
        records = []
    
    # Vocabulary 정보는 출력 생략
    
    return records, vocab_data


def _target_blenders_to_positive_set_tensor(target_blenders: List, device: torch.device) -> torch.Tensor:
    """
    (레시피, [A, B, ...]) → [batch_size, max_positives] 패딩 텐서.
    One-to-Many: 레시피 하나에 여러 조향사를 하나의 Positive Set으로 묶어 loss에 전달.
    패딩값 -1 (loss 내부에서 무시).
    """
    batch_size = len(target_blenders)
    lists = []
    for t in target_blenders:
        if isinstance(t, list):
            lists.append([int(x) for x in t if x is not None])
        else:
            lists.append([int(t)] if t is not None else [])
    max_p = max(len(l) for l in lists) if lists else 1
    if max_p == 0:
        max_p = 1
    padded = []
    for l in lists:
        if not l:
            padded.append([-1] * max_p)
        else:
            padded.append(l + [-1] * (max_p - len(l)))
    return torch.tensor(padded, device=device, dtype=torch.long)


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    loss_fn: Optional[nn.Module] = None,
    k: int = 10,
    use_blender_input: bool = False,
    temperature: float = 0.07
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set.
    
    Supports multi-label evaluation (multiple target blenders per recipe).
    전체 검증을 단일 torch.no_grad()로 감싸 그래디언트 그래프가 전혀 쌓이지 않도록 함 (메모리 폭발 방지).
    """
    model.eval()
    
    total_samples = 0
    top_1_hits = 0
    top_5_hits = 0
    top_k_hits = 0
    sum_recall_1 = 0.0
    sum_recall_5 = 0.0
    sum_recall_k = 0.0
    reciprocal_ranks = []
    ndcg_scores = []
    all_val_loss = 0.0
    num_batches = 0
    
    # Diversity tracking
    recommended_blenders = []
    # z_recipe norm 모니터링 (하이퍼볼릭 수치 불안정: 원점 붕괴 ~0 또는 boundary ~0.999)
    recipe_norm_list = []
    
    # 메모리 안정: 검증 전체를 한 번의 no_grad()로 감싸서 그래디언트 계산 그래프가 절대 쌓이지 않도록 함
    # 평가 Top-K: model.rank_blenders() → compute_poincare_distance() → manifold.dist() 사용 (유클리드 L2/내적 아님)
    with torch.no_grad():
        all_blender_embs = model.blender_anchors()  # [num_blenders, embedding_dim]
        
        pbar = tqdm(val_loader, desc="검증 중", unit="batch", leave=False, file=sys.stderr, dynamic_ncols=True, mininterval=0.5)
        for batch in pbar:
            # Move to device (MPS에서는 non_blocking이 오히려 느릴 수 있으므로 조건부 적용)
            use_non_blocking = device.type == 'cuda'  # CUDA에서만 non_blocking 사용
            note_indices = batch['note_indices'].to(device, non_blocking=use_non_blocking)
            blender_indices = batch['blender_indices'].to(device, non_blocking=use_non_blocking) if use_blender_input else None
            molecule_mask = batch['molecule_mask'].to(device, non_blocking=use_non_blocking).float()  # float32 고정 (float64 시 메모리 2배)
            smiles_graphs = batch['smiles_graphs']
            smiles_batch = batch['smiles_batch'].to(device, non_blocking=use_non_blocking)
            
            # Handle SMILES graphs
            if hasattr(smiles_graphs, 'to'):
                smiles_graphs = smiles_graphs.to(device, non_blocking=use_non_blocking)
            else:
                smiles_graphs.x = smiles_graphs.x.to(device, non_blocking=use_non_blocking)
                smiles_graphs.edge_index = smiles_graphs.edge_index.to(device, non_blocking=use_non_blocking)
                if hasattr(smiles_graphs, 'edge_attr') and smiles_graphs.edge_attr is not None:
                    smiles_graphs.edge_attr = smiles_graphs.edge_attr.to(device, non_blocking=use_non_blocking)
            
            # Forward pass (Content* 모델은 precomputed_mol_embs 지원)
            forward_kw = dict(
                smiles_graphs=smiles_graphs,
                smiles_batch=smiles_batch,
                note_indices=note_indices,
                blender_indices=blender_indices,
                molecule_mask=molecule_mask,
            )
            # 지문 전용: 항상 mol_fingerprints 전달
            if 'mol_fingerprints' in batch:
                forward_kw['mol_fingerprints'] = batch['mol_fingerprints'].to(device, non_blocking=use_non_blocking)
            z_recipe = model(**forward_kw)
            
            # z_recipe norm 수집 (collapse/boundary 감지용)
            recipe_norm_list.append(z_recipe.norm(dim=1).cpu())
            
            # ⚡ 재사용: 검증 중에는 파라미터가 업데이트되지 않으므로 한 번 계산한 값 재사용 가능
            
            # Compute loss if loss function provided (One-to-Many: 전체 Positive Set 사용)
            if loss_fn is not None:
                target_blenders = batch['target_blenders']
                target_blender_tensor = _target_blenders_to_positive_set_tensor(target_blenders, device)
                batch_loss = loss_fn(z_recipe, target_blender_tensor, all_blender_embs)
                all_val_loss += batch_loss.item()
                num_batches += 1
            
            # Rank blenders: MPS OOM 방지를 위해 z_recipe를 청크 단위로 처리
            batch_size = z_recipe.size(0)
            rank_chunk_size = 64 if device.type == 'mps' else 512  # MPS OOM 방지
            top_k_indices_list = []
            top_k_distances_list = []
            for start in range(0, batch_size, rank_chunk_size):
                end = min(start + rank_chunk_size, batch_size)
                z_chunk = z_recipe[start:end]
                idx_chunk, dist_chunk = model.rank_blenders(
                    z_chunk,
                    k=k,
                    blender_embs=all_blender_embs,
                    use_temperature_scaling=True,
                    temperature=temperature
                )
                top_k_indices_list.append(idx_chunk)
                top_k_distances_list.append(dist_chunk)
            top_k_indices = torch.cat(top_k_indices_list, dim=0)
            top_k_distances = torch.cat(top_k_distances_list, dim=0)
            
            # Get target blenders (support multi-label)
            target_blenders = batch['target_blenders']  # List of lists or single values
            
            # ⚡ 최적화: 배치 단위로 처리하여 CPU 전송 최소화
            top_k_indices_cpu = top_k_indices.cpu().numpy()
            
            for i in range(batch_size):
                # Handle multi-label: target_blenders[i] can be a list or single value
                if isinstance(target_blenders[i], list):
                    true_blenders = set(target_blenders[i])
                else:
                    true_blenders = {target_blenders[i]}
                
                if not true_blenders:
                    continue
                
                # Get predicted blenders (이미 CPU로 전송됨)
                pred_blenders = top_k_indices_cpu[i]
                
                # Track diversity (검증 시에만, 학습 속도에 영향 최소화)
                if len(recommended_blenders) < 10000:  # 샘플링으로 제한
                    recommended_blenders.extend(pred_blenders.tolist())
                
                # Hit Rate@1
                if pred_blenders[0] in true_blenders:
                    top_1_hits += 1
                
                # Hit Rate@5
                if any(p in true_blenders for p in pred_blenders[:5]):
                    top_5_hits += 1
                
                # Hit Rate@K (binary: 1 if any positive in top-K)
                if len(set(pred_blenders) & true_blenders) > 0:
                    top_k_hits += 1

                # Recall@K (비율: top-K 내 정답 개수 / 해당 샘플 정답 개수)
                n_true = len(true_blenders)
                if n_true > 0:
                    sum_recall_1 += 1.0 if pred_blenders[0] in true_blenders else 0.0
                    sum_recall_5 += len(set(pred_blenders[:5]) & true_blenders) / n_true
                    sum_recall_k += len(set(pred_blenders) & true_blenders) / n_true
                
                # MRR: Find rank of first correct blender
                rr = 0.0
                for rank, pred_blender in enumerate(pred_blenders, start=1):
                    if pred_blender in true_blenders:
                        rr = 1.0 / rank
                        break
                reciprocal_ranks.append(rr)
                
                # NDCG@K
                dcg = 0.0
                for rank, pred_blender in enumerate(pred_blenders, start=1):
                    if pred_blender in true_blenders:
                        dcg += 1.0 / np.log2(rank + 1)
                
                idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(k, len(true_blenders)) + 1))
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_scores.append(ndcg)
                
                total_samples += 1
        
        # Calculate metrics (no_grad 블록 안에서 완료)
        hit_rate_1 = top_1_hits / total_samples if total_samples > 0 else 0.0
        hit_rate_5 = top_5_hits / total_samples if total_samples > 0 else 0.0
        hit_rate_k = top_k_hits / total_samples if total_samples > 0 else 0.0
        recall_1 = sum_recall_1 / total_samples if total_samples > 0 else 0.0
        recall_5 = sum_recall_5 / total_samples if total_samples > 0 else 0.0
        recall_k = sum_recall_k / total_samples if total_samples > 0 else 0.0
        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        ndcg_k = np.mean(ndcg_scores) if ndcg_scores else 0.0
        
        # Diversity metrics
        if recommended_blenders:
            unique_blenders = len(set(recommended_blenders))
            blender_counts = {}
            for b in recommended_blenders:
                blender_counts[b] = blender_counts.get(b, 0) + 1
            diversity_std = np.std(list(blender_counts.values()))
            diversity_mean = np.mean(list(blender_counts.values()))
            diversity_coefficient = diversity_std / diversity_mean if diversity_mean > 0 else 0.0
        else:
            unique_blenders = 0
            diversity_std = 0.0
            diversity_coefficient = 0.0

        val_loss = all_val_loss / num_batches if num_batches > 0 else 0.0

        # z_recipe norm 통계 (하이퍼볼릭 수치 안정성 모니터링)
        if recipe_norm_list:
            all_norms = torch.cat(recipe_norm_list, dim=0)
            recipe_norm_mean = all_norms.mean().item()
            recipe_norm_min = all_norms.min().item()
            recipe_norm_max = all_norms.max().item()
        else:
            recipe_norm_mean = recipe_norm_min = recipe_norm_max = 0.0

        return {
            'hit_rate@1': hit_rate_1,
            'hit_rate@5': hit_rate_5,
            'hit_rate@k': hit_rate_k,
            'recall@1': recall_1,
            'recipe_norm_mean': recipe_norm_mean,
            'recipe_norm_min': recipe_norm_min,
            'recipe_norm_max': recipe_norm_max,
            'recall@5': recall_5,
            'recall@k': recall_k,
            'mrr': mrr,
            'ndcg@k': ndcg_k,
            'unique_recommended_blenders': unique_blenders,
            'diversity_std': diversity_std,
            'diversity_coefficient': diversity_coefficient,
            'val_loss': val_loss,
            'total_samples': total_samples
        }


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    epoch: int,
    warmup_epochs: int = 7,
    initial_lr: float = 1e-6,
    target_lr: float = 0.0006,
    gradient_clip_value: float = 1.0,
    use_blender_input: bool = False,
    gradient_accumulation_steps: int = 1,
    scaler: Optional[object] = None,
    clear_cache_every: int = 0,
    group_loss_weight: float = 0.0,
    inter_group_separation_weight: float = 0.0,
    inter_group_separation_margin: float = 0.3,
    anti_collapse_weight: float = 0.0,
    anti_collapse_min_norm: float = 0.2,
    norm_penalty_weight: float = 0.0,
) -> Tuple[float, Optional[Dict[str, float]]]:
    """
    Train for one epoch with warmup, gradient clipping, and soft margin.
    gradient_accumulation_steps > 1 이면 N step마다 optimizer.step() 호출 (메모리 절감).
    Returns (train_loss, train_recipe_norm_stats) for collapse/boundary 모니터링.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    accumulation_steps = gradient_accumulation_steps
    use_amp = scaler is not None and device.type == 'cuda'
    train_recipe_norms = []  # z_recipe.norm(dim=1) per batch (하이퍼볼릭 수치 안정성)
    
    # LR은 main에서 optimizer/scheduler가 관리. train_epoch에서는 건드리지 않음.
    
    # 진행 표시줄: 한 줄에서 갱신 (stderr 사용, refresh 제거로 중복 출력 방지)
    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1} 학습",
        unit="batch",
        leave=False,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.5,
        maxinterval=2.0,
    )
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(pbar):
        # Move to device (MPS에서는 non_blocking이 오히려 느릴 수 있으므로 조건부 적용)
        use_non_blocking = device.type == 'cuda'  # CUDA에서만 non_blocking 사용
        note_indices = batch['note_indices'].to(device, non_blocking=use_non_blocking)
        blender_indices = batch['blender_indices'].to(device, non_blocking=use_non_blocking) if use_blender_input else None
        molecule_mask = batch['molecule_mask'].to(device, non_blocking=use_non_blocking).float()  # float32 고정 (float64 시 메모리 2배)
        smiles_graphs = batch['smiles_graphs']
        smiles_batch = batch['smiles_batch'].to(device, non_blocking=use_non_blocking)
        
        # Handle SMILES graphs
        if hasattr(smiles_graphs, 'to'):
            smiles_graphs = smiles_graphs.to(device, non_blocking=use_non_blocking)
        else:
            smiles_graphs.x = smiles_graphs.x.to(device, non_blocking=use_non_blocking)
            smiles_graphs.edge_index = smiles_graphs.edge_index.to(device, non_blocking=use_non_blocking)
            if hasattr(smiles_graphs, 'edge_attr') and smiles_graphs.edge_attr is not None:
                smiles_graphs.edge_attr = smiles_graphs.edge_attr.to(device, non_blocking=use_non_blocking)
        
        forward_kw = dict(
            smiles_graphs=smiles_graphs,
            smiles_batch=smiles_batch,
            note_indices=note_indices,
            blender_indices=blender_indices,
            molecule_mask=molecule_mask,
        )
        # 지문 전용: 항상 mol_fingerprints 전달
        if 'mol_fingerprints' in batch:
            forward_kw['mol_fingerprints'] = batch['mol_fingerprints'].to(device, non_blocking=use_non_blocking)
        
        # Forward pass (AMP 사용 시 autocast; PyTorch 1.x/2.x 호환)
        if use_amp:
            autocast_ctx = (
                torch.amp.autocast(device_type='cuda', dtype=torch.float16)
                if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast()
            )
            with autocast_ctx:
                z_recipe = model(**forward_kw)
                train_recipe_norms.append(z_recipe.norm(dim=1).float().mean().item())
                target_blenders = batch['target_blenders']
                target_blender_tensor = _target_blenders_to_positive_set_tensor(target_blenders, device)
                all_blender_embs = model.blender_anchors()
                loss = loss_fn(z_recipe, target_blender_tensor, all_blender_embs) / accumulation_steps
                if group_loss_weight > 0 and getattr(model, 'group_anchors', None) is not None and 'target_group' in batch:
                    target_group = batch['target_group'].to(device, non_blocking=use_non_blocking)
                    group_embs = model.group_anchors(target_group)
                    group_loss = model.manifold.dist(z_recipe, group_embs).mean() / accumulation_steps
                    loss = loss + group_loss_weight * group_loss
                if inter_group_separation_weight > 0 and getattr(model, 'group_anchors', None) is not None:
                    loss = loss + inter_group_separation_weight * model.group_anchors.separation_loss(margin=inter_group_separation_margin) / accumulation_steps
                if anti_collapse_weight > 0:
                    mean_norm = z_recipe.norm(dim=1).mean()
                    loss = loss + anti_collapse_weight * (anti_collapse_min_norm - mean_norm).clamp(min=0.0) / accumulation_steps
                if norm_penalty_weight > 0:
                    loss = loss + norm_penalty_weight * z_recipe.norm(dim=-1).mean() / accumulation_steps
        else:
            z_recipe = model(**forward_kw)
            train_recipe_norms.append(z_recipe.norm(dim=1).mean().item())
            target_blenders = batch['target_blenders']
            target_blender_tensor = _target_blenders_to_positive_set_tensor(target_blenders, device)
            all_blender_embs = model.blender_anchors()
            loss = loss_fn(z_recipe, target_blender_tensor, all_blender_embs) / accumulation_steps
            if group_loss_weight > 0 and getattr(model, 'group_anchors', None) is not None and 'target_group' in batch:
                target_group = batch['target_group'].to(device, non_blocking=use_non_blocking)
                group_embs = model.group_anchors(target_group)
                group_loss = model.manifold.dist(z_recipe, group_embs).mean() / accumulation_steps
                loss = loss + group_loss_weight * group_loss
            if inter_group_separation_weight > 0 and getattr(model, 'group_anchors', None) is not None:
                loss = loss + inter_group_separation_weight * model.group_anchors.separation_loss(margin=inter_group_separation_margin) / accumulation_steps
            if anti_collapse_weight > 0:
                mean_norm = z_recipe.norm(dim=1).mean()
                loss = loss + anti_collapse_weight * (anti_collapse_min_norm - mean_norm).clamp(min=0.0) / accumulation_steps
            if norm_penalty_weight > 0:
                loss = loss + norm_penalty_weight * z_recipe.norm(dim=-1).mean() / accumulation_steps

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation: N step마다 step & zero_grad
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                if gradient_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)  # 하이퍼볼릭 수치 안정
                scaler.step(optimizer)
                scaler.update()
            else:
                if gradient_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)  # 하이퍼볼릭 수치 안정
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        # 메모리 정리 (OOM 반복 시 유효, MPS/CUDA 모두 캐시 비움)
        if clear_cache_every > 0 and (batch_idx + 1) % clear_cache_every == 0:
            clear_device_memory(device)
        
        # 진행 표시줄 postfix만 갱신 (refresh 호출하지 않음 → 한 줄로 유지)
        if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
            avg_loss = total_loss / num_batches if num_batches > 0 else loss.item() * accumulation_steps
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # 마지막 불완전 accumulation step 처리
    if (batch_idx + 1) % accumulation_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
            if gradient_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            if gradient_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            optimizer.step()
        optimizer.zero_grad()
    
    # 평균 loss 계산
    train_loss = total_loss / num_batches if num_batches > 0 else 0.0
    if train_recipe_norms:
        train_norm_mean = float(np.mean(train_recipe_norms))
        train_norm_min = float(np.min(train_recipe_norms))
        train_norm_max = float(np.max(train_recipe_norms))
        norm_stats = {'recipe_norm_mean': train_norm_mean, 'recipe_norm_min': train_norm_min, 'recipe_norm_max': train_norm_max}
    else:
        norm_stats = None
    return train_loss, norm_stats


def main():
    """Main training function."""
    import argparse
    
    # 스크립트 파일의 디렉토리 경로를 기준으로 기본 경로 설정
    script_dir = Path(__file__).parent.absolute()
    default_data_path = script_dir.parent / 'cleaned_data' / 'cleaned_complete_data.json'
    default_vocab_path = script_dir.parent / 'feature_encoding' / 'vocabularies.json'
    default_output_dir = script_dir / 'results' / 'checkpoints_fingerprint'  # 지문 전용 결과 디렉터리
    
    parser = argparse.ArgumentParser(description='Train Hyperbolic Hypergraph Model')
    parser.add_argument('--data_path', type=str, default=str(default_data_path),
                       help=f'Path to cleaned_complete_data.json (default: {default_data_path})')
    parser.add_argument('--vocab_path', type=str, default=str(default_vocab_path),
                       help=f'Path to vocabularies.json (default: {default_vocab_path})')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training (default: 128). OOM 시 32~64, gradient_accumulation_steps로 유효 배치 유지')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps. 유효 배치 = batch_size * 이 값 (메모리 부족 시 batch_size 줄이고 4~8로 설정)')
    parser.add_argument('--low_memory', action='store_true',
                       help='저메모리 모드: batch_size=32, gradient_accumulation_steps=4, val_batch_size=32, 소규모 검증')
    parser.add_argument('--val_batch_size', type=int, default=None,
                       help='Batch size for validation (default: same as batch_size, larger = faster validation)')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate (하이퍼볼릭 3e-5 고정 권장, Warmup/Scheduler 없이)')
    parser.add_argument('--margin', type=float, default=0.4,
                       help='Triplet margin (기본 0.4, 학습 신호 강화)')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='랭킹/대조 학습 temperature (낮을수록 조향사 구분 날카로움, 기본 0.07)')
    parser.add_argument('--loss', type=str, default='triplet',
                       choices=['triplet', 'bpr', 'circle', 'mean_pos', 'max_margin'],
                       help='Loss: triplet, bpr, circle, mean_pos=Mean Positive Distance, max_margin=Max-Margin Ranking')
    parser.add_argument('--group_loss_weight', type=float, default=0.1,
                       help='계층 보조 loss 가중치: 레시피→그룹 앵커 당김. 너무 크면 유저-아이템 거리보다 그룹 뭉침에만 급해 HR 하락 (기본 0.1)')
    parser.add_argument('--inter_group_separation_weight', type=float, default=0.005,
                       help='그룹 앵커 분리 loss 가중치 (과적합 완화, 0=끔, 기본 0.005)')
    parser.add_argument('--inter_group_separation_margin', type=float, default=0.2,
                       help='Inter-group separation 목표 최소 거리 (기본 0.2)')
    parser.add_argument('--anti_collapse_weight', type=float, default=0.01,
                       help='레시피 norm 하한 유도: 평균 norm이 min 이하면 패널티 (중심 밀림 완화, 0=비활성, 기본 0.01)')
    parser.add_argument('--anti_collapse_min_norm', type=float, default=0.2,
                       help='Anti-collapse 목표 최소 평균 norm (기본 0.2)')
    parser.add_argument('--norm_penalty_weight', type=float, default=0.025,
                       help='z_recipe norm 패널티 (0.025=권장·경계 쏠림 억제, 0.01=완화)')
    parser.add_argument('--margin_start', type=float, default=0.2,
                       help='Starting margin for scheduling (0.2=학습 신호 강화)')
    parser.add_argument('--distance_scale', type=float, default=1.0,
                       help='Scale for hyperbolic distances in loss (1.0=full gradient)')
    parser.add_argument('--num_hard_negatives', type=int, default=8,
                       help='Number of hard negatives (8=안정, 12=더 hard)')
    parser.add_argument('--c', type=float, default=0.5,
                       help='Poincaré curvature (0.5=권장·수치 안정·유클리드 섞임, 0.3=더 완만)')
    parser.add_argument('--learnable_curvature', action='store_true', default=False,
                       help='Learnable curvature (기본: False, 고정 권장)')
    parser.add_argument('--no_learnable_curvature', action='store_false', dest='learnable_curvature',
                       help='Curvature 고정 (기본값, 수치 안정)')
    parser.add_argument('--dropout', type=float, default=0.45,
                       help='Dropout rate (기본 0.45: 일반화·과적합 억제)')
    parser.add_argument('--weight_decay', type=float, default=2e-4,
                       help='Weight decay (기본 2e-4: 일반화·과적합 억제)')
    parser.add_argument('--gradient_clip_value', type=float, default=1.0,
                       help='Gradient clipping (하이퍼볼릭 필수 권장 1.0, 0=비활성화)')
    parser.add_argument('--use_riemannian_optimizer', action='store_true', default=False,
                       help='Use RiemannianAdam (geoopt): 매니폴드 밖으로 나가면 proj로 복귀')
    parser.add_argument('--lr_max', type=float, default=2e-5,
                       help='Maximum learning rate (하이퍼볼릭 3e-5 고정 권장)')
    parser.add_argument('--lr_min', type=float, default=1e-5,
                       help='Minimum learning rate (기본: 1e-5)')
    parser.add_argument('--plateau_min_lr', type=float, default=0.0001,
                       help='ReduceLROnPlateau 최소 LR (default: 0.0001)')
    parser.add_argument('--plateau_patience', type=int, default=4,
                       help='LR 줄이기 전 기다릴 에폭 수 (default: 4, 15에폭 전에 LR 유지)')
    parser.add_argument('--no_lr_scheduler', action='store_false', dest='use_lr_scheduler',
                       help='ReduceLROnPlateau 비활성화 (LR 고정)')
    parser.add_argument('--scheduler_metric', type=str, default='hr10', choices=['hr10', 'val_loss'],
                       help='ReduceLROnPlateau 기준: hr10=HR@10(기본), val_loss=Val Loss 발산 시 LR 빨리 낮춤')
    parser.add_argument('--lr_lock', action='store_true',
                       help='Lock learning rate to lr_max')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Early stopping patience (기본 5: 개선 없으면 바로 멈추고 그때 가중치 저장)')
    parser.add_argument('--early_stopping_metric', type=str, default='val_loss', choices=['val_loss', 'hr10'],
                       help='Early stopping 기준: val_loss=Loss 낮을 때 멈춤(수치 안정·Test 일반화 권장), hr10=Val HR@10')
    parser.add_argument('--fast_mode', action='store_true',
                       help='속도 우선 모드 (cudnn benchmark on, deterministic off)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='DataLoader num_workers (기본: MPS=0, CUDA=최대8). OOM/killed 시 0 권장')
    parser.add_argument('--pin_memory', action='store_true',
                       help='DataLoader pin_memory 사용')
    parser.add_argument('--persistent_workers', action='store_true',
                       help='DataLoader persistent_workers 사용')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                       help='DataLoader prefetch_factor (num_workers>0, default: 2, MPS에서는 낮게 권장)')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                       help='Warmup 에폭 수 (0=비활성 권장, 높은 LR이 boundary로 튐)')
    parser.add_argument('--warmup_initial_lr', type=float, default=1e-6,
                       help='Warmup 시작 학습률 (기본 1e-6)')
    parser.add_argument('--use_soft_margin', action='store_true',
                       help='Use soft margin loss')
    parser.add_argument('--no_soft_margin', dest='use_soft_margin', action='store_false',
                       help='Disable soft margin loss (default: soft margin is enabled)')
    parser.set_defaults(use_soft_margin=True, use_lr_scheduler=False)
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for checkpoints (default: script_dir/results/checkpoints)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/mps/cpu, auto-detect if None)')
    parser.add_argument('--train_size', type=int, default=TRAIN_SIZE,
                       help=f'Training set size (default: {TRAIN_SIZE})')
    parser.add_argument('--val_size', type=int, default=VAL_SIZE,
                       help=f'Validation set size (default: {VAL_SIZE})')
    parser.add_argument('--test_size', type=int, default=TEST_SIZE,
                       help=f'Test set size (default: {TEST_SIZE})')
    parser.add_argument('--val_use_original', action='store_true',
                       help='Use original (non-augmented) data for validation set')
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED,
                       help=f'Random seed for reproducibility (default: {RANDOM_SEED})')
    parser.add_argument('--load_dataset', type=str, default=None,
                       help='Load pre-saved datasets from directory (skips data generation)')
    parser.add_argument('--fp_dim', type=int, default=2048,
                        help='Morgan/ECFP 지문 비트 수 (기본 2048)')
    parser.add_argument('--fp_radius', type=int, default=2,
                        help='Morgan radius (2=ECFP4, 3=ECFP6)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing (0=no smoothing, 성능 저하 시 0.1 시도)')
    parser.add_argument('--use_confused_negatives', action='store_true', default=False,
                       help='Use confused blenders as hard negatives (시나리오 C)')
    parser.add_argument('--no_use_confused_negatives', dest='use_confused_negatives', action='store_false',
                       help='Do not use confused negatives (default)')
    parser.add_argument('--enable_cheat_key_detection', action='store_true', default=True,
                       help='Enable cheat key molecule detection and masking (시나리오 A)')
    parser.add_argument('--enable_duplicate_filtering', action='store_true', default=True,
                       help='Enable duplicate combination filtering (시나리오 B)')
    parser.add_argument('--molecule_dropout_rate', type=float, default=0.15,
                       help='Molecule dropout rate for training (0.1~0.2 recommended, default: 0.15)')
    parser.add_argument('--filter_common_molecules', action='store_true', default=False,
                       help='Filter out common molecules (used by >80% blenders) during training')
    parser.add_argument('--use_blender_input', action='store_true', default=False,
                       help='Use blender indices as input (주의: 라벨 누수 가능성)')
    parser.add_argument('--downsample_top_blender', action='store_true', default=True,
                       help='Downsample top blender to reduce imbalance (default: True)')
    parser.add_argument('--downsample_val_top_blender', action='store_true', default=True,
                       help='Also downsample validation top blender (default: True)')
    parser.add_argument('--top_blender_id', type=int, default=0,
                       help='Top blender ID to downsample (default: 0)')
    parser.add_argument('--top_blender_max_ratio', type=float, default=0.1,
                       help='Max ratio for top blender after downsampling (default: 0.1)')
    parser.add_argument('--val_interval', type=int, default=1,
                       help='Validation interval (default: 1 = 매 에폭마다 검증)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau'],
                       help='Learning rate scheduler type (default: cosine)')
    parser.add_argument('--embed_dim', type=int, default=256,
                       help='임베딩 차원(그릇 크기). 64=작게(과적합 완화), 128=기본, 192/256=크게(데이터 많을 때)')
    parser.add_argument('--small_model', action='store_true',
                       help='소형 모델 (embed_dim 64): 메모리 절감, VRAM 부족 시 사용')
    parser.add_argument('--use_amp', action='store_true',
                       help='CUDA에서만: Mixed precision (FP16) 사용 → 메모리 절감 및 속도 향상')
    parser.add_argument('--clear_cache_every', type=int, default=0,
                       help='매 N 배치마다 CUDA cache 정리 (0=비활성화). OOM 반복 시 20~50 권장')
    
    args = parser.parse_args()
    
    # output_dir 기본값 설정 (스크립트 위치 기준)
    if args.output_dir is None:
        args.output_dir = str(default_output_dir)

    # Learning rate 범위 (하이퍼볼릭: 3e-5 고정 권장, lr_min은 1e-5까지 허용)
    args.lr_min = max(1e-5, args.lr_min)
    args.lr_max = min(0.001, args.lr_max)
    if args.lr_lock:
        args.learning_rate = args.lr_max
    else:
        args.learning_rate = max(args.lr_min, min(args.learning_rate, args.lr_max))
    
    if not hasattr(args, 'use_lr_scheduler'):
        args.use_lr_scheduler = False  # 하이퍼볼릭: LR 고정 권장
    
    # 모든 random seed 설정 (데이터 무결성 보장)
    set_all_seeds(args.random_seed, fast_mode=args.fast_mode)
    
    # Device setup (우선순위: CUDA > MPS > CPU)
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    # 저메모리 모드: 배치 32 + accumulation 2 → 유효 배치 64, 메모리 안전권
    if getattr(args, 'low_memory', False):
        args.batch_size = 32
        if not hasattr(args, 'gradient_accumulation_steps') or args.gradient_accumulation_steps == 1:
            args.gradient_accumulation_steps = 2
        args.val_batch_size = args.val_batch_size or 32
        print(f"  📉 저메모리 모드: batch_size=32, gradient_accumulation_steps={args.gradient_accumulation_steps} (유효 배치={32 * args.gradient_accumulation_steps}, 안정성 확보)")
    if not hasattr(args, 'gradient_accumulation_steps'):
        args.gradient_accumulation_steps = 1
    if not hasattr(args, 'clear_cache_every'):
        args.clear_cache_every = 0
    if not hasattr(args, 'use_amp'):
        args.use_amp = False
    if args.use_amp and device.type != 'cuda':
        args.use_amp = False
        print(f"  ⚠️  AMP는 CUDA에서만 지원. 비활성화합니다.")
    # MPS: 배치 상한 64 (OOM 시 --low_memory 또는 --batch_size 32 사용). LR은 올리지 않음 (하이퍼볼릭 안정)
    if device.type == 'mps':
        args.batch_size = min(args.batch_size, 64)
        if args.batch_size < 32:
            args.batch_size = 32
        print(f"  ⚠️  MPS: batch_size 상한 64 (현재 {args.batch_size}) | LR: {args.learning_rate:.6f} (고정)")
    # DataLoader 기본값: CUDA만 멀티워커, MPS/CPU는 0 (killed·loky 경고 방지)
    if args.num_workers is None:
        cpu_count = os.cpu_count() or 2
        if device.type == 'cuda':
            args.num_workers = min(8, cpu_count)
        else:
            args.num_workers = 0  # MPS·CPU: 메모리·멀티프로세싱 이슈 방지
    
    # Device 이름 표시
    device_name = str(device)
    if device.type == 'mps':
        device_name = 'MPS (Apple GPU)'
    elif device.type == 'cuda':
        device_name = 'CUDA (NVIDIA GPU)'
    else:
        device_name = 'CPU'
    print(f"Device: {device_name}")
    
    # 좀비 메모리 정리: 껐다 켜서 조각난 메모리 비우기 (2GB×4조각 → 4GB 할당 실패 방지)
    clear_device_memory(device)
    print(f"   ✓ 메모리 캐시 정리 완료 (이전 실행 잔여/조각난 메모리 비움)")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 로드: 이미 만든 results/checkpoints/datasets 우선 사용 (새로 생성하지 않음)
    dataset_dir = None
    if args.load_dataset:
        dataset_dir = Path(args.load_dataset)
    else:
        script_dir = Path(__file__).parent.absolute()
        shared_datasets = script_dir / 'results' / 'checkpoints' / 'datasets'
        out = Path(args.output_dir)
        # 1) 공용 경로 먼저 → 2) output_dir/datasets (GNN/지문 모두 동일 데이터 사용)
        auto_candidates = [shared_datasets, out / 'datasets']
        for auto_dataset_dir in auto_candidates:
            has_train = (auto_dataset_dir / 'train_combinations.json').exists()
            has_val = (auto_dataset_dir / 'val_combinations.json').exists()
            has_test_orig = (auto_dataset_dir / 'test_combinations_original.json').exists()
            has_test = (auto_dataset_dir / 'test_combinations.json').exists()
            if has_train and has_val and (has_test_orig or has_test):
                split_info_path = auto_dataset_dir / 'split_info.json'
                if split_info_path.exists():
                    with open(split_info_path, 'r', encoding='utf-8') as f:
                        split_info = json.load(f)
                    split_train_size = split_info.get('train_size')
                    split_val_size = split_info.get('val_size')
                    split_test_size = split_info.get('test_original_size', split_info.get('test_size', 0))
                    split_seed = split_info.get('random_seed')
                    args.train_size = split_train_size if split_train_size is not None else args.train_size
                    args.val_size = split_val_size if split_val_size is not None else args.val_size
                    args.test_size = split_test_size if split_test_size is not None else args.test_size
                    args.random_seed = split_seed if split_seed is not None else args.random_seed
                dataset_dir = auto_dataset_dir
                print(f"💾 저장된 데이터셋 자동 발견: {dataset_dir}")
                break
            elif has_train and has_val:
                dataset_dir = auto_dataset_dir
                print(f"💾 저장된 데이터셋 발견 (split_info 없음): {dataset_dir}")
                break
    
    if dataset_dir and dataset_dir.exists():
        with open(dataset_dir / 'train_combinations.json', 'r', encoding='utf-8') as f:
            train_combinations = json.load(f)
        with open(dataset_dir / 'val_combinations.json', 'r', encoding='utf-8') as f:
            val_combinations = json.load(f)
        with open(dataset_dir / 'test_combinations.json', 'r', encoding='utf-8') as f:
            test_combinations = json.load(f)
        
        # 순수 원본 테스트셋 로드 (필수)
        original_test_path = dataset_dir / 'test_combinations_original.json'
        if original_test_path.exists():
            with open(original_test_path, 'r', encoding='utf-8') as f:
                test_combinations_original = json.load(f)
            test_combinations = test_combinations_original  # 원본 데이터 사용
            print(f"데이터셋 로드: Train={len(train_combinations):,}, Val={len(val_combinations):,}, Test(원본)={len(test_combinations_original):,} ⭐")
        else:
            print(f"데이터셋 로드: Train={len(train_combinations):,}, Val={len(val_combinations):,}, Test={len(test_combinations):,}")
            print(f"  ⚠️  순수 원본 테스트셋이 없습니다. 새로 생성됩니다.")
            test_combinations_original = []
        
        # Load vocabulary
        records, vocab_data = load_data(args.data_path, args.vocab_path)

        # ✅ 저장된 데이터셋에서도 누수 진단 실행 (한 번만 실행)
        # ⚡ 성능 최적화: export는 필요할 때만 실행 (주석 처리)
        # export_combination_format(train_combinations, dataset_dir / "train_export.json")
        # export_combination_format(val_combinations, dataset_dir / "val_export.json")
        # export_combination_format(test_combinations, dataset_dir / "test_export.json")

        # ① 다운샘플링 적용 (저장된 데이터에도 적용 가능)
        if args.downsample_top_blender:
            train_combinations = downsample_blender_combinations(
                train_combinations,
                blender_id=args.top_blender_id,
                max_ratio=args.top_blender_max_ratio,
                seed=args.random_seed
            )
        if args.downsample_val_top_blender:
            val_combinations = downsample_blender_combinations(
                val_combinations,
                blender_id=args.top_blender_id,
                max_ratio=args.top_blender_max_ratio,
                seed=args.random_seed + 1
            )
    else:
        # Load data
        records, vocab_data = load_data(args.data_path, args.vocab_path)
        # 그룹 vocab 및 blender→group 매핑 구축 (계층 보조 loss용)
        build_group_vocab_and_blender_to_group(records, vocab_data)

        # Create recipe combinations
        # 분할: Train (증강) 60,000 / Val (원본 또는 증강) 10,000 / Test (원본) 11,248

        # 원본 분자 레코드를 Train/Val/Test로 먼저 분리 (엄격한 분리 보장)
        print(f"\n원본 분자 레코드 분리 중...")
        np.random.seed(args.random_seed)
        all_molecule_indices = list(range(len(records)))
        np.random.shuffle(all_molecule_indices)
        
        # Train/Val/Test로 분자 레코드 분리
        # Train: 75%, Val: 12.5%, Test: 12.5%
        train_mol_end = int(len(all_molecule_indices) * 0.75)
        val_mol_end = train_mol_end + int(len(all_molecule_indices) * 0.125)
        
        train_molecule_indices = set(all_molecule_indices[:train_mol_end])
        val_molecule_indices = set(all_molecule_indices[train_mol_end:val_mol_end])
        test_molecule_indices = set(all_molecule_indices[val_mol_end:])
        
        print(f"  Train 분자: {len(train_molecule_indices):,}개")
        print(f"  Val 분자: {len(val_molecule_indices):,}개")
        print(f"  Test 분자: {len(test_molecule_indices):,}개")
        
        # 겹침 확인
        train_val_overlap = train_molecule_indices & val_molecule_indices
        train_test_overlap = train_molecule_indices & test_molecule_indices
        val_test_overlap = val_molecule_indices & test_molecule_indices
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            raise ValueError(f"❌ 분자 레코드 분리 실패: 겹침이 있습니다!")
        print(f"  ✅ 분자 레코드 완전 분리 확인")
        
        # 각 세트에 해당하는 레코드만 사용하여 조합 생성
        train_records = [records[i] for i in train_molecule_indices]
        val_records = [records[i] for i in val_molecule_indices]
        test_records = [records[i] for i in test_molecule_indices]
        
        # 인덱스 매핑 생성 (원본 인덱스 추적)
        train_idx_mapping = {new_idx: orig_idx for new_idx, orig_idx in enumerate(sorted(train_molecule_indices))}
        val_idx_mapping = {new_idx: orig_idx for new_idx, orig_idx in enumerate(sorted(val_molecule_indices))}
        test_idx_mapping = {new_idx: orig_idx for new_idx, orig_idx in enumerate(sorted(test_molecule_indices))}
        
        # 1. Train용 증강된 조합 생성
        print(f"\nTrain 조합 생성 중 (목표: {args.train_size:,}개, 증강 포함)...", end=" ", flush=True)
        train_combinations_raw = create_recipe_combinations(
            train_records, 
            vocab_data, 
            max_samples=args.train_size + int(args.train_size * 0.1),  # 10% 여유분
            seed=args.random_seed,
            use_augmentation=True,  # 증강 포함
            molecule_idx_mapping=train_idx_mapping,  # 원본 인덱스 매핑
            enable_cheat_key_detection=getattr(args, 'enable_cheat_key_detection', True),  # 시나리오 A
            enable_duplicate_filtering=getattr(args, 'enable_duplicate_filtering', True),  # 시나리오 B
            use_inverse_blender_sampling=True  # ② 희귀 조향사 우선 샘플링
        )
        
        if len(train_combinations_raw) == 0:
            raise ValueError("❌ Train 조합 생성 실패: 조합이 생성되지 않았습니다.")
        
        # 정확한 크기로 제한
        np.random.seed(args.random_seed)
        train_indices = np.random.permutation(len(train_combinations_raw))
        train_combinations = [train_combinations_raw[i] for i in train_indices[:min(args.train_size, len(train_combinations_raw))]]
        print(f"완료: {len(train_combinations):,}개")
        
        # 2. Validation용 조합 생성 (원본 또는 증강)
        if args.val_use_original:
            print(f"Val 조합 생성 중 (목표: {args.val_size:,}개, 순수 원본)...", end=" ", flush=True)
            val_combinations_raw = create_recipe_combinations(
                val_records,
                vocab_data,
                max_samples=args.val_size + int(args.val_size * 0.1),
                seed=args.random_seed + 1000,  # 다른 seed
                use_augmentation=False,  # 원본 데이터
                molecule_idx_mapping=val_idx_mapping,
                use_inverse_blender_sampling=True
            )
        else:
            print(f"Val 조합 생성 중 (목표: {args.val_size:,}개, 증강 포함)...", end=" ", flush=True)
            val_combinations_raw = create_recipe_combinations(
                val_records,
                vocab_data,
                max_samples=args.val_size + int(args.val_size * 0.1),
                seed=args.random_seed + 1000,  # 다른 seed
                use_augmentation=True,  # 증강 포함
                molecule_idx_mapping=val_idx_mapping,
                use_inverse_blender_sampling=True
            )
        
        if len(val_combinations_raw) == 0:
            raise ValueError("❌ Val 조합 생성 실패: 조합이 생성되지 않았습니다.")
        
        # 정확한 크기로 제한
        np.random.seed(args.random_seed + 1000)
        val_indices = np.random.permutation(len(val_combinations_raw))
        val_combinations = [val_combinations_raw[i] for i in val_indices[:min(args.val_size, len(val_combinations_raw))]]
        print(f"완료: {len(val_combinations):,}개")
        
        # 3. Test용 순수 원본 데이터 조합 생성 (최종 평가용)
        print(f"Test 조합 생성 중 (목표: {args.test_size:,}개, 순수 원본)...", end=" ", flush=True)
        test_combinations_original_raw = create_recipe_combinations(
            test_records,
            vocab_data,
            max_samples=args.test_size + int(args.test_size * 0.1),
            seed=args.random_seed + 9999,  # 다른 seed로 생성
            use_augmentation=False,  # 증강 없음 (순수 원본)
            molecule_idx_mapping=test_idx_mapping,
            use_inverse_blender_sampling=True
        )
        
        if len(test_combinations_original_raw) == 0:
            raise ValueError("❌ Test 조합 생성 실패: 조합이 생성되지 않았습니다.")
        
        # 정확한 크기로 제한
        np.random.seed(args.random_seed + 9999)
        test_indices = np.random.permutation(len(test_combinations_original_raw))
        test_combinations_original = [test_combinations_original_raw[i] for i in test_indices[:min(args.test_size, len(test_combinations_original_raw))]]
        print(f"완료: {len(test_combinations_original):,}개")
        
        # 하위 호환성을 위해 test_combinations 설정
        test_combinations = test_combinations_original
        
        print(f"\n데이터 분할 완료:")
        print(f"  Train (증강): {len(train_combinations):,}개")
        print(f"  Val ({'순수 원본' if args.val_use_original else '증강'}): {len(val_combinations):,}개")
        print(f"  Test (순수 원본): {len(test_combinations_original):,}개 ⭐ 최종 평가용")
        
        # ① 다운샘플링 (최빈 블렌더 비율 10% 이하로)
        if args.downsample_top_blender:
            train_combinations = downsample_blender_combinations(
                train_combinations,
                blender_id=args.top_blender_id,
                max_ratio=args.top_blender_max_ratio,
                seed=args.random_seed
            )
        if args.downsample_val_top_blender:
            val_combinations = downsample_blender_combinations(
                val_combinations,
                blender_id=args.top_blender_id,
                max_ratio=args.top_blender_max_ratio,
                seed=args.random_seed + 1
            )

        # ⚡ 성능 최적화: 누수 진단은 첫 실행 시에만 (주석 처리하여 속도 향상)
        # run_leakage_diagnostics(train_combinations, val_combinations, vocab_data)
        
        # 원본 분자 레코드 ID 분리 검증
        def extract_molecule_ids(combinations_list):
            """조합에서 사용된 원본 분자 레코드 ID 추출"""
            molecule_ids = set()
            for combo in combinations_list:
                if 'original_molecule_ids' in combo:
                    molecule_ids.update(combo['original_molecule_ids'])
                elif 'molecules' in combo:
                    # 하위 호환성: molecules에서 원본 ID 추출 시도
                    for mol in combo['molecules']:
                        if 'original_idx' in mol:
                            molecule_ids.add(mol['original_idx'])
            return molecule_ids
        
        train_molecule_ids = extract_molecule_ids(train_combinations)
        val_molecule_ids = extract_molecule_ids(val_combinations)
        test_molecule_ids = extract_molecule_ids(test_combinations_original)
        
        # 겹침 확인
        train_val_overlap = train_molecule_ids & val_molecule_ids
        train_test_overlap = train_molecule_ids & test_molecule_ids
        val_test_overlap = val_molecule_ids & test_molecule_ids
        
        print(f"\n📊 원본 분자 레코드 ID 분리 검증:")
        print(f"  Train 사용 분자: {len(train_molecule_ids):,}개")
        print(f"  Val 사용 분자: {len(val_molecule_ids):,}개")
        print(f"  Test 사용 분자: {len(test_molecule_ids):,}개")
        print(f"  Train-Val 겹침: {len(train_val_overlap):,}개", end="")
        if len(train_val_overlap) > 0:
            print(f" ⚠️  경고: 겹치는 분자가 있습니다!")
        else:
            print(f" ✅ 완전 분리")
        print(f"  Train-Test 겹침: {len(train_test_overlap):,}개", end="")
        if len(train_test_overlap) > 0:
            print(f" ⚠️  경고: 겹치는 분자가 있습니다!")
        else:
            print(f" ✅ 완전 분리")
        print(f"  Val-Test 겹침: {len(val_test_overlap):,}개", end="")
        if len(val_test_overlap) > 0:
            print(f" ⚠️  경고: 겹치는 분자가 있습니다!")
        else:
            print(f" ✅ 완전 분리")
        
        # 데이터 검증 (경고만 출력, 학습은 진행)
        if len(train_combinations) < args.train_size * 0.9:
            print(f"⚠️  경고: 학습 데이터가 부족합니다. 목표: {args.train_size:,}개, 생성: {len(train_combinations):,}개")
        if len(val_combinations) < args.val_size * 0.9:
            print(f"⚠️  경고: 검증 데이터가 부족합니다. 목표: {args.val_size:,}개, 생성: {len(val_combinations):,}개")
        if len(test_combinations_original) < args.test_size * 0.9:
            print(f"⚠️  경고: 테스트 데이터가 부족합니다. 목표: {args.test_size:,}개, 생성: {len(test_combinations_original):,}개")
        
        # 최소한의 데이터는 있어야 함
        if len(train_combinations) == 0:
            raise ValueError("❌ 학습 데이터가 없습니다. 조합 생성에 실패했습니다.")
        if len(val_combinations) == 0:
            raise ValueError("❌ 검증 데이터가 없습니다. 조합 생성에 실패했습니다.")
        
        # 데이터셋 자동 저장 (항상 저장하여 다음 실행 시 재사용) - 비교 실험을 위해 필수
        dataset_dir = Path(args.output_dir) / 'datasets'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 데이터셋 저장 중... (비교 실험을 위해 필수)")
        print(f"   저장 경로: {dataset_dir}")

        # 출력 포맷 저장 (요청된 형식)
        export_combination_format(train_combinations, dataset_dir / "train_export.json")
        export_combination_format(val_combinations, dataset_dir / "val_export.json")
        export_combination_format(test_combinations, dataset_dir / "test_export.json")
        
        # NumPy 타입을 Python 기본 타입으로 변환
        train_combinations_serializable = convert_to_json_serializable(train_combinations)
        val_combinations_serializable = convert_to_json_serializable(val_combinations)
        test_combinations_serializable = convert_to_json_serializable(test_combinations)
        test_combinations_original_serializable = convert_to_json_serializable(test_combinations_original)
        
        # 파일 저장 (명시적으로 flush하고 검증)
        train_path = dataset_dir / 'train_combinations.json'
        val_path = dataset_dir / 'val_combinations.json'
        test_path = dataset_dir / 'test_combinations.json'
        test_orig_path = dataset_dir / 'test_combinations_original.json'
        
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_combinations_serializable, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())  # 디스크에 강제 쓰기
        print(f"   ✓ Train 저장 완료: {len(train_combinations):,}개 → {train_path}")
        
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_combinations_serializable, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        print(f"   ✓ Val 저장 완료: {len(val_combinations):,}개 → {val_path}")
        
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_combinations_serializable, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        print(f"   ✓ Test 저장 완료: {len(test_combinations):,}개 → {test_path}")
        
        with open(test_orig_path, 'w', encoding='utf-8') as f:
            json.dump(test_combinations_original_serializable, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        print(f"   ✓ Test(원본) 저장 완료: {len(test_combinations_original):,}개 → {test_orig_path}")
        
        # 분할 정보 저장
        split_info = {
            'train_size': len(train_combinations),
            'val_size': len(val_combinations),
            'test_size': len(test_combinations),
            'test_original_size': len(test_combinations_original),
            'random_seed': args.random_seed,
            'total_combinations': len(train_combinations) + len(val_combinations) + len(test_combinations_original),
            'output_dir': str(args.output_dir),  # 저장 위치 기록
            'note': 'test_combinations_original.json은 증강되지 않은 순수 원본 데이터입니다. 최종 평가에 사용하세요.'
        }
        split_info_path = dataset_dir / 'split_info.json'
        with open(split_info_path, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        print(f"   ✓ Split info 저장 완료: {split_info_path}")
        
        # 저장 검증 (파일이 실제로 존재하는지 확인)
        all_files_exist = (
            train_path.exists() and train_path.stat().st_size > 0 and
            val_path.exists() and val_path.stat().st_size > 0 and
            test_path.exists() and test_path.stat().st_size > 0 and
            test_orig_path.exists() and test_orig_path.stat().st_size > 0 and
            split_info_path.exists() and split_info_path.stat().st_size > 0
        )
        
        if all_files_exist:
            print(f"\n✅ 데이터셋 저장 완료 및 검증 성공!")
            print(f"   저장 위치: {dataset_dir.absolute()}")
            print(f"   다음 실행 시 자동으로 이 데이터셋을 사용합니다.")
            print(f"   ⭐ 비교 실험 시 동일한 데이터셋을 사용하려면 이 경로를 지정하세요: --load_dataset {dataset_dir}")
            # 베이스라인이 이 데이터를 쓰도록 안내 (dataset_dir의 부모 = output_dir)
            _parent = dataset_dir.parent
            print(f"   ⭐ 베이스라인과 동일 데이터로 실행: python run_baselines.py --dataset_dir {_parent}")
        else:
            print(f"\n⚠️  경고: 일부 파일이 저장되지 않았을 수 있습니다. 확인이 필요합니다.")
            missing = []
            if not train_path.exists() or train_path.stat().st_size == 0:
                missing.append('train_combinations.json')
            if not val_path.exists() or val_path.stat().st_size == 0:
                missing.append('val_combinations.json')
            if not test_path.exists() or test_path.stat().st_size == 0:
                missing.append('test_combinations.json')
            if not test_orig_path.exists() or test_orig_path.stat().st_size == 0:
                missing.append('test_combinations_original.json')
            if not split_info_path.exists() or split_info_path.stat().st_size == 0:
                missing.append('split_info.json')
            print(f"   누락된 파일: {', '.join(missing)}")
            raise RuntimeError(f"❌ 데이터셋 저장 실패: 일부 파일이 저장되지 않았습니다.")
    
    # Create datasets (지문 전용: ECFP/Morgan, GNN/프리컴퓨팅 미사용)
    molecule_dropout_rate = getattr(args, 'molecule_dropout_rate', 0.15)
    fp_dim = getattr(args, 'fp_dim', 2048)
    fp_radius = getattr(args, 'fp_radius', 2)
    print(f"  🧪 분자 지문 모드: ECFP/Morgan (fp_dim={fp_dim}, radius={fp_radius})")
    train_dataset = HyperbolicRecipeDataset(
        records=train_combinations,
        vocab_data=vocab_data,
        max_molecules=10,
        max_notes_per_molecule=20,
        max_blenders_per_molecule=10,
        mode="train",
        molecule_dropout_rate=molecule_dropout_rate,
        precomputed_path=None,
        use_fingerprint=True,
        fp_dim=fp_dim,
        fp_radius=fp_radius,
    )
    
    val_dataset = HyperbolicRecipeDataset(
        records=val_combinations,
        vocab_data=vocab_data,
        max_molecules=10,
        max_notes_per_molecule=20,
        max_blenders_per_molecule=10,
        mode="val",
        precomputed_path=None,
        use_fingerprint=True,
        fp_dim=fp_dim,
        fp_radius=fp_radius,
    )
    
    # Create data loaders
    # Train loader의 shuffle을 위해 generator 설정 (재현성 보장)
    generator = torch.Generator()
    generator.manual_seed(args.random_seed)
    
    train_loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_hyperbolic_recipes,
        generator=generator,  # 재현성을 위한 generator
        drop_last=True  # ⚡ 마지막 불완전 배치 제거로 성능 향상
    )
    # MPS에서는 num_workers=0 강제 (멀티프로세싱이 문제를 일으킬 수 있음)
    if device.type == 'mps':
        train_loader_kwargs["num_workers"] = 0
        print(f"  ⚠️  MPS 사용: num_workers=0으로 설정 (안정성 우선)")
    elif args.num_workers > 0:
        train_loader_kwargs.update({
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory if device.type != 'mps' else False,  # MPS는 pin_memory 비활성화
            "persistent_workers": args.persistent_workers if args.num_workers > 0 else False,
            "prefetch_factor": args.prefetch_factor
        })
    else:
        train_loader_kwargs["num_workers"] = 0

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    
    # DataLoader 설정 출력 (디버깅용)
    acc = getattr(args, 'gradient_accumulation_steps', 1)
    print(f"📦 DataLoader 설정:")
    print(f"   - 배치 크기: {args.batch_size}" + (f" (유효 배치: {args.batch_size * acc})" if acc > 1 else ""))
    print(f"   - num_workers: {train_loader_kwargs.get('num_workers', 0)}")
    print(f"   - 총 배치 수: {len(train_loader)}")
    print(f"   - 첫 배치 로딩 중...")
    
    # 검증 배치 크기: MPS는 메모리 제한으로 64 이하
    if args.val_batch_size is not None:
        val_batch_size = args.val_batch_size
    else:
        val_batch_size = args.batch_size * 4
        if device.type == 'mps':
            val_batch_size = min(val_batch_size, 64)  # MPS OOM 방지
    
    val_loader_kwargs = dict(
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_hyperbolic_recipes
    )
    # MPS에서는 num_workers=0 강제 (멀티프로세싱이 문제를 일으킬 수 있음)
    if device.type == 'mps':
        val_loader_kwargs["num_workers"] = 0
    elif args.num_workers > 0:
        val_loader_kwargs.update({
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory if device.type != 'mps' else False,  # MPS는 pin_memory 비활성화
            "persistent_workers": args.persistent_workers if args.num_workers > 0 else False,
            "prefetch_factor": args.prefetch_factor
        })
    else:
        val_loader_kwargs["num_workers"] = 0

    val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    
    # 테스트 세트 로더 (베이스라인과 공정 비교용: 동일 split의 test에서 HR@10 보고)
    test_dataset = HyperbolicRecipeDataset(
        records=test_combinations,
        vocab_data=vocab_data,
        max_molecules=10,
        max_notes_per_molecule=20,
        max_blenders_per_molecule=10,
        mode="test",
        precomputed_path=None,
        use_fingerprint=True,
        fp_dim=fp_dim,
        fp_radius=fp_radius,
    )
    test_loader = DataLoader(test_dataset, **val_loader_kwargs)
    
    # Get vocabulary sizes (다양한 구조 지원)
    blender_data = vocab_data.get('blenders', {})
    if isinstance(blender_data, dict):
        if 'vocab' in blender_data:
            num_blenders = len(blender_data['vocab'])
        elif 'to_idx' in blender_data:
            num_blenders = len(blender_data['to_idx'])
        else:
            num_blenders = len(blender_data)
    elif isinstance(blender_data, list):
        num_blenders = len(blender_data)
    else:
        num_blenders = 0
    
    notes_data = vocab_data.get('notes', {})
    if isinstance(notes_data, dict):
        if 'vocab' in notes_data:
            vocab_size = len(notes_data['vocab'])
        elif 'to_idx' in notes_data:
            vocab_size = len(notes_data['to_idx'])
        else:
            vocab_size = len(notes_data)
    elif isinstance(notes_data, list):
        vocab_size = len(notes_data)
    else:
        vocab_size = 0
    
    num_groups = 1
    if isinstance(vocab_data.get('groups'), dict):
        num_groups = max(1, vocab_data['groups'].get('size', 1))
    print(f"Vocabulary 크기: Notes={vocab_size}, Blenders={num_blenders}, Groups={num_groups}")
    # 추천(검색): 후보 조향사 N명 중 상위 k명 랭킹. 랜덤 HR@10 ≈ 10/N% → 모델이 2~8%면 랜덤 대비 10배 이상
    random_hr10 = (10.0 / num_blenders * 100) if num_blenders else 0.0
    print(f"   📊 랜덤 기준선: HR@10 ≈ {random_hr10:.2f}% (후보 {num_blenders}명 중 top-10 추천)")

    # 임베딩 차원(그릇): 데이터 양에 맞춰 조절 (작으면 underfitting, 크면 overfitting)
    embed_dim = getattr(args, 'embed_dim', 256)
    small = getattr(args, 'small_model', False)
    if small:
        embed_dim = 64
    embed_dim = max(64, min(256, embed_dim))  # 64~256 허용
    gnn_h = gnn_out = note_hyp = blend_dim = ch1_dim = ch2_dim = embed_dim
    if small:
        print(f"   📉 소형 모델 사용 (embed_dim=64, 메모리 절감)")
    elif embed_dim != 128:
        print(f"   📐 임베딩 차원: {embed_dim} (그릇 크기)")
    
    # Create model (지문 전용: FingerprintEncoder, GNN 미사용)
    model = HierarchicalFragranceHypergraph(
        node_dim=9,
        edge_dim=3,
        gnn_hidden_dim=gnn_h,
        gnn_output_dim=gnn_out,
        gnn_num_layers=3,
        gnn_architecture='GCN',
        vocab_size=vocab_size,
        note_embedding_dim=300,
        note_hyperbolic_dim=note_hyp,
        num_blenders=num_blenders,
        blender_dim=blend_dim,
        num_groups=num_groups,
        channel1_output_dim=ch1_dim,
        channel2_output_dim=ch2_dim,
        c=args.c,
        learnable_curvature=args.learnable_curvature,
        dropout=args.dropout,
        use_fingerprint=True,
        fp_dim=fp_dim,
    ).to(device)
    
    print(f"모델 생성 완료: {sum(p.numel() for p in model.parameters()):,} 파라미터")
    
    # 시나리오 C: Confusion Matrix 계산 (헷갈리는 조향사 찾기)
    # ⚡ 최적화: 혼동행렬을 파일로 저장/로드하여 재사용
    confusion_matrix = None
    if hasattr(args, 'use_confused_negatives') and args.use_confused_negatives:
        confusion_matrix_path = output_dir / 'confusion_matrix.pt'
        confusion_matrix_hash_path = output_dir / 'confusion_matrix_hash.txt'
        
        # 데이터셋 해시 계산 (데이터셋이 변경되었는지 확인)
        import hashlib
        dataset_hash = hashlib.md5(
            json.dumps([len(train_combinations), num_blenders], sort_keys=True).encode()
        ).hexdigest()
        
        # 저장된 혼동행렬이 있고 데이터셋이 동일한지 확인
        if confusion_matrix_path.exists() and confusion_matrix_hash_path.exists():
            try:
                with open(confusion_matrix_hash_path, 'r') as f:
                    saved_hash = f.read().strip()
                
                if saved_hash == dataset_hash:
                    print(f"\n  💾 저장된 Confusion Matrix 로드 중...")
                    confusion_matrix = torch.load(confusion_matrix_path, map_location='cpu', weights_only=False)
                    print(f"     ✓ Confusion Matrix 로드 완료: {confusion_matrix.shape}")
                    high_confusion_pairs = (confusion_matrix > 0.3).sum().item()
                    print(f"     ⚠️  헷갈리는 조향사 쌍: {high_confusion_pairs:,}개 (confusion > 0.3)")
                else:
                    print(f"\n  🔄 데이터셋이 변경되어 Confusion Matrix를 재계산합니다...")
                    confusion_matrix = None
            except Exception as e:
                print(f"     ⚠️  저장된 Confusion Matrix 로드 실패: {e}")
                print(f"     → 재계산합니다...")
                confusion_matrix = None
        
        # 혼동행렬이 없으면 계산
        if confusion_matrix is None:
            print(f"\n  🔍 Confusion Matrix 계산 중 (헷갈리는 조향사 탐지)...")
            print(f"     ⚠️  이 작업은 시간이 걸릴 수 있습니다. 계산 후 저장됩니다.")
            confusion_matrix = compute_confusion_matrix_from_data(
                train_combinations,
                num_blenders,
                vocab_data
            )
            if confusion_matrix is not None:
                print(f"     ✓ Confusion Matrix 계산 완료: {confusion_matrix.shape}")
                # 높은 confusion 값이 있는 쌍의 수 확인
                high_confusion_pairs = (confusion_matrix > 0.3).sum().item()
                print(f"     ⚠️  헷갈리는 조향사 쌍: {high_confusion_pairs:,}개 (confusion > 0.3)")
                
                # 파일로 저장
                print(f"     💾 Confusion Matrix 저장 중...")
                torch.save(confusion_matrix, confusion_matrix_path)
                with open(confusion_matrix_hash_path, 'w') as f:
                    f.write(dataset_hash)
                print(f"     ✓ 저장 완료: {confusion_matrix_path}")
            else:
                print(f"     ⚠️  Confusion Matrix 계산 실패 - 기본 Hard Negative Mining 사용")
    
    # Create loss function
    # confusion_matrix를 모델과 같은 디바이스로 이동
    if confusion_matrix is not None:
        confusion_matrix = confusion_matrix.to(device)

    # ② 손실 함수: triplet(기본) / bpr / circle
    loss_type = getattr(args, 'loss', 'triplet')
    if loss_type == 'triplet':
        class_weights = compute_blender_class_weights(train_combinations, num_blenders)
        class_weights = class_weights.to(device)
        loss_fn = HyperbolicTripletMarginLoss(
            manifold=model.manifold,
            margin=args.margin_start,
            num_hard_negatives=args.num_hard_negatives,
            distance_scale=getattr(args, 'distance_scale', 1.0),
            use_soft_margin=args.use_soft_margin,
            confusion_matrix=confusion_matrix,
            use_confused_negatives=getattr(args, 'use_confused_negatives', False),
            class_weights=class_weights,
            label_smoothing=getattr(args, 'label_smoothing', 0.0)
        )
        print(f"   📉 Loss: Triplet Margin (margin 스케줄링)")
    elif loss_type == 'bpr':
        loss_fn = HyperbolicBPRLoss(
            manifold=model.manifold,
            num_hard_negatives=args.num_hard_negatives,
            distance_scale=0.2,
        )
        print(f"   📉 Loss: BPR (Bayesian Personalized Ranking, 상대적 순위 학습)")
    elif loss_type == 'circle':
        loss_fn = HyperbolicCircleLoss(
            manifold=model.manifold,
            temperature=args.temperature,
            distance_scale=0.2,
        )
        print(f"   📉 Loss: Circle (Multi-label Softmax, 복수 정답 뭉치기)")
    elif loss_type == 'mean_pos':
        loss_fn = HyperbolicMeanPositiveDistanceLoss(
            manifold=model.manifold,
            margin=args.margin,
            num_hard_negatives=args.num_hard_negatives,
            distance_scale=0.2,
            use_margin_vs_negatives=True,
        )
        print(f"   📉 Loss: Mean Positive Distance (Positive Set 평균 거리 최소화)")
    else:  # max_margin
        loss_fn = HyperbolicMaxMarginRankingLoss(
            manifold=model.manifold,
            margin=args.margin,
            num_hard_negatives=args.num_hard_negatives,
            distance_scale=0.2,
        )
        print(f"   📉 Loss: Max-Margin Ranking (가장 먼 정답도 오답보다 가깝게)")
    
    # Create optimizer (Riemannian Adam 옵션: 매니폴드 밖으로 나가면 proj로 안전히 복귀)
    use_riemannian = getattr(args, 'use_riemannian_optimizer', False) and _RIEMANNIAN_AVAILABLE
    if use_riemannian:
        optimizer = RiemannianAdam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            stabilize=10,
        )
        optimizer._default_manifold = Euclidean()
        print(f"   🔧 Optimizer: RiemannianAdam (stabilize=10, 유클리드 파라미터는 Euclidean 매니폴드)")
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        if getattr(args, 'use_riemannian_optimizer', False) and not _RIEMANNIAN_AVAILABLE:
            print(f"   ⚠️  RiemannianAdam 요청했으나 geoopt 없음 → Adam 사용")
    
    # Mixed precision (CUDA only)
    scaler = None
    if getattr(args, 'use_amp', False) and device.type == 'cuda':
        try:
            scaler = torch.amp.GradScaler('cuda')
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler()
        print(f"   ⚡ AMP(FP16) 활성화: 메모리 절감 및 속도 향상")
    
    # Learning Rate 초기화 (정밀 모드: 0.0003~0.0004)
    initial_lr = args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr
    
    # ReduceLROnPlateau: 기준 지표가 N에폭 개선 없으면 LR 절반 (Val Loss 발산 시 val_loss 기준 권장)
    plateau_min = getattr(args, 'plateau_min_lr', 0.0001)
    plateau_patience = getattr(args, 'plateau_patience', 3)
    scheduler_metric = getattr(args, 'scheduler_metric', 'hr10')
    use_val_loss_scheduler = (scheduler_metric == 'val_loss')
    if args.use_lr_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min' if use_val_loss_scheduler else 'max',
            factor=0.5,
            patience=plateau_patience,
            min_lr=plateau_min
        )
        metric_desc = 'val_loss(발산 시 LR 감소)' if use_val_loss_scheduler else 'HR@10'
        print(f"📉 Learning Rate: {initial_lr:.6f} | ReduceLROnPlateau(기준={metric_desc}, patience={plateau_patience}, min_lr={plateau_min})")
    else:
        scheduler = None
        print(f"📉 Learning Rate: {initial_lr:.6f} (고정, 스케줄러 없음)")
    
    current_lr = initial_lr
    
    start_epoch = 0
    early_stopping_metric = getattr(args, 'early_stopping_metric', 'val_loss')
    best_val_hit_rate = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []
    last_val_metrics = None
    val_loss_warning_shown = False  # Val Loss 발산 시 한 번만 안내 출력
    
    print(f"\n학습 시작...")
    print(f"💡 Early stopping 기준: {early_stopping_metric} (val_loss=수치 안정·Test 일반화 권장)")
    print(f"💡 세팅: embed_dim(그릇) | lr 0.0001 | dropout {args.dropout} | norm_penalty {getattr(args, 'norm_penalty_weight', 0.1)} | weight_decay 1e-4")
    print(f"💡 OOM/메모리 부족 시: --low_memory 또는 --batch_size 32 --gradient_accumulation_steps 4 --small_model --use_amp")
    if args.val_interval == 1:
        print(f"⚡ 검증: 매 에폭마다 검증 (느릴 수 있음)")
    else:
        print(f"⚡ 검증 주기: {args.val_interval} 에폭마다 검증 (에폭 {args.val_interval}, {args.val_interval*2}, {args.val_interval*3}...)")
        print(f"   💡 검증은 느리므로 주기를 늘리면 학습 속도가 향상됩니다")
    
    # 첫 배치 테스트 (DataLoader가 정상 작동하는지 확인)
    print(f"\n🔍 첫 배치 로딩 테스트 중...")
    try:
        first_batch = next(iter(train_loader))
        print(f"   ✓ 첫 배치 로딩 성공: 배치 크기={first_batch['note_indices'].size(0)}")
        del first_batch  # 메모리 해제
    except Exception as e:
        print(f"   ❌ 첫 배치 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # output_dir은 이미 위에서 생성됨
    
    warmup_epochs = getattr(args, 'warmup_epochs', 0)
    warmup_initial_lr = getattr(args, 'warmup_initial_lr', 1e-6)
    
    for epoch in range(start_epoch, args.num_epochs):
        # Warmup: 초반 3~5 에폭 저학습률로 하이퍼볼릭 원점 근처에서 안정화 (성능 응급처치)
        if epoch < warmup_epochs:
            warmup_lr = warmup_initial_lr + (args.learning_rate - warmup_initial_lr) * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
            if epoch == 0:
                print(f"   🔥 Warmup: 에폭 1~{warmup_epochs} 학습률 {warmup_initial_lr:.2e} → {args.learning_rate:.2e}")
        
        # Margin scheduling (Triplet Margin Loss만 해당)
        current_margin = getattr(args, 'margin_start', 0.15)
        if hasattr(loss_fn, 'margin') and epoch < args.num_epochs:
            progress = epoch / args.num_epochs
            current_margin = args.margin_start + (args.margin - args.margin_start) * progress
            loss_fn.margin = current_margin
        
        # Train
        train_loss, train_norm_stats = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            initial_lr=warmup_initial_lr,
            target_lr=args.learning_rate,
            gradient_clip_value=args.gradient_clip_value,
            use_blender_input=args.use_blender_input,
            gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
            scaler=scaler,
            clear_cache_every=getattr(args, 'clear_cache_every', 0),
            group_loss_weight=getattr(args, 'group_loss_weight', 0.1),
            inter_group_separation_weight=getattr(args, 'inter_group_separation_weight', 0.005),
            inter_group_separation_margin=getattr(args, 'inter_group_separation_margin', 0.2),
            anti_collapse_weight=getattr(args, 'anti_collapse_weight', 0.01),
            anti_collapse_min_norm=getattr(args, 'anti_collapse_min_norm', 0.2),
            norm_penalty_weight=getattr(args, 'norm_penalty_weight', 0.025),
        )
        
        # Evaluate (검증 주기 적용: 매 에폭마다 검증)
        should_validate = (epoch + 1) % args.val_interval == 0  # 검증 주기마다 검증
        if should_validate:
            val_metrics = evaluate_model(
                model=model,
                val_loader=val_loader,
                device=device,
                loss_fn=loss_fn,
                k=10,
                use_blender_input=args.use_blender_input,
                temperature=getattr(args, 'temperature', 0.07)
            )
            last_val_metrics = val_metrics  # 마지막 검증 결과 저장
        else:
            # 검증하지 않는 에폭에서는 이전 검증 결과 재사용 또는 기본값
            if last_val_metrics:
                val_metrics = last_val_metrics
            else:
                # 첫 에폭 등 검증 결과가 없는 경우
                val_metrics = {
                    'hit_rate@1': 0.0,
                    'hit_rate@5': 0.0,
                    'hit_rate@k': 0.0,
                    'recall@1': 0.0,
                    'recall@5': 0.0,
                    'recall@k': 0.0,
                    'mrr': 0.0,
                    'ndcg@k': 0.0,
                    'val_loss': train_loss,
                    'diversity_std': 0.0,
                    'diversity_coefficient': 0.0,
                    'total_samples': 0
                }
        
        val_hit_rate = val_metrics['hit_rate@k']
        val_loss = val_metrics['val_loss']
        
        # Gradient Clipping은 1.0 이상 유지 (0.30 등으로 강화하지 않음, 하이퍼볼릭 움직임 허용)
        args.gradient_clip_value = max(1.0, args.gradient_clip_value)
        
        # ⚡ ReduceLROnPlateau: 기준 지표(HR@10 또는 val_loss)로 LR 자동 조절 (검증한 에폭에만 step, warmup 중에는 step 생략)
        if args.use_lr_scheduler and scheduler is not None and should_validate and epoch >= warmup_epochs:
            step_metric = val_loss if getattr(args, 'scheduler_metric', 'hr10') == 'val_loss' else val_hit_rate
            scheduler.step(step_metric)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        # Loss가 너무 크면 경고 표시 (Val Loss 발산 = 과적합 또는 LR 과다 신호)
        loss_warning = ""
        if val_loss > 10.0:
            loss_warning = " ⚠️"
        elif val_loss > 5.0:
            loss_warning = " ⚡"  # Val Loss 발산 시: --learning_rate 5e-5 또는 --scheduler_metric val_loss 권장
        
        # 검증 여부 표시
        val_marker = "✓" if should_validate else "⏭"
        
        # 출력 포맷: Hit Rate(HR) + Recall 둘 다 표시 (검증 안 한 경우 "-")
        val_info = ""
        # 학습 쪽 z_recipe norm (검증 안 하는 에폭에도 붕괴/경계 감지)
        if train_norm_stats:
            tn_mean = train_norm_stats.get('recipe_norm_mean', 0.0)
            tn_min = train_norm_stats.get('recipe_norm_min', 0.0)
            tn_max = train_norm_stats.get('recipe_norm_max', 0.0)
            if tn_mean < 0.01 or tn_mean > 0.95:
                print(f"   ⚠️  [Train] z_recipe.norm: mean={tn_mean:.4f} min={tn_min:.4f} max={tn_max:.4f} → 붕괴/경계 의심")
            else:
                print(f"   📐 [Train] z_recipe.norm: mean={tn_mean:.4f} min={tn_min:.4f} max={tn_max:.4f}")
        if should_validate:
            r1 = val_metrics.get('recall@1', 0.0)
            r5 = val_metrics.get('recall@5', 0.0)
            r10 = val_metrics.get('recall@k', val_metrics.get('recall@10', 0.0))
            val_info = (f"Val Loss: {val_loss:.4f}{loss_warning} | "
                f"HR@1/5/10: {val_metrics['hit_rate@1']:.4f}/{val_metrics['hit_rate@5']:.4f}/{val_hit_rate:.4f} | "
                f"Recall@1/5/10: {r1:.4f}/{r5:.4f}/{r10:.4f} | MRR: {val_metrics['mrr']:.4f} NDCG@10: {val_metrics['ndcg@k']:.4f}")
            # 하이퍼볼릭 수치 안정성: z_recipe norm (0 근처=붕괴, 0.999 근처=boundary 튐)
            rn_mean = val_metrics.get('recipe_norm_mean', 0.0)
            rn_min = val_metrics.get('recipe_norm_min', 0.0)
            rn_max = val_metrics.get('recipe_norm_max', 0.0)
            if rn_mean < 0.01 or rn_mean > 0.95:
                print(f"   ⚠️  [Val] z_recipe.norm: mean={rn_mean:.4f} min={rn_min:.4f} max={rn_max:.4f} → 붕괴(≈0) 또는 boundary(≈1) 의심. c/LR/초기화 확인.")
            else:
                print(f"   📐 [Val] z_recipe.norm: mean={rn_mean:.4f} min={rn_min:.4f} max={rn_max:.4f}")
        else:
            val_info = "Val Loss: - | HR@1/5/10: - | Recall@1/5/10: - | MRR: - NDCG@10: -"
        
        print(f"Epoch {epoch+1}/{args.num_epochs} [{val_marker}] | "
              f"Train Loss: {train_loss:.4f} | {val_info} | LR: {current_lr:.6f}")
        # Val Loss 발산(⚡) 시 한 번만 안내: LR 낮추기 또는 스케줄러를 val_loss 기준으로
        if should_validate and val_loss > 5.0 and not val_loss_warning_shown:
            val_loss_warning_shown = True
            print(f"   💡 Val Loss 발산 시: --learning_rate 5e-5 또는 --scheduler_metric val_loss 권장")
        
        # 첫 에포크에 HR@1이 너무 높으면 경고 및 진단 (검증한 경우에만, 첫 에폭은 검증 안 함)
        if should_validate and val_metrics['hit_rate@1'] > 0.7:
            print(f"\n  ⚠️  경고: 첫 에포크에 HR@1이 {val_metrics['hit_rate@1']*100:.1f}%입니다!")
            print(f"     가능한 원인:")
            print(f"     1. 데이터 누수: Train/Val 조합이 겹칠 수 있음")
            print(f"     2. 블렌더 수가 적음: 랜덤 추측 성능이 높을 수 있음")
            print(f"     3. 데이터 불균형: 특정 블렌더가 대부분을 차지할 수 있음")
            print(f"     4. 모델 초기화 문제: 초기 가중치가 이미 좋을 수 있음")
            print(f"     → 위의 '데이터 누수 진단' 결과를 확인하세요.")
            
            # 실제 예측 분석 (어떤 블렌더를 예측하는지 확인)
            print(f"\n  🔍 첫 에포크 예측 분석:")
            print(f"     HR@1: {val_metrics['hit_rate@1']*100:.1f}%")
            print(f"     HR@5: {val_metrics['hit_rate@5']*100:.1f}%")
            hr10_key = 'hit_rate@k' if 'hit_rate@k' in val_metrics else 'hit_rate@10'
            hr10_value = val_metrics.get(hr10_key, val_metrics.get('hit_rate@k', 0.0))
            print(f"     HR@10: {hr10_value*100:.1f}%")
            print(f"     NDCG@10: {val_metrics['ndcg@k']*100:.1f}%")
            print(f"     → HR@1과 HR@5의 차이가 크면 모델이 상위 몇 개 블렌더만 외우고 있을 수 있습니다.")
        
        # Save history (Hit Rate + Recall 둘 다)
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_hit_rate@1': val_metrics['hit_rate@1'],
            'val_hit_rate@5': val_metrics['hit_rate@5'],
            'val_hit_rate@10': val_hit_rate,
            'val_recall@1': val_metrics.get('recall@1', 0.0),
            'val_recall@5': val_metrics.get('recall@5', 0.0),
            'val_recall@10': val_metrics.get('recall@k', val_metrics.get('recall@10', 0.0)),
            'val_mrr': val_metrics['mrr'],
            'val_ndcg@10': val_metrics['ndcg@k'],
            'learning_rate': current_lr,
            'margin': current_margin
        })
        
        if should_validate:
            improved = False
            if early_stopping_metric == 'val_loss':
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    improved = True
            else:
                if val_hit_rate > best_val_hit_rate:
                    best_val_hit_rate = val_hit_rate
                    improved = True
            if improved:
                patience_counter = 0
                best_model_path = output_dir / 'best_model.pt'
                best_checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'args': vars(args),
                    'training_history': training_history,
                    'epoch': epoch + 1,
                    'val_metrics': val_metrics,
                }
                torch.save(best_checkpoint, best_model_path)
            else:
                patience_counter += 1
        
        # Early stopping (검증한 경우에만 체크)
        if should_validate and patience_counter >= args.early_stopping_patience:
            print(f"\n조기 종료 (patience: {args.early_stopping_patience}, 기준: {early_stopping_metric})")
            break
        
        # 에폭 끝 메모리 정리 (장시간 학습 시 OOM·killed 방지)
        gc.collect()
    
    # Save final model only (no per-epoch checkpoints)
    final_model_path = output_dir / 'final_model.pt'
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'training_history': training_history,
        'epoch': (training_history[-1]['epoch'] if training_history else 0),
        'val_metrics': last_val_metrics,
    }
    torch.save(final_checkpoint, final_model_path)
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # 베이스라인과 공정 비교: best 모델로 테스트 세트 평가 (베이스라인은 test HR@10 보고)
    best_model_path = output_dir / 'best_model.pt'
    if best_model_path.exists():
        cp = torch.load(best_model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in cp:
            model.load_state_dict(cp['model_state_dict'], strict=True)
        test_metrics = evaluate_model(model, test_loader, device, k=10)
        print(f"\n📊 Test 세트 (베이스라인 비교용) | Best 모델 기준:")
        r1 = test_metrics.get('recall@1', 0.0)
        r5 = test_metrics.get('recall@5', 0.0)
        r10 = test_metrics.get('recall@k', test_metrics.get('recall@10', 0.0))
        print(f"   HR@1={test_metrics['hit_rate@1']:.4f}  HR@5={test_metrics['hit_rate@5']:.4f}  HR@10={test_metrics['hit_rate@k']:.4f}  |  Recall@1={r1:.4f}  Recall@5={r5:.4f}  Recall@10={r10:.4f}  |  MRR={test_metrics['mrr']:.4f}  NDCG@10={test_metrics['ndcg@k']:.4f}")
    
    if early_stopping_metric == 'val_loss':
        print(f"\n학습 완료 | Best Val Loss: {best_val_loss:.4f} | 저장 위치: {output_dir}")
    else:
        print(f"\n학습 완료 | Best Val HR@10: {best_val_hit_rate:.4f} | 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()