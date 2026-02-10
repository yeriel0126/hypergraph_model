"""
조향사 혼동 패턴 분석 스크립트

모델이 어떤 조향사를 가장 헷갈려 하는지 분석합니다.
- Confusion Matrix 생성
- 가장 자주 혼동되는 조향사 쌍 찾기
- 각 조향사별 오분류 패턴 분석
"""

# Fix OpenMP error - MUST be before any torch imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import model components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import (
    HierarchicalFragranceHypergraph,
    HyperbolicRecipeDataset,
    collate_hyperbolic_recipes,
    load_data
)
from torch.utils.data import DataLoader


def convert_to_native_types(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj

# Import create_recipe_combinations from train script
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from train_hyperbolic_hypergraph import create_recipe_combinations

# Import create_recipe_combinations from train script
sys.path.insert(0, str(Path(__file__).parent.parent))
from train_hyperbolic_hypergraph import create_recipe_combinations

def analyze_blender_confusion(
    checkpoint_path: str,
    data_path: str,
    vocab_path: str,
    device: str = "cpu",
    k: int = 10,
    num_samples: int = None
):
    """
    조향사 혼동 패턴 분석
    
    Args:
        checkpoint_path: 모델 체크포인트 경로
        data_path: 데이터 파일 경로
        vocab_path: Vocabulary 파일 경로
        device: 디바이스 (cpu/cuda/mps)
        k: Top-K 조향사 고려
        num_samples: 분석할 샘플 수 (None이면 전체)
    """
    
    print("=" * 70)
    print("조향사 혼동 패턴 분석")
    print("=" * 70)
    
    # Load checkpoint
    print(f"\n1. 체크포인트 로딩: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load data
    print(f"2. 데이터 로딩...")
    records, vocab_data = load_data(data_path, vocab_path)
    
    # Get blender vocabulary
    blender_vocab = vocab_data.get('blenders', {})
    blender_idx_to_name = blender_vocab.get('idx_to_name', {})
    blender_name_to_idx = blender_vocab.get('name_to_idx', {})
    num_blenders = blender_vocab.get('size', len(blender_idx_to_name))
    
    print(f"   총 조향사 수: {num_blenders}")
    print(f"   총 레시피 수: {len(records):,}")
    
    # Create model
    print(f"\n3. 모델 초기화...")
    vocab_size = vocab_data.get('notes', {}).get('size', 435)
    
    model = HierarchicalFragranceHypergraph(
        node_dim=9,
        edge_dim=3,
        gnn_hidden_dim=128,
        gnn_output_dim=128,
        gnn_num_layers=3,
        vocab_size=vocab_size,
        note_embedding_dim=300,
        note_hyperbolic_dim=128,
        num_blenders=num_blenders,
        blender_dim=128,
        channel1_output_dim=128,
        channel2_output_dim=128,
        c=0.3,
        learnable_curvature=True,
        dropout=0.1
    ).to(device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        try:
            # Try strict loading first
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f"   ✓ 모델 가중치 로드 완료 (strict mode)")
        except RuntimeError as e:
            # If strict loading fails, try non-strict (for compatibility with older checkpoints)
            print(f"   ⚠️  Strict loading failed, trying non-strict mode...")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys:
                print(f"   ⚠️  Missing keys (will use default values): {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"   ⚠️  Unexpected keys (ignored): {unexpected_keys[:5]}...")
            print(f"   ✓ 모델 가중치 로드 완료 (non-strict mode)")
    else:
        print(f"   ⚠️  모델 가중치를 찾을 수 없습니다.")
        return
    
    model.eval()
    
    # Create dataset and dataloader
    print(f"\n4. 데이터셋 생성...")
    combinations = create_recipe_combinations(records, vocab_data, max_samples=50000)
    
    if num_samples:
        combinations = combinations[:num_samples]
    
    dataset = HyperbolicRecipeDataset(combinations, vocab_data)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_hyperbolic_recipes,
        num_workers=0
    )
    
    print(f"   분석할 샘플 수: {len(combinations):,}")
    
    # Analyze confusion patterns
    print(f"\n5. 혼동 패턴 분석 중...")
    
    # Confusion tracking
    confusion_matrix = np.zeros((num_blenders, num_blenders), dtype=np.int32)
    correct_predictions = defaultdict(int)
    incorrect_predictions = defaultdict(list)  # blender_idx -> [predicted_indices]
    blender_error_counts = defaultdict(int)  # blender_idx -> error_count
    blender_total_counts = defaultdict(int)  # blender_idx -> total_count
    
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move to device
                note_indices = batch['note_indices'].to(device)
                blender_indices = batch['blender_indices'].to(device)
                molecule_mask = batch['molecule_mask'].to(device)
                
                smiles_graphs = batch['smiles_graphs']
                if hasattr(smiles_graphs, 'to'):
                    smiles_graphs = smiles_graphs.to(device)
                else:
                    smiles_graphs.x = smiles_graphs.x.to(device)
                    smiles_graphs.edge_index = smiles_graphs.edge_index.to(device)
                    if hasattr(smiles_graphs, 'edge_attr') and smiles_graphs.edge_attr is not None:
                        smiles_graphs.edge_attr = smiles_graphs.edge_attr.to(device)
                
                smiles_batch = batch['smiles_batch'].to(device)
                
                # Forward pass
                z_recipe = model(
                    smiles_graphs=smiles_graphs,
                    smiles_batch=smiles_batch,
                    note_indices=note_indices,
                    blender_indices=blender_indices,
                    molecule_mask=molecule_mask
                )
                
                # Get predictions
                all_blender_embs = model.blender_anchors()
                distances = model.compute_poincare_distance(z_recipe, all_blender_embs)
                scores = model.compute_temperature_scores(distances, temperature=0.07)
                
                # Get top-k predictions
                _, top_k_indices = torch.topk(scores, k=k, dim=1, largest=True)
                
                # Compare with ground truth
                target_blenders = batch['target_blenders']
                
                for i, target_blender in enumerate(target_blenders):
                    if target_blender is None or target_blender == 0:
                        continue
                    
                    total_samples += 1
                    target_idx = target_blender
                    predicted_indices = top_k_indices[i].cpu().numpy()
                    
                    # Update counts
                    blender_total_counts[target_idx] += 1
                    
                    # Check if correct
                    if target_idx in predicted_indices:
                        correct_predictions[target_idx] += 1
                        # Find rank
                        rank = np.where(predicted_indices == target_idx)[0][0] + 1
                        confusion_matrix[target_idx, target_idx] += 1
                    else:
                        # Incorrect prediction
                        blender_error_counts[target_idx] += 1
                        incorrect_predictions[target_idx].extend(predicted_indices[:3])  # Top-3만 기록
                        
                        # Update confusion matrix (most confused with top-1 prediction)
                        top_predicted = predicted_indices[0]
                        confusion_matrix[target_idx, top_predicted] += 1
                
            except Exception as e:
                if batch_idx < 3:
                    print(f"   Error at batch {batch_idx}: {e}")
                continue
    
    print(f"   분석 완료: {total_samples:,} 샘플")
    
    # Analyze results
    print(f"\n6. 분석 결과:")
    print("=" * 70)
    
    # Most confused blenders
    print(f"\n📊 가장 자주 혼동되는 조향사 (Top-10):")
    error_rates = {}
    for blender_idx, error_count in blender_error_counts.items():
        total_count = blender_total_counts[blender_idx]
        if total_count > 0:
            error_rate = error_count / total_count
            error_rates[blender_idx] = error_rate
    
    sorted_errors = sorted(error_rates.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for rank, (blender_idx, error_rate) in enumerate(sorted_errors, 1):
        blender_name = blender_idx_to_name.get(str(blender_idx), f"Blender_{blender_idx}")
        total_count = blender_total_counts[blender_idx]
        error_count = blender_error_counts[blender_idx]
        correct_count = correct_predictions[blender_idx]
        
        print(f"   {rank}. {blender_name} (ID: {blender_idx})")
        print(f"      오분류율: {error_rate*100:.1f}% ({error_count}/{total_count})")
        print(f"      정확도: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
        
        # Most confused with
        if blender_idx in incorrect_predictions:
            confused_with = Counter(incorrect_predictions[blender_idx])
            top_confused = confused_with.most_common(3)
            if top_confused:
                print(f"      가장 자주 혼동되는 조향사:")
                for confused_idx, count in top_confused:
                    confused_name = blender_idx_to_name.get(str(confused_idx), f"Blender_{confused_idx}")
                    print(f"        - {confused_name} (ID: {confused_idx}): {count}회")
        print()
    
    # Most confusing pairs
    print(f"\n🔗 가장 자주 혼동되는 조향사 쌍 (Top-10):")
    pair_confusions = []
    for true_idx in range(num_blenders):
        for pred_idx in range(num_blenders):
            if true_idx != pred_idx and confusion_matrix[true_idx, pred_idx] > 0:
                count = confusion_matrix[true_idx, pred_idx]
                true_name = blender_idx_to_name.get(str(true_idx), f"Blender_{true_idx}")
                pred_name = blender_idx_to_name.get(str(pred_idx), f"Blender_{pred_idx}")
                pair_confusions.append((true_name, pred_name, true_idx, pred_idx, count))
    
    pair_confusions.sort(key=lambda x: x[4], reverse=True)
    
    for rank, (true_name, pred_name, true_idx, pred_idx, count) in enumerate(pair_confusions[:10], 1):
        print(f"   {rank}. {true_name} (ID: {true_idx}) → {pred_name} (ID: {pred_idx}): {count}회")
    
    # Save results
    output_dir = Path(checkpoint_path).parent
    results = {
        'total_samples': total_samples,
        'most_confused_blenders': [
            {
                'blender_idx': idx,
                'blender_name': blender_idx_to_name.get(str(idx), f"Blender_{idx}"),
                'error_rate': error_rates[idx],
                'error_count': blender_error_counts[idx],
                'total_count': blender_total_counts[idx],
                'correct_count': correct_predictions[idx]
            }
            for idx, _ in sorted_errors
        ],
        'most_confusing_pairs': [
            {
                'true_blender': true_name,
                'true_idx': true_idx,
                'predicted_blender': pred_name,
                'predicted_idx': pred_idx,
                'confusion_count': count
            }
            for true_name, pred_name, true_idx, pred_idx, count in pair_confusions[:20]
        ]
    }
    
    results_path = output_dir / "blender_confusion_analysis.json"
    # Convert NumPy types to Python native types for JSON serialization
    results_serializable = convert_to_native_types(results)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 분석 결과 저장: {results_path}")
    
    # Create confusion matrix visualization (for top confused blenders)
    if len(sorted_errors) > 0:
        top_confused_indices = [idx for idx, _ in sorted_errors[:20]]
        
        # Create submatrix
        submatrix = confusion_matrix[np.ix_(top_confused_indices, top_confused_indices)]
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            submatrix,
            xticklabels=[blender_idx_to_name.get(str(idx), f"B_{idx}") for idx in top_confused_indices],
            yticklabels=[blender_idx_to_name.get(str(idx), f"B_{idx}") for idx in top_confused_indices],
            annot=True,
            fmt='d',
            cmap='YlOrRd',
            cbar_kws={'label': 'Confusion Count'}
        )
        plt.title('Confusion Matrix: Top 20 Most Confused Blenders', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Blender', fontsize=12)
        plt.ylabel('True Blender', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        confusion_plot_path = output_dir / "blender_confusion_matrix.png"
        plt.savefig(confusion_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 혼동 행렬 시각화 저장: {confusion_plot_path}")
    
    print("\n" + "=" * 70)
    print("분석 완료!")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    
    # Set default paths relative to script location
    script_dir = Path(__file__).parent
    # scripts/ -> hyperbolic_model/ -> newcode2/ -> cleaned_data/
    default_checkpoint = script_dir.parent / "results" / "checkpoints" / "final_model.pt"
    default_data_path = script_dir.parent.parent / "cleaned_data" / "cleaned_complete_data.json"
    default_vocab_path = script_dir.parent.parent / "feature_encoding" / "vocabularies.json"
    
    parser = argparse.ArgumentParser(description="Analyze blender confusion patterns")
    parser.add_argument("--checkpoint", type=str, default=str(default_checkpoint), help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default=str(default_data_path), help="Path to data file")
    parser.add_argument("--vocab_path", type=str, default=str(default_vocab_path), help="Path to vocab file")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--k", type=int, default=10, help="Top-K for evaluation")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of samples to analyze (default: 5000)")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {args.checkpoint}")
        print(f"\n사용 방법:")
        print(f"  python analyze_blender_confusion.py \\")
        print(f"    --checkpoint <체크포인트_경로> \\")
        print(f"    --data_path <데이터_경로> \\")
        print(f"    --vocab_path <vocab_경로>")
        sys.exit(1)
    
    analyze_blender_confusion(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        vocab_path=args.vocab_path,
        device=args.device,
        k=args.k,
        num_samples=args.num_samples
    )
