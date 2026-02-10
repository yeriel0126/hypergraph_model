"""
기존 체크포인트로 Top-1, Top-5 평가 실행 스크립트
"""

import torch
import json
from pathlib import Path
import sys
import argparse

# 모델 import를 위한 경로 추가
model_dir = Path(__file__).parent.parent
sys.path.insert(0, str(model_dir))
sys.path.insert(0, str(model_dir / "model"))

from model.hierarchical_hyperbolic_hypergraph import HierarchicalFragranceHypergraph
from model.hyperbolic_data_loader import load_data
from torch.utils.data import DataLoader
from model.hyperbolic_data_loader import HyperbolicRecipeDataset, collate_hyperbolic_recipes
from train_hyperbolic_hypergraph import evaluate_model, create_recipe_combinations
import numpy as np

def evaluate_checkpoint(
    checkpoint_path: str,
    vocab_path: str,
    data_path: str,
    device: str = None
):
    """
    체크포인트를 로드하여 평가 실행
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("체크포인트 평가 실행")
    print("=" * 70)
    
    # 체크포인트 로드
    print(f"\n1. 체크포인트 로드: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Vocabulary 로드
    print(f"2. Vocabulary 로드: {vocab_path}")
    records, vocab_data = load_data(data_path, vocab_path)
    print(f"   ✓ 로드된 레코드 수: {len(records):,}")
    
    # 모델 파라미터 추출
    args = checkpoint.get('args', {})
    num_blenders = vocab_data.get('blenders', {}).get('size', 100)
    vocab_size = vocab_data.get('notes', {}).get('size', 435)
    
    # 모델 생성
    print(f"\n3. 모델 생성")
    model = HierarchicalFragranceHypergraph(
        node_dim=9,
        edge_dim=3,
        gnn_hidden_dim=128,
        gnn_output_dim=128,
        gnn_num_layers=3,
        gnn_architecture=args.get('gnn_architecture', 'GCN'),
        vocab_size=vocab_size,
        note_embedding_dim=300,
        note_hyperbolic_dim=128,
        num_blenders=num_blenders,
        blender_dim=128,
        channel1_output_dim=128,
        channel2_output_dim=128,
        c=args.get('c', 0.2),
        learnable_curvature=True,
        dropout=args.get('dropout', 0.1)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✓ 모델 로드 완료 (Epoch {checkpoint.get('epoch', 'unknown')})")
    
    # 데이터 준비
    print(f"\n4. 데이터 준비")
    combinations = create_recipe_combinations(records, vocab_data, max_samples=50000)
    print(f"   ✓ 생성된 조합: {len(combinations):,}")
    
    # Train/Val/Test 분할
    np.random.seed(42)
    indices = np.random.permutation(len(combinations))
    train_size = int(0.7 * len(combinations))
    val_size = int(0.15 * len(combinations))
    
    test_combinations = [combinations[i] for i in indices[train_size+val_size:]]
    print(f"   ✓ Test 세트: {len(test_combinations):,} 샘플")
    
    # Test Dataset 생성
    test_dataset = HyperbolicRecipeDataset(
        records=test_combinations,
        vocab_data=vocab_data,
        max_molecules=args.get('max_molecules', 10),
        max_notes_per_molecule=args.get('max_notes', 20),
        max_blenders_per_molecule=args.get('max_blenders', 10),
        mode="test"
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.get('batch_size', 32),
        shuffle=False,
        collate_fn=collate_hyperbolic_recipes,
        num_workers=0
    )
    
    # 평가 실행
    print(f"\n5. 평가 실행 (Top-1, Top-5, Top-10)")
    print("-" * 70)
    
    test_metrics = evaluate_model(model, test_loader, device, k=10)
    
    # 결과 출력
    print(f"\n  📊 Test Set 평가 결과:")
    print(f"  {'Metric':<20} {'Value':<15} {'Description':<30}")
    print(f"  {'-'*65}")
    print(f"  {'Hit Rate@1':<20} {test_metrics.get('hit_rate@1', 0.0):<15.4f} {'Top-1 accuracy':<30}")
    print(f"  {'Hit Rate@5':<20} {test_metrics.get('hit_rate@5', 0.0):<15.4f} {'Top-5 accuracy':<30}")
    print(f"  {'Hit Rate@10':<20} {test_metrics['hit_rate@k']:<15.4f} {'Top-10 accuracy':<30}")
    print(f"  {'MRR':<20} {test_metrics.get('mrr', 0.0):<15.4f} {'Mean Reciprocal Rank':<30}")
    print(f"  {'NDCG@10':<20} {test_metrics.get('ndcg@k', 0.0):<15.4f} {'Normalized DCG@10':<30}")
    print(f"  {'Diversity (Std)':<20} {test_metrics.get('diversity_std', 0.0):<15.4f} {'Recommendation std dev':<30}")
    print(f"  {'Diversity (CV)':<20} {test_metrics.get('diversity_coefficient', 0.0):<15.4f} {'Coefficient of variation':<30}")
    print(f"  {'Total Samples':<20} {test_metrics.get('total_samples', 0):<15}")
    
    # 결과 저장
    output_dir = Path("checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'checkpoint': checkpoint_path,
        'epoch': checkpoint.get('epoch', 'unknown'),
        'metrics': {
            'hit_rate@1': float(test_metrics.get('hit_rate@1', 0.0)),
            'hit_rate@5': float(test_metrics.get('hit_rate@5', 0.0)),
            'hit_rate@10': float(test_metrics.get('hit_rate@k', 0.0)),
            'mrr': float(test_metrics.get('mrr', 0.0)),
            'ndcg@10': float(test_metrics.get('ndcg@k', 0.0)),
            'diversity_std': float(test_metrics.get('diversity_std', 0.0)),
            'diversity_coefficient': float(test_metrics.get('diversity_coefficient', 0.0)),
            'total_samples': int(test_metrics.get('total_samples', 0))
        }
    }
    
    checkpoint_name = Path(checkpoint_path).stem
    results_path = output_dir / f"evaluation_{checkpoint_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ 평가 결과 저장: {results_path}")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate checkpoint with Top-1, Top-5, Top-10 metrics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data file")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu/mps)")
    
    args = parser.parse_args()
    
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab_path,
        data_path=args.data_path,
        device=args.device
    )
