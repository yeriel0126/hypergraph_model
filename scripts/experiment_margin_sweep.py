"""
Margin 값 실험 스크립트

여러 margin 값으로 실험을 실행하고 결과를 비교합니다.
"""

import subprocess
import json
from pathlib import Path
import sys

def run_experiment(margin, num_hard_negatives=10, epochs=5, output_suffix=""):
    """
    특정 margin 값으로 실험 실행
    
    Args:
        margin: margin 값
        num_hard_negatives: hard negative mining 개수
        epochs: 학습할 에폭 수
        output_suffix: 출력 디렉토리 suffix
    """
    print("=" * 70)
    print(f"실험 시작: Margin={margin}, Hard Negatives={num_hard_negatives}")
    print("=" * 70)
    
    output_dir = f"./results/checkpoints/margin_{margin}{output_suffix}"
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "train_hyperbolic_hypergraph.py"),
        "--data_path", "cleaned_data/cleaned_complete_data.json",
        "--vocab_path", "feature_encoding/vocabularies.json",
        "--batch_size", "32",
        "--num_epochs", str(epochs),
        "--learning_rate", "0.001",
        "--margin", str(margin),
        "--num_hard_negatives", str(num_hard_negatives),
        "--output_dir", output_dir
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={**subprocess.os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
        )
        
        if result.returncode == 0:
            print(f"✓ 실험 완료: Margin={margin}")
            return True, output_dir
        else:
            print(f"✗ 실험 실패: Margin={margin}")
            print(f"Error: {result.stderr[:500]}")
            return False, output_dir
    except Exception as e:
        print(f"✗ 실험 오류: Margin={margin}, Error: {e}")
        return False, output_dir

def compare_results(margin_values, base_dir="results/checkpoints"):
    """
    여러 margin 값의 결과 비교
    """
    print("\n" + "=" * 70)
    print("실험 결과 비교")
    print("=" * 70)
    
    results = {}
    
    for margin in margin_values:
        margin_dir = Path(base_dir) / f"margin_{margin}"
        history_path = margin_dir / "training_history.json"
        
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            if history:
                final_epoch = history[-1]
                results[margin] = {
                    'final_loss': final_epoch.get('train_loss', 0),
                    'final_hit_rate@1': final_epoch.get('val_hit_rate@1', 0),
                    'final_hit_rate@5': final_epoch.get('val_hit_rate@5', 0),
                    'final_hit_rate@10': final_epoch.get('val_hit_rate@10', 0),
                    'final_mrr': final_epoch.get('val_mrr', 0),
                    'final_ndcg': final_epoch.get('val_ndcg', 0),
                }
    
    # 결과 출력
    if results:
        print(f"\n{'Margin':<10} {'Loss':<10} {'Hit@1':<10} {'Hit@5':<10} {'Hit@10':<10} {'MRR':<10} {'NDCG':<10}")
        print("-" * 80)
        for margin in sorted(results.keys()):
            r = results[margin]
            print(f"{margin:<10.2f} {r['final_loss']:<10.4f} {r['final_hit_rate@1']:<10.4f} "
                  f"{r['final_hit_rate@5']:<10.4f} {r['final_hit_rate@10']:<10.4f} "
                  f"{r['final_mrr']:<10.4f} {r['final_ndcg']:<10.4f}")
        
        # 최고 성능 찾기
        best_hit10 = max(results.items(), key=lambda x: x[1]['final_hit_rate@10'])
        best_mrr = max(results.items(), key=lambda x: x[1]['final_mrr'])
        best_ndcg = max(results.items(), key=lambda x: x[1]['final_ndcg'])
        
        print(f"\n최고 성능:")
        print(f"  Hit Rate@10: Margin={best_hit10[0]:.2f} ({best_hit10[1]['final_hit_rate@10']:.4f})")
        print(f"  MRR: Margin={best_mrr[0]:.2f} ({best_mrr[1]['final_mrr']:.4f})")
        print(f"  NDCG: Margin={best_ndcg[0]:.2f} ({best_ndcg[1]['final_ndcg']:.4f})")
    
    return results

def main():
    """
    여러 margin 값으로 실험 실행
    """
    # 실험할 margin 값들
    margin_values = [0.30, 0.35, 0.40]
    num_hard_negatives = 10
    epochs = 5  # 빠른 실험을 위해 5 에폭만
    
    print("=" * 70)
    print("Margin 값 실험 스위프")
    print("=" * 70)
    print(f"실험할 Margin 값: {margin_values}")
    print(f"Hard Negatives: {num_hard_negatives}")
    print(f"에폭 수: {epochs}")
    print("=" * 70)
    
    # 각 margin 값으로 실험 실행
    for margin in margin_values:
        success, output_dir = run_experiment(
            margin=margin,
            num_hard_negatives=num_hard_negatives,
            epochs=epochs
        )
        if not success:
            print(f"⚠️  Margin={margin} 실험 실패, 다음으로 진행...")
    
    # 결과 비교
    compare_results(margin_values)
    
    print("\n" + "=" * 70)
    print("실험 완료!")
    print("=" * 70)
    print("\n각 실험 결과는 다음 디렉토리에 저장되었습니다:")
    for margin in margin_values:
        print(f"  - results/checkpoints/margin_{margin}/")

if __name__ == "__main__":
    main()
