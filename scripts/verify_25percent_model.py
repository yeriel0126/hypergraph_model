"""
25% 성능 모델 확인 스크립트

final_model.pt에서 test_metrics를 확인하여 25% 성능을 달성했는지 확인합니다.
"""

import torch
import os
from pathlib import Path
import sys

# Fix OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def verify_25percent_model(checkpoint_path="results/checkpoints/final_model.pt"):
    """25% 성능 모델 확인"""
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {checkpoint_path}")
        return False
    
    try:
        print("=" * 70)
        print("25% 성능 모델 확인")
        print("=" * 70)
        print(f"\n📁 체크포인트: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check test metrics
        test_metrics = checkpoint.get('test_metrics', {})
        if test_metrics:
            hit_rate_10 = test_metrics.get('hit_rate@k', test_metrics.get('hit_rate', 0))
            hit_rate_1 = test_metrics.get('hit_rate@1', 0)
            hit_rate_5 = test_metrics.get('hit_rate@5', 0)
            mrr = test_metrics.get('mrr', 0)
            ndcg = test_metrics.get('ndcg@k', 0)
            
            print(f"\n📊 Test Set 성능:")
            print(f"   Hit Rate@1: {hit_rate_1:.4f} ({hit_rate_1*100:.2f}%)")
            print(f"   Hit Rate@5: {hit_rate_5:.4f} ({hit_rate_5*100:.2f}%)")
            print(f"   Hit Rate@10: {hit_rate_10:.4f} ({hit_rate_10*100:.2f}%)")
            print(f"   MRR: {mrr:.4f}")
            print(f"   NDCG@10: {ndcg:.4f}")
            
            if hit_rate_10 >= 0.25:
                print(f"\n🎉 25% 이상 성능 달성!")
                print(f"   ✓ 이 모델은 저장되어 있습니다: {checkpoint_path}")
                return True
            elif hit_rate_10 >= 0.20:
                print(f"\n✓ 좋은 성능입니다! (20% 이상)")
                print(f"   모델이 저장되어 있습니다: {checkpoint_path}")
                return True
            else:
                print(f"\n⚠️  성능이 25% 미만입니다.")
                return False
        else:
            print(f"\n⚠️  Test metrics가 저장되지 않았습니다.")
            print(f"   모델은 저장되어 있지만 성능 정보가 없습니다.")
            
            # Check val metrics as fallback
            val_metrics = checkpoint.get('val_metrics', {})
            if val_metrics:
                val_hit_rate = val_metrics.get('hit_rate@k', val_metrics.get('hit_rate', 0))
                print(f"\n   Validation Hit Rate@10: {val_hit_rate:.4f} ({val_hit_rate*100:.2f}%)")
            
            return False
        
    except Exception as e:
        print(f"❌ 읽기 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "results/checkpoints/final_model.pt"
    verify_25percent_model(checkpoint_path)
