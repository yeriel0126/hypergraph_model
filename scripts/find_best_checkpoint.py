"""
최고 성능 체크포인트 찾기 스크립트

training_history.json을 분석하여 최고 성능을 달성한 체크포인트를 찾습니다.
"""

import json
import torch
from pathlib import Path

def find_best_checkpoint(checkpoint_dir="results/checkpoints"):
    """최고 성능 체크포인트 찾기"""
    
    checkpoint_dir = Path(checkpoint_dir)
    history_path = checkpoint_dir / "training_history.json"
    
    if not history_path.exists():
        print(f"❌ 학습 히스토리 파일을 찾을 수 없습니다: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    if not history:
        print("❌ 학습 히스토리가 비어있습니다.")
        return
    
    # Find best performance
    best_hit_rate = 0.0
    best_epoch = 0
    best_epoch_info = None
    
    for h in history:
        hit_rate = h.get('val_hit_rate@10', h.get('val_hit_rate', 0))
        if hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            best_epoch = h.get('epoch', 0)
            best_epoch_info = h
    
    print("=" * 70)
    print("최고 성능 체크포인트 찾기")
    print("=" * 70)
    print(f"\n📊 최고 성능:")
    print(f"   Hit Rate@10: {best_hit_rate:.4f} ({best_hit_rate*100:.2f}%)")
    print(f"   Epoch: {best_epoch}")
    
    if best_epoch_info:
        print(f"\n📈 해당 에폭의 상세 정보:")
        print(f"   Train Loss: {best_epoch_info.get('train_loss', 0):.4f}")
        print(f"   Val Hit Rate@1: {best_epoch_info.get('val_hit_rate@1', 0):.4f}")
        print(f"   Val Hit Rate@5: {best_epoch_info.get('val_hit_rate@5', 0):.4f}")
        print(f"   Val Hit Rate@10: {best_hit_rate:.4f}")
        print(f"   MRR: {best_epoch_info.get('val_mrr', 0):.4f}")
        print(f"   NDCG@10: {best_epoch_info.get('val_ndcg', 0):.4f}")
    
    # Check available checkpoints
    print(f"\n📁 사용 가능한 체크포인트:")
    checkpoint_files = {
        f"checkpoint_epoch_{best_epoch}.pt": checkpoint_dir / f"checkpoint_epoch_{best_epoch}.pt",
        f"best_model_epoch_{best_epoch}.pt": checkpoint_dir / f"best_model_epoch_{best_epoch}.pt",
        "final_model.pt": checkpoint_dir / "final_model.pt"
    }
    
    found_checkpoints = []
    for name, path in checkpoint_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   ✓ {name} ({size_mb:.1f} MB)")
            found_checkpoints.append((name, path))
        else:
            print(f"   ✗ {name} (없음)")
    
    # Check all checkpoints
    print(f"\n📋 모든 체크포인트 파일:")
    all_checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    for cp in all_checkpoints:
        size_mb = cp.stat().st_size / (1024 * 1024)
        print(f"   - {cp.name} ({size_mb:.1f} MB)")
    
    # Recommend best checkpoint to use
    print(f"\n💡 권장 사항:")
    if best_hit_rate >= 0.20:  # 20% 이상
        print(f"   🎉 최고 성능이 {best_hit_rate*100:.2f}%로 우수합니다!")
        if found_checkpoints:
            print(f"   ✓ 다음 체크포인트를 사용하세요:")
            print(f"     {found_checkpoints[0][1]}")
    elif best_hit_rate >= 0.10:  # 10% 이상
        print(f"   ⚠️  성능이 {best_hit_rate*100:.2f}%로 개선이 필요합니다.")
        print(f"   ✓ 다음 학습에서 설정을 조정하세요.")
    else:
        print(f"   ⚠️  성능이 {best_hit_rate*100:.2f}%로 낮습니다.")
        print(f"   ✓ 학습 설정을 재검토하세요.")
    
    print("=" * 70)
    
    return found_checkpoints[0][1] if found_checkpoints else None

if __name__ == "__main__":
    import sys
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "results/checkpoints"
    find_best_checkpoint(checkpoint_dir)
