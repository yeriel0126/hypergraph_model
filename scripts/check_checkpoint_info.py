"""
체크포인트 정보 확인 스크립트

체크포인트 파일에서 직접 성능 정보를 추출합니다.
"""

import torch
import json
from pathlib import Path
import sys
import os

# Fix OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def check_checkpoint_info(checkpoint_path):
    """체크포인트 파일에서 정보 추출"""
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"\n📁 체크포인트: {checkpoint_path.name}")
        print(f"   크기: {checkpoint_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Extract information
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"   Epoch: {epoch}")
        
        # Check for metrics
        val_metrics = checkpoint.get('val_metrics', {})
        train_metrics = checkpoint.get('train_metrics', {})
        
        if val_metrics:
            print(f"\n   📊 Validation Metrics:")
            hit_rate = val_metrics.get('hit_rate@k', val_metrics.get('hit_rate', 'N/A'))
            hit_rate_1 = val_metrics.get('hit_rate@1', 'N/A')
            hit_rate_5 = val_metrics.get('hit_rate@5', 'N/A')
            mrr = val_metrics.get('mrr', 'N/A')
            ndcg = val_metrics.get('ndcg@k', 'N/A')
            
            print(f"      Hit Rate@1: {hit_rate_1}")
            print(f"      Hit Rate@5: {hit_rate_5}")
            print(f"      Hit Rate@10: {hit_rate}")
            print(f"      MRR: {mrr}")
            print(f"      NDCG@10: {ndcg}")
            
            if isinstance(hit_rate, (int, float)) and hit_rate >= 0.20:
                print(f"\n   🎉 우수한 성능! (Hit Rate@10: {hit_rate*100:.2f}%)")
                return {'epoch': epoch, 'hit_rate': hit_rate, 'path': checkpoint_path}
        
        if train_metrics:
            print(f"\n   📊 Training Metrics:")
            train_loss = train_metrics.get('loss', 'N/A')
            print(f"      Train Loss: {train_loss}")
        
        # Check args
        args = checkpoint.get('args', {})
        if args:
            print(f"\n   ⚙️  학습 설정:")
            print(f"      Learning Rate: {args.get('learning_rate', 'N/A')}")
            print(f"      Margin: {args.get('margin', 'N/A')}")
            print(f"      Batch Size: {args.get('batch_size', 'N/A')}")
        
        return None
        
    except Exception as e:
        print(f"   ❌ 읽기 실패: {e}")
        return None

def scan_all_checkpoints(checkpoint_dir="results/checkpoints"):
    """모든 체크포인트 스캔"""
    
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"❌ 디렉토리를 찾을 수 없습니다: {checkpoint_dir}")
        return
    
    print("=" * 70)
    print("모든 체크포인트 스캔")
    print("=" * 70)
    
    checkpoint_files = sorted(checkpoint_dir.glob("*.pt"))
    
    if not checkpoint_files:
        print("❌ 체크포인트 파일이 없습니다.")
        return
    
    print(f"\n총 {len(checkpoint_files)}개의 체크포인트 파일 발견\n")
    
    best_checkpoint = None
    best_hit_rate = 0.0
    
    for cp_file in checkpoint_files:
        result = check_checkpoint_info(cp_file)
        if result and result['hit_rate'] > best_hit_rate:
            best_hit_rate = result['hit_rate']
            best_checkpoint = result
    
    print("\n" + "=" * 70)
    if best_checkpoint:
        print(f"🏆 최고 성능 체크포인트:")
        print(f"   파일: {best_checkpoint['path'].name}")
        print(f"   Epoch: {best_checkpoint['epoch']}")
        print(f"   Hit Rate@10: {best_checkpoint['hit_rate']:.4f} ({best_checkpoint['hit_rate']*100:.2f}%)")
        print(f"   경로: {best_checkpoint['path']}")
    else:
        print("⚠️  25% 이상 성능을 달성한 체크포인트를 찾지 못했습니다.")
        print("   모든 체크포인트의 성능 정보가 없거나 낮은 성능입니다.")
    print("=" * 70)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        check_checkpoint_info(checkpoint_path)
    else:
        checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "results/checkpoints"
        scan_all_checkpoints(checkpoint_dir)
