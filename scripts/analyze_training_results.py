"""
Analyze training results and provide evaluation
"""

import torch
import json
from pathlib import Path

def analyze_training():
    checkpoint_dir = Path("checkpoints")
    
    print("=" * 70)
    print("학습 결과 분석")
    print("=" * 70)
    print()
    
    # Load training history
    history_path = checkpoint_dir / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        print("📊 학습 히스토리 요약:")
        print("-" * 70)
        if history:
            train_losses = [h.get('train_loss', 0) for h in history]
            val_hit_rates = [h.get('val_hit_rate', 0) for h in history]
            
            print(f"총 에폭 수: {len(history)}")
            print(f"평균 Train Loss: {sum(train_losses) / len(train_losses):.6f}")
            print(f"평균 Val Hit Rate@10: {sum(val_hit_rates) / len(val_hit_rates):.6f}")
            print(f"최종 Train Loss: {train_losses[-1]:.6f}")
            print(f"최종 Val Hit Rate@10: {val_hit_rates[-1]:.6f}")
            
            # Check if loss is decreasing
            if len(train_losses) > 1:
                loss_trend = "감소" if train_losses[-1] < train_losses[0] else "증가" if train_losses[-1] > train_losses[0] else "변화 없음"
                print(f"Loss 추세: {loss_trend}")
            
            # Check if hit rate is increasing
            if len(val_hit_rates) > 1:
                hit_rate_trend = "증가" if val_hit_rates[-1] > val_hit_rates[0] else "감소" if val_hit_rates[-1] < val_hit_rates[0] else "변화 없음"
                print(f"Hit Rate 추세: {hit_rate_trend}")
        print()
    
    # Load final model
    final_path = checkpoint_dir / "final_model.pt"
    if final_path.exists():
        print("📁 Final Model 체크포인트:")
        print("-" * 70)
        checkpoint = torch.load(final_path, map_location='cpu', weights_only=False)
        print(f"에폭: {checkpoint.get('epoch', 'N/A')}")
        
        if 'train_metrics' in checkpoint and checkpoint['train_metrics']:
            tm = checkpoint['train_metrics']
            print(f"  Train Loss: {tm.get('loss', 'N/A')}")
            print(f"  Train Batches: {tm.get('num_batches', 'N/A')}")
        
        if 'val_metrics' in checkpoint and checkpoint['val_metrics']:
            vm = checkpoint['val_metrics']
            print(f"  Val Hit Rate@10: {vm.get('hit_rate@k', 'N/A'):.4f}")
            print(f"  Val Samples: {vm.get('total_samples', 'N/A')}")
            print(f"  Val Top-K Hits: {vm.get('top_k_hits', 'N/A')}")
        
        if 'test_metrics' in checkpoint and checkpoint['test_metrics']:
            testm = checkpoint['test_metrics']
            print(f"  Test Hit Rate@10: {testm.get('hit_rate@k', 'N/A'):.4f}")
            print(f"  Test Samples: {testm.get('total_samples', 'N/A')}")
        print()
    
    # Load best model
    best_files = list(checkpoint_dir.glob("best_model_epoch_*.pt"))
    if best_files:
        best_file = sorted(best_files, key=lambda x: int(x.stem.split('_')[-1]))[-1]
        print("🏆 Best Model 체크포인트:")
        print("-" * 70)
        checkpoint = torch.load(best_file, map_location='cpu', weights_only=False)
        print(f"파일: {best_file.name}")
        print(f"에폭: {checkpoint.get('epoch', 'N/A')}")
        
        if 'train_metrics' in checkpoint and checkpoint['train_metrics']:
            tm = checkpoint['train_metrics']
            print(f"  Train Loss: {tm.get('loss', 'N/A')}")
        
        if 'val_metrics' in checkpoint and checkpoint['val_metrics']:
            vm = checkpoint['val_metrics']
            print(f"  Val Hit Rate@10: {vm.get('hit_rate@k', 'N/A'):.4f}")
        
        print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
        print()
    
    # Evaluation
    print("=" * 70)
    print("📈 학습 품질 평가")
    print("=" * 70)
    
    # Check for issues
    issues = []
    warnings = []
    
    if history:
        train_losses = [h.get('train_loss', 0) for h in history]
        val_hit_rates = [h.get('val_hit_rate', 0) for h in history]
        
        # Check if all losses are zero
        if all(loss == 0.0 for loss in train_losses):
            issues.append("⚠️  모든 에폭의 Train Loss가 0.0입니다. 학습이 제대로 되지 않았을 수 있습니다.")
        
        # Check if all hit rates are zero
        if all(hr == 0.0 for hr in val_hit_rates):
            issues.append("⚠️  모든 에폭의 Val Hit Rate가 0.0입니다. 모델이 예측을 하지 못하고 있습니다.")
        
        # Check if loss is not decreasing
        if len(train_losses) > 1 and train_losses[-1] >= train_losses[0]:
            warnings.append("⚠️  Loss가 감소하지 않습니다. 학습률이나 모델 구조를 확인하세요.")
        
        # Check if hit rate is not improving
        if len(val_hit_rates) > 1 and val_hit_rates[-1] <= val_hit_rates[0]:
            warnings.append("⚠️  Hit Rate가 개선되지 않습니다. 모델이 학습되지 않고 있을 수 있습니다.")
        
        # Check if loss is NaN or Inf
        if any(not isinstance(loss, (int, float)) or loss != loss or abs(loss) == float('inf') for loss in train_losses):
            issues.append("❌ Loss에 NaN 또는 Inf 값이 있습니다. 수치적 불안정성이 발생했습니다.")
    
    if issues:
        print("\n🚨 발견된 문제:")
        for issue in issues:
            print(f"  {issue}")
    
    if warnings:
        print("\n⚠️  주의사항:")
        for warning in warnings:
            print(f"  {warning}")
    
    if not issues and not warnings:
        print("\n✅ 학습이 정상적으로 진행된 것으로 보입니다!")
        if history:
            train_losses = [h.get('train_loss', 0) for h in history]
            val_hit_rates = [h.get('val_hit_rate', 0) for h in history]
            if train_losses[-1] < 1.0 and val_hit_rates[-1] > 0.1:
                print("   Loss가 낮고 Hit Rate가 양호합니다.")
    
    print()
    print("=" * 70)

if __name__ == "__main__":
    analyze_training()

