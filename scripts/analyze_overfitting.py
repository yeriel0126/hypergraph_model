"""
과적합 분석 스크립트

학습 결과를 분석하여 과적합 여부를 판단합니다.
"""

import json
from pathlib import Path
import numpy as np

def analyze_overfitting(history_path="results/checkpoints/training_history.json"):
    """과적합 분석"""
    
    history_path = Path(history_path)
    if not history_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    if not history:
        print("❌ 학습 히스토리가 비어있습니다.")
        return
    
    epochs = [h.get('epoch', i+1) for i, h in enumerate(history)]
    train_losses = [h.get('train_loss', 0) for h in history]
    val_hit_rates = [h.get('val_hit_rate@10', h.get('val_hit_rate', 0)) for h in history]
    
    print("=" * 70)
    print("과적합 분석 결과")
    print("=" * 70)
    
    # 1. Train Loss 추이
    print(f"\n1. Train Loss 추이:")
    print(f"   초기: {train_losses[0]:.4f}")
    print(f"   최종: {train_losses[-1]:.4f}")
    print(f"   변화: {train_losses[-1] - train_losses[0]:.4f}")
    print(f"   최소: {min(train_losses):.4f} (Epoch {epochs[train_losses.index(min(train_losses))]})")
    
    # 2. Validation Hit Rate 추이
    print(f"\n2. Validation Hit Rate@10 추이:")
    print(f"   초기: {val_hit_rates[0]:.4f}")
    print(f"   최종: {val_hit_rates[-1]:.4f}")
    print(f"   변화: {val_hit_rates[-1] - val_hit_rates[0]:.4f}")
    print(f"   최대: {max(val_hit_rates):.4f} (Epoch {epochs[val_hit_rates.index(max(val_hit_rates))]})")
    
    # 3. 과적합 패턴 분석
    best_epoch = epochs[val_hit_rates.index(max(val_hit_rates))]
    best_val_hit_rate = max(val_hit_rates)
    final_val_hit_rate = val_hit_rates[-1]
    
    print(f"\n3. 과적합 패턴 분석:")
    print(f"   최고 성능: Epoch {best_epoch} ({best_val_hit_rate:.4f})")
    print(f"   최종 성능: Epoch {epochs[-1]} ({final_val_hit_rate:.4f})")
    
    if best_epoch < epochs[-1]:
        performance_drop = best_val_hit_rate - final_val_hit_rate
        drop_percentage = (performance_drop / best_val_hit_rate) * 100 if best_val_hit_rate > 0 else 0
        print(f"   성능 하락: {performance_drop:.4f} ({drop_percentage:.1f}%)")
        
        if performance_drop > 0.01:  # 1% 이상 하락
            print(f"\n   ⚠️  과적합 감지!")
            print(f"      - Epoch {best_epoch} 이후 성능이 {drop_percentage:.1f}% 하락했습니다.")
            print(f"      - Train Loss는 계속 감소하지만 Val Hit Rate는 하락했습니다.")
            print(f"      - 권장: Epoch {best_epoch}에서 학습을 중단하거나 정규화를 강화하세요.")
        else:
            print(f"\n   ✓ 성능 하락이 미미합니다. 과적합 가능성 낮음.")
    else:
        print(f"\n   ✓ 최종 에폭에서 최고 성능을 달성했습니다.")
    
    # 4. Train Loss vs Val Hit Rate 비교
    print(f"\n4. Train Loss vs Val Hit Rate 비교:")
    train_loss_at_best = train_losses[val_hit_rates.index(max(val_hit_rates))]
    train_loss_final = train_losses[-1]
    train_loss_improvement = train_loss_at_best - train_loss_final
    
    print(f"   최고 Val 성능 시 Train Loss: {train_loss_at_best:.4f}")
    print(f"   최종 Train Loss: {train_loss_final:.4f}")
    print(f"   Train Loss 추가 개선: {train_loss_improvement:.4f}")
    
    if train_loss_improvement > 0.05 and performance_drop > 0.01:
        print(f"\n   🚨 명백한 과적합 패턴!")
        print(f"      - Train Loss는 {train_loss_improvement:.4f} 추가로 감소")
        print(f"      - Val Hit Rate는 {performance_drop:.4f} 하락")
        print(f"      - 모델이 학습 데이터에 과도하게 적합되었습니다.")
    
    # 5. 권장 사항
    print(f"\n5. 권장 사항:")
    if performance_drop > 0.01:
        print(f"   ✓ Epoch {best_epoch}의 체크포인트를 사용하세요 (최고 성능)")
        print(f"   ✓ Early Stopping patience를 줄이세요 (현재: 5 → 3 권장)")
        print(f"   ✓ 정규화 강화:")
        print(f"     - Dropout 증가: 0.1 → 0.2")
        print(f"     - Weight Decay 증가: 1e-5 → 1e-4")
        print(f"     - Learning Rate 감소: 현재 LR의 1/2")
        print(f"   ✓ 데이터 증강 또는 더 많은 데이터 사용")
    else:
        print(f"   ✓ 현재 설정이 적절합니다.")
        print(f"   ✓ 더 많은 에폭으로 학습을 계속할 수 있습니다.")
    
    print("=" * 70)

if __name__ == "__main__":
    import sys
    history_path = sys.argv[1] if len(sys.argv) > 1 else "results/checkpoints/training_history.json"
    analyze_overfitting(history_path)
