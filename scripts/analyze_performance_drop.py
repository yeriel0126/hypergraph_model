"""
성능 하락 원인 분석 스크립트

이전 성능(25%) 대비 현재 성능이 낮아진 원인을 분석합니다.
"""

import json
from pathlib import Path
import numpy as np

def analyze_performance_drop(history_path="results/checkpoints/training_history.json"):
    """성능 하락 원인 분석"""
    
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
    learning_rates = [h.get('learning_rate', 0) for h in history]
    
    print("=" * 70)
    print("성능 하락 원인 분석")
    print("=" * 70)
    
    # 현재 성능
    best_val_hit_rate = max(val_hit_rates)
    best_epoch = epochs[val_hit_rates.index(best_val_hit_rate)]
    final_val_hit_rate = val_hit_rates[-1]
    
    print(f"\n📊 현재 성능:")
    print(f"   최고 Hit Rate@10: {best_val_hit_rate:.4f} ({best_val_hit_rate*100:.2f}%) - Epoch {best_epoch}")
    print(f"   최종 Hit Rate@10: {final_val_hit_rate:.4f} ({final_val_hit_rate*100:.2f}%)")
    print(f"   이전 목표 성능: 0.25 (25%)")
    print(f"   성능 차이: {0.25 - best_val_hit_rate:.4f} ({((0.25 - best_val_hit_rate) / 0.25 * 100):.1f}% 낮음)")
    
    # 원인 분석
    print(f"\n🔍 성능 하락 가능 원인:")
    
    # 1. Learning Rate 분석
    if learning_rates and learning_rates[0] > 0:
        initial_lr = learning_rates[0]
        print(f"\n1. Learning Rate:")
        print(f"   초기 LR: {initial_lr:.6f}")
        if initial_lr < 0.0005:
            print(f"   ⚠️  Learning Rate가 너무 낮습니다!")
            print(f"      - 현재: {initial_lr:.6f}")
            print(f"      - 권장: 0.0005 ~ 0.001")
            print(f"      - 영향: 학습 속도가 느려서 충분한 학습이 이루어지지 않음")
        else:
            print(f"   ✓ Learning Rate는 적절합니다.")
    
    # 2. 학습 속도 분석
    print(f"\n2. 학습 속도 분석:")
    if len(train_losses) > 1:
        loss_reduction_rate = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
        print(f"   Loss 감소율: {loss_reduction_rate:.1f}%")
        if loss_reduction_rate < 50:
            print(f"   ⚠️  Loss 감소가 느립니다!")
            print(f"      - 충분한 학습이 이루어지지 않았을 수 있음")
        else:
            print(f"   ✓ Loss는 충분히 감소했습니다.")
    
    # 3. 과적합 분석
    print(f"\n3. 과적합 분석:")
    if best_epoch < epochs[-1]:
        performance_drop = best_val_hit_rate - final_val_hit_rate
        print(f"   최고 성능 이후 하락: {performance_drop:.4f}")
        if performance_drop > 0.01:
            print(f"   🚨 명백한 과적합 발생!")
            print(f"      - Epoch {best_epoch} 이후 성능 하락")
            print(f"      - Early Stopping이 제대로 작동하지 않음")
    
    # 4. Margin 스케줄링 영향
    print(f"\n4. Margin 스케줄링 영향:")
    print(f"   현재 설정: 0.2 → 0.4 (선형 증가)")
    print(f"   ⚠️  초기 Margin이 낮아서 학습이 느릴 수 있음")
    print(f"      - Epoch 1-5: Margin ≈ 0.2 (너무 쉬운 문제)")
    print(f"      - 권장: 초기 Margin을 0.3으로 시작")
    
    # 5. Warmup 영향
    print(f"\n5. Warmup 영향:")
    print(f"   현재 설정: 5 에폭")
    print(f"   ⚠️  Warmup이 길어서 초기 학습이 느릴 수 있음")
    print(f"      - Epoch 1-5: Learning Rate가 서서히 증가")
    print(f"      - 권장: 3 에폭으로 단축")
    
    # 권장 사항
    print(f"\n💡 성능 개선 권장 사항:")
    print(f"   1. Learning Rate 증가: 0.0002 → 0.0005 (2.5배)")
    print(f"   2. Warmup 단축: 5 → 3 에폭")
    print(f"   3. Margin 스케줄링 조정: 0.3 → 0.4 (초기값 상향)")
    print(f"   4. Early Stopping patience: 5 → 3 (과적합 방지)")
    print(f"   5. 더 많은 에폭 학습: 20 → 30 에폭")
    
    print("=" * 70)

if __name__ == "__main__":
    import sys
    history_path = sys.argv[1] if len(sys.argv) > 1 else "results/checkpoints/training_history.json"
    analyze_performance_drop(history_path)
