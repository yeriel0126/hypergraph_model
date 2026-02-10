"""
과적합 원인 분석 스크립트

데이터 증강과 에폭 증가에도 불구하고 과적합이 발생하는 원인을 분석합니다.
"""

import json
import numpy as np
from pathlib import Path
import sys

def analyze_overfitting_cause(history_path="results/checkpoints/training_history.json"):
    """
    과적합 원인 분석
    
    일반적인 기대: 데이터 증강 + 에폭 증가 → 과적합 감소
    실제 현상: 과적합 발생
    
    가능한 원인:
    1. 조합 생성이 실제로는 다양성을 줄임 (같은 분자 조합 반복)
    2. 정규화가 부족함 (Dropout 0.1, Weight Decay 1e-5)
    3. 모델 용량이 데이터에 비해 큼
    4. 학습률이 너무 높아서 빠르게 수렴 후 과적합
    5. 조합 생성 시 Label Noise (잘못된 blender 할당)
    """
    
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
    
    print("=" * 80)
    print("🔍 과적합 원인 분석: 데이터 증강 + 에폭 증가에도 불구하고 과적합 발생")
    print("=" * 80)
    
    # 1. 과적합 패턴 확인
    best_epoch = epochs[val_hit_rates.index(max(val_hit_rates))]
    best_val_hit_rate = max(val_hit_rates)
    final_val_hit_rate = val_hit_rates[-1]
    performance_drop = best_val_hit_rate - final_val_hit_rate if best_epoch < epochs[-1] else 0
    
    print(f"\n1. 과적합 패턴 확인:")
    print(f"   최고 성능: Epoch {best_epoch} ({best_val_hit_rate:.4f})")
    print(f"   최종 성능: Epoch {epochs[-1]} ({final_val_hit_rate:.4f})")
    if performance_drop > 0:
        print(f"   성능 하락: {performance_drop:.4f} ({performance_drop/best_val_hit_rate*100:.1f}%)")
        print(f"   🚨 명백한 과적합 발생!")
    else:
        print(f"   ✓ 과적합 패턴이 명확하지 않습니다.")
    
    # 2. 학습 속도 분석
    print(f"\n2. 학습 속도 분석:")
    if len(train_losses) > 1:
        loss_reduction_rate = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
        print(f"   Loss 감소율: {loss_reduction_rate:.1f}%")
        
        # Loss가 빠르게 감소하면 과적합 가능성 높음
        if loss_reduction_rate > 80:
            print(f"   ⚠️  Loss가 너무 빠르게 감소합니다!")
            print(f"      - 모델이 학습 데이터에 빠르게 적합됨")
            print(f"      - 일반화 능력이 떨어질 수 있음")
        
        # Loss 감소 vs Val 성능 개선 비교
        loss_improvement = train_losses[0] - train_losses[-1]
        val_improvement = final_val_hit_rate - val_hit_rates[0]
        
        if loss_improvement > 0.5 and val_improvement < 0.05:
            print(f"   🚨 Train Loss는 크게 개선되지만 Val 성능은 거의 개선되지 않음!")
            print(f"      - Train Loss 개선: {loss_improvement:.4f}")
            print(f"      - Val Hit Rate 개선: {val_improvement:.4f}")
            print(f"      - 이는 과적합의 명확한 신호입니다.")
    
    # 3. 정규화 강도 분석
    print(f"\n3. 정규화 강도 분석:")
    print(f"   현재 설정:")
    print(f"   - Dropout: 0.1 (매우 낮음)")
    print(f"   - Weight Decay: 1e-5 (매우 낮음)")
    print(f"   - Gradient Clipping: 0.5 (적절)")
    print(f"   ⚠️  정규화가 부족합니다!")
    print(f"      - 60K 데이터셋에 대해 Dropout 0.1은 너무 낮음")
    print(f"      - Weight Decay 1e-5는 거의 효과가 없음")
    print(f"      - 권장: Dropout 0.2~0.3, Weight Decay 1e-4~1e-3")
    
    # 4. 학습률 분석
    print(f"\n4. 학습률 분석:")
    if learning_rates and learning_rates[0] > 0:
        initial_lr = learning_rates[0]
        final_lr = learning_rates[-1] if learning_rates else initial_lr
        print(f"   초기 LR: {initial_lr:.6f}")
        print(f"   최종 LR: {final_lr:.6f}")
        
        if initial_lr > 0.001:
            print(f"   ⚠️  Learning Rate가 높습니다!")
            print(f"      - 높은 LR은 빠른 수렴을 유도하지만 과적합 위험 증가")
            print(f"      - 권장: 0.0002~0.0005 (더 천천히 학습)")
        
        # LR 감소 패턴 확인
        if len(learning_rates) > 1:
            lr_reduction = (initial_lr - final_lr) / initial_lr * 100
            print(f"   LR 감소율: {lr_reduction:.1f}%")
            if lr_reduction < 20:
                print(f"   ⚠️  LR이 충분히 감소하지 않았습니다!")
                print(f"      - Scheduler가 제대로 작동하지 않았을 수 있음")
    
    # 5. 데이터 증강 효과 분석
    print(f"\n5. 데이터 증강 효과 분석:")
    print(f"   현재 방식: 조합 생성 (Combination Generation)")
    print(f"   ⚠️  잠재적 문제점:")
    print(f"      1. 조합 생성이 실제로는 다양성을 줄일 수 있음")
    print(f"         - 같은 분자 조합이 반복될 수 있음")
    print(f"         - 원본 데이터의 패턴을 단순히 복제할 수 있음")
    print(f"      2. Label Noise 가능성")
    print(f"         - 공통 blender가 없을 때 첫 번째 분자의 blender 사용")
    print(f"         - 잘못된 라벨로 인한 학습 혼란")
    print(f"      3. 실제 증강 효과가 제한적")
    print(f"         - 원본 데이터에 노이즈를 추가하는 것이 아니라")
    print(f"         - 단순히 조합만 생성하므로 일반화 능력 향상이 제한적")
    print(f"   권장:")
    print(f"      - 원본 데이터에 Gaussian Noise 추가 (실제 증강)")
    print(f"      - Mixup/CutMix 같은 고급 증강 기법 적용")
    print(f"      - Label Smoothing 적용")
    
    # 6. 모델 용량 분석
    print(f"\n6. 모델 용량 분석:")
    print(f"   현재 설정:")
    print(f"   - GNN Layers: 3")
    print(f"   - Hidden Dim: 128")
    print(f"   - Output Dim: 128")
    print(f"   - Attention Heads: 4")
    print(f"   ⚠️  모델이 데이터에 비해 클 수 있음")
    print(f"      - 60K 데이터셋에 대해 현재 모델은 적절할 수 있으나")
    print(f"      - 정규화가 부족하면 과적합 위험 증가")
    
    # 7. 해결 방안
    print(f"\n" + "=" * 80)
    print("💡 해결 방안 (우선순위 순)")
    print("=" * 80)
    
    print(f"\n1. 정규화 강화 (가장 중요):")
    print(f"   - Dropout: 0.1 → 0.25~0.3")
    print(f"   - Weight Decay: 1e-5 → 1e-4~1e-3")
    print(f"   - Label Smoothing: 0.1 추가")
    
    print(f"\n2. 학습률 조정:")
    print(f"   - 초기 LR: 0.001 → 0.0003~0.0005")
    print(f"   - Warmup: 7 → 10 epochs")
    print(f"   - LR Scheduler: ReduceLROnPlateau patience 5 → 3")
    
    print(f"\n3. 데이터 증강 개선:")
    print(f"   - 원본 데이터에 Gaussian Noise 추가 (σ=0.005)")
    print(f"   - Mixup 적용 (α=0.2)")
    print(f"   - Label Smoothing (ε=0.1)")
    
    print(f"\n4. Early Stopping 강화:")
    print(f"   - Patience: 7 → 5")
    print(f"   - Val Loss 기준으로 조기 중단")
    
    print(f"\n5. 모델 구조 조정 (선택적):")
    print(f"   - Dropout을 각 레이어에 추가")
    print(f"   - Batch Normalization 추가 (하이퍼볼릭 공간에서는 제한적)")
    
    print("=" * 80)

if __name__ == "__main__":
    import sys
    history_path = sys.argv[1] if len(sys.argv) > 1 else "results/checkpoints/training_history.json"
    analyze_overfitting_cause(history_path)
