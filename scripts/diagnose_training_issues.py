"""
학습 문제 진단 스크립트

데이터 증강과 에폭 증가에도 불구하고 loss가 줄어들지 않고 과적합이 발생하는 원인 분석
"""

import json
import numpy as np
from pathlib import Path
import sys

def diagnose_training_issues(history_path="results/checkpoints/training_history.json"):
    """
    학습 문제 진단
    
    가능한 원인:
    1. 데이터 증강이 실제로는 다양성을 줄임 (같은 조합 반복)
    2. Gaussian Noise가 너무 작아서 효과가 없음 (σ=0.005)
    3. Label Noise 발생 (잘못된 blender 할당)
    4. 학습률이 너무 낮아서 학습이 안 됨
    5. 정규화가 너무 강해서 학습이 안 됨
    6. 조합 생성이 실패해서 실제로는 적은 데이터만 사용
    """
    
    history_path = Path(history_path)
    if not history_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {history_path}")
        print(f"   학습 히스토리 파일이 없습니다. 학습이 시작되지 않았거나 중단되었을 수 있습니다.")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    if not history:
        print("❌ 학습 히스토리가 비어있습니다.")
        return
    
    epochs = [h.get('epoch', i+1) for i, h in enumerate(history)]
    train_losses = [h.get('train_loss', 0) for h in history]
    val_losses = [h.get('val_loss', 0) for h in history]
    val_hit_rates = [h.get('val_hit_rate@10', h.get('val_hit_rate', 0)) for h in history]
    learning_rates = [h.get('learning_rate', 0) for h in history]
    
    print("=" * 80)
    print("🔍 학습 문제 진단: Loss가 줄어들지 않고 과적합 발생")
    print("=" * 80)
    
    # 1. Loss 추이 분석
    print(f"\n1. Loss 추이 분석:")
    if len(train_losses) > 1:
        initial_loss = train_losses[0]
        final_loss = train_losses[-1]
        loss_change = final_loss - initial_loss
        loss_change_pct = (loss_change / initial_loss * 100) if initial_loss > 0 else 0
        
        print(f"   Train Loss:")
        print(f"     초기: {initial_loss:.4f}")
        print(f"     최종: {final_loss:.4f}")
        print(f"     변화: {loss_change:+.4f} ({loss_change_pct:+.1f}%)")
        
        if loss_change_pct > -5:  # 5% 미만 감소
            print(f"   🚨 문제: Train Loss가 거의 감소하지 않습니다!")
            print(f"      - 학습이 제대로 진행되지 않고 있음")
            print(f"      - 가능한 원인:")
            print(f"        1. 학습률이 너무 낮음 (현재: {learning_rates[0] if learning_rates else 'N/A'})")
            print(f"        2. 정규화가 너무 강함 (Dropout 0.25, Weight Decay 1e-4)")
            print(f"        3. Gradient Clipping이 너무 타이트함 (0.5)")
            print(f"        4. 데이터 증강이 실제로는 다양성을 줄임")
        elif loss_change_pct < -50:  # 50% 이상 감소
            print(f"   ✓ Train Loss는 충분히 감소했습니다.")
        else:
            print(f"   ⚠️  Train Loss는 감소하지만 속도가 느립니다.")
        
        # Loss가 증가하는지 확인
        if loss_change > 0:
            print(f"   🚨 심각: Train Loss가 증가하고 있습니다!")
            print(f"      - 모델이 학습하지 못하고 있음")
            print(f"      - 즉시 학습률, 정규화, 데이터 증강 방식을 재검토 필요")
    
    # 2. Validation 성능 분석
    print(f"\n2. Validation 성능 분석:")
    if len(val_hit_rates) > 1:
        initial_hit_rate = val_hit_rates[0]
        final_hit_rate = val_hit_rates[-1]
        best_hit_rate = max(val_hit_rates)
        best_epoch = epochs[val_hit_rates.index(best_hit_rate)]
        
        print(f"   Val Hit Rate@10:")
        print(f"     초기: {initial_hit_rate:.4f}")
        print(f"     최종: {final_hit_rate:.4f}")
        print(f"     최고: {best_hit_rate:.4f} (Epoch {best_epoch})")
        
        if best_epoch < len(epochs) - 1:
            performance_drop = best_hit_rate - final_hit_rate
            print(f"     성능 하락: {performance_drop:.4f}")
            if performance_drop > 0.01:
                print(f"   🚨 과적합 발생!")
                print(f"      - Epoch {best_epoch} 이후 성능 하락")
        else:
            print(f"   ✓ 최종 에폭에서 최고 성능 달성")
        
        if final_hit_rate < 0.1:
            print(f"   🚨 심각: Val Hit Rate가 10% 미만입니다!")
            print(f"      - 모델이 거의 학습하지 못하고 있음")
            print(f"      - 데이터 증강이나 학습 설정에 문제가 있을 가능성")
    
    # 3. 학습률 분석
    print(f"\n3. 학습률 분석:")
    if learning_rates and learning_rates[0] > 0:
        initial_lr = learning_rates[0]
        final_lr = learning_rates[-1] if learning_rates else initial_lr
        
        print(f"   초기 LR: {initial_lr:.6f}")
        print(f"   최종 LR: {final_lr:.6f}")
        
        if initial_lr < 0.0003:
            print(f"   ⚠️  학습률이 너무 낮습니다!")
            print(f"      - 현재: {initial_lr:.6f}")
            print(f"      - 권장: 0.0005 ~ 0.001")
            print(f"      - 영향: 학습 속도가 매우 느려서 loss가 줄어들지 않음")
        elif initial_lr > 0.001:
            print(f"   ⚠️  학습률이 높을 수 있습니다.")
            print(f"      - 과적합 위험 증가")
        else:
            print(f"   ✓ 학습률은 적절합니다.")
        
        # LR 감소 패턴 확인
        if len(learning_rates) > 1:
            lr_reduction = (initial_lr - final_lr) / initial_lr * 100
            print(f"   LR 감소율: {lr_reduction:.1f}%")
            if lr_reduction > 50:
                print(f"   ⚠️  LR이 너무 많이 감소했습니다!")
                print(f"      - Scheduler가 너무 공격적으로 작동")
                print(f"      - 학습이 조기에 멈췄을 가능성")
    
    # 4. 데이터 증강 효과 분석
    print(f"\n4. 데이터 증강 효과 분석:")
    print(f"   현재 방식:")
    print(f"     - 조합 생성: 60,000개 (2~3개 40%, 4~6개 40%, 7~10개+ 20%)")
    print(f"     - Gaussian Noise: σ=0.005 (매우 작음)")
    print(f"   ⚠️  잠재적 문제:")
    print(f"     1. Gaussian Noise가 너무 작아서 효과가 거의 없음")
    print(f"        - σ=0.005는 정규화 후 거의 무시될 수 있음")
    print(f"        - 실제 데이터 다양성 증가가 미미함")
    print(f"     2. 조합 생성이 실제로는 같은 패턴 반복")
    print(f"        - 공통 blender가 없을 때 여러 blender 풀 결합")
    print(f"        - 하지만 여전히 제한된 분자만 사용될 수 있음")
    print(f"     3. Label Noise 가능성")
    print(f"        - 여러 blender 풀 결합 시 잘못된 blender 할당")
    print(f"     4. 실제 생성된 조합 수가 목표보다 적을 수 있음")
    print(f"        - 복잡한 조합 생성 실패로 인해 심플 조합만 생성")
    
    # 5. 정규화 강도 분석
    print(f"\n5. 정규화 강도 분석:")
    print(f"   현재 설정:")
    print(f"     - Dropout: 0.25 (높음)")
    print(f"     - Weight Decay: 1e-4 (높음)")
    print(f"     - Gradient Clipping: 0.5 (타이트함)")
    print(f"   ⚠️  정규화가 너무 강할 수 있습니다!")
    print(f"      - 학습을 방해할 수 있음")
    print(f"      - 특히 학습률이 낮을 때 문제가 됨")
    print(f"      - 권장: Dropout 0.15~0.2, Weight Decay 5e-5")
    
    # 6. 해결 방안
    print(f"\n" + "=" * 80)
    print("💡 해결 방안 (우선순위 순)")
    print("=" * 80)
    
    print(f"\n1. 학습률 조정 (가장 중요):")
    print(f"   - 현재: {learning_rates[0] if learning_rates else 'N/A'}")
    print(f"   - 권장: 0.0005 ~ 0.0008 (약간 상향)")
    print(f"   - 이유: 현재 LR이 너무 낮아서 학습이 안 될 수 있음")
    
    print(f"\n2. 정규화 완화:")
    print(f"   - Dropout: 0.25 → 0.15~0.2")
    print(f"   - Weight Decay: 1e-4 → 5e-5")
    print(f"   - Gradient Clipping: 0.5 → 1.0")
    print(f"   - 이유: 정규화가 너무 강해서 학습을 방해할 수 있음")
    
    print(f"\n3. 데이터 증강 개선:")
    print(f"   - Gaussian Noise: σ=0.005 → σ=0.01~0.02 (증가)")
    print(f"   - 또는 Gaussian Noise 제거하고 다른 증강 방식 사용")
    print(f"   - 이유: 현재 노이즈가 너무 작아서 효과가 거의 없음")
    
    print(f"\n4. 조합 생성 검증:")
    print(f"   - 실제 생성된 조합 수 확인")
    print(f"   - 분자 개수 분포 확인 (목표: 40%, 40%, 20%)")
    print(f"   - 복잡한 조합이 실제로 생성되었는지 확인")
    
    print(f"\n5. 학습 설정 재검토:")
    print(f"   - Warmup Epochs: 7 → 5 (단축)")
    print(f"   - Margin Scheduling: 시작 margin 0.1 → 0.2 (증가)")
    print(f"   - Early Stopping Patience: 5 → 7 (여유)")
    
    print("=" * 80)

if __name__ == "__main__":
    import sys
    history_path = sys.argv[1] if len(sys.argv) > 1 else "results/checkpoints/training_history.json"
    diagnose_training_issues(history_path)
