"""
성능 진단: HR@10이 낮게 나올 때 데이터/난이도 원인 파악.

- 조향사 수 → 랜덤 HR@10 기준선
- train/val 조향사 분포 (상위 N개 비율)
- 권장: early_stopping_patience, HR@50 확인 등
"""

import json
import sys
from pathlib import Path
from collections import Counter


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # vocabularies.json
    vocab_path = project_root / "feature_encoding" / "vocabularies.json"
    if not vocab_path.exists():
        vocab_path = project_root.parent / "feature_encoding" / "vocabularies.json"
    if not vocab_path.exists():
        print("❌ vocabularies.json을 찾을 수 없습니다.")
        return
    
    with open(vocab_path) as f:
        vocab = json.load(f)
    
    blenders = vocab.get("blenders", {})
    if isinstance(blenders, dict):
        if "vocab" in blenders:
            num_blenders = len(blenders["vocab"])
        elif "to_idx" in blenders:
            num_blenders = len(blenders["to_idx"])
        else:
            num_blenders = len(blenders)
    else:
        num_blenders = len(blenders)
    
    print("=" * 60)
    print("📊 성능 진단 (HR@10이 낮을 때)")
    print("=" * 60)
    print(f"\n1. 태스크 난이도")
    print(f"   조향사 수: {num_blenders:,}명 (추천 후보 수, 분류 클래스 아님)")
    random_hr1 = 100.0 / num_blenders
    random_hr5 = 500.0 / num_blenders
    random_hr10 = 1000.0 / num_blenders
    print(f"   랜덤 기준선: HR@1 ≈ {random_hr1:.2f}% | HR@5 ≈ {random_hr5:.2f}% | HR@10 ≈ {random_hr10:.2f}%")
    print(f"   → 추천 태스크: 레시피 → 조향사 후보 {num_blenders}명 중 상위 k명 랭킹. HR@10 2~8%면 랜덤 대비 **약 10~40배** 학습된 상태.")
    
    # 저장된 데이터셋에서 조향사 분포 확인
    dataset_dirs = [
        project_root / "hyperbolic_model" / "results" / "checkpoints" / "datasets",
        project_root.parent / "results" / "checkpoints" / "datasets",
    ]
    train_combos = None
    val_combos = None
    for d in dataset_dirs:
        train_path = d / "train_combinations.json"
        val_path = d / "val_combinations.json"
        if train_path.exists():
            with open(train_path) as f:
                train_combos = json.load(f)
            if val_path.exists():
                with open(val_path) as f:
                    val_combos = json.load(f)
            break
    
    if train_combos is not None:
        print(f"\n2. 학습 데이터 조향사 분포 (샘플 수 기준)")
        counts = Counter(c.get("target_blender") for c in train_combos if c.get("target_blender") is not None)
        total = len(train_combos)
        top10 = counts.most_common(10)
        print(f"   총 조합 수: {total:,} | 등장 조향사 수: {len(counts):,}")
        for i, (bid, cnt) in enumerate(top10, 1):
            pct = 100.0 * cnt / total
            print(f"   #{i} blender_id={bid}: {cnt:,}개 ({pct:.1f}%)")
        top1_ratio = top10[0][1] / total if top10 else 0
        if top1_ratio > 0.2:
            print(f"   ⚠️  상위 1개 조향사가 {top1_ratio*100:.0f}% 차지 → 불균형. downsample_top_blender=True 권장.")
        else:
            print(f"   ✓ 상위 조향사 비율 양호 (다운샘플링 적용 시 더 균형)")
        
        if val_combos is not None:
            val_counts = Counter(c.get("target_blender") for c in val_combos if c.get("target_blender") is not None)
            val_total = len(val_combos)
            overlap = len(set(counts) & set(val_counts))
            print(f"\n3. 검증 데이터")
            print(f"   검증 조합 수: {val_total:,} | 검증에 등장한 조향사 수: {len(val_counts):,}")
            print(f"   train과 겹치는 조향사 수: {overlap:,} / {len(counts):,}")
            if overlap < len(counts) * 0.5:
                print(f"   ⚠️  검증에 train에 없는 조향사가 많을 수 있음 (cold-start) → HR@10 낮을 수 있음.")
    else:
        print(f"\n2. 저장된 train/val 조합 없음 (학습 1회 실행 후 다시 실행하면 분포 출력)")
    
    print(f"\n4. 권장 사항")
    print(f"   - 조기 종료 여유: --early_stopping_patience 8 (기본값) 또는 10")
    print(f"   - 후보 5천 명급 추천에서 HR@10 8%는 이미 랜덤 대비 ~45배; 더 끌어올리려면 margin/dropout/LR 튜닝")
    print(f"   - 데이터 불균형 의심 시: scripts/analyze_blender_distribution.py 실행")
    print("=" * 60)


if __name__ == "__main__":
    main()
