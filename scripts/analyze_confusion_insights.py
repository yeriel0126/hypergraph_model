"""
조향사 혼동 패턴 분석 인사이트 생성 스크립트

blender_confusion_analysis.json 파일을 읽어서:
1. 허브 조향사 식별 (여러 조향사가 혼동되는 타겟)
2. 완전 오분류 조향사 식별
3. 혼동 쌍 클러스터 분석
4. 개선 방안 제시
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

def analyze_confusion_insights(confusion_file: str):
    """혼동 분석 결과에서 인사이트 추출"""
    
    with open(confusion_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("🔍 조향사 혼동 패턴 인사이트 분석")
    print("=" * 80)
    
    # 1. 허브 조향사 분석 (여러 조향사가 혼동되는 타겟)
    print("\n📌 1. 허브 조향사 (Hub Blenders) 분석")
    print("-" * 80)
    
    predicted_counter = Counter()
    confusion_map = defaultdict(list)
    
    for pair in data['most_confusing_pairs']:
        pred_idx = pair['predicted_idx']
        true_idx = pair['true_idx']
        count = pair['confusion_count']
        predicted_counter[pred_idx] += count
        confusion_map[pred_idx].append((true_idx, count))
    
    # 상위 허브 조향사
    top_hubs = predicted_counter.most_common(10)
    print("\n가장 자주 혼동되는 타겟 조향사 (허브):")
    for rank, (hub_idx, total_confusions) in enumerate(top_hubs, 1):
        confused_from = confusion_map[hub_idx]
        print(f"\n   {rank}. Blender_{hub_idx} (총 {total_confusions}회 혼동)")
        print(f"      ← 혼동되는 조향사들:")
        for true_idx, count in sorted(confused_from, key=lambda x: x[1], reverse=True)[:5]:
            print(f"         - Blender_{true_idx}: {count}회")
    
    # 2. 완전 오분류 조향사 분석
    print("\n\n📌 2. 완전 오분류 조향사 (100% 오류율) 분석")
    print("-" * 80)
    
    confused_blenders = data['most_confused_blenders']
    print(f"\n총 {len(confused_blenders)}개 조향사가 100% 오류율을 보입니다:")
    
    # 샘플 수 기준으로 정렬
    sorted_by_samples = sorted(confused_blenders, key=lambda x: x['total_count'], reverse=True)
    
    print("\n샘플 수가 많은 순서:")
    for rank, blender in enumerate(sorted_by_samples[:10], 1):
        print(f"   {rank}. Blender_{blender['blender_idx']}: "
              f"{blender['total_count']}개 샘플 모두 오분류")
    
    # 3. 혼동 쌍 클러스터 분석
    print("\n\n📌 3. 혼동 쌍 클러스터 분석")
    print("-" * 80)
    
    # 양방향 혼동 쌍 찾기 (A→B와 B→A가 모두 존재)
    bidirectional_pairs = defaultdict(int)
    pair_set = set()
    
    for pair in data['most_confusing_pairs']:
        true_idx = pair['true_idx']
        pred_idx = pair['predicted_idx']
        if true_idx != pred_idx:
            pair_key = tuple(sorted([true_idx, pred_idx]))
            pair_set.add(pair_key)
            bidirectional_pairs[pair_key] += pair['confusion_count']
    
    print(f"\n총 {len(pair_set)}개의 고유 혼동 쌍이 발견되었습니다.")
    
    # 가장 빈번한 혼동 쌍
    top_pairs = sorted(bidirectional_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n가장 빈번한 혼동 쌍:")
    for rank, ((idx1, idx2), count) in enumerate(top_pairs, 1):
        print(f"   {rank}. Blender_{idx1} ↔ Blender_{idx2}: {count}회")
    
    # 4. 개선 방안 제시
    print("\n\n📌 4. 개선 방안 제시")
    print("-" * 80)
    
    print("\n🎯 우선순위별 개선 전략:")
    
    print("\n   1️⃣  허브 조향사 분리 강화")
    print(f"      - Blender_{top_hubs[0][0]}이 {top_hubs[0][1]}회 혼동의 타겟이 됨")
    print(f"      - 이 조향사의 임베딩이 과도하게 중심적일 가능성")
    print(f"      - 해결책:")
    print(f"        • Blender_{top_hubs[0][0]}에 대한 Hard Negative Mining 강화")
    print(f"        • 혼동되는 조향사들과의 거리 명시적 확대")
    print(f"        • 해당 조향사 데이터 증강")
    
    print("\n   2️⃣  완전 오분류 조향사 데이터 보강")
    top_confused = sorted_by_samples[0]
    print(f"      - Blender_{top_confused['blender_idx']}: {top_confused['total_count']}개 샘플 모두 오분류")
    print(f"      - 해결책:")
    print(f"        • 해당 조향사들의 학습 데이터 추가 수집")
    print(f"        • 클래스 불균형 해소 (Class Weight 적용)")
    print(f"        • Focal Loss 또는 다른 불균형 대응 손실 함수 고려")
    
    print("\n   3️⃣  혼동 쌍 명시적 구분 학습")
    print(f"      - 상위 {len(top_pairs)}개 혼동 쌍에 대한 특별 학습")
    print(f"      - 해결책:")
    print(f"        • 혼동 쌍에 대한 가중치 손실 적용")
    print(f"        • Contrastive Learning으로 유사 조향사 구분 강화")
    print(f"        • Triplet Loss의 margin을 혼동 쌍에 대해 증가")
    
    # 5. 통계 요약
    print("\n\n📊 통계 요약")
    print("-" * 80)
    print(f"   • 총 분석 샘플 수: {data['total_samples']:,}개")
    print(f"   • 완전 오분류 조향사 수: {len(confused_blenders)}개")
    print(f"   • 고유 혼동 쌍 수: {len(pair_set)}개")
    print(f"   • 허브 조향사 수 (5회 이상 혼동): {len([x for x in top_hubs if x[1] >= 5])}개")
    
    # 결과 저장
    insights = {
        'hub_blenders': [
            {
                'blender_idx': hub_idx,
                'total_confusions': total_confusions,
                'confused_from': [{'blender_idx': idx, 'count': count} 
                                 for idx, count in confusion_map[hub_idx]]
            }
            for hub_idx, total_confusions in top_hubs
        ],
        'completely_confused_blenders': sorted_by_samples,
        'top_confusion_pairs': [
            {
                'blender_1': idx1,
                'blender_2': idx2,
                'confusion_count': count
            }
            for (idx1, idx2), count in top_pairs
        ],
        'recommendations': {
            'priority_1': f"Blender_{top_hubs[0][0]} 허브 분리 강화",
            'priority_2': f"Blender_{top_confused['blender_idx']} 데이터 보강",
            'priority_3': f"{len(top_pairs)}개 혼동 쌍 명시적 구분 학습"
        }
    }
    
    output_file = Path(confusion_file).parent / "confusion_insights.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 인사이트 결과 저장: {output_file}")
    except PermissionError:
        # Fallback to current directory if permission denied
        output_file = Path("confusion_insights.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 인사이트 결과 저장 (현재 디렉토리): {output_file.absolute()}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="조향사 혼동 패턴 인사이트 분석")
    parser.add_argument(
        '--confusion_file',
        type=str,
        default='../results/checkpoints/blender_confusion_analysis.json',
        help='혼동 분석 결과 JSON 파일 경로'
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    confusion_path = script_dir / args.confusion_file if not Path(args.confusion_file).is_absolute() else Path(args.confusion_file)
    
    if not confusion_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {confusion_path}")
        exit(1)
    
    analyze_confusion_insights(str(confusion_path))
