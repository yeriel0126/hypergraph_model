"""
데이터 증강(Recipe Combinations) 검증 스크립트

확인 사항:
1. 총 조합 수가 6만개인지
2. 분자 개수별 분포: 2~3개(40%), 4~6개(40%), 7~10개+(20%)
3. 조향사별 샘플 수 불균형 보정 여부
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import sys

def verify_combinations(combinations_file: str = None):
    """조합 데이터 검증"""
    
    # 기본 경로 설정
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # 조합 파일 찾기
    if combinations_file is None:
        # 학습 스크립트가 생성한 조합을 직접 확인할 수 없으므로,
        # 학습 로그나 체크포인트에서 확인하거나, 학습 스크립트를 실행해서 확인해야 함
        print("⚠️  조합 파일이 지정되지 않았습니다.")
        print("   학습 스크립트를 실행하면 조합 생성 로그가 출력됩니다.")
        print("   또는 학습 스크립트에 조합 저장 기능을 추가해야 합니다.")
        return
    
    # 조합 파일 로드
    if not Path(combinations_file).exists():
        print(f"❌ 조합 파일을 찾을 수 없습니다: {combinations_file}")
        return
    
    with open(combinations_file, 'r', encoding='utf-8') as f:
        combinations = json.load(f)
    
    if isinstance(combinations, dict):
        combinations = combinations.get('data', combinations.get('combinations', []))
    
    print("=" * 80)
    print("📊 Recipe Combinations 검증 리포트")
    print("=" * 80)
    
    # 1. 총 조합 수 확인
    total_combinations = len(combinations)
    print(f"\n1. 총 조합 수: {total_combinations:,}개")
    if total_combinations >= 60000:
        print(f"   ✅ 목표 달성 (6만개 이상)")
    else:
        print(f"   ⚠️  목표 미달성 (목표: 60,000개, 현재: {total_combinations:,}개)")
    
    # 2. 분자 개수별 분포 확인
    print(f"\n2. 분자 개수별 분포:")
    print("-" * 80)
    
    molecule_counts = []
    for combo in combinations:
        num_mols = len(combo.get('molecules', []))
        molecule_counts.append(num_mols)
    
    molecule_counter = Counter(molecule_counts)
    
    # 분류
    simple_count = sum(count for num_mols, count in molecule_counter.items() if 2 <= num_mols <= 3)
    medium_count = sum(count for num_mols, count in molecule_counter.items() if 4 <= num_mols <= 6)
    complex_count = sum(count for num_mols, count in molecule_counter.items() if 7 <= num_mols <= 10)
    other_count = total_combinations - simple_count - medium_count - complex_count
    
    simple_pct = (simple_count / total_combinations * 100) if total_combinations > 0 else 0
    medium_pct = (medium_count / total_combinations * 100) if total_combinations > 0 else 0
    complex_pct = (complex_count / total_combinations * 100) if total_combinations > 0 else 0
    
    print(f"   {'구분':<20} {'개수':<15} {'비율':<15} {'목표 비율':<15} {'상태'}")
    print("-" * 80)
    print(f"   {'2~3개 (심플)':<20} {simple_count:<15,} {simple_pct:<14.1f}% {'40%':<15} {'✅' if 35 <= simple_pct <= 45 else '⚠️'}")
    print(f"   {'4~6개 (중간)':<20} {medium_count:<15,} {medium_pct:<14.1f}% {'40%':<15} {'✅' if 35 <= medium_pct <= 45 else '⚠️'}")
    print(f"   {'7~10개+ (복잡)':<20} {complex_count:<15,} {complex_pct:<14.1f}% {'20%':<15} {'✅' if 15 <= complex_pct <= 25 else '⚠️'}")
    if other_count > 0:
        other_pct = (other_count / total_combinations * 100)
        print(f"   {'기타':<20} {other_count:<15,} {other_pct:<14.1f}% {'0%':<15} {'⚠️'}")
    
    # 상세 분포
    print(f"\n   상세 분포:")
    for num_mols in sorted(molecule_counter.keys()):
        count = molecule_counter[num_mols]
        pct = (count / total_combinations * 100) if total_combinations > 0 else 0
        print(f"     {num_mols}개 분자: {count:,}개 ({pct:.1f}%)")
    
    # 3. 조향사별 샘플 수 확인 (불균형 보정 여부)
    print(f"\n3. 조향사별 샘플 수 분포:")
    print("-" * 80)
    
    blender_counts = defaultdict(int)
    for combo in combinations:
        blender_id = combo.get('target_blender')
        if blender_id is not None:
            blender_counts[blender_id] += 1
    
    if blender_counts:
        counts_list = list(blender_counts.values())
        min_count = min(counts_list)
        max_count = max(counts_list)
        mean_count = np.mean(counts_list)
        median_count = np.median(counts_list)
        std_count = np.std(counts_list)
        
        print(f"   전체 조향사 수: {len(blender_counts):,}개")
        print(f"   최소 샘플 수: {min_count}개")
        print(f"   최대 샘플 수: {max_count}개")
        print(f"   평균 샘플 수: {mean_count:.1f}개")
        print(f"   중앙값 샘플 수: {median_count:.1f}개")
        print(f"   표준편차: {std_count:.1f}개")
        
        # 불균형 지표
        cv = (std_count / mean_count) if mean_count > 0 else 0
        print(f"   변동계수 (CV): {cv:.2f}")
        if cv < 0.5:
            print(f"   ✅ 불균형이 잘 보정되었습니다 (CV < 0.5)")
        elif cv < 1.0:
            print(f"   ⚠️  불균형이 부분적으로 보정되었습니다 (CV < 1.0)")
        else:
            print(f"   ❌ 불균형이 여전히 존재합니다 (CV >= 1.0)")
        
        # 샘플 수별 조향사 분포
        bins = [
            (0, 10, "1-9개"),
            (10, 50, "10-49개"),
            (50, 100, "50-99개"),
            (100, 200, "100-199개"),
            (200, float('inf'), "200개 이상")
        ]
        
        print(f"\n   샘플 수별 조향사 분포:")
        for min_val, max_val, label in bins:
            count = sum(1 for c in counts_list if min_val <= c < max_val)
            pct = (count / len(blender_counts) * 100) if blender_counts else 0
            print(f"     {label:<15}: {count:>5}개 조향사 ({pct:>5.1f}%)")
    
    print("\n" + "=" * 80)
    
    # 요약
    print("\n📋 검증 요약:")
    issues = []
    if total_combinations < 60000:
        issues.append(f"총 조합 수 부족 ({total_combinations:,} < 60,000)")
    if not (35 <= simple_pct <= 45):
        issues.append(f"심플 조합 비율 불일치 ({simple_pct:.1f}%, 목표: 40%)")
    if not (35 <= medium_pct <= 45):
        issues.append(f"중간 조합 비율 불일치 ({medium_pct:.1f}%, 목표: 40%)")
    if not (15 <= complex_pct <= 25):
        issues.append(f"복잡 조합 비율 불일치 ({complex_pct:.1f}%, 목표: 20%)")
    
    if issues:
        print("   ⚠️  발견된 문제:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("   ✅ 모든 검증 항목 통과!")
    
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="데이터 증강 검증 스크립트")
    parser.add_argument("--combinations_file", type=str, default=None, 
                       help="조합 파일 경로 (JSON)")
    
    args = parser.parse_args()
    
    verify_combinations(args.combinations_file)
    
    print("\n💡 조합 파일이 없는 경우:")
    print("   학습 스크립트를 실행하면 조합 생성 로그가 출력됩니다.")
    print("   또는 학습 스크립트에 조합 저장 기능을 추가하세요.")
