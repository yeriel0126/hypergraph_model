"""
전체 조향사 샘플 수 분포 분석
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List


def extract_blender_from_record(record: Dict, blender_to_idx: Dict[str, int] = None) -> set:
    """
    레코드에서 조향사 ID 추출 (다양한 형식 지원)
    
    Args:
        record: 레코드 딕셔너리
        blender_to_idx: 조향사 이름 -> ID 매핑 (vocabularies.json에서)
    
    Returns:
        조향사 ID 집합
    """
    blender_ids = set()
    blenders = record.get('blenders', [])
    
    if isinstance(blenders, list):
        for blender in blenders:
            blender_name = None
            
            # 다양한 형식 처리
            if isinstance(blender, list) and len(blender) > 0:
                # [['name', 'group'], ...] 형식
                blender_name = str(blender[0]).strip().lower()
            elif isinstance(blender, dict):
                # {'id': 123} 형식
                if 'id' in blender:
                    try:
                        blender_ids.add(int(blender['id']))
                        continue
                    except:
                        pass
                # {'name': '...'} 형식
                elif 'name' in blender:
                    blender_name = str(blender['name']).strip().lower()
            elif isinstance(blender, str):
                blender_name = blender.strip().lower()
            elif isinstance(blender, int):
                blender_ids.add(blender)
                continue
            
            # 이름을 ID로 변환
            if blender_name and blender_to_idx:
                blender_id = blender_to_idx.get(blender_name)
                if blender_id is not None:
                    blender_ids.add(blender_id)
    
    elif isinstance(blenders, dict):
        if 'id' in blenders:
            try:
                blender_ids.add(int(blenders['id']))
            except:
                pass
    
    return blender_ids


def analyze_blender_distribution(data_path: str, vocab_path: str = None):
    """조향사별 샘플 수 분포 분석"""
    
    print("=" * 80)
    print("📊 전체 조향사 샘플 수 분포 분석")
    print("=" * 80)
    
    # 데이터 로드
    print(f"\n1. 데이터 로드 중...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        records = data.get('data', [])
    
    print(f"   ✓ 로드된 레코드 수: {len(records):,}개")
    
    # Vocabulary 로드 (조향사 이름 -> ID 매핑)
    blender_to_idx = {}
    if vocab_path and Path(vocab_path).exists():
        print(f"\n2. Vocabulary 로드 중...")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        blender_dict = vocab_data.get('blenders', {})
        blender_to_idx = (
            blender_dict.get('to_idx') or 
            blender_dict.get('to_index') or 
            blender_dict.get('item_to_idx') or 
            blender_dict.get('name_to_idx') or
            {}
        )
        print(f"   ✓ 조향사 Vocabulary 크기: {len(blender_to_idx)}개")
    else:
        print(f"\n2. ⚠️  Vocabulary 파일을 찾을 수 없습니다. 이름 기반 추출을 시도합니다.")
    
    # 조향사별 데이터 수집
    print(f"\n3. 조향사별 데이터 수집 중...")
    blender_to_records = defaultdict(list)
    
    for i, record in enumerate(records):
        blender_ids = extract_blender_from_record(record, blender_to_idx)
        for blender_id in blender_ids:
            blender_to_records[blender_id].append(i)  # 레코드 인덱스 저장
    
    if len(blender_to_records) == 0:
        print(f"   ⚠️  조향사를 찾을 수 없습니다. 데이터 구조를 확인합니다...")
        if records:
            sample_record = records[0]
            print(f"   샘플 레코드의 'blenders' 필드: {sample_record.get('blenders', 'N/A')[:200]}")
        return {}
    
    print(f"   ✓ 전체 조향사 수: {len(blender_to_records):,}개")
    
    # 조향사별 샘플 수 계산
    blender_counts = {bid: len(recs) for bid, recs in blender_to_records.items()}
    
    # 통계
    print(f"\n4. 통계 분석")
    print("-" * 80)
    counts_list = list(blender_counts.values())
    
    print(f"   최소 샘플 수: {min(counts_list)}개")
    print(f"   최대 샘플 수: {max(counts_list)}개")
    print(f"   평균 샘플 수: {np.mean(counts_list):.2f}개")
    print(f"   중앙값 샘플 수: {np.median(counts_list):.2f}개")
    print(f"   표준편차: {np.std(counts_list):.2f}개")
    
    # 샘플 수별 조향사 분포
    print(f"\n5. 샘플 수별 조향사 분포")
    print("-" * 80)
    
    bins = [
        (0, 1, "0개"),
        (1, 2, "1개"),
        (2, 5, "2-4개"),
        (5, 10, "5-9개"),
        (10, 20, "10-19개"),
        (20, 50, "20-49개"),
        (50, 100, "50-99개"),
        (100, 200, "100-199개"),
        (200, float('inf'), "200개 이상")
    ]
    
    bin_counts = {label: 0 for _, _, label in bins}
    bin_examples = {label: [] for _, _, label in bins}
    
    for blender_id, count in blender_counts.items():
        for min_val, max_val, label in bins:
            if min_val <= count < max_val:
                bin_counts[label] += 1
                if len(bin_examples[label]) < 5:  # 각 구간당 최대 5개 예시
                    bin_examples[label].append(blender_id)
                break
    
    for label in bins:
        _, _, label_name = label
        count = bin_counts[label_name]
        percentage = (count / len(blender_counts)) * 100 if blender_counts else 0
        print(f"   {label_name:15s}: {count:5d}개 조향사 ({percentage:5.2f}%)")
        if bin_examples[label_name]:
            examples_str = ", ".join([f"Blender_{bid}" for bid in bin_examples[label_name][:5]])
            print(f"      예시: {examples_str}")
    
    # 상위/하위 조향사
    print(f"\n6. 샘플 수 상위/하위 조향사")
    print("-" * 80)
    
    sorted_blenders = sorted(blender_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n   상위 10개 조향사 (가장 많은 샘플):")
    for rank, (blender_id, count) in enumerate(sorted_blenders[:10], 1):
        print(f"     {rank:2d}. Blender_{blender_id}: {count}개 샘플")
    
    print(f"\n   하위 10개 조향사 (가장 적은 샘플):")
    for rank, (blender_id, count) in enumerate(sorted_blenders[-10:], 1):
        print(f"     {rank:2d}. Blender_{blender_id}: {count}개 샘플")
    
    # 특정 임계값 이하 조향사
    print(f"\n7. 특정 임계값 이하 조향사 수")
    print("-" * 80)
    
    thresholds = [1, 5, 10, 20, 50]
    for threshold in thresholds:
        below_threshold = sum(1 for count in counts_list if count < threshold)
        percentage = (below_threshold / len(blender_counts)) * 100 if blender_counts else 0
        print(f"   {threshold}개 미만: {below_threshold:5d}개 조향사 ({percentage:5.2f}%)")
    
    print("\n" + "=" * 80)
    
    return blender_counts


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="조향사별 샘플 수 분포 분석")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="데이터 파일 경로 (cleaned_complete_data.json)"
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="Vocabulary 파일 경로 (vocabularies.json)"
    )
    
    args = parser.parse_args()
    
    # 기본 경로 설정
    script_dir = Path(__file__).parent
    
    if args.data_path is None:
        default_path = script_dir.parent.parent / "cleaned_data" / "cleaned_complete_data.json"
        if default_path.exists():
            args.data_path = str(default_path)
        else:
            print(f"❌ 데이터 파일을 찾을 수 없습니다: {default_path}")
            print(f"   --data_path 옵션으로 경로를 지정해주세요.")
            exit(1)
    
    if args.vocab_path is None:
        default_vocab = script_dir.parent.parent / "feature_encoding" / "vocabularies.json"
        if default_vocab.exists():
            args.vocab_path = str(default_vocab)
    
    if not Path(args.data_path).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {args.data_path}")
        exit(1)
    
    analyze_blender_distribution(args.data_path, args.vocab_path)
