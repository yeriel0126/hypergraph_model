"""
데이터 증강 스크립트

현재 조향 데이터셋의 불균형을 해결하기 위한 정교한 증강 전략:
1. 혼동 행렬 기반 가중치 매핑
2. 구간별 타겟 증강
3. 성분 벡터 노이즈 추가 및 Normalization
4. 상세 리포트 생성
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import copy


def load_confusion_analysis(confusion_file: str) -> Dict:
    """혼동 분석 결과 로드"""
    with open(confusion_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_weight_mapping(confusion_data: Dict, default_weight: float = 1.5) -> Dict[int, float]:
    """
    혼동 행렬 결과를 가중치(1.0~3.0)로 변환
    
    Args:
        confusion_data: 혼동 분석 결과 딕셔너리
        default_weight: 분석에 포함되지 않은 조향사의 기본 가중치 (기본값: 1.5)
    
    Returns:
        {blender_id: weight} 딕셔너리
    """
    weight_mapping = {}
    
    # 혼동 분석 결과에서 오분류율 추출
    for blender_info in confusion_data.get('most_confused_blenders', []):
        blender_id = blender_info['blender_idx']
        error_rate = blender_info['error_rate']  # 0.0 ~ 1.0
        
        # 오분류율을 가중치로 변환 (1.0 ~ 3.0)
        # error_rate 0.0 -> 1.0, error_rate 1.0 -> 3.0
        weight = 1.0 + error_rate * 2.0
        weight_mapping[blender_id] = weight
    
    # 혼동 쌍에서도 가중치 추가 (혼동이 잦을수록 가중치 증가)
    confusion_count_map = defaultdict(int)
    for pair in confusion_data.get('most_confusing_pairs', []):
        true_idx = pair['true_idx']
        confusion_count_map[true_idx] += pair['confusion_count']
    
    # 혼동 횟수에 따라 추가 가중치 (최대 +0.5)
    for blender_id, count in confusion_count_map.items():
        if blender_id in weight_mapping:
            # 기존 가중치에 추가 (최대 3.0 제한)
            weight_mapping[blender_id] = min(3.0, weight_mapping[blender_id] + min(0.5, count * 0.1))
        else:
            # 새로운 조향사는 혼동 횟수 기반 가중치
            weight_mapping[blender_id] = min(3.0, 1.0 + min(2.0, count * 0.2))
    
    return weight_mapping


def extract_blender_from_record(record: Dict, blender_to_idx: Dict[str, int] = None) -> set:
    """레코드에서 조향사 ID 추출"""
    blender_ids = set()
    blenders = record.get('blenders', [])
    
    if isinstance(blenders, list):
        for blender in blenders:
            blender_name = None
            
            if isinstance(blender, list) and len(blender) > 0:
                blender_name = str(blender[0]).strip().lower()
            elif isinstance(blender, dict):
                if 'id' in blender:
                    try:
                        blender_ids.add(int(blender['id']))
                        continue
                    except:
                        pass
                elif 'name' in blender:
                    blender_name = str(blender['name']).strip().lower()
            elif isinstance(blender, str):
                blender_name = blender.strip().lower()
            elif isinstance(blender, int):
                blender_ids.add(blender)
                continue
            
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


def normalize_vector(vector: List[float]) -> List[float]:
    """
    벡터를 Normalization하여 합계가 1이 되도록 함
    
    Args:
        vector: 정규화할 벡터
    
    Returns:
        정규화된 벡터 (합계 = 1.0)
    """
    vector = np.array(vector, dtype=float)
    total = np.sum(vector)
    
    if total == 0:
        # 합계가 0이면 균등 분배
        return [1.0 / len(vector)] * len(vector)
    
    # 합계로 나누어 정규화
    normalized = vector / total
    return normalized.tolist()


def add_noise_to_recipe_vector(record: Dict, noise_level: float = 0.005) -> Dict:
    """
    성분 벡터(Recipe Vector)에 노이즈 추가 및 Normalization
    
    Args:
        record: 레코드 딕셔너리
        noise_level: 노이즈 수준 (기본값: 0.005)
    
    Returns:
        노이즈가 추가되고 정규화된 레코드
    """
    augmented = copy.deepcopy(record)
    
    # notes를 성분 벡터로 간주하고 노이즈 추가
    if 'notes' in augmented and isinstance(augmented['notes'], list):
        notes = augmented['notes']
        
        # notes가 숫자 리스트인 경우
        if notes and isinstance(notes[0], (int, float)):
            # 노이즈 추가
            noisy_notes = [max(0.0, note + np.random.normal(0, noise_level)) for note in notes]
            # Normalization (합계 = 1)
            augmented['notes'] = normalize_vector(noisy_notes)
        
        # notes가 dict 리스트인 경우 (예: {'name': '...', 'intensity': 0.5})
        elif notes and isinstance(notes[0], dict):
            intensities = []
            for note in notes:
                if isinstance(note, dict):
                    # intensity, weight, amount, concentration 등의 수치 필드 찾기
                    for key in ['intensity', 'weight', 'amount', 'concentration']:
                        if key in note and isinstance(note[key], (int, float)):
                            intensities.append(note[key])
                            break
                    else:
                        intensities.append(0.0)
                else:
                    intensities.append(0.0)
            
            if intensities:
                # 노이즈 추가
                noisy_intensities = [max(0.0, intensity + np.random.normal(0, noise_level)) for intensity in intensities]
                # Normalization
                normalized_intensities = normalize_vector(noisy_intensities)
                
                # 원본 구조 유지하면서 값 업데이트
                for i, note in enumerate(augmented['notes']):
                    if isinstance(note, dict) and i < len(normalized_intensities):
                        for key in ['intensity', 'weight', 'amount', 'concentration']:
                            if key in note:
                                note[key] = normalized_intensities[i]
                                break
    
    # molecules 내부의 notes도 처리
    if 'molecules' in augmented and isinstance(augmented['molecules'], list):
        for mol in augmented['molecules']:
            if isinstance(mol, dict) and 'notes' in mol:
                mol['notes'] = add_noise_to_recipe_vector({'notes': mol['notes']}, noise_level)['notes']
    
    return augmented


def augment_data(
    records: List[Dict],
    confusion_file: str = None,
    vocab_path: str = None,
    target_total: int = 60000,
    min_samples: int = 20,
    default_weight: float = 1.5,
    dense_blender_threshold: int = 500,  # 초과 밀집 조향사 임계값
    noise_level: float = 0.005
) -> Tuple[List[Dict], Dict]:
    """
    데이터 증강 수행
    
    Args:
        records: 원본 레코드 리스트
        confusion_file: 혼동 분석 결과 파일 경로 (선택적)
        vocab_path: Vocabulary 파일 경로
        target_total: 목표 총 샘플 수 (기본값: 60000)
        min_samples: 최소 샘플 수 (기본값: 20)
        default_weight: 기본 가중치 (기본값: 1.5)
        dense_blender_threshold: 초과 밀집 조향사 임계값 (기본값: 500)
        noise_level: 노이즈 수준 (기본값: 0.005)
    
    Returns:
        (증강된 레코드 리스트, 통계 딕셔너리)
    """
    print("=" * 80)
    print("📊 데이터 증강 시작")
    print("=" * 80)
    
    # Vocabulary 로드
    blender_to_idx = {}
    if vocab_path and Path(vocab_path).exists():
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
    
    # 조향사별 데이터 수집
    print(f"\n1. 조향사별 데이터 수집 중...")
    blender_to_records = defaultdict(list)
    for i, record in enumerate(records):
        blender_ids = extract_blender_from_record(record, blender_to_idx)
        for blender_id in blender_ids:
            blender_to_records[blender_id].append(i)
    
    blender_counts_before = {bid: len(recs) for bid, recs in blender_to_records.items()}
    print(f"   ✓ 전체 조향사 수: {len(blender_to_records):,}개")
    
    # 가중치 매핑 생성
    print(f"\n2. 가중치 매핑 생성 중...")
    weight_mapping = {}
    if confusion_file and Path(confusion_file).exists():
        confusion_data = load_confusion_analysis(confusion_file)
        weight_mapping = create_weight_mapping(confusion_data, default_weight)
        print(f"   ✓ 혼동 분석 기반 가중치: {len(weight_mapping)}개 조향사")
    else:
        print(f"   ⚠️  혼동 분석 파일이 없습니다. 모든 조향사에 기본 가중치 {default_weight} 적용")
    
    # 모든 조향사에 가중치 할당 (분석에 포함되지 않은 조향사는 기본 가중치)
    all_blender_weights = {}
    for blender_id in blender_to_records.keys():
        all_blender_weights[blender_id] = weight_mapping.get(blender_id, default_weight)
    
    print(f"   ✓ 전체 조향사 가중치 할당 완료")
    
    # 증강 계획 수립
    print(f"\n3. 증강 계획 수립 중...")
    augmentation_plan = {}
    
    # 초과 밀집 조향사 식별 (증강 제외)
    dense_blenders = {bid for bid, count in blender_counts_before.items() if count >= dense_blender_threshold}
    print(f"   ✓ 초과 밀집 조향사 (증강 제외): {len(dense_blenders)}개")
    if dense_blenders:
        print(f"      예시: {list(dense_blenders)[:10]}")
    
    # 구간별 증강 계획
    for blender_id, current_count in blender_counts_before.items():
        if blender_id in dense_blenders:
            # 초과 밀집 조향사는 증강 제외
            augmentation_plan[blender_id] = {
                'current': current_count,
                'target': current_count,  # 유지
                'needed': 0,
                'weight': 1.0,
                'reason': 'dense'
            }
        elif current_count < min_samples:
            # 샘플 20개 미만: 최소 20개까지 최우선 증강
            augmentation_plan[blender_id] = {
                'current': current_count,
                'target': min_samples,
                'needed': min_samples - current_count,
                'weight': all_blender_weights[blender_id],
                'reason': 'below_min'
            }
        else:
            # 그 외: 원본 수 * 혼동 가중치
            weight = all_blender_weights[blender_id]
            target = int(current_count * weight)
            augmentation_plan[blender_id] = {
                'current': current_count,
                'target': target,
                'needed': max(0, target - current_count),
                'weight': weight,
                'reason': 'weighted'
            }
    
    # 총 필요 샘플 수 계산
    total_needed = sum(plan['needed'] for plan in augmentation_plan.values())
    print(f"   ✓ 총 필요 샘플 수: {total_needed:,}개")
    
    # 60,000개 리밋에 맞춰 조정
    current_total = len(records)
    available_slots = target_total - current_total
    
    if total_needed > available_slots:
        # 비율로 축소
        scale_factor = available_slots / total_needed
        print(f"   ⚠️  리밋 초과: {total_needed:,}개 → {available_slots:,}개로 축소 (비율: {scale_factor:.3f})")
        for plan in augmentation_plan.values():
            plan['needed'] = int(plan['needed'] * scale_factor)
            plan['target'] = plan['current'] + plan['needed']
    
    # 증강 실행
    print(f"\n4. 데이터 증강 실행 중...")
    augmented_records = list(records)  # 원본 유지
    
    augmentation_stats = {
        'below_min': 0,
        'weighted': 0,
        'dense': 0,
        'total': 0
    }
    
    for blender_id, plan in sorted(augmentation_plan.items(), key=lambda x: x[1]['needed'], reverse=True):
        if plan['needed'] == 0:
            continue
        
        blender_records = blender_to_records[blender_id]
        if len(blender_records) == 0:
            continue
        
        # 샘플링
        sampled_indices = np.random.choice(blender_records, size=plan['needed'], replace=True)
        
        for idx in sampled_indices:
            original_record = records[idx]
            # 노이즈 추가 및 정규화
            augmented_record = add_noise_to_recipe_vector(original_record, noise_level)
            augmented_records.append(augmented_record)
            
            augmentation_stats[plan['reason']] += 1
            augmentation_stats['total'] += 1
    
    print(f"   ✓ 증강 완료:")
    print(f"      - 최소 샘플 미만 조향사: {augmentation_stats['below_min']:,}개")
    print(f"      - 가중치 기반 증강: {augmentation_stats['weighted']:,}개")
    print(f"      - 초과 밀집 조향사 (제외): {augmentation_stats['dense']:,}개")
    print(f"      - 총 증강 샘플: {augmentation_stats['total']:,}개")
    
    # 증강 후 통계
    blender_counts_after = {}
    temp_blender_to_records = defaultdict(list)
    for i, record in enumerate(augmented_records):
        blender_ids = extract_blender_from_record(record, blender_to_idx)
        for blender_id in blender_ids:
            temp_blender_to_records[blender_id].append(i)
    blender_counts_after = {bid: len(recs) for bid, recs in temp_blender_to_records.items()}
    
    # 리포트 생성
    stats = {
        'before': blender_counts_before,
        'after': blender_counts_after,
        'augmentation_plan': augmentation_plan,
        'augmentation_stats': augmentation_stats,
        'total_before': len(records),
        'total_after': len(augmented_records)
    }
    
    return augmented_records, stats


def generate_report(stats: Dict):
    """증강 전/후 분포 변화 리포트 생성"""
    print("\n" + "=" * 80)
    print("📊 증강 전/후 조향사별 샘플 수 분포 변화 리포트")
    print("=" * 80)
    
    before_counts = stats['before']
    after_counts = stats['after']
    
    # 구간별 통계
    bins = [
        (0, 1, "1개"),
        (1, 5, "2-4개"),
        (5, 10, "5-9개"),
        (10, 20, "10-19개"),
        (20, 50, "20-49개"),
        (50, 100, "50-99개"),
        (100, 200, "100-199개"),
        (200, float('inf'), "200개 이상")
    ]
    
    print(f"\n{'구간':<15} {'증강 전':<15} {'증강 후':<15} {'변화':<15} {'변화율':<15}")
    print("-" * 80)
    
    for min_val, max_val, label in bins:
        before_count = sum(1 for count in before_counts.values() if min_val <= count < max_val)
        after_count = sum(1 for count in after_counts.values() if min_val <= count < max_val)
        change = after_count - before_count
        change_rate = (change / before_count * 100) if before_count > 0 else 0.0
        
        print(f"{label:<15} {before_count:<15} {after_count:<15} {change:+<15} {change_rate:+6.2f}%")
    
    # 전체 통계
    print(f"\n{'항목':<30} {'증강 전':<20} {'증강 후':<20} {'변화':<20}")
    print("-" * 90)
    
    total_before = stats['total_before']
    total_after = stats['total_after']
    print(f"{'총 레코드 수':<30} {total_before:<20,} {total_after:<20,} {total_after - total_before:+<20,}")
    
    avg_before = np.mean(list(before_counts.values()))
    avg_after = np.mean(list(after_counts.values()))
    print(f"{'평균 샘플 수':<30} {avg_before:<20.2f} {avg_after:<20.2f} {avg_after - avg_before:+<20.2f}")
    
    median_before = np.median(list(before_counts.values()))
    median_after = np.median(list(after_counts.values()))
    print(f"{'중앙값 샘플 수':<30} {median_before:<20.2f} {median_after:<20.2f} {median_after - median_before:+<20.2f}")
    
    below_20_before = sum(1 for count in before_counts.values() if count < 20)
    below_20_after = sum(1 for count in after_counts.values() if count < 20)
    print(f"{'20개 미만 조향사 수':<30} {below_20_before:<20} {below_20_after:<20} {below_20_after - below_20_before:+<20}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="데이터 증강 스크립트")
    parser.add_argument("--data_path", type=str, required=True, help="원본 데이터 경로")
    parser.add_argument("--confusion_file", type=str, default=None, help="혼동 분석 결과 경로 (선택적)")
    parser.add_argument("--vocab_path", type=str, default=None, help="Vocabulary 파일 경로")
    parser.add_argument("--output_path", type=str, required=True, help="증강된 데이터 저장 경로")
    parser.add_argument("--target_total", type=int, default=60000, help="목표 총 샘플 수")
    parser.add_argument("--min_samples", type=int, default=20, help="최소 샘플 수")
    parser.add_argument("--default_weight", type=float, default=1.5, help="기본 가중치")
    parser.add_argument("--dense_threshold", type=int, default=500, help="초과 밀집 조향사 임계값")
    parser.add_argument("--noise_level", type=float, default=0.005, help="노이즈 수준")
    
    args = parser.parse_args()
    
    # 기본 경로 설정
    script_dir = Path(__file__).parent
    if args.vocab_path is None:
        default_vocab = script_dir.parent.parent / "feature_encoding" / "vocabularies.json"
        if default_vocab.exists():
            args.vocab_path = str(default_vocab)
    
    # 데이터 로드
    print("데이터 로드 중...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        records = data.get('data', [])
    
    print(f"로드된 레코드 수: {len(records):,}개")
    
    # 데이터 증강
    augmented_records, stats = augment_data(
        records=records,
        confusion_file=args.confusion_file,
        vocab_path=args.vocab_path,
        target_total=args.target_total,
        min_samples=args.min_samples,
        default_weight=args.default_weight,
        dense_blender_threshold=args.dense_threshold,
        noise_level=args.noise_level
    )
    
    # 리포트 생성
    generate_report(stats)
    
    # 저장
    output_data = {
        'data': augmented_records,
        'metadata': {
            'augmented': True,
            'original_count': len(records),
            'augmented_count': len(augmented_records),
            'augmentation_stats': stats['augmentation_stats']
        }
    }
    
    # 원본 데이터의 다른 필드들도 유지
    if 'data' in data:
        for key, value in data.items():
            if key != 'data':
                output_data[key] = value
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 증강된 데이터 저장: {args.output_path}")
