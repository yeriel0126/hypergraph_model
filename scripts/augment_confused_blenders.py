"""
정교한 데이터 증강 전략

1. 정답률 0% 조향사 (Targeted Oversampling): 5~10배 집중 증강
2. Blender_351로 오해받는 조향사들: 미세한 성분 변화(±1~2%)를 준 변형본 생성
3. 데이터 희소성이 높은 조향사: 최소 15~20개 샘플 확보
4. 노이즈 증강: Gaussian Noise 0.01 수준
5. 총 6만 개 구성 시나리오
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
import copy


def load_confusion_analysis(confusion_file: str) -> Dict:
    """혼동 분석 결과 로드"""
    with open(confusion_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def add_gaussian_noise_to_notes(notes: List, noise_level: float = 0.01) -> List:
    """
    노트에 Gaussian Noise 추가 (성분 함량 미세 변화)
    
    Args:
        notes: 노트 리스트
        noise_level: 노이즈 수준 (기본값: 0.01 = 1%)
    
    Returns:
        노이즈가 추가된 노트 리스트
    """
    if not notes or not isinstance(notes, list):
        return notes
    
    noisy_notes = []
    for note in notes:
        if isinstance(note, dict):
            # 노트가 dict인 경우 (예: {'name': '...', 'intensity': 0.5})
            noisy_note = copy.deepcopy(note)
            # intensity나 weight 같은 수치 필드에 노이즈 추가
            for key in ['intensity', 'weight', 'amount', 'concentration']:
                if key in noisy_note and isinstance(noisy_note[key], (int, float)):
                    noise = np.random.normal(0, noise_level)
                    noisy_note[key] = max(0.0, min(1.0, noisy_note[key] + noise))
            noisy_notes.append(noisy_note)
        elif isinstance(note, (int, float)):
            # 노트가 숫자인 경우
            noise = np.random.normal(0, noise_level)
            noisy_notes.append(max(0.0, min(1.0, note + noise)))
        else:
            # 노트가 문자열이거나 다른 타입인 경우 그대로 유지
            noisy_notes.append(note)
    
    return noisy_notes


def add_fine_variation_to_record(record: Dict, variation_level: float = 0.02) -> Dict:
    """
    레코드에 미세한 변형 추가 (±1~2% 성분 변화)
    
    Blender_351로 오해받는 조향사들을 위한 변형본 생성
    """
    augmented = copy.deepcopy(record)
    
    # 1. 노트 순서 섞기 (약간의 변형)
    if 'notes' in augmented and isinstance(augmented['notes'], list):
        notes = augmented['notes'].copy()
        # 10% 확률로 순서 섞기
        if random.random() < 0.1:
            random.shuffle(notes)
        augmented['notes'] = notes
    
    # 2. 분자 레벨에서 노트 변형 (molecules가 있는 경우)
    if 'molecules' in augmented and isinstance(augmented['molecules'], list):
        for mol in augmented['molecules']:
            if isinstance(mol, dict) and 'notes' in mol:
                mol['notes'] = add_gaussian_noise_to_notes(mol['notes'], variation_level)
    
    # 3. 레코드 레벨 노트 변형
    if 'notes' in augmented:
        augmented['notes'] = add_gaussian_noise_to_notes(augmented['notes'], variation_level)
    
    return augmented


def extract_blender_from_record(record: Dict) -> Set[int]:
    """
    레코드에서 조향사 ID 추출 (다양한 형식 지원)
    
    Returns:
        조향사 ID 집합
    """
    blender_ids = set()
    blenders = record.get('blenders', [])
    
    if isinstance(blenders, list):
        for blender in blenders:
            if isinstance(blender, dict):
                # {'id': 123} 형식
                if 'id' in blender:
                    blender_ids.add(int(blender['id']))
                # {'name': 'Blender_123'} 형식에서 ID 추출 시도
                elif 'name' in blender:
                    name = str(blender['name'])
                    if 'Blender_' in name:
                        try:
                            blender_id = int(name.replace('Blender_', '').strip())
                            blender_ids.add(blender_id)
                        except:
                            pass
            elif isinstance(blender, (int, str)):
                try:
                    blender_id = int(blender)
                    blender_ids.add(blender_id)
                except:
                    pass
    elif isinstance(blenders, dict):
        if 'id' in blenders:
            blender_ids.add(int(blenders['id']))
    
    return blender_ids


def augment_confused_blenders(
    records: List[Dict],
    confusion_file: str = None,  # 선택적: 혼동 분석 결과 (가중치용)
    target_total: int = 60000,  # 목표 총 샘플 수
    zero_accuracy_factor: float = 7.0,  # 정답률 0% 조향사 증강 배수 (5~10배 중간값)
    min_samples_per_blender: int = 20,  # 최소 샘플 수
    noise_augmentation_ratio: float = 0.5,  # 노이즈 증강 비율 (전체의 50%)
    fine_variation_ratio: float = 0.25  # 미세 변형 비율 (전체의 25%)
) -> List[Dict]:
    """
    정교한 데이터 증강 전략 적용
    
    전체 조향사 데이터 분포를 먼저 분석하고, 혼동 분석 결과는 가중치로만 사용
    
    Args:
        records: 원본 데이터 레코드 리스트
        confusion_file: 혼동 분석 결과 JSON 파일 경로 (선택적, 가중치용)
        target_total: 목표 총 샘플 수 (기본값: 60000)
        zero_accuracy_factor: 정답률 0% 조향사 증강 배수 (기본값: 7.0)
        min_samples_per_blender: 조향사당 최소 샘플 수 (기본값: 20)
        noise_augmentation_ratio: 노이즈 증강 비율 (기본값: 0.5)
        fine_variation_ratio: 미세 변형 비율 (기본값: 0.25)
    
    Returns:
        증강된 레코드 리스트
    """
    # 혼동 분석 결과 로드 (선택적, 가중치용)
    confusion_data = None
    confusion_weights = {}  # {blender_id: weight}
    if confusion_file:
        try:
            confusion_data = load_confusion_analysis(confusion_file)
            # 혼동 분석 결과를 가중치로 변환
            for blender_info in confusion_data.get('most_confused_blenders', []):
                blender_id = blender_info['blender_idx']
                error_rate = blender_info['error_rate']
                # 오류율이 높을수록 가중치 증가
                confusion_weights[blender_id] = 1.0 + error_rate * 2.0  # 1.0 ~ 3.0
            
            # 혼동 쌍에서도 가중치 추가
            for pair in confusion_data.get('most_confusing_pairs', []):
                true_idx = pair['true_idx']
                confusion_weights[true_idx] = confusion_weights.get(true_idx, 1.0) + 0.5
        except Exception as e:
            print(f"  ⚠️  혼동 분석 로드 실패 (무시하고 계속): {e}")
    
    # 전체 조향사별 데이터 수집 (모든 조향사 분석)
    blender_to_records = defaultdict(list)
    for record in records:
        blender_ids = extract_blender_from_record(record)
        for blender_id in blender_ids:
            blender_to_records[blender_id].append(record)
    
    print(f"원본 데이터: {len(records):,}개 레코드")
    print(f"전체 조향사 수: {len(blender_to_records):,}개")
    
    # 전체 조향사 데이터 분포 분석
    blender_counts = {bid: len(recs) for bid, recs in blender_to_records.items()}
    sorted_blenders = sorted(blender_counts.items(), key=lambda x: x[1])
    
    print(f"\n📊 전체 조향사 데이터 분포 분석:")
    print(f"   최소 샘플 수: {min(blender_counts.values())}개")
    print(f"   최대 샘플 수: {max(blender_counts.values())}개")
    print(f"   평균 샘플 수: {np.mean(list(blender_counts.values())):.1f}개")
    print(f"   중앙값 샘플 수: {np.median(list(blender_counts.values())):.1f}개")
    
    # 샘플 수별 조향사 분포
    bins = [0, 5, 10, 20, 50, 100, float('inf')]
    bin_labels = ['0-5', '5-10', '10-20', '20-50', '50-100', '100+']
    bin_counts = {label: 0 for label in bin_labels}
    for count in blender_counts.values():
        for i, (label, max_val) in enumerate(zip(bin_labels, bins[1:])):
            if count <= max_val:
                bin_counts[label] += 1
                break
    
    print(f"\n   샘플 수별 조향사 분포:")
    for label, count in bin_counts.items():
        print(f"     {label}개: {count}개 조향사")
    
    # 1. 데이터 희소성이 높은 조향사 식별 (전체 조향사 대상)
    sparse_blenders = {}
    for blender_id, blender_records in blender_to_records.items():
        current_count = len(blender_records)
        if current_count < min_samples_per_blender and current_count > 0:
            # 혼동 분석 가중치 적용
            weight = confusion_weights.get(blender_id, 1.0)
            target_count = int(min_samples_per_blender * weight)
            sparse_blenders[blender_id] = {
                'current_count': current_count,
                'target_count': target_count,
                'weight': weight
            }
    
    print(f"\n1️⃣  데이터 희소성 높은 조향사: {len(sparse_blenders)}개")
    print(f"   (샘플 수 < {min_samples_per_blender}개)")
    if sparse_blenders:
        print(f"   예시 (상위 10개):")
        sorted_sparse = sorted(sparse_blenders.items(), key=lambda x: x[1]['current_count'])[:10]
        for blender_id, info in sorted_sparse:
            print(f"     Blender_{blender_id}: {info['current_count']} → {info['target_count']}개 (가중치: {info['weight']:.2f})")
    
    # 2. 정답률 0% 조향사 식별 (혼동 분석 결과가 있는 경우만)
    zero_accuracy_blenders = {}
    if confusion_data:
        for blender_info in confusion_data.get('most_confused_blenders', []):
            if blender_info['error_rate'] == 1.0 and blender_info['total_count'] >= 5:
                blender_id = blender_info['blender_idx']
                # 학습 데이터에서 실제 샘플 수 확인
                actual_count = len(blender_to_records.get(blender_id, []))
                if actual_count > 0:
                    zero_accuracy_blenders[blender_id] = {
                        'current_count': actual_count,
                        'target_count': int(actual_count * zero_accuracy_factor),
                        'test_count': blender_info['total_count']  # 테스트 세트에서의 샘플 수
                    }
    
    print(f"\n2️⃣  정답률 0% 조향사 (혼동 분석 기반): {len(zero_accuracy_blenders)}개")
    if zero_accuracy_blenders:
        for blender_id, info in list(zero_accuracy_blenders.items())[:10]:
            print(f"   Blender_{blender_id}: {info['current_count']} → {info['target_count']}개 (×{zero_accuracy_factor:.1f})")
    
    # 3. Blender_351로 오해받는 조향사들 식별 (혼동 분석 결과가 있는 경우만)
    confused_by_351 = set()
    if confusion_data:
        predicted_counter = Counter()
        confusion_map = defaultdict(list)
        
        for pair in confusion_data.get('most_confusing_pairs', []):
            true_idx = pair['true_idx']
            pred_idx = pair['predicted_idx']
            count = pair['confusion_count']
            predicted_counter[pred_idx] += count
            confusion_map[pred_idx].append((true_idx, count))
        
        hub_blender_351 = 351
        if hub_blender_351 in confusion_map:
            for true_idx, conf_count in confusion_map[hub_blender_351]:
                if conf_count >= 3:  # 3회 이상 혼동
                    # 학습 데이터에 실제로 존재하는지 확인
                    if true_idx in blender_to_records:
                        confused_by_351.add(true_idx)
    
    print(f"\n3️⃣  Blender_351로 오해받는 조향사: {len(confused_by_351)}개")
    if confused_by_351:
        print(f"   예시: {list(confused_by_351)[:10]}")
    
    # 4. 전체 조향사 균형 맞추기 (모든 조향사에 대해 최소 샘플 수 확보)
    all_blenders_needing_augmentation = {}
    for blender_id, blender_records in blender_to_records.items():
        current_count = len(blender_records)
        if current_count < min_samples_per_blender:
            weight = confusion_weights.get(blender_id, 1.0)
            target_count = int(min_samples_per_blender * weight)
            all_blenders_needing_augmentation[blender_id] = {
                'current_count': current_count,
                'target_count': target_count,
                'weight': weight
            }
    
    print(f"\n4️⃣  전체 조향사 균형 맞추기: {len(all_blenders_needing_augmentation)}개 조향사가 증강 필요")
    
    # 증강된 레코드 생성
    augmented_records = list(records)  # 원본 유지
    augmentation_stats = {
        'zero_accuracy': 0,
        'confused_by_351': 0,
        'sparse': 0,
        'noise': 0,
        'fine_variation': 0
    }
    
    # 전략 1: 전체 조향사 균형 맞추기 (모든 조향사에 대해 최소 샘플 수 확보)
    print(f"\n📊 전략 1: 전체 조향사 균형 맞추기 (최소 {min_samples_per_blender}개)")
    for blender_id, info in all_blenders_needing_augmentation.items():
        blender_records = blender_to_records.get(blender_id, [])
        if len(blender_records) == 0:
            continue
        
        needed_count = max(0, info['target_count'] - info['current_count'])
        if needed_count > 0:
            # 혼동 분석 가중치가 높은 조향사는 더 강한 변형 적용
            variation_level = 0.02 if info['weight'] > 2.0 else 0.01
            
            sampled = random.choices(blender_records, k=needed_count)
            for record in sampled:
                augmented_record = add_fine_variation_to_record(record, variation_level=variation_level)
                augmented_records.append(augmented_record)
                augmentation_stats['sparse'] += 1
    
    print(f"   ✓ 증강 완료: {augmentation_stats['sparse']:,}개")
    
    # 전략 2: 정답률 0% 조향사 집중 증강 (약 1만 개 목표)
    print(f"\n📊 전략 2: 정답률 0% 조향사 집중 증강")
    for blender_id, info in zero_accuracy_blenders.items():
        blender_records = blender_to_records.get(blender_id, [])
        if len(blender_records) == 0:
            continue
        
        needed_count = max(0, info['target_count'] - info['current_count'])
        if needed_count > 0:
            # 다양한 변형으로 증강
            sampled = random.choices(blender_records, k=needed_count)
            
            for record in sampled:
                # 50% 확률로 미세 변형, 50% 확률로 노이즈 추가
                if random.random() < 0.5:
                    augmented_record = add_fine_variation_to_record(record, variation_level=0.02)
                    augmentation_stats['fine_variation'] += 1
                else:
                    augmented_record = add_fine_variation_to_record(record, variation_level=0.01)
                    augmentation_stats['noise'] += 1
                
                augmented_records.append(augmented_record)
                augmentation_stats['zero_accuracy'] += 1
    
    print(f"   ✓ 증강 완료: {augmentation_stats['zero_accuracy']:,}개")
    
    # 전략 3: Blender_351로 오해받는 조향사 미세 변형 (약 1.5만 개 목표)
    print(f"\n📊 전략 2: Blender_351로 오해받는 조향사 미세 변형")
    confused_351_augmented = 0
    target_confused_351 = int(len(records) * fine_variation_ratio)
    
    for blender_id in confused_by_351:
        blender_records = blender_to_records.get(blender_id, [])
        if len(blender_records) == 0:
            continue
        
        # 각 조향사당 3~5배 증강
        factor = random.uniform(3.0, 5.0)
        target_count = int(len(blender_records) * factor)
        needed_count = max(0, target_count - len(blender_records))
        
        if needed_count > 0 and confused_351_augmented < target_confused_351:
            sampled = random.choices(blender_records, k=min(needed_count, target_confused_351 - confused_351_augmented))
            
            for record in sampled:
                # 미세한 성분 변화 (±1~2%)
                augmented_record = add_fine_variation_to_record(record, variation_level=0.015)
                augmented_records.append(augmented_record)
                confused_351_augmented += 1
                augmentation_stats['confused_by_351'] += 1
    
    print(f"   ✓ 증강 완료: {augmentation_stats['confused_by_351']:,}개")
    
    # 전략 4: 전체 데이터 노이즈 증강 (약 1.5만 개 목표)
    print(f"\n📊 전략 4: 전체 데이터 노이즈 증강")
    target_noise = int(len(records) * noise_augmentation_ratio)
    noise_augmented = 0
    
    # 전체 레코드에서 랜덤 샘플링
    random.shuffle(records)
    for record in records:
        if noise_augmented >= target_noise:
            break
        
        # Gaussian Noise 추가 (0.01 수준)
        augmented_record = add_fine_variation_to_record(record, variation_level=0.01)
        augmented_records.append(augmented_record)
        noise_augmented += 1
        augmentation_stats['noise'] += 1
    
    print(f"   ✓ 증강 완료: {augmentation_stats['noise']:,}개")
    
    # 최종 통계
    print(f"\n{'='*80}")
    print(f"📊 최종 증강 통계")
    print(f"{'='*80}")
    print(f"   원본 데이터: {len(records):,}개")
    print(f"   증강 후 데이터: {len(augmented_records):,}개")
    print(f"   증강된 레코드 수: {len(augmented_records) - len(records):,}개")
    print(f"\n   증강 세부 내역:")
    print(f"   - 정답률 0% 조향사: {augmentation_stats['zero_accuracy']:,}개")
    print(f"   - Blender_351 혼동 조향사: {augmentation_stats['confused_by_351']:,}개")
    print(f"   - 데이터 희소성 조향사: {augmentation_stats['sparse']:,}개")
    print(f"   - 노이즈 증강: {augmentation_stats['noise']:,}개")
    print(f"   - 미세 변형: {augmentation_stats['fine_variation']:,}개")
    print(f"{'='*80}")
    
    return augmented_records


def save_augmented_data(
    augmented_records: List[Dict],
    output_path: str,
    original_data_path: str = None
):
    """증강된 데이터 저장"""
    # 원본 데이터의 메타데이터 유지
    output_data = {
        'data': augmented_records,
        'metadata': {
            'augmented': True,
            'original_count': len([r for r in augmented_records if 'augmented' not in str(r)]),
            'augmented_count': len(augmented_records)
        }
    }
    
    # 원본 데이터의 다른 필드들도 유지
    if original_data_path:
        try:
            with open(original_data_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
                for key, value in original_data.items():
                    if key != 'data':
                        output_data[key] = value
        except:
            pass
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 증강된 데이터 저장: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="정교한 데이터 증강 전략")
    parser.add_argument("--data_path", type=str, required=True, help="원본 데이터 경로 (cleaned_complete_data.json)")
    parser.add_argument("--confusion_file", type=str, required=True, help="혼동 분석 결과 경로 (blender_confusion_analysis.json)")
    parser.add_argument("--output_path", type=str, required=True, help="증강된 데이터 저장 경로")
    parser.add_argument("--target_total", type=int, default=60000, help="목표 총 샘플 수 (기본값: 60000)")
    parser.add_argument("--zero_accuracy_factor", type=float, default=7.0, help="정답률 0% 조향사 증강 배수 (기본값: 7.0)")
    parser.add_argument("--min_samples", type=int, default=20, help="조향사당 최소 샘플 수 (기본값: 20)")
    
    args = parser.parse_args()
    
    # 데이터 로드
    print("데이터 로드 중...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        records = data.get('data', [])
    
    print(f"로드된 레코드 수: {len(records):,}개")
    
    # 데이터 증강
    print("\n데이터 증강 시작...")
    augmented_records = augment_confused_blenders(
        records=records,
        confusion_file=args.confusion_file,
        target_total=args.target_total,
        zero_accuracy_factor=args.zero_accuracy_factor,
        min_samples_per_blender=args.min_samples
    )
    
    # 저장
    save_augmented_data(
        augmented_records=augmented_records,
        output_path=args.output_path,
        original_data_path=args.data_path
    )
    
    print("\n✓ 데이터 증강 완료!")
