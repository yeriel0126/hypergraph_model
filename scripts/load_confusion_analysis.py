"""
혼동 분석 결과를 로드하여 손실 함수에 사용할 수 있는 형태로 변환
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict


def load_confusion_analysis(confusion_file: str) -> Dict:
    """
    혼동 분석 결과를 로드하고 손실 함수에 사용할 수 있는 형태로 변환
    
    Returns:
        {
            'confusion_pairs': {(true_idx, pred_idx): weight},
            'hub_blenders': [hub_idx, ...],
            'confused_blenders': [confused_idx, ...]
        }
    """
    with open(confusion_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. 혼동 쌍 추출 및 가중치 계산
    confusion_pairs = {}
    predicted_counter = Counter()
    
    for pair in data['most_confusing_pairs']:
        true_idx = pair['true_idx']
        pred_idx = pair['predicted_idx']
        count = pair['confusion_count']
        
        # 혼동 횟수를 가중치로 사용 (정규화)
        weight = count / data['total_samples']  # 혼동 비율
        confusion_pairs[(true_idx, pred_idx)] = weight
        predicted_counter[pred_idx] += count
    
    # 2. 허브 조향사 식별 (여러 조향사가 혼동되는 타겟)
    # 상위 허브 조향사 (5회 이상 혼동)
    hub_blenders = [hub_idx for hub_idx, count in predicted_counter.most_common(10) if count >= 5]
    
    # 3. 완전 오분류 조향사 식별 (100% 오류율)
    confused_blenders = [
        blender['blender_idx'] 
        for blender in data['most_confused_blenders']
        if blender['error_rate'] == 1.0 and blender['total_count'] >= 5  # 최소 5개 샘플
    ]
    
    return {
        'confusion_pairs': confusion_pairs,
        'hub_blenders': hub_blenders,
        'confused_blenders': confused_blenders,
        'stats': {
            'total_confusion_pairs': len(confusion_pairs),
            'num_hub_blenders': len(hub_blenders),
            'num_confused_blenders': len(confused_blenders),
            'top_hub': predicted_counter.most_common(1)[0] if predicted_counter else None
        }
    }


if __name__ == "__main__":
    # 테스트
    script_dir = Path(__file__).parent
    confusion_file = script_dir.parent / "results" / "checkpoints" / "blender_confusion_analysis.json"
    
    if confusion_file.exists():
        result = load_confusion_analysis(str(confusion_file))
        print("혼동 분석 결과 로드 완료:")
        print(f"  - 혼동 쌍 수: {result['stats']['total_confusion_pairs']}")
        print(f"  - 허브 조향사 수: {result['stats']['num_hub_blenders']}")
        print(f"  - 완전 오분류 조향사 수: {result['stats']['num_confused_blenders']}")
        if result['stats']['top_hub']:
            print(f"  - 최고 허브 조향사: Blender_{result['stats']['top_hub'][0]} ({result['stats']['top_hub'][1]}회 혼동)")
        print(f"\n허브 조향사: {result['hub_blenders']}")
        print(f"완전 오분류 조향사: {result['confused_blenders'][:10]}...")  # 처음 10개만
    else:
        print(f"파일을 찾을 수 없습니다: {confusion_file}")
