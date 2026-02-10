"""
데이터셋 크기 확인 스크립트
- 원본 레코드 수
- 증강 후 레코드 수 (증강이 실행된 경우)
- Recipe Combinations 수
"""

import json
import sys
from pathlib import Path

def load_data(data_path: str):
    """데이터 로드"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        records = data.get('data', [])
    return records, data

def main():
    # 기본 경로 설정
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # 데이터 경로 찾기
    possible_paths = [
        project_root / "cleaned_data" / "cleaned_complete_data.json",
        project_root.parent / "cleaned_data" / "cleaned_complete_data.json",
        project_root / "complete_data" / "complete_data.json",
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        print("❌ 데이터 파일을 찾을 수 없습니다.")
        print("다음 경로들을 확인했습니다:")
        for path in possible_paths:
            print(f"  - {path}")
        return
    
    print(f"📁 데이터 파일: {data_path}")
    print("=" * 80)
    
    # 데이터 로드
    records, data = load_data(str(data_path))
    print(f"📊 원본 레코드 수: {len(records):,}개")
    
    # 메타데이터 확인
    metadata = data.get('metadata', {})
    if metadata.get('augmented'):
        print(f"✅ 증강된 데이터셋입니다!")
        print(f"   - 원본 레코드 수: {metadata.get('original_count', 'N/A'):,}개")
        print(f"   - 증강 후 레코드 수: {metadata.get('augmented_count', len(records)):,}개")
        print(f"   - 증강된 샘플 수: {metadata.get('augmented_count', len(records)) - metadata.get('original_count', 0):,}개")
        
        aug_stats = metadata.get('augmentation_stats', {})
        if aug_stats:
            print(f"\n📈 증강 통계:")
            print(f"   - 최소 샘플 미만 조향사: {aug_stats.get('below_min', 0):,}개")
            print(f"   - 가중치 기반 증강: {aug_stats.get('weighted', 0):,}개")
            print(f"   - 총 증강 샘플: {aug_stats.get('total', 0):,}개")
    else:
        print(f"⚠️  증강되지 않은 원본 데이터입니다.")
    
    print("=" * 80)
    
    # Recipe Combinations 수는 학습 스크립트 실행 시 확인 가능
    print("\n💡 Recipe Combinations 수는 학습 스크립트 실행 시 확인할 수 있습니다.")
    print("   학습 스크립트에서 'Created X recipe combinations (target: Y)' 메시지를 확인하세요.")

if __name__ == "__main__":
    main()
