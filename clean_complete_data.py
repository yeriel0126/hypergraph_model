"""
complete_data 기본 정리 및 클렌징 스크립트

1. 중복 분자 필터링 (SMILES, 이름 기준)
2. 결측치 및 불일치 데이터 필터링
3. 노트 정보 정규화 (중복 제거, 표기 일관성)
4. 블렌더 정보 정규화 (이름 및 그룹 표기 일관성)
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPLETE_DATA_DIR = PROJECT_ROOT / "complete_data"
SUMMARY_PATH = COMPLETE_DATA_DIR / "complete_data_summary.json"
OUTPUT_DIR = PROJECT_ROOT / "newcode2" / "cleaned_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_complete_data() -> List[Dict]:
    """complete_data 로드"""
    if SUMMARY_PATH.exists():
        with SUMMARY_PATH.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        data = payload.get("data", [])
        if data:
            return data

    records: List[Dict] = []
    for entry in COMPLETE_DATA_DIR.glob("*.json"):
        if entry.name == "complete_data_summary.json":
            continue
        with entry.open("r", encoding="utf-8") as f:
            try:
                obj = json.load(f)
            except json.JSONDecodeError:
                continue
        records.append(obj)
    return records


def normalize_string(s: str) -> str:
    """문자열 정규화: 공백 제거, 소문자 변환"""
    if not s:
        return ""
    return " ".join(s.strip().split()).lower()


def normalize_note(note: str) -> str:
    """노트 정규화"""
    if not note:
        return ""
    # 기본 정규화
    normalized = normalize_string(note)
    # 추가 정규화 규칙 (필요시 확장)
    return normalized


def normalize_notes(notes: List[str]) -> List[str]:
    """노트 리스트 정규화 및 중복 제거"""
    if not isinstance(notes, list):
        return []
    
    normalized_notes = []
    seen = set()
    
    for note in notes:
        if not note:
            continue
        norm_note = normalize_note(str(note))
        if norm_note and norm_note not in seen:
            normalized_notes.append(norm_note)
            seen.add(norm_note)
    
    # 알파벳 순으로 정렬 (일관성 확보)
    return sorted(normalized_notes)


def normalize_blender_name(name: str) -> str:
    """블렌더 이름 정규화"""
    if not name:
        return ""
    return normalize_string(name)


def normalize_group_name(group: str) -> str:
    """블렌더 그룹 이름 정규화"""
    if not group:
        return "unknown"
    
    normalized = normalize_string(group)
    
    # 특수 케이스 처리
    if not normalized or normalized in ["no flavor group found for these", "no flavor group"]:
        return "unknown"
    
    return normalized


def normalize_blenders(blenders: List) -> List[Tuple[str, str]]:
    """블렌더 리스트 정규화 및 중복 제거"""
    if not isinstance(blenders, list):
        return []
    
    normalized_blenders = []
    seen = set()
    
    for item in blenders:
        if not isinstance(item, list) or len(item) < 1:
            continue
        
        name = str(item[0]).strip() if item[0] is not None else ""
        group = str(item[1]).strip() if len(item) > 1 and item[1] is not None else ""
        
        if not name:
            continue
        
        norm_name = normalize_blender_name(name)
        norm_group = normalize_group_name(group)
        
        # 중복 제거 (이름+그룹 조합 기준)
        key = (norm_name, norm_group)
        if key not in seen:
            normalized_blenders.append([norm_name, norm_group])
            seen.add(key)
    
    # 이름 순으로 정렬 (일관성 확보)
    normalized_blenders.sort(key=lambda x: (x[1], x[0]))  # 그룹 먼저, 그 다음 이름
    
    return normalized_blenders


def validate_record(record: Dict) -> Tuple[bool, str]:
    """레코드 유효성 검증"""
    name = record.get("name", "").strip()
    cas = record.get("cas", "").strip()
    smiles = record.get("smiles", "").strip()
    notes = record.get("notes", [])
    blenders = record.get("blenders", [])
    
    # 필수 필드 검증
    if not smiles:
        return False, "SMILES 없음"
    
    if not isinstance(notes, list) or len(notes) == 0:
        return False, "노트 없음"
    
    if not isinstance(blenders, list) or len(blenders) == 0:
        return False, "블렌더 없음"
    
    # 노트 유효성 검증
    valid_notes = [n for n in notes if n and str(n).strip()]
    if len(valid_notes) == 0:
        return False, "유효한 노트 없음"
    
    # 블렌더 유효성 검증
    valid_blenders = 0
    for item in blenders:
        if isinstance(item, list) and len(item) >= 1 and item[0]:
            valid_blenders += 1
    if valid_blenders == 0:
        return False, "유효한 블렌더 없음"
    
    return True, "OK"


def remove_duplicates(records: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    """중복 분자 제거 (SMILES 우선, 이름 차순)"""
    stats = {
        "total": len(records),
        "duplicate_smiles": 0,
        "duplicate_names": 0,
        "removed": 0,
    }
    
    # SMILES 기준으로 그룹화
    smiles_to_records = defaultdict(list)
    for record in records:
        smiles = record.get("smiles", "").strip()
        if smiles:
            smiles_to_records[smiles].append(record)
    
    # 이름 기준으로도 그룹화 (SMILES가 없는 경우 대비)
    name_to_records = defaultdict(list)
    for record in records:
        name = normalize_string(record.get("name", ""))
        if name:
            name_to_records[name].append(record)
    
    # 중복 통계
    for smiles, recs in smiles_to_records.items():
        if len(recs) > 1:
            stats["duplicate_smiles"] += len(recs) - 1
    
    for name, recs in name_to_records.items():
        if len(recs) > 1:
            stats["duplicate_names"] += len(recs) - 1
    
    # 중복 제거: SMILES 기준으로 첫 번째 레코드만 유지
    seen_smiles = set()
    seen_names = set()
    cleaned_records = []
    
    for record in records:
        smiles = record.get("smiles", "").strip()
        name = normalize_string(record.get("name", ""))
        
        # SMILES가 있으면 SMILES 기준으로 중복 체크
        if smiles:
            if smiles in seen_smiles:
                stats["removed"] += 1
                continue
            seen_smiles.add(smiles)
        # SMILES가 없으면 이름 기준으로 중복 체크
        elif name:
            if name in seen_names:
                stats["removed"] += 1
                continue
            seen_names.add(name)
        
        cleaned_records.append(record)
    
    stats["final"] = len(cleaned_records)
    return cleaned_records, stats


def clean_record(record: Dict) -> Dict:
    """개별 레코드 클렌징"""
    cleaned = {
        "name": record.get("name", "").strip(),
        "cas": record.get("cas", "").strip(),
        "smiles": record.get("smiles", "").strip(),
        "notes": normalize_notes(record.get("notes", [])),
        "blenders": normalize_blenders(record.get("blenders", [])),
    }
    
    # original_file 정보 유지 (있는 경우)
    if "original_file" in record:
        cleaned["original_file"] = record["original_file"]
    
    return cleaned


def clean_all_data(records: List[Dict]) -> Tuple[List[Dict], Dict]:
    """전체 데이터 클렌징"""
    stats = {
        "total_input": len(records),
        "validated": 0,
        "invalid": 0,
        "invalid_reasons": defaultdict(int),
        "duplicates_removed": 0,
        "final_count": 0,
    }
    
    # 1. 유효성 검증
    valid_records = []
    for record in records:
        is_valid, reason = validate_record(record)
        if is_valid:
            valid_records.append(record)
            stats["validated"] += 1
        else:
            stats["invalid"] += 1
            stats["invalid_reasons"][reason] += 1
    
    # 2. 중복 제거
    deduplicated_records, dup_stats = remove_duplicates(valid_records)
    stats["duplicates_removed"] = dup_stats["removed"]
    stats["duplicate_smiles_count"] = dup_stats["duplicate_smiles"]
    stats["duplicate_names_count"] = dup_stats["duplicate_names"]
    
    # 3. 개별 레코드 클렌징
    cleaned_records = []
    for record in deduplicated_records:
        cleaned = clean_record(record)
        cleaned_records.append(cleaned)
    
    stats["final_count"] = len(cleaned_records)
    
    return cleaned_records, stats


def save_cleaned_data(records: List[Dict], stats: Dict):
    """클렌징된 데이터 저장"""
    # 통합 파일 저장
    output_file = OUTPUT_DIR / "cleaned_complete_data.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "statistics": stats,
                "data": records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    
    # 개별 파일 저장
    individual_dir = OUTPUT_DIR / "individual"
    individual_dir.mkdir(exist_ok=True)
    
    for record in records:
        cas = record.get("cas", "").strip()
        name = record.get("name", "").strip()
        
        # 파일명 생성
        if cas:
            filename = f"{cas}_{name[:50] if name else 'unknown'}.json"
        elif name:
            filename = f"{name[:50]}.json"
        else:
            filename = f"unknown_{hash(record.get('smiles', ''))}.json"
        
        # 파일명 정리 (특수문자 제거)
        filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
        filename = filename.replace(" ", "_")
        
        filepath = individual_dir / filename
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    
    # 통계 리포트 저장
    stats_file = OUTPUT_DIR / "cleaning_statistics.json"
    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return output_file, individual_dir, stats_file


def print_statistics(stats: Dict):
    """통계 출력"""
    print("=" * 60)
    print("데이터 클렌징 통계")
    print("=" * 60)
    print(f"입력 레코드 수: {stats['total_input']:,}")
    print(f"유효한 레코드: {stats['validated']:,}")
    print(f"무효한 레코드: {stats['invalid']:,}")
    
    if stats["invalid_reasons"]:
        print("\n무효 레코드 사유:")
        for reason, count in sorted(stats["invalid_reasons"].items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}건")
    
    print(f"\n중복 제거:")
    print(f"  - SMILES 기준 중복: {stats.get('duplicate_smiles_count', 0)}건")
    print(f"  - 이름 기준 중복: {stats.get('duplicate_names_count', 0)}건")
    print(f"  - 제거된 레코드: {stats['duplicates_removed']:,}건")
    
    print(f"\n최종 클렌징된 레코드 수: {stats['final_count']:,}")
    print(f"제거율: {(1 - stats['final_count'] / stats['total_input']) * 100:.2f}%")
    print("=" * 60)


def main():
    """메인 함수"""
    print("데이터 클렌징 시작...")
    print(f"입력 디렉토리: {COMPLETE_DATA_DIR}")
    print(f"출력 디렉토리: {OUTPUT_DIR}")
    print()
    
    # 데이터 로드
    records = load_complete_data()
    if not records:
        raise RuntimeError("로드할 수 있는 레코드가 없습니다.")
    
    print(f"로드된 레코드 수: {len(records):,}")
    print()
    
    # 클렌징
    cleaned_records, stats = clean_all_data(records)
    
    # 통계 출력
    print_statistics(stats)
    
    # 저장
    output_file, individual_dir, stats_file = save_cleaned_data(cleaned_records, stats)
    
    print()
    print("클렌징 완료!")
    print(f"통합 파일: {output_file}")
    print(f"개별 파일 디렉토리: {individual_dir}")
    print(f"통계 파일: {stats_file}")
    print(f"개별 파일 수: {len(cleaned_records):,}개")


if __name__ == "__main__":
    main()

