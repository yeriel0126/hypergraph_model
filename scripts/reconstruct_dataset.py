"""
데이터 재설계: 필터링 + 다운샘플링 + 층화 분할.

[필터] 조향사당 레시피가 min_count 미만인 조향사는 제외 (노이즈 제거).
[다운샘플링] 조향사당 max_count 초과분은 무작위로 제거 (편향 방지).
[층화 분할] Train/Val을 blender_id 기준 Stratified Split (검증에 "안 배운 조향사" 방지).

사용 예:
  python scripts/reconstruct_dataset.py
  python scripts/reconstruct_dataset.py --min_count 10 --max_count 400 --val_ratio 0.2 --output_dir results/checkpoints/datasets_v2
"""

# OpenMP 중복 로딩 방지 (macOS, numpy 등)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import argparse
from pathlib import Path
from collections import Counter

# 스크립트 기준 상위 경로에서 utils import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    filter_combinations_by_blender_count,
    stratified_train_val_split,
    convert_to_json_serializable,
)


def load_combinations(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data.get("combinations", [])


def main():
    parser = argparse.ArgumentParser(description="레시피 데이터셋 재구성: 필터 + 다운샘플 + 층화 분할")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="train/val 조합 JSON이 있는 디렉터리 (기본: results/checkpoints/datasets)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="저장 디렉터리 (기본: input_dir와 동일, 덮어쓰기)")
    parser.add_argument("--min_count", type=int, default=10,
                        help="조향사당 최소 레시피 수, 미만이면 해당 조향사 전체 제외 (기본 10)")
    parser.add_argument("--max_count", type=int, default=400,
                        help="조향사당 최대 레시피 수, 초과분 무작위 제거 (기본 400)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="검증 비율 0~1 (기본 0.2 = 8:2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    base = script_dir.parent
    default_input = base / "results" / "checkpoints" / "datasets"
    input_dir = Path(args.input_dir) if args.input_dir else default_input
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    train_path = input_dir / "train_combinations.json"
    val_path = input_dir / "val_combinations.json"
    if not train_path.exists():
        print(f"❌ 입력 파일 없음: {train_path}")
        print("   학습 1회 실행 후 results/checkpoints/datasets/ 에 생성된 JSON을 사용하거나 --input_dir 지정.")
        return 1

    print("=" * 60)
    print("📦 데이터 재구성: 필터 + 다운샘플 + 층화 분할")
    print("=" * 60)

    # 1) 풀 로드 (train + val 합침)
    train_raw = load_combinations(train_path)
    if val_path.exists():
        val_raw = load_combinations(val_path)
        pool = train_raw + val_raw
        print(f"\n[입력] Train: {len(train_raw):,}개 | Val: {len(val_raw):,}개 → 풀: {len(pool):,}개")
    else:
        pool = train_raw
        print(f"\n[입력] 풀: {len(pool):,}개 (val 없음)")

    # 조향사별 카운트 (필터 전)
    counts_before = Counter(c.get("target_blender") for c in pool if c.get("target_blender") is not None)
    blenders_before = len(counts_before)
    total_before = len(pool)
    avg_before = total_before / blenders_before if blenders_before else 0

    # 2) 필터 + 다운샘플
    filtered = filter_combinations_by_blender_count(
        pool,
        min_count=args.min_count,
        max_count=args.max_count,
        seed=args.seed,
    )
    counts_after = Counter(c.get("target_blender") for c in filtered if c.get("target_blender") is not None)
    blenders_after = len(counts_after)
    total_after = len(filtered)
    avg_after = total_after / blenders_after if blenders_after else 0

    print(f"\n[필터] min_count={args.min_count} → 조향사 {blenders_before:,} → {blenders_after:,} (제외: {blenders_before - blenders_after:,})")
    print(f"[다운샘플] max_count={args.max_count} → 레시피 {total_before:,} → {total_after:,}")
    if total_after == total_before and args.max_count > 400 and total_before < 50000:
        print(f"   ⚠️  입력이 이미 이전 재구성(max_count=400 등) 결과라 개수가 안 늘었습니다.")
        print(f"   💡 Train 5만+ 쓰려면: 학습 스크립트를 --load_dataset 없이 한 번 돌려 원본 6만+1만 저장 후, 이 스크립트를 다시 실행하세요.")

    # 3) 층화 분할
    train_list, val_list = stratified_train_val_split(
        filtered,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"[층화 분할] val_ratio={args.val_ratio} → Train: {len(train_list):,} | Val: {len(val_list):,}")

    # 4) 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = output_dir / "train_combinations.json"
    val_out = output_dir / "val_combinations.json"
    train_ser = convert_to_json_serializable(train_list)
    val_ser = convert_to_json_serializable(val_list)
    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_ser, f, ensure_ascii=False, indent=2)
    with open(val_out, "w", encoding="utf-8") as f:
        json.dump(val_ser, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {train_out}")
    print(f"       {val_out}")

    # 5) 결과 리포트
    print("\n" + "=" * 60)
    print("📊 결과 확인")
    print("=" * 60)
    print(f"   최종 조향사 수:     {blenders_after:,}")
    print(f"   최종 레시피 수:     Train {len(train_list):,} | Val {len(val_list):,} | 합 {total_after:,}")
    print(f"   조향사당 평균(전):  {avg_before:.1f}개")
    print(f"   조향사당 평균(후):  {avg_after:.1f}개")
    print("=" * 60)
    print("\n💡 학습 시 이 데이터 사용: --load_dataset", output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
