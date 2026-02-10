#!/usr/bin/env python3
"""
하이퍼볼릭과 동일한 데이터로 RecBole 베이스라인을 돌리기 위한 변환 스크립트.

사용 파일 (dataset_dir 안):
  - train_combinations.json  → part1 (train)
  - val_combinations.json    → part2 (valid)
  - test_combinations_original.json 있으면 → part3 (test, 하이퍼볼릭 Test(원본)과 동일)
  - 없으면 test_combinations.json → part3

사용법:
  python prepare_recbole_data.py --dataset_dir /path/to/datasets
  python prepare_recbole_data.py --dataset_dir /path/to/datasets --out_dir ./odor
"""
import argparse
import json
from pathlib import Path


def load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def combinations_to_inter_rows(combinations: list, user_id_offset: int) -> list[tuple[int, int]]:
    """
    조합 리스트에서 (user_id, item_id) 행 목록 생성.
    - user_id: 1-based, recipe별 고유 (offset + 1, offset + 2, ...)
    - item_id: 1-based (target_blenders가 0-based 정수이면 +1)
    """
    rows = []
    for i, combo in enumerate(combinations):
        user_id = user_id_offset + i + 1  # 1-based
        target_blenders = combo.get("target_blenders") or combo.get("target_blender")
        if isinstance(target_blenders, int):
            target_blenders = [target_blenders]
        if not target_blenders:
            continue
        for b in target_blenders:
            bid = int(b)
            rows.append((user_id, bid + 1))  # item_id 1-based
    return rows


def main():
    script_dir = Path(__file__).resolve().parent
    default_dataset = script_dir / ".." / "hyperbolic_model" / "results" / "checkpoints" / "datasets"
    parser = argparse.ArgumentParser(description="RecBole용 .inter 파일 생성 (benchmark split)")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(default_dataset.resolve()),
        help="train_combinations.json, val_combinations.json, test_combinations.json 가 있는 디렉터리 (기본: ../hyperbolic_model/results/checkpoints/datasets)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="RecBole 데이터셋 출력 디렉터리 (기본: dataset_dir/recbole_data/데이터셋이름)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="odor",
        help="RecBole 데이터셋 이름 (파일 접두사, 기본: odor)",
    )
    args = parser.parse_args()

    base = Path(args.dataset_dir)
    required = ["train_combinations.json", "val_combinations.json"]
    optional_test = ["test_combinations_original.json", "test_combinations.json"]
    for name in required:
        if not (base / name).exists():
            raise FileNotFoundError(f"필요한 파일 없음: {base / name}")
    if not any((base / n).exists() for n in optional_test):
        raise FileNotFoundError(f"테스트 파일 없음: {base} 에 test_combinations_original.json 또는 test_combinations.json 필요")

    # 하이퍼볼릭과 동일 분할임을 확인하기 위해 split_info.json 출력
    split_info_path = base / "split_info.json"
    if split_info_path.exists():
        try:
            with open(split_info_path, "r", encoding="utf-8") as f:
                split_info = json.load(f)
            print(f"[데이터셋 일치] 하이퍼볼릭 split_info 사용: seed={split_info.get('random_seed')}, "
                  f"train={split_info.get('train_size')}, val={split_info.get('val_size')}, "
                  f"test_original={split_info.get('test_original_size', split_info.get('test_size'))}")
        except Exception:
            pass

    # RecBole: data_path 아래에 {dataset_name}/ 폴더가 있어야 함 → {dataset_name}.part1.inter 등
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = base / "recbole_data" / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.dataset_name

    train = load_json(base / "train_combinations.json")
    val = load_json(base / "val_combinations.json")
    test = load_json(base / "test_combinations.json")

    # 하이퍼볼릭과 동일: Test(원본) 사용 (Train/Val/Test(원본) = 동일 split)
    test_orig_path = base / "test_combinations_original.json"
    if test_orig_path.exists():
        test = load_json(test_orig_path)
        print(f"[동일 데이터] Train={len(train):,}, Val={len(val):,}, Test(원본)={len(test):,} ⭐ (하이퍼볼릭과 동일)")
    else:
        print(f"[동일 데이터] Train={len(train):,}, Val={len(val):,}, Test={len(test):,} (test_combinations.json 사용; 원본 있으면 동일 권장)")

    offset_train = 0
    offset_val = len(train)
    offset_test = offset_val + len(val)

    rows_train = combinations_to_inter_rows(train, offset_train)
    rows_val = combinations_to_inter_rows(val, offset_val)
    rows_test = combinations_to_inter_rows(test, offset_test)

    header = "user_id:token\titem_id:token"
    sep = "\t"

    def write_inter(path: Path, rows: list) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            for uid, iid in rows:
                f.write(f"{uid}{sep}{iid}\n")

    write_inter(out_dir / f"{name}.part1.inter", rows_train)
    write_inter(out_dir / f"{name}.part2.inter", rows_val)
    write_inter(out_dir / f"{name}.part3.inter", rows_test)
    # RecBole가 요구하는 메인 파일: part1+part2+part3 병합
    all_rows = rows_train + rows_val + rows_test
    write_inter(out_dir / f"{name}.inter", all_rows)

    num_items = max(
        (iid for rows in [rows_train, rows_val, rows_test] for _, iid in rows),
        default=0,
    )
    info = {
        "dataset_name": name,
        "out_dir": str(out_dir),
        "data_path_for_recbole": str(out_dir.parent),
        "n_train_users": len(train),
        "n_valid_users": len(val),
        "n_test_users": len(test),
        "n_train_inter": len(rows_train),
        "n_valid_inter": len(rows_val),
        "n_test_inter": len(rows_test),
        "n_items": num_items,
        "n_users_total": len(train) + len(val) + len(test),
    }
    with open(out_dir / "recbole_data_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"RecBole 데이터 생성 완료: {out_dir}")
    print(f"  {name}.part1.inter (train): {len(rows_train)} interactions, {len(train)} users")
    print(f"  {name}.part2.inter (valid): {len(rows_val)} interactions, {len(val)} users")
    print(f"  {name}.part3.inter (test):  {len(rows_test)} interactions, {len(test)} users")
    print(f"  items (blenders): 1 ~ {num_items}")
    return 0


if __name__ == "__main__":
    exit(main())
