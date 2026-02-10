#!/usr/bin/env python3
"""
공식 HGCF (https://github.com/layer6ai-labs/HGCF) 레포에서 사용할 데이터 생성.

우리 실험의 edges_train.json / edges_test.json (또는 RecBole part1·part3)을
HGCF 형식으로 변환: train_dict (user_id -> list of item_id), test_dict 동일.
출력: data/odor/train.pkl, data/odor/test.pkl → HGCF의 Data 클래스에서 'odor' 로 로드 가능하도록.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from collections import defaultdict


def load_edges_json(path: Path) -> list[tuple[int, int]]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        return [tuple(p) for p in json.load(f)]


def load_inter_file(path: Path) -> list[tuple[int, int]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                rows.append((int(parts[0]), int(parts[1])))
    return rows


def edges_to_user_item_dicts(
    train_edges: list[tuple[int, int]],
    test_edges: list[tuple[int, int]],
) -> tuple[dict, dict, int, int]:
    """
    (user_id, item_id) 엣지 리스트 -> train_dict, test_dict (모든 user id가 키로 존재).
    user_id, item_id는 0-based 연속이라고 가정.
    """
    train_dict = defaultdict(list)
    test_dict = defaultdict(list)
    for u, i in train_edges:
        train_dict[u].append(i)
    for u, i in test_edges:
        test_dict[u].append(i)
    all_users = set(train_dict) | set(test_dict)
    all_items = set()
    for items in train_dict.values():
        all_items.update(items)
    for items in test_dict.values():
        all_items.update(items)
    num_users = max(all_users) + 1 if all_users else 0
    num_items = max(all_items) + 1 if all_items else 0
    for u in range(num_users):
        if u not in train_dict:
            train_dict[u] = []
        if u not in test_dict:
            test_dict[u] = []
    return dict(train_dict), dict(test_dict), num_users, num_items


def main():
    parser = argparse.ArgumentParser(description="HGCF 공식 레포용 train/test pkl 생성")
    parser.add_argument(
        "--edges_dir",
        type=str,
        default=None,
        help="edges_train.json, edges_test.json 있는 폴더 (기본: recbole_baselines/data)",
    )
    parser.add_argument(
        "--recbole_inter_dir",
        type=str,
        default=None,
        help="RecBole .inter가 있는 디렉터리 (예: .../recbole_data/odor). part1=train, part3=test 사용. 베이스라인·제안 모델 동일 데이터용.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="odor",
        help="--recbole_inter_dir 사용 시 파일 접두사 (기본: odor → odor.part1.inter, odor.part3.inter)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="출력 디렉터리 (기본: recbole_baselines/data/odor_hgcf)",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else (base / "data" / "odor_hgcf")

    if args.recbole_inter_dir:
        # 하이퍼볼릭/RecBole과 동일한 레시피→블렌더 분할 사용 (공정 비교)
        inter_dir = Path(args.recbole_inter_dir)
        prefix = args.dataset_name
        p1 = inter_dir / f"{prefix}.part1.inter"
        p3 = inter_dir / f"{prefix}.part3.inter"
        if not p1.exists() or not p3.exists():
            raise FileNotFoundError(
                f"RecBole .inter 없음: {p1}, {p3}. --recbole_inter_dir 경로를 확인하세요."
            )
        train_edges = load_inter_file(p1)
        test_edges = load_inter_file(p3)
    else:
        edges_dir = Path(args.edges_dir) if args.edges_dir else (base / "data")
        train_path = edges_dir / "edges_train.json"
        test_path = edges_dir / "edges_test.json"
        if not train_path.exists() or not test_path.exists():
            recbole_dir = edges_dir / "recbole" / "odor"
            if (recbole_dir / "odor.part1.inter").exists() and (recbole_dir / "odor.part3.inter").exists():
                train_edges = load_inter_file(recbole_dir / "odor.part1.inter")
                test_edges = load_inter_file(recbole_dir / "odor.part3.inter")
            else:
                raise FileNotFoundError(
                    f"데이터 없음: {train_path}, {test_path}. "
                    "먼저 build_id_mapping_and_edges.py 를 실행하세요."
                )
        else:
            train_edges = load_edges_json(train_path)
            test_edges = load_edges_json(test_path)

    train_dict, test_dict, num_users, num_items = edges_to_user_item_dicts(train_edges, test_edges)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "train.pkl", "wb") as f:
        pickle.dump(train_dict, f)
    with open(out_dir / "test.pkl", "wb") as f:
        pickle.dump(test_dict, f)

    print(f"HGCF용 데이터 저장: {out_dir}")
    print(f"  num_users={num_users}, num_items={num_items}")
    print(f"  train.pkl, test.pkl → HGCF 레포의 data/odor/ 에 복사 후 dataset='odor' 로 사용")
    return 0


if __name__ == "__main__":
    exit(main())
