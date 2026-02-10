#!/usr/bin/env python3
"""
Step 1: 데이터 '아이디' 매핑 (ID Mapping)
  - Perfumer ID: 0, 1, 2, …, (num_blenders - 1)  [조향사/블렌더]
  - Ingredient ID: 0, 1, 2, …, (num_ingredients - 1)  [성분/분자]

Step 2: 상호작용 행렬(엣지 리스트) 생성
  - (perfumer_id, ingredient_id) 형태의 엣지 리스트
  - train / valid / test 분할 (동일 시드로 고정 → 모든 베이스라인이 같은 분할 사용)

출력:
  - data/id_mapping.json    : 매핑 테이블 (모든 모델이 동일 ID 공유)
  - data/edges_train.json   : train 엣지 리스트
  - data/edges_valid.json   : valid 엣지 리스트
  - data/edges_test.json    : test 엣지 리스트
  - data/adjacency_info.json: 행렬 크기 등 (LightGCN/HGCF용)
  - data/recbole/odor/*.inter: RecBole benchmark 형식 (part1/2/3)

사용법:
  python build_id_mapping_and_edges.py --data_path ../cleaned_data/cleaned_complete_data.json --vocab_path ../feature_encoding/vocabularies.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_blender_name_to_id(vocab_data: dict) -> tuple[dict[str, int], list[str]]:
    """vocab에서 Perfumer(블렌더) ID 매핑: name -> 0..P-1, id_to_name list."""
    blender_data = vocab_data.get("blenders") or {}
    name_to_id = {}
    id_to_name = []
    if isinstance(blender_data, dict) and "vocab" in blender_data:
        for idx, name in enumerate(blender_data["vocab"]):
            if isinstance(name, str):
                key = name.lower()
                name_to_id[key] = idx
                id_to_name.append(name)
    elif isinstance(blender_data, list):
        for idx, name in enumerate(blender_data):
            if isinstance(name, str):
                key = name.lower()
                name_to_id[key] = idx
                id_to_name.append(name)
    return name_to_id, id_to_name


def build_ingredient_list_and_edges(
    data: list | dict,
    blender_name_to_id: dict[str, int],
) -> tuple[list[dict], list[tuple[int, int]]]:
    """
    레코드에서 성분(ingredient) 순서와 (perfumer_id, ingredient_id) 엣지 리스트 생성.
    - ingredient_id = 0..len(valid_records)-1 (blenders가 하나라도 있는 레코드만)
    """
    records = data if isinstance(data, list) else data.get("data", data.get("records", []))
    if not isinstance(records, list):
        records = []
    id_to_key = []  # ingredient_id -> (name, cas) or smiles
    edges = []
    for record in records:
        if record.get("molecules"):
            continue
        blenders = record.get("blenders") or []
        if not blenders:
            continue
        ingredient_id = len(id_to_key)
        key = {
            "name": record.get("name"),
            "cas": record.get("cas"),
            "smiles": record.get("smiles"),
        }
        id_to_key.append(key)
        seen_perfumer = set()
        for item in blenders:
            if isinstance(item, list) and len(item) > 0:
                name = item[0]
            elif isinstance(item, str):
                name = item
            else:
                continue
            pid = blender_name_to_id.get(name.lower())
            if pid is not None and pid not in seen_perfumer:
                seen_perfumer.add(pid)
                edges.append((pid, ingredient_id))
    return id_to_key, edges


def split_edges(
    edges: list[tuple[int, int]],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 2024,
) -> tuple[list, list, list]:
    """동일 시드로 train/valid/test 분할 (모든 베이스라인 공통)."""
    rng = random.Random(seed)
    indices = list(range(len(edges)))
    rng.shuffle(indices)
    n = len(indices)
    t = int(n * train_ratio)
    v = int(n * valid_ratio)
    train_idx = indices[:t]
    valid_idx = indices[t : t + v]
    test_idx = indices[t + v :]
    return (
        [edges[i] for i in train_idx],
        [edges[i] for i in valid_idx],
        [edges[i] for i in test_idx],
    )


def main():
    parser = argparse.ArgumentParser(description="ID 매핑 및 상호작용 엣지 생성")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="cleaned_complete_data.json 경로",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="vocabularies.json 경로",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="출력 디렉터리 (기본: recbole_baselines/data)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train 비율 (기본 0.8)",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.1,
        help="Valid 비율 (기본 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="분할 시드 (모든 베이스라인 공통)",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    root = base.parent
    data_path = Path(args.data_path or str(root / "cleaned_data" / "cleaned_complete_data.json"))
    vocab_path = Path(args.vocab_path or str(root / "feature_encoding" / "vocabularies.json"))
    out_dir = Path(args.out_dir) if args.out_dir else (base / "data")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"데이터 파일 없음: {data_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"어휘 파일 없음: {vocab_path}")

    # Step 1: ID 매핑
    vocab_data = load_json(vocab_path)
    blender_name_to_id, perfumer_id_to_name = build_blender_name_to_id(vocab_data)
    n_perfumers = len(perfumer_id_to_name)
    print(f"Perfumer(블렌더) 수: {n_perfumers} (ID 0..{n_perfumers - 1})")

    data = load_json(data_path)
    ingredient_id_to_key, edges = build_ingredient_list_and_edges(data, blender_name_to_id)
    n_ingredients = len(ingredient_id_to_key)
    print(f"Ingredient(성분) 수: {n_ingredients} (ID 0..{n_ingredients - 1})")
    print(f"총 엣지 수: {len(edges)}")

    id_mapping = {
        "n_perfumers": n_perfumers,
        "n_ingredients": n_ingredients,
        "perfumer_id_to_name": perfumer_id_to_name,
        "ingredient_id_to_key": ingredient_id_to_key,
    }
    mapping_path = out_dir / "id_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(id_mapping, f, indent=2, ensure_ascii=False)
    print(f"저장: {mapping_path}")

    # Step 2: 상호작용 train/valid/test 분할
    train_edges, valid_edges, test_edges = split_edges(
        edges,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    print(f"분할: train={len(train_edges)}, valid={len(valid_edges)}, test={len(test_edges)}")

    for name, edge_list in [
        ("edges_train", train_edges),
        ("edges_valid", valid_edges),
        ("edges_test", test_edges),
    ]:
        path = out_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(edge_list, f, indent=0)
        print(f"저장: {path}")

    adjacency_info = {
        "n_perfumers": n_perfumers,
        "n_ingredients": n_ingredients,
        "n_edges_train": len(train_edges),
        "n_edges_valid": len(valid_edges),
        "n_edges_test": len(test_edges),
        "seed": args.seed,
    }
    with open(out_dir / "adjacency_info.json", "w", encoding="utf-8") as f:
        json.dump(adjacency_info, f, indent=2)

    # RecBole .inter 형식 (part1=train, part2=valid, part3=test) — 0-based ID 그대로 사용
    recbole_dir = out_dir / "recbole" / "odor"
    recbole_dir.mkdir(parents=True, exist_ok=True)
    header = "user_id:token\titem_id:token"
    for part_name, edge_list in [
        ("part1", train_edges),
        ("part2", valid_edges),
        ("part3", test_edges),
    ]:
        inter_path = recbole_dir / f"odor.{part_name}.inter"
        with open(inter_path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            for u, i in edge_list:
                f.write(f"{u}\t{i}\n")
        print(f"저장: {inter_path}")

    print("\n완료. 모든 베이스라인은 data/id_mapping.json 과 data/edges_*.json 을 사용하세요.")
    return 0


if __name__ == "__main__":
    exit(main())
