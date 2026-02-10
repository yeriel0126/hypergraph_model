"""
Step 2 보조: 동일 ID 매핑으로부터
- LightGCN/HGCF: 인접 행렬(Adjacency Matrix) 생성
- CML: Positive pair / Negative pair 샘플링용 로더

build_id_mapping_and_edges.py 실행 후 data/edges_*.json, data/id_mapping.json 을 사용합니다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_id_mapping(data_dir: Path) -> dict:
    with open(data_dir / "id_mapping.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_edges(data_dir: Path, split: str = "train") -> List[Tuple[int, int]]:
    """split in ('train','valid','test')."""
    with open(data_dir / f"edges_{split}.json", "r", encoding="utf-8") as f:
        return [tuple(pair) for pair in json.load(f)]


def build_adjacency_from_edges(
    edges: List[Tuple[int, int]],
    n_perfumers: int,
    n_ingredients: int,
):
    """
    (perfumer_id, ingredient_id) 엣지 리스트로부터 user-item 이분 인접 행렬 구성.
    LightGCN/HGCF: [0..P-1]=users(perfumers), [0..N-1]=items(ingredients).
    반환: (rows, cols, shape) 또는 scipy.sparse 행렬 (scipy 있으면).
    """
    np_edges = np.array(edges, dtype=np.int64)
    if len(np_edges) == 0:
        rows, cols = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    else:
        rows, cols = np_edges[:, 0], np_edges[:, 1]
    shape = (n_perfumers, n_ingredients)
    try:
        from scipy.sparse import coo_matrix
        A = coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=shape,
        )
        return A
    except ImportError:
        return rows, cols, shape


def get_positive_negative_pairs(
    edges_train: List[Tuple[int, int]],
    n_perfumers: int,
    n_ingredients: int,
    neg_per_pos: int = 1,
    rng=None,
):
    """
    CML용: Positive (perfumer_id, ingredient_id) / Negative (perfumer_id, neg_ingredient_id) 샘플.
    neg_per_pos: positive 하나당 샘플할 negative 개수.
    """
    if rng is None:
        rng = np.random.default_rng(2024)
    pos_set = set(edges_train)
    positives = list(pos_set)
    pairs = []
    for (u, i) in positives:
        pairs.append((int(u), int(i), 1))
        for _ in range(neg_per_pos):
            neg_i = int(rng.integers(0, n_ingredients))
            while (u, neg_i) in pos_set:
                neg_i = int(rng.integers(0, n_ingredients))
            pairs.append((int(u), neg_i, 0))
    return pairs


def main():
    data_dir = Path(__file__).resolve().parent / "data"
    if not (data_dir / "id_mapping.json").exists():
        print("먼저 build_id_mapping_and_edges.py 를 실행하세요.")
        return 1
    m = load_id_mapping(data_dir)
    n_perfumers = m["n_perfumers"]
    n_ingredients = m["n_ingredients"]
    train_edges = load_edges(data_dir, "train")
    A = build_adjacency_from_edges(train_edges, n_perfumers, n_ingredients)
    print("Adjacency (train):", A if not hasattr(A, "shape") else A.shape)
    pairs = get_positive_negative_pairs(train_edges, n_perfumers, n_ingredients, neg_per_pos=1)
    print("CML-style pairs (pos+neg) sample:", len(pairs), "ex.", pairs[:3])
    return 0


if __name__ == "__main__":
    exit(main())
