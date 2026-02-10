#!/usr/bin/env python3
"""
프리컴퓨팅된 분자 임베딩 .pt의 분산(Variance) 검사.

변별력 상실 시 레시피 A/B가 같은 입력으로 보여 '인기 블렌더'만 추천 → HR@10 고정.
이 스크립트로 .pt 로드 직후 임베딩이 서로 얼마나 다른지 확인하세요.

사용법:
  cd hyperbolic_model
  python scripts/check_precomputed_variance.py --pt results/checkpoints/datasets/precomputed_mol_embs.pt
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # OpenMP libomp 중복 초기화 시 abort 방지

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
HYPERBOLIC_MODEL = SCRIPT_DIR.parent
if str(HYPERBOLIC_MODEL) not in sys.path:
    sys.path.insert(0, str(HYPERBOLIC_MODEL))


def report_variance(embeddings: dict, embed_dim: int, label: str = "") -> None:
    if not embeddings:
        print(f"{label}[분산 검사] 임베딩 없음")
        return
    embs = torch.stack([
        embeddings[k] if isinstance(embeddings[k], torch.Tensor) else torch.tensor(embeddings[k], dtype=torch.float)
        for k in list(embeddings.keys())
    ], dim=0)
    n = embs.shape[0]
    per_dim_std = embs.std(dim=0)
    mean_std = per_dim_std.mean().item()
    min_std = per_dim_std.min().item()
    max_std = per_dim_std.max().item()
    norms = embs.norm(dim=1)
    mean_norm = norms.mean().item()
    std_norm = norms.std().item()
    sample_size = min(500, n)
    idx = torch.randperm(n)[:sample_size] if n > sample_size else torch.arange(n)
    sub = embs[idx]
    dists = torch.cdist(sub, sub, p=2)
    triu = dists.triu(diagonal=1)
    mean_pair_dist = triu[triu > 0].mean().item() if triu.numel() > 0 else 0.0

    print(f"{label}[분산 검사] GNN 출력 변별력 (분자 수={n:,}, dim={embed_dim})")
    print(f"    차원당 std: 평균={mean_std:.4f}, min={min_std:.4f}, max={max_std:.4f}")
    print(f"    L2 norm: 평균={mean_norm:.4f}, std={std_norm:.4f}")
    print(f"    샘플({sample_size}개) 쌍별 L2 거리 평균={mean_pair_dist:.4f}")
    if mean_std < 0.01 or mean_pair_dist < 0.1:
        print("    ⚠ 분산/거리 매우 작음 → 레시피 구분 어려움, HR@10 고정 가능성 큼. --no_precomputed 또는 학습된 GNN으로 재프리컴퓨팅 권장.")
    print()


def main():
    p = argparse.ArgumentParser(description="프리컴퓨팅 .pt 임베딩 분산 검사")
    default_pt = str(HYPERBOLIC_MODEL / "results" / "checkpoints" / "datasets" / "precomputed_mol_embs.pt")
    p.add_argument("--pt", type=str, default=default_pt, help=f".pt 경로 (기본: {default_pt})")
    args = p.parse_args()

    path = Path(args.pt).resolve()
    if not path.exists():
        print(f"ERROR: 파일 없음: {path}")
        return 1

    data = torch.load(path, map_location="cpu", weights_only=False)
    embeddings = data.get("smiles_to_emb", data)
    embed_dim = data.get("embed_dim", 128)

    if isinstance(embeddings, torch.Tensor):
        print("ERROR: smiles_to_emb가 딕셔너리가 아닌 텐서입니다.")
        return 1

    print(f"로드: {path}")
    report_variance(embeddings, embed_dim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
