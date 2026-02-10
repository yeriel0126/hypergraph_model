#!/usr/bin/env python3
"""
분자 GNN 임베딩 프리컴퓨팅: 학습 전 한 번만 GNN 통과 → .pt 저장.
학습 시 --precomputed_mol_embs 로 로드하면 GNN forward를 생략해 속도 대폭 향상 (GNN fine-tuning 불가).

① 모델 가중치: 기본은 새로 생성된(랜덤 초기화) SMILESGNNEncoder 사용 = "고정 랜덤 투영".
   학습된 GNN을 쓰려면 --encoder_checkpoint 로 ContentLightGCN/제안 모델 체크포인트를 주면
   smiles_encoder 부분만 로드합니다.
② SMILES 키: 데이터 로더(HyperbolicRecipeDataset)와 100% 동일한 정규화 필수.
   (strip 후, InChI가 아니면 "." 기준 첫 조각 사용 → collect_unique_smiles / 데이터 로더 __getitem__ 동일)
③ embed_dim: 프리컴퓨팅 시 --embed_dim(기본 128)과 학습 모델의 gnn_output_dim이 일치해야 합니다.

사용법:
  cd hyperbolic_model
  python scripts/precompute_molecule_embeddings.py --dataset_dir results/checkpoints/datasets --output results/checkpoints/datasets/precomputed_mol_embs.pt
  # 학습된 GNN 사용 시:
  python scripts/precompute_molecule_embeddings.py --dataset_dir ... --encoder_checkpoint path/to/best.pt
"""
import os
import sys
import json
import argparse
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch_geometric.data import Batch, Data

SCRIPT_DIR = Path(__file__).resolve().parent
HYPERBOLIC_MODEL = SCRIPT_DIR.parent
if str(HYPERBOLIC_MODEL) not in sys.path:
    sys.path.insert(0, str(HYPERBOLIC_MODEL))

from model.hyperbolic_data_loader import molecule_to_graph_standalone, normalize_smiles_key
from model.hierarchical_hyperbolic_hypergraph import SMILESGNNEncoder


def _report_embedding_variance(embeddings: dict, embed_dim: int) -> None:
    """
    프리컴퓨팅 임베딩의 분산/변별력 보고.
    분산이 너무 작으면 레시피 구분이 안 되어 '인기 블렌더'만 추천 → HR@10 고정.
    """
    if not embeddings:
        print("  [분산 검사] 임베딩 없음, 스킵")
        return
    embs = torch.stack([embeddings[k] if isinstance(embeddings[k], torch.Tensor) else torch.tensor(embeddings[k]) for k in list(embeddings.keys())], dim=0)
    n = embs.shape[0]
    # per-dimension std (전체 분자에 대해)
    per_dim_std = embs.std(dim=0)
    mean_std = per_dim_std.mean().item()
    min_std = per_dim_std.min().item()
    max_std = per_dim_std.max().item()
    # L2 norm 통계
    norms = embs.norm(dim=1)
    mean_norm = norms.mean().item()
    std_norm = norms.std().item()
    # 샘플 기반 평균 쌍별 L2 거리 (전체 N^2는 비용 큼)
    sample_size = min(500, n)
    idx = torch.randperm(n)[:sample_size] if n > sample_size else torch.arange(n)
    sub = embs[idx]
    # (S, S) 거리 행렬 상삼각만 사용
    dists = torch.cdist(sub, sub, p=2)
    triu = dists.triu(diagonal=1)
    mean_pair_dist = triu[triu > 0].mean().item() if triu.numel() > 0 else 0.0
    print("  [분산 검사] GNN 출력 변별력:")
    print(f"      분자 수={n:,} | 임베딩 차원당 std: 평균={mean_std:.4f}, min={min_std:.4f}, max={max_std:.4f}")
    print(f"      L2 norm: 평균={mean_norm:.4f}, std={std_norm:.4f}")
    print(f"      샘플({sample_size}개) 쌍별 L2 거리 평균={mean_pair_dist:.4f}")
    if mean_std < 0.01 or mean_pair_dist < 0.1:
        print("      ⚠ 분산/거리가 매우 작음 → 레시피 구분 어려움, HR@10 고정 가능성 큼. --no_precomputed 또는 학습된 GNN으로 재프리컴퓨팅 권장.")


def collect_unique_smiles(combinations: list) -> set:
    """조합에서 고유 SMILES 추출. 키는 normalize_smiles_key로 데이터 로더와 동일하게 유지."""
    out = set()
    for recipe in combinations:
        for mol in recipe.get("molecules", []):
            s = normalize_smiles_key(mol.get("smiles") or "")
            if s:
                out.add(s)
    return out


def main():
    default_dataset = str(HYPERBOLIC_MODEL / "results" / "checkpoints" / "datasets")
    p = argparse.ArgumentParser(description="Precompute GNN molecule embeddings")
    p.add_argument("--dataset_dir", type=str, default=default_dataset,
                   help=f"train/val/test_combinations.json 디렉터리 (기본: {default_dataset})")
    p.add_argument("--output", type=str, default=None, help="출력 .pt 경로 (기본: dataset_dir/precomputed_mol_embs.pt)")
    p.add_argument("--embed_dim", type=int, default=128,
                   help="GNN 출력 차원 (학습 모델 gnn_output_dim과 일치해야 함, 기본 128)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=256, help="SMILES 배치 단위로 인코딩 (메모리 절약)")
    p.add_argument("--encoder_checkpoint", type=str, default=None,
                   help="선택: ContentLightGCN/제안 모델 체크포인트 .pt → smiles_encoder 가중치만 로드 (미지정 시 랜덤 초기화 고정 투영)")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    for name in ("train_combinations.json", "val_combinations.json", "test_combinations.json"):
        if not (dataset_dir / name).exists() and (dataset_dir / "test_combinations_original.json").exists() and name == "test_combinations.json":
            continue
        if not (dataset_dir / name).exists():
            print(f"ERROR: {dataset_dir / name} 없음")
            sys.exit(1)

    combinations = []
    for name in ("train_combinations.json", "val_combinations.json", "test_combinations.json"):
        path = dataset_dir / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                combinations.extend(json.load(f))
    test_orig = dataset_dir / "test_combinations_original.json"
    if test_orig.exists():
        with open(test_orig, "r", encoding="utf-8") as f:
            combinations.extend(json.load(f))

    unique_smiles = collect_unique_smiles(combinations)
    print(f"고유 SMILES: {len(unique_smiles):,}개 (조합 총 {len(combinations):,}건에서 추출)")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)

    encoder = SMILESGNNEncoder(
        node_dim=9,
        edge_dim=3,
        hidden_dim=args.embed_dim,
        output_dim=args.embed_dim,
        num_layers=3,
        architecture="GCN",
        dropout=0.0,
    ).to(device)
    if args.encoder_checkpoint:
        ckpt_path = Path(args.encoder_checkpoint).resolve()
        if not ckpt_path.exists():
            print(f"ERROR: encoder_checkpoint 없음: {ckpt_path}")
            sys.exit(1)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            state = getattr(state, "state_dict", lambda: state)()
        encoder_state = {k.replace("smiles_encoder.", ""): v for k, v in state.items() if k.startswith("smiles_encoder.")}
        if encoder_state:
            encoder.load_state_dict(encoder_state, strict=False)
            print(f"  ✓ Encoder 가중치 로드: {ckpt_path} (smiles_encoder 키 {len(encoder_state):,}개)")
        else:
            print(f"  WARN: 체크포인트에 'smiles_encoder.' 키 없음 → 랜덤 초기화 유지")
    else:
        print("  Encoder: 랜덤 초기화 (고정 랜덤 투영). 학습 가중치 사용 시 --encoder_checkpoint 지정.")
    encoder.eval()

    cache = {}
    smiles_list = list(unique_smiles)
    embeddings = {}
    batch_size = args.batch_size
    failed_smiles = []  # 파싱 실패 또는 예외 로깅용

    def _is_dummy_graph(g: Data) -> bool:
        """RDKit/ogb 파싱 실패 시 반환되는 더미 그래프 여부 (단일 원자 유효 분자는 제외)."""
        if g is None:
            return True
        return (
            g.edge_index.numel() == 0
            and g.x.shape[0] == 1
            and g.x.abs().sum().item() < 1e-6
        )

    with torch.no_grad():
        zero_emb = torch.zeros(args.embed_dim, dtype=torch.float)
        for start in range(0, len(smiles_list), batch_size):
            end = min(start + batch_size, len(smiles_list))
            batch_smiles = smiles_list[start:end]
            graphs = []
            valid_indices = []
            for i, s in enumerate(batch_smiles):
                try:
                    g = molecule_to_graph_standalone(s, cache)
                    if _is_dummy_graph(g):
                        failed_smiles.append(("unparseable", s))
                        embeddings[s] = zero_emb.clone()
                    else:
                        graphs.append(g)
                        valid_indices.append((i, s))
                except Exception as e:
                    failed_smiles.append(("exception", s, str(e)))
                    embeddings[s] = zero_emb.clone()
            for _, s in valid_indices:
                pass  # 아래 배치 인코딩 후 채움
            if graphs:
                batch_data = Batch.from_data_list(graphs)
                batch_data = batch_data.to(device)
                embs = encoder(
                    batch_data.x,
                    batch_data.edge_index,
                    getattr(batch_data, "edge_attr", None),
                    batch_data.batch,
                )
                for j, (i, s) in enumerate(valid_indices):
                    embeddings[s] = embs[j].cpu().clone()
            if (end - start) == batch_size or end % 500 == 0 or end == len(smiles_list):
                print(f"  인코딩: {end:,}/{len(smiles_list):,}")

    if failed_smiles:
        n_fail = len(failed_smiles)
        print(f"  ⚠ 파싱 실패/더미 그래프: {n_fail:,}건 (해당 SMILES는 0 벡터로 저장, 키는 유지)")
        for kind, *rest in failed_smiles[:10]:
            if kind == "exception":
                print(f"      exception: {rest[0][:60]}... -> {rest[1][:40]}")
            else:
                print(f"      unparseable: {rest[0][:70]}")
        if n_fail > 10:
            print(f"      ... 외 {n_fail - 10}건")

    # ① GNN 출력 분산(Variance) 확인: 임베딩이 서로 얼마나 다른지 (변별력 상실 시 수치 고정 원인)
    _report_embedding_variance(embeddings, args.embed_dim)

    out_path = Path(args.output) if args.output else dataset_dir / "precomputed_mol_embs.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"smiles_to_emb": embeddings, "embed_dim": args.embed_dim}, out_path)
    print(f"저장: {out_path} (고유 분자 {len(embeddings):,}개, embed_dim={args.embed_dim})")
    print("  → 학습 시 확인: embed_dim(128)과 모델 gnn_output_dim 일치, SMILES 키는 데이터 로더와 동일 정규화 사용.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
