#!/usr/bin/env python3
"""
유클리드 거리 기반 CML (Collaborative Metric Learning) 단독 실행.
하이퍼볼릭과 동일 실험 세팅: 같은 데이터(train/val/test), seed=42, embed_dim=128,
full ranking 평가, HR@1/5/10, MRR, NDCG@10.

사용법:
  python run_cml_euclidean.py
  python run_cml_euclidean.py --dataset_dir ../hyperbolic_model/results/checkpoints/datasets
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# OpenMP / NumPy 2.0 (run_baselines와 동일)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64

# 프로젝트 루트
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from cml_euclidean_model import CML_Euclidean, cml_margin_loss  # noqa: E402


RANDOM_SEED = 42
EMBED_DIM = 128
K = 10


def load_combinations(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pairs_from_combinations(combinations: list, user_offset: int) -> list[tuple[int, int]]:
    """(user_id 0-based, item_id 0-based) 리스트. 복수 정답 전부 포함."""
    pairs = []
    for i, c in enumerate(combinations):
        u = user_offset + i
        blenders = c.get("target_blenders") or c.get("target_blender")
        if isinstance(blenders, int):
            blenders = [blenders]
        if not blenders:
            continue
        for b in blenders:
            pairs.append((u, int(b)))
    return pairs


def get_num_items(train_pairs: list, val_pairs: list, test_pairs: list) -> int:
    all_items = set()
    for a, b in train_pairs + val_pairs + test_pairs:
        all_items.add(b)
    return max(all_items) + 1 if all_items else 0


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[int, int]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def collate_pairs(batch):
    users = torch.tensor([b[0] for b in batch], dtype=torch.long)
    items = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return users, items


def evaluate(
    model: CML_Euclidean,
    test_pairs: list[tuple[int, int]],
    num_items: int,
    device: torch.device,
    k: int = 10,
    batch_size: int = 256,
) -> dict[str, float]:
    """Full ranking: 각 테스트 유저에 대해 전체 아이템과 거리 계산 후 Top-K. 하이퍼볼릭과 동일 지표."""
    model.eval()
    user_emb = model.get_user_embeddings().to(device)
    item_emb = model.get_item_embeddings().to(device)

    # test_pairs에서 유저별 정답 아이템 집합
    user_to_items: dict[int, set[int]] = {}
    for u, i in test_pairs:
        user_to_items.setdefault(u, set()).add(i)
    test_users = sorted(user_to_items.keys())

    top_1_hits = top_5_hits = top_k_hits = 0
    sum_recall_1 = sum_recall_5 = sum_recall_k = 0.0
    reciprocal_ranks = []
    ndcg_scores = []
    total = 0

    with torch.no_grad():
        for start in range(0, len(test_users), batch_size):
            end = min(start + batch_size, len(test_users))
            u_batch = test_users[start:end]
            u_ids = torch.tensor(u_batch, dtype=torch.long, device=device)
            u_embs = user_emb[u_ids]  # [B, D]
            dist = torch.cdist(u_embs, item_emb, p=2)  # [B, num_items]
            _, top_k_indices = torch.topk(dist, k=k, dim=1, largest=False)  # 거리 작은 순
            top_k_indices = top_k_indices.cpu().numpy()

            for idx, u in enumerate(u_batch):
                true_items = user_to_items[u]
                pred = top_k_indices[idx]
                total += 1
                n_true = len(true_items)
                # Hit Rate (binary)
                if pred[0] in true_items:
                    top_1_hits += 1
                if any(p in true_items for p in pred[:5]):
                    top_5_hits += 1
                if len(set(pred) & true_items) > 0:
                    top_k_hits += 1
                # Recall (비율)
                if n_true > 0:
                    sum_recall_1 += 1.0 if pred[0] in true_items else 0.0
                    sum_recall_5 += len(set(pred[:5]) & true_items) / n_true
                    sum_recall_k += len(set(pred) & true_items) / n_true
                rr = 0.0
                for rank, p in enumerate(pred, start=1):
                    if p in true_items:
                        rr = 1.0 / rank
                        break
                reciprocal_ranks.append(rr)
                dcg = sum(1.0 / np.log2(r + 1) for r, p in enumerate(pred, start=1) if p in true_items)
                idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(k, len(true_items)) + 1))
                ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    n = total if total else 1
    return {
        "hit_rate@1": top_1_hits / n,
        "hit_rate@5": top_5_hits / n,
        "hit_rate@k": top_k_hits / n,
        "hit_rate@10": top_k_hits / n,
        "recall@1": sum_recall_1 / n,
        "recall@5": sum_recall_5 / n,
        "recall@k": sum_recall_k / n,
        "recall@10": sum_recall_k / n,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "ndcg@k": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "ndcg@10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="유클리드 CML (하이퍼볼릭과 동일 세팅)")
    p.add_argument("--dataset_dir", type=str, default=None, help="train/val/test_combinations.json 위치")
    p.add_argument("--embed_dim", type=int, default=EMBED_DIM, help="임베딩 차원 (기본 128)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="랜덤 시드 (기본 42)")
    p.add_argument("--epochs", type=int, default=200, help="에폭")
    p.add_argument("--batch_size", type=int, default=2048, help="배치 크기")
    p.add_argument("--lr", type=float, default=1e-3, help="학습률")
    p.add_argument("--margin", type=float, default=0.5, help="CML margin")
    p.add_argument("--neg_per_pos", type=int, default=1, help="positive당 negative 샘플 수")
    p.add_argument("--results_dir", type=str, default=None, help="결과 JSON 저장 디렉터리")
    p.add_argument("--stopping_step", type=int, default=15, help="val 개선 없으면 조기 종료 (기본 15)")
    args = p.parse_args()

    # run_baselines와 동일 경로: 정제 우선 datasets_refined → datasets_v2 → datasets
    _checkpoints = (SCRIPT_DIR / ".." / "hyperbolic_model" / "results" / "checkpoints").resolve()
    if args.dataset_dir is None:
        for cand in [_checkpoints / "datasets_refined", _checkpoints / "datasets_v2", _checkpoints / "datasets"]:
            if (cand / "train_combinations.json").exists():
                args.dataset_dir = str(cand)
                break
        if args.dataset_dir is None:
            args.dataset_dir = str(_checkpoints / "datasets")
    base = Path(args.dataset_dir).resolve()
    if not (base / "train_combinations.json").exists() and (base / "datasets" / "train_combinations.json").exists():
        base = (base / "datasets").resolve()

    train_path = base / "train_combinations.json"
    val_path = base / "val_combinations.json"
    test_path = base / "test_combinations.json"
    test_orig_path = base / "test_combinations_original.json"
    if test_orig_path.exists():
        test_path = test_orig_path

    print("CML Euclidean 실행 중 (유클리드 CML, 하이퍼볼릭과 동일 지표)")
    print(f"  dataset_dir: {base}")

    for p in (train_path, val_path, test_path):
        if not p.exists():
            print(f"ERROR: 데이터 파일 없음 — {p}", file=sys.stderr)
            print(f"  → 먼저 하이퍼볼릭 학습을 한 번 돌리거나, --dataset_dir 로 조합 JSON이 있는 폴더를 지정하세요.", file=sys.stderr)
            print(f"  예: python run_cml_euclidean.py --dataset_dir ../hyperbolic_model/results/checkpoints/datasets", file=sys.stderr)
            return 1

    train_combo = load_combinations(train_path)
    val_combo = load_combinations(val_path)
    test_combo = load_combinations(test_path)
    n_train, n_val, n_test = len(train_combo), len(val_combo), len(test_combo)
    num_users = n_train + n_val + n_test

    train_pairs = pairs_from_combinations(train_combo, 0)
    val_pairs = pairs_from_combinations(val_combo, n_train)
    test_pairs = pairs_from_combinations(test_combo, n_train + n_val)
    num_items = get_num_items(train_pairs, val_pairs, test_pairs)

    print(f"데이터: train 유저 {n_train}, val {n_val}, test {n_test} | 아이템(조향사) {num_items}")
    print(f"Train pairs(복수 정답 포함): {len(train_pairs)} | seed={args.seed}, embed_dim={args.embed_dim}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = CML_Euclidean(
        num_users=num_users,
        num_items=num_items,
        embed_dim=args.embed_dim,
        margin=args.margin,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = PairDataset(train_pairs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pairs, num_workers=0)

    best_val_hr10 = 0.0
    best_state = None
    no_improve = 0
    print(f"\n학습 시작 (최대 {args.epochs} 에폭, 조기 종료={args.stopping_step} 에폭)", flush=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_b = 0
        for users, pos_items in train_loader:
            users = users.to(device)
            pos_items = pos_items.to(device)
            B = users.size(0)
            neg_items = torch.randint(0, num_items, (B, args.neg_per_pos), device=device)
            loss = 0.0
            for j in range(args.neg_per_pos):
                loss = loss + cml_margin_loss(users, pos_items, neg_items[:, j], model, margin=args.margin)
            loss = loss / args.neg_per_pos
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_b += 1
        train_loss = total_loss / n_b if n_b else 0.0

        val_metrics = evaluate(model, val_pairs, num_items, device, k=K, batch_size=256)
        val_hr10 = val_metrics["hit_rate@10"]
        if val_hr10 > best_val_hr10:
            best_val_hr10 = val_hr10
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # 매 에폭 진행 상황 출력 (stdout 버퍼 즉시 반영)
        print(f"Epoch {epoch+1}/{args.epochs}  loss={train_loss:.4f}  val HR@10={val_hr10:.4f}  MRR={val_metrics['mrr']:.4f}", flush=True)
        if no_improve >= args.stopping_step:
            print(f"조기 종료 (val HR@10 {args.stopping_step} 에폭 개선 없음)", flush=True)
            break

    if best_state:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_pairs, num_items, device, k=K, batch_size=256)
    print("\n--- Test (하이퍼볼릭과 동일 지표: HR + Recall) ---")
    print(f"  HR@1={test_metrics['hit_rate@1']:.4f}  HR@5={test_metrics['hit_rate@5']:.4f}  HR@10={test_metrics['hit_rate@10']:.4f}")
    print(f"  Recall@1={test_metrics['recall@1']:.4f}  Recall@5={test_metrics['recall@5']:.4f}  Recall@10={test_metrics['recall@10']:.4f}")
    print(f"  MRR={test_metrics['mrr']:.4f}  NDCG@10={test_metrics['ndcg@10']:.4f}")

    out = {
        "model": "CML_Euclidean",
        "hit_rate@1": test_metrics["hit_rate@1"],
        "hit_rate@5": test_metrics["hit_rate@5"],
        "hit_rate@k": test_metrics["hit_rate@k"],
        "hit_rate@10": test_metrics["hit_rate@10"],
        "recall@1": test_metrics["recall@1"],
        "recall@5": test_metrics["recall@5"],
        "recall@k": test_metrics["recall@k"],
        "recall@10": test_metrics["recall@10"],
        "mrr": test_metrics["mrr"],
        "ndcg@k": test_metrics["ndcg@k"],
        "ndcg@10": test_metrics["ndcg@10"],
        "embed_dim": args.embed_dim,
        "seed": args.seed,
        "dataset_dir": str(base),
    }
    results_dir = Path(args.results_dir or SCRIPT_DIR / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "recbole_CML_Euclidean.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
