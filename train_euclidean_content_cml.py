#!/usr/bin/env python3
"""
Content-aware CML (분자 GNN + 노트 → 유클리드, CML margin loss).

하이퍼볼릭·ContentLightGCN과 동일한 입력·데이터·평가, 손실만 CML margin으로 공정 비교.
실행: hyperbolic_model 디렉터리에서
  python train_euclidean_content_cml.py --load_dataset results/checkpoints/datasets
결과: recbole_baselines/results/recbole_ContentCML.json

에폭마다 Val HR@10이 일정하면 z_recipe 붕괴 가능 → --variance_reg 0.05, --lr 2e-4, --diagnose 시도.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model.euclidean_content_lightgcn import ContentLightGCN
from model.hyperbolic_data_loader import HyperbolicRecipeDataset, collate_hyperbolic_recipes
from model.hyperbolic_losses import HyperbolicBPRLoss

from train_hyperbolic_hypergraph import (
    load_data,
    evaluate_model,
    set_all_seeds,
)
from tqdm import tqdm


def content_cml_margin_loss(
    z_recipe: torch.Tensor,
    pos_blender_ids: torch.Tensor,
    neg_blender_ids: torch.Tensor,
    model: ContentLightGCN,
    margin: float,
) -> torch.Tensor:
    """CML margin: d(z_recipe, pos) - d(z_recipe, neg) + margin. z_recipe [B,D], ids [B]."""
    pos_emb = model.blender_anchors(pos_blender_ids)
    neg_emb = model.blender_anchors(neg_blender_ids)
    d_pos = (z_recipe - pos_emb).norm(dim=1)
    d_neg = (z_recipe - neg_emb).norm(dim=1)
    return F.relu(d_pos - d_neg + margin).mean()


def sample_pos_neg_from_batch(target_blenders: list, num_blenders: int, device: torch.device, neg_per_pos: int = 1):
    """배치별로 positive 1개, negative 1개(또는 neg_per_pos개) 샘플. (pos_ids, neg_ids) 각 [B] 또는 [B, neg_per_pos]."""
    B = len(target_blenders)
    pos_list = []
    neg_list = []
    for i in range(B):
        blenders = target_blenders[i]
        if isinstance(blenders, int):
            blenders = [blenders]
        if not blenders:
            pos_list.append(0)
            neg_list.append([0])
            continue
        pos_list.append(blenders[0] % num_blenders)
        pos_set = set(int(b) % num_blenders for b in blenders)
        negs = []
        for _ in range(neg_per_pos):
            for _ in range(50):
                n = torch.randint(0, num_blenders, (1,), device=device).item()
                if n not in pos_set:
                    negs.append(n)
                    break
            else:
                negs.append(0)
        neg_list.append(negs)
    pos_ids = torch.tensor(pos_list, dtype=torch.long, device=device)
    neg_ids = torch.tensor(neg_list, dtype=torch.long, device=device)
    return pos_ids, neg_ids


def main():
    p = argparse.ArgumentParser(description="Content CML (분자 GNN + 노트, CML margin)")
    p.add_argument("--load_dataset", type=str, required=True,
                   help="train/val/test_combinations.json 이 있는 디렉터리")
    p.add_argument("--data_path", type=str, default=None,
                   help="cleaned_complete_data.json (기본: 프로젝트 cleaned_data)")
    p.add_argument("--vocab_path", type=str, default=None,
                   help="vocabularies.json (기본: 프로젝트 feature_encoding)")
    p.add_argument("--results_dir", type=str, default=None,
                   help="결과 JSON 저장 경로 (기본: recbole_baselines/results)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4, help="학습률 (붕괴 시 1e-4, 1e-3은 붕괴 유발)")
    p.add_argument("--margin", type=float, default=0.5, help="CML margin")
    p.add_argument("--neg_per_pos", type=int, default=1)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_interval", type=int, default=1)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--variance_reg", type=float, default=0.05, help="z_recipe 붕괴 완화: 배치 내 쌍별 L2 거리 최대화 (에폭마다 성능 일정 시 권장)")
    p.add_argument("--diagnose", action="store_true", help="z_recipe 거리·std 진단 로그")
    p.add_argument("--precomputed_mol_embs", type=str, default=None,
                   help="프리컴퓨팅된 분자 임베딩 .pt 경로 (기본: 미사용, GNN end-to-end 학습). 지정 시 해당 .pt 사용")
    p.add_argument("--no_precomputed", action="store_true",
                   help="프리컴퓨팅 비활성화 → GNN end-to-end 학습 (성능 고정 시 비교용)")
    args = p.parse_args()

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    dataset_dir = Path(args.load_dataset).resolve()
    if not (dataset_dir / "train_combinations.json").exists():
        print(f"ERROR: {dataset_dir}에 train_combinations.json 없음")
        sys.exit(1)

    default_data = SCRIPT_DIR.parent / "cleaned_data" / "cleaned_complete_data.json"
    default_vocab = SCRIPT_DIR.parent / "feature_encoding" / "vocabularies.json"
    data_path = args.data_path or str(default_data)
    vocab_path = args.vocab_path or str(default_vocab)
    records, vocab_data = load_data(data_path, vocab_path)

    with open(dataset_dir / "train_combinations.json", "r", encoding="utf-8") as f:
        train_combinations = json.load(f)
    with open(dataset_dir / "val_combinations.json", "r", encoding="utf-8") as f:
        val_combinations = json.load(f)
    with open(dataset_dir / "test_combinations.json", "r", encoding="utf-8") as f:
        test_combinations = json.load(f)
    if (dataset_dir / "test_combinations_original.json").exists():
        with open(dataset_dir / "test_combinations_original.json", "r", encoding="utf-8") as f:
            test_combinations = json.load(f)
    print(f"데이터: Train={len(train_combinations):,}, Val={len(val_combinations):,}, Test={len(test_combinations):,}")

    def vocab_len(key, default=0):
        d = vocab_data.get(key, {})
        if isinstance(d, dict):
            return len(d.get("to_idx") or d.get("vocab") or d)
        return len(d) if isinstance(d, list) else default

    num_blenders = max(vocab_len("blenders", 100), 1)
    vocab_size = max(vocab_len("notes", 435), 1)

    precomputed_path = None if getattr(args, "no_precomputed", False) else args.precomputed_mol_embs
    if precomputed_path and Path(precomputed_path).exists():
        print(f"프리컴퓨팅 분자 임베딩 사용: {precomputed_path} (GNN forward 생략)")
    elif getattr(args, "no_precomputed", False):
        print("프리컴퓨팅 비활성화 → GNN end-to-end 학습")
    train_dataset = HyperbolicRecipeDataset(
        records=train_combinations,
        vocab_data=vocab_data,
        max_molecules=10,
        max_notes_per_molecule=20,
        max_blenders_per_molecule=10,
        mode="train",
        precomputed_path=precomputed_path,
    )
    val_dataset = HyperbolicRecipeDataset(
        records=val_combinations,
        vocab_data=vocab_data,
        max_molecules=10,
        max_notes_per_molecule=20,
        max_blenders_per_molecule=10,
        mode="val",
        precomputed_path=precomputed_path,
    )
    test_dataset = HyperbolicRecipeDataset(
        records=test_combinations,
        vocab_data=vocab_data,
        max_molecules=10,
        max_notes_per_molecule=20,
        max_blenders_per_molecule=10,
        mode="test",
        precomputed_path=precomputed_path,
    )

    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_hyperbolic_recipes,
        generator=gen,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(args.batch_size * 4, 64),
        shuffle=False,
        collate_fn=collate_hyperbolic_recipes,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(args.batch_size * 4, 64),
        shuffle=False,
        collate_fn=collate_hyperbolic_recipes,
        num_workers=0,
    )

    model = ContentLightGCN(
        node_dim=9,
        edge_dim=3,
        gnn_hidden_dim=args.embed_dim,
        gnn_output_dim=args.embed_dim,
        gnn_num_layers=3,
        gnn_architecture="GCN",
        vocab_size=vocab_size,
        note_embedding_dim=300,
        note_output_dim=args.embed_dim,
        num_blenders=num_blenders,
        embedding_dim=args.embed_dim,
        dropout=0.2,
    ).to(device)
    # 프리컴퓨팅 사용 시 GNN 파인튜닝 비활성화 (해당 .pt 파일이 있을 때만)
    use_precomputed = bool(precomputed_path) and Path(precomputed_path).exists()
    if use_precomputed and hasattr(model, "smiles_encoder"):
        for p in model.smiles_encoder.parameters():
            p.requires_grad = False
    trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(trainable if use_precomputed else model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn_bpr = HyperbolicBPRLoss(manifold=model.manifold).to(device)

    if use_precomputed:
        _check_loader = DataLoader(train_dataset, batch_size=min(4, len(train_dataset)), shuffle=False, collate_fn=collate_hyperbolic_recipes, num_workers=0)
        _check_batch = next(iter(_check_loader))
        if "precomputed_mol_embs" not in _check_batch:
            raise RuntimeError(
                "precomputed_mol_embs .pt 파일은 있으나 배치에 precomputed_mol_embs가 없습니다. "
                "데이터셋 생성 시 precomputed_path가 올바르게 전달되었는지 확인하세요."
            )
        print("  ✓ 배치에 precomputed_mol_embs 포함 확인 → forward 시 GNN 스킵")

    best_hr10 = 0.0
    best_state = None
    patience = 0
    print("  💡 에폭마다 Val HR@10이 일정하면 z_recipe 붕괴 가능 → --diagnose, --variance_reg 0.05~0.1, --no_precomputed 시도")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_b = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            note_indices = batch["note_indices"].to(device)
            molecule_mask = batch["molecule_mask"].to(device).float()
            smiles_graphs = batch["smiles_graphs"]
            smiles_batch = batch["smiles_batch"].to(device)
            if hasattr(smiles_graphs, "to"):
                smiles_graphs = smiles_graphs.to(device)
            else:
                smiles_graphs.x = smiles_graphs.x.to(device)
                smiles_graphs.edge_index = smiles_graphs.edge_index.to(device)
                if getattr(smiles_graphs, "edge_attr", None) is not None:
                    smiles_graphs.edge_attr = smiles_graphs.edge_attr.to(device)

            kw = dict(
                smiles_graphs=smiles_graphs,
                smiles_batch=smiles_batch,
                note_indices=note_indices,
                molecule_mask=molecule_mask,
            )
            if "precomputed_mol_embs" in batch:
                kw["precomputed_mol_embs"] = batch["precomputed_mol_embs"].to(device)
            z_recipe = model(**kw)
            pos_ids, neg_ids = sample_pos_neg_from_batch(
                batch["target_blenders"], num_blenders, device, args.neg_per_pos
            )
            if getattr(args, "diagnose", False):
                B = z_recipe.size(0)
                if B >= 2:
                    d01 = (z_recipe[0] - z_recipe[1]).norm(p=2).item()
                    z_std = z_recipe.std(dim=0).mean().item()
                    z_norm_mean = z_recipe.norm(dim=1).mean().item()
                else:
                    d01, z_std, z_norm_mean = 0.0, 0.0, z_recipe.norm(dim=1).mean().item()
                if epoch == 0 and n_b == 0:
                    print(f"[진단 CML] Epoch 1 첫 배치 | Recipe 거리(z[0]-z[1]): {d01:.4f} | z_recipe batch std: {z_std:.6f} | z L2 평균: {z_norm_mean:.4f}")
                    print(f"           pos_ids[:3]={pos_ids[:3].tolist()} neg_ids[:3]={neg_ids[:3].tolist()}")
                elif n_b == 0:
                    print(f"[진단 CML] Epoch {epoch+1} 첫 배치 | Recipe 거리: {d01:.4f} | z std: {z_std:.6f} | z L2: {z_norm_mean:.4f}")
            loss = 0.0
            for j in range(args.neg_per_pos):
                loss = loss + content_cml_margin_loss(
                    z_recipe, pos_ids, neg_ids[:, j] if neg_ids.dim() > 1 else neg_ids,
                    model, args.margin,
                )
            loss = loss / args.neg_per_pos
            variance_reg = getattr(args, "variance_reg", 0.0)
            if variance_reg > 0 and z_recipe.size(0) >= 2:
                B = z_recipe.size(0)
                sample = min(64, B)
                idx = torch.randperm(B, device=z_recipe.device)[:sample]
                zs = z_recipe[idx]
                dists = torch.cdist(zs, zs, p=2)
                triu = dists.triu(diagonal=1)
                mean_dist = triu[triu > 0].mean()
                loss = loss - variance_reg * mean_dist
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_b += 1
        train_loss = total_loss / max(n_b, 1)

        if (epoch + 1) % args.val_interval == 0:
            val_metrics = evaluate_model(
                model=model,
                val_loader=val_loader,
                device=device,
                loss_fn=loss_fn_bpr,
                k=10,
                use_blender_input=False,
                temperature=0.07,
            )
            hr10 = val_metrics.get("hit_rate@k", val_metrics.get("hit_rate@10", 0.0))
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_state = {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val HR@10: {hr10:.6f} | Best: {best_hr10:.6f}")
            if patience >= args.early_stopping_patience:
                print(f"Early stopping (patience {args.early_stopping_patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(
        model=model,
        val_loader=test_loader,
        device=device,
        loss_fn=loss_fn_bpr,
        k=10,
        use_blender_input=False,
        temperature=0.07,
    )

    results_dir = Path(args.results_dir) if args.results_dir else (SCRIPT_DIR.parent / "recbole_baselines" / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "model": "ContentCML",
        "test_result": test_metrics,
        "hit_rate@1": test_metrics.get("hit_rate@1", 0.0),
        "hit_rate@5": test_metrics.get("hit_rate@5", 0.0),
        "hit_rate@k": test_metrics.get("hit_rate@k", 0.0),
        "hit_rate@10": test_metrics.get("hit_rate@10", test_metrics.get("hit_rate@k", 0.0)),
        "recall@1": test_metrics.get("recall@1", 0.0),
        "recall@5": test_metrics.get("recall@5", 0.0),
        "recall@k": test_metrics.get("recall@k", 0.0),
        "recall@10": test_metrics.get("recall@10", test_metrics.get("recall@k", 0.0)),
        "mrr": test_metrics.get("mrr", 0.0),
        "ndcg@k": test_metrics.get("ndcg@k", 0.0),
        "ndcg@10": test_metrics.get("ndcg@10", test_metrics.get("ndcg@k", 0.0)),
    }
    out_path = results_dir / "recbole_ContentCML.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
