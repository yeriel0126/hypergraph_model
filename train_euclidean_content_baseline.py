#!/usr/bin/env python3
"""
Content-aware LightGCN (분자 GNN + 노트 → 유클리드 랭킹).

하이퍼볼릭과 동일한 입력·데이터·평가를 쓰고, 공간만 유클리드로 두어 공정 비교.
실행: hyperbolic_model 디렉터리에서
  python train_euclidean_content_baseline.py --load_dataset results/checkpoints/datasets
결과: recbole_baselines/results/recbole_ContentLightGCN.json (--results_dir로 변경 가능)

Train loss는 줄어드는데 Val HR@10·추천 다양성이 일정한 경우:
  BPR이 '인기 블렌더' 위주로 학습되어 동일 top-10만 반복될 수 있음.
  → variance_reg, blender_norm_reg, rank 시 blender L2 정규화로 완화 시도.

에폭마다 성능(Val HR@10 등)이 거의 안 변하고 일정한 숫자로 나오면:
  z_recipe 붕괴(모든 레시피가 비슷한 임베딩 → 항상 같은 top-K) 가능성.
  → --variance_reg 0.2~0.3, --lr 2e-4, --diagnose 로 확인. 프리컴퓨팅 사용 시 --no_precomputed 로 GNN end-to-end 시도.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 프로젝트 루트
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model.euclidean_content_lightgcn import ContentLightGCN
from model.hyperbolic_data_loader import HyperbolicRecipeDataset, collate_hyperbolic_recipes
from model.hyperbolic_losses import HyperbolicBPRLoss

# train_hyperbolic_hypergraph에서 데이터 로딩·평가 재사용
from train_hyperbolic_hypergraph import (
    load_data,
    evaluate_model,
    train_epoch,
    _target_blenders_to_positive_set_tensor,
    set_all_seeds,
    clear_device_memory,
)
from tqdm import tqdm


def _stage_variance_stats(t: torch.Tensor, stage_name: str) -> str:
    """단일 단계 텐서의 분산 요약 (입력→z_recipe 파이프라인 점검용)."""
    if t is None or t.numel() == 0:
        return f"    {stage_name}: (없음)"
    x = t.detach().float()
    if x.dim() == 3:
        x = x.reshape(-1, x.size(-1))  # (B,M,D) -> (B*M, D)
    n = x.size(0)
    per_dim_std = x.std(dim=0)
    mean_std = per_dim_std.mean().item()
    sample_size = min(200, max(2, n))
    idx = torch.randperm(n, device=x.device)[:sample_size]
    sub = x[idx]
    dists = torch.cdist(sub, sub, p=2)
    triu = dists.triu(diagonal=1)
    mean_pair = triu[triu > 0].mean().item() if triu.numel() > 0 and triu[triu > 0].numel() > 0 else 0.0
    return f"    {stage_name}: 차원당 std 평균={mean_std:.4f}, 쌍별 L2 평균={mean_pair:.4f} (n={n})"


def main():
    p = argparse.ArgumentParser(description="Content LightGCN (분자 GNN + 노트, 유클리드)")
    _default_dataset = str(SCRIPT_DIR / "results" / "checkpoints" / "datasets")
    p.add_argument("--load_dataset", type=str, default=_default_dataset,
                   help=f"train/val/test_combinations.json 이 있는 디렉터리 (기본: {_default_dataset})")
    p.add_argument("--data_path", type=str, default=None,
                   help="cleaned_complete_data.json (vocab/레코드용, 기본: 프로젝트 cleaned_data)")
    p.add_argument("--vocab_path", type=str, default=None,
                   help="vocabularies.json (기본: 프로젝트 feature_encoding)")
    p.add_argument("--results_dir", type=str, default=None,
                   help="결과 JSON 저장 경로 (기본: recbole_baselines/results)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4,
                   help="학습률 (2e-4=권장·붕괴 시 1e-4, 1e-3은 붕괴 유발)")
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_interval", type=int, default=1)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--model_name", type=str, default="ContentLightGCN", choices=["ContentLightGCN", "ContentBPR"],
                   help="출력용 모델 이름 (ContentBPR = 분자 GNN+노트 BPR, 동일 구조·손실)")
    p.add_argument("--precomputed_mol_embs", type=str, default=None,
                   help="프리컴퓨팅된 분자 임베딩 .pt 경로 (기본: 미사용, GNN end-to-end 학습). 지정 시 해당 .pt 사용")
    p.add_argument("--no_precomputed", action="store_true",
                   help="프리컴퓨팅 비활성화 → GNN end-to-end 학습 (성능 고정 시 원인 비교용)")
    p.add_argument("--diagnose", action="store_true",
                   help="z_recipe 붕괴·target_tensor 다양성 진단 로그 (첫 배치 1회 + 매 에폭 첫 배치)")
    p.add_argument("--variance_reg", type=float, default=0.15,
                   help="z_recipe 붕괴 완화: 배치 내 쌍별 L2 거리 최대화 (Val mean_pairwise_dist≈0이면 0.2~0.3으로 올리기)")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="검증/테스트 시 랭킹 점수 스케일 (낮으면 상위 몇 개만 선택됨, 기본 0.1, 0.07=더 극단적)")
    p.add_argument("--blender_norm_reg", type=float, default=0.01,
                   help="블렌더 임베딩 노름 분산 패널티 (소수 블렌더가 노름으로 상위 독점 방지, 0=끔)")
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

    num_blenders = vocab_len("blenders", 100)
    vocab_size = vocab_len("notes", 435)
    num_blenders = max(num_blenders, 1)
    vocab_size = max(vocab_size, 1)

    precomputed_path = None if getattr(args, "no_precomputed", False) else args.precomputed_mol_embs
    if precomputed_path and Path(precomputed_path).exists():
        print(f"프리컴퓨팅 분자 임베딩 사용: {precomputed_path} (GNN forward 생략)")
    elif getattr(args, "no_precomputed", False):
        print("프리컴퓨팅 비활성화 → GNN end-to-end 학습 (분자 임베딩이 학습됨)")
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
        shuffle=True,  # 표현력 붕괴 방지: 매 에폭 순서 셔플
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
    # 프리컴퓨팅 사용 시 GNN 파인튜닝 비활성화 (--no_precomputed 시 경로 무시)
    use_precomputed = bool(precomputed_path) and Path(precomputed_path).exists()
    if use_precomputed and hasattr(model, "smiles_encoder"):
        for p in model.smiles_encoder.parameters():
            p.requires_grad = False
    trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(trainable if use_precomputed else model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = HyperbolicBPRLoss(manifold=model.manifold).to(device)

    # (1) 프리컴퓨팅 사용 시: 배치에 precomputed_mol_embs가 들어오는지 검증 (GNN 미사용 확실히)
    if use_precomputed:
        _check_loader = DataLoader(train_dataset, batch_size=min(4, len(train_dataset)), shuffle=False, collate_fn=collate_hyperbolic_recipes, num_workers=0)
        _check_batch = next(iter(_check_loader))
        if "precomputed_mol_embs" not in _check_batch:
            raise RuntimeError(
                "precomputed_mol_embs .pt 파일은 있으나 배치에 precomputed_mol_embs가 없습니다. "
                "데이터셋 생성 시 precomputed_path가 올바르게 전달되었는지 확인하세요."
            )
        print("  ✓ 배치에 precomputed_mol_embs 포함 확인 → forward 시 GNN 스킵, 고정 임베딩 사용")

    best_hr10 = 0.0
    best_state = None
    patience = 0
    print("  💡 에폭마다 Val HR@10이 일정하면 z_recipe 붕괴 가능 → --diagnose, --variance_reg 0.05~0.1, --no_precomputed 시도")
    if getattr(args, "diagnose", False):
        print("  [진단 모드] DataLoader shuffle=True | 매 에폭 첫 배치: z_recipe 거리·std·target_tensor 다양성 출력")

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
            if getattr(smiles_graphs, "batch", None) is None and hasattr(smiles_graphs, "ptr"):
                pass
            kw = dict(
                smiles_graphs=smiles_graphs,
                smiles_batch=smiles_batch,
                note_indices=note_indices,
                molecule_mask=molecule_mask,
            )
            if "precomputed_mol_embs" in batch:
                kw["precomputed_mol_embs"] = batch["precomputed_mol_embs"].to(device)
            # 진단: 입력→z_recipe 파이프라인 단계별 분산 (첫 배치 1회만, GNN 임베딩은 정상인데 z_recipe 붕괴 시 원인 확인)
            if getattr(args, "diagnose", False) and epoch == 0 and n_b == 0 and "precomputed_mol_embs" in batch:
                kw_debug = {**kw, "return_debug": True}
                z_recipe, debug = model(**kw_debug)
                print("[진단] GNN 임베딩 → z_recipe 파이프라인 (첫 배치 단계별 분산)")
                print(_stage_variance_stats(debug["mol_embs"], "1. mol_embs (입력)"))
                print(_stage_variance_stats(debug["note_embs"], "2. note_embs"))
                print(_stage_variance_stats(debug["fused"], "3. fused (mol+note 퓨전 후)"))
                print(_stage_variance_stats(debug["z_before_ln"], "4. z_before_ln (레이어놈 전)"))
                print(_stage_variance_stats(z_recipe, "5. z_recipe (최종)"))
                target_tensor = _target_blenders_to_positive_set_tensor(batch["target_blenders"], device)
                all_blender_embs = model.blender_anchors()
                B = z_recipe.size(0)
                pos_sample = target_tensor.cpu().numpy()
                uniq = set(pos_sample[pos_sample >= 0].tolist())
                print(f"       target_tensor | 배치 내 서로 다른 정답 blender 수: {len(uniq)}")
            else:
                if getattr(args, "diagnose", False) and epoch == 0 and n_b == 0:
                    z_recipe, _ = model(**{**kw, "return_debug": True})
                else:
                    z_recipe = model(**kw)
                target_tensor = _target_blenders_to_positive_set_tensor(batch["target_blenders"], device)
                all_blender_embs = model.blender_anchors()
                # 진단: z_recipe 요약 (매 에폭 첫 배치)
                if getattr(args, "diagnose", False):
                    B = z_recipe.size(0)
                    if B >= 2:
                        d01 = (z_recipe[0] - z_recipe[1]).norm(p=2).item()
                        z_std = z_recipe.std(dim=0).mean().item()
                        z_norm_mean = z_recipe.norm(dim=1).mean().item()
                    else:
                        d01, z_std, z_norm_mean = 0.0, 0.0, z_recipe.norm(dim=1).mean().item()
                    if epoch == 0 and n_b == 0 and "precomputed_mol_embs" not in batch:
                        pos_sample = target_tensor.cpu().numpy()
                        uniq = set(pos_sample[pos_sample >= 0].tolist())
                        print(f"[진단] Epoch 1 첫 배치 | Recipe 거리: {d01:.4f} | z_recipe std: {z_std:.6f} | z L2: {z_norm_mean:.4f} | 정답 blender 수: {len(uniq)}")
                    elif n_b == 0:
                        print(f"[진단] Epoch {epoch+1} 첫 배치 | Recipe 거리: {d01:.4f} | z std: {z_std:.6f} | z L2: {z_norm_mean:.4f}")
            loss = loss_fn(z_recipe, target_tensor, all_blender_embs)
            # 분산 정규화: z_recipe가 한 점으로 모이지 않도록 배치 내 쌍별 L2 거리 최대화
            variance_reg = getattr(args, "variance_reg", 0.0)
            if variance_reg > 0 and z_recipe.size(0) >= 2:
                B = z_recipe.size(0)
                sample = min(64, B)
                idx = torch.randperm(B, device=z_recipe.device)[:sample]
                zs = z_recipe[idx]
                dists = torch.cdist(zs, zs, p=2)
                triu = dists.triu(diagonal=1)
                mean_dist = triu[triu > 0].mean()
                loss = loss - variance_reg * mean_dist  # 거리 최대화 = loss에 음수 부여
            # 블렌더 노름 분산 패널티: 소수 블렌더가 L2 노름으로 상위 독점하는 것 완화
            blender_norm_reg = getattr(args, "blender_norm_reg", 0.0)
            if blender_norm_reg > 0 and hasattr(model, "blender_emb"):
                w = model.blender_emb.weight
                norms = w.norm(dim=1)
                loss = loss + blender_norm_reg * norms.var()
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
                loss_fn=loss_fn,
                k=10,
                use_blender_input=False,
                temperature=args.temperature,
                return_z_recipe_stats=True,
            )
            hr10 = val_metrics.get("hit_rate@k", val_metrics.get("hit_rate@10", 0.0))
            uniq_rec = val_metrics.get("unique_recommended_blenders", 0)
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_state = {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            # Val z_recipe 붕괴 진단: mean_pairwise_dist ≈ 0 이면 모든 레시피가 같은 벡터 → 동일 top-10 → HR@10 고정
            mpd = val_metrics.get("z_recipe_mean_pairwise_dist")
            zstd = val_metrics.get("z_recipe_std_norm")
            if mpd is not None:
                print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val HR@10: {hr10:.6f} | Best: {best_hr10:.6f} | Val 추천 다양성(서로 다른 블렌더 수): {uniq_rec} | Val z_recipe mean_pairwise_dist: {mpd:.4f} | z_recipe std_norm: {zstd:.4f}")
                if mpd < 0.02 and uniq_rec == 10:
                    print("  ⚠️ Val z_recipe가 거의 동일(mean_pairwise_dist≈0) → 모든 레시피에 같은 top-10만 나와 HR@10 고정. --variance_reg 0.2~0.3 또는 --no_precomputed 시도.")
            else:
                print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val HR@10: {hr10:.6f} | Best: {best_hr10:.6f} | Val 추천 다양성(서로 다른 블렌더 수): {uniq_rec}")
            if epoch == 0 and uniq_rec <= 20 and uniq_rec > 0:
                print("  → 추천이 소수 블렌더로 고정된 상태일 수 있음.")
                if uniq_rec == 10:
                    if use_precomputed:
                        print("  → 모든 레시피에 동일 top-10만 추천 중. 아래로 GNN end-to-end 학습:")
                        print("     python train_euclidean_content_baseline.py --no_precomputed")
                    else:
                        print("  → GNN end-to-end인데도 동일 top-10: z_recipe 붕괴 또는 blender_emb 편향 가능.")
                        print("     --diagnose 로 z_recipe 거리 확인. rank_blenders에서 blender L2 정규화 사용 중이면 다양성 개선 기대.")
            if patience >= args.early_stopping_patience:
                print(f"Early stopping (patience {args.early_stopping_patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(
        model=model,
        val_loader=test_loader,
        device=device,
        loss_fn=loss_fn,
        k=10,
        use_blender_input=False,
        temperature=args.temperature,
    )

    results_dir = Path(args.results_dir) if args.results_dir else (SCRIPT_DIR.parent / "recbole_baselines" / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "model": args.model_name,
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
    out_path = results_dir / f"recbole_{args.model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
