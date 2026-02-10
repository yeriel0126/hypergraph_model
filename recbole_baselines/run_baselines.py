#!/usr/bin/env python3
"""
베이스라인 실행: 기본은 분자 GNN 입력(Content*)만 실행 — ContentLightGCN, ContentBPR, ContentCML, ContentHGCF.
공정 비교는 Content*만 해당 (동일 입력: 분자+노트). ID 기반(LightGCN, BPR, HGCF, CML)은 분자 입력 없음 → 비교실험이 아님.
하이퍼볼릭과 공정 비교를 위해 같은 데이터(--dataset_dir) 사용.

[데이터셋 맞추기 — 필수]
  - 기본값 --dataset_dir = ../hyperbolic_model/results/checkpoints/datasets (하이퍼볼릭이 저장하는 경로).
  - 구비된 데이터가 이미 다른 경로에 있으면 반드시 그 경로를 지정해야 함. 지정 안 하면 위 기본 경로만 사용 →
    그 경로에 데이터 없으면 에러. (다른 데이터셋을 쓰면 공정 비교가 깨짐.)
  예: python run_baselines.py
      python run_baselines.py --dataset_dir /path/to/train_combinations.json이_있는_폴더
  ※ --recbole_data_path / --use_unified_id 는 하이퍼볼릭과 다른 데이터일 수 있어 비교 시 비권장.
"""
from __future__ import annotations

import os
# OpenMP 중복 초기화 방지 (macOS, RecBole/numpy 등) — 다른 import 전에 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import subprocess
import sys
from pathlib import Path

# NumPy 2.0 호환: RecBole 등이 제거된 타입 사용 시 매핑
import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

# PyTorch 2.6+: RecBole 체크포인트 로드 시 weights_only=True 오류 방지
import torch
_orig_torch_load = torch.load
def _torch_load_safe(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_safe

# scipy 1.14+: dok_matrix._update 제거로 RecBole/LightGCN 오류 방지 (소스 빌드 없이)
try:
    import scipy.sparse as _sp
    if not hasattr(_sp.dok_matrix, "_update"):
        def _dok_update(self, other):
            if hasattr(other, "keys"):
                for k in other:
                    self[k] = other[k]
            else:
                for k, v in other:
                    self[k] = v
        _sp.dok_matrix._update = _dok_update
except Exception:
    pass


def _ensure_odor_inter_exists(out: Path, dataset_name: str) -> None:
    """RecBole가 요구하는 {name}.inter가 없으면 part1+part2+part3 병합해서 생성."""
    main_inter = out / f"{dataset_name}.inter"
    if main_inter.exists():
        return
    p1, p2, p3 = out / f"{dataset_name}.part1.inter", out / f"{dataset_name}.part2.inter", out / f"{dataset_name}.part3.inter"
    if not (p1.exists() and p2.exists() and p3.exists()):
        return
    lines = []
    for p in (p1, p2, p3):
        with open(p, "r", encoding="utf-8") as f:
            part_lines = f.readlines()
        if part_lines and not lines:
            lines.append(part_lines[0])  # header
        lines.extend(part_lines[1:] if len(part_lines) > 1 else [])
    with open(main_inter, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"RecBole 메인 파일 생성: {main_inter} (part1+part2+part3 병합)")


def run_prepare_if_needed(dataset_dir: Path, dataset_name: str, out_dir: Path | None) -> tuple[Path, Path]:
    """필요 시 prepare_recbole_data.py 실행 후 RecBole 데이터 경로 반환."""
    script_dir = Path(__file__).resolve().parent
    out = out_dir or (dataset_dir / "recbole_data" / dataset_name)
    inter_file = out / f"{dataset_name}.part1.inter"
    if inter_file.exists():
        print(f"  → RecBole 형식 변환 생략 (위 데이터셋에서 이미 생성됨): {out}")
        _ensure_odor_inter_exists(out, dataset_name)
        return out.parent, out
    prepare = script_dir / "prepare_recbole_data.py"
    if not prepare.exists():
        raise FileNotFoundError(f"스크립트 없음: {prepare}")
    cmd = [
        sys.executable,
        str(prepare),
        "--dataset_dir",
        str(dataset_dir),
        "--dataset_name",
        dataset_name,
    ]
    if out_dir:
        cmd += ["--out_dir", str(out_dir)]
    subprocess.run(cmd, check=True)
    return out.parent, out


def run_recbole_model(
    model_name: str,
    data_path: Path,
    dataset_name: str,
    config_path: Path,
    results_dir: Path,
    seed: int = 2024,
) -> dict:
    """RecBole 모델 한 개 학습·평가 후 지표 반환."""
    try:
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.evaluator import Evaluator
        from recbole.trainer import Trainer
        from recbole.utils import init_seed, init_logger
    except ImportError as e:
        raise ImportError(
            "RecBole이 설치되어 있지 않습니다. 다음으로 설치하세요: pip install recbole"
        ) from e

    config_dict = {
        "model": model_name,
        "dataset": dataset_name,
        "data_path": str(data_path),
        "seed": seed,
        # RecBole가 recall@1, recall@5를 계산·반환하도록 topk 명시 (기본값 10만 쓰는 버전 대비)
        "topk": [1, 5, 10],
    }
    if config_path.exists():
        config_dict["config_file_list"] = [str(config_path)]
    config = Config(model=model_name, dataset=dataset_name, config_dict=config_dict)

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model_class = _get_model_class(model_name)
    model = model_class(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)

    trainer.fit(train_data, valid_data, test_data, show_progress=config["show_progress"])
    result = trainer.evaluate(test_data, load_best_model=True, show_progress=config["show_progress"])

    # Hit Rate(HR)와 Recall 둘 다 파싱 — 하이퍼볼릭과 동일하게 HR/Recall 모두 출력
    # HR = binary (해당 샘플에 정답 1개라도 top-K에 있으면 1), Recall = (top-K 내 정답 수)/(해당 샘플 정답 수) 평균
    def _get_metric(res: dict, *keys: str, default: float = 0.0) -> float:
        for k in keys:
            v = res.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return default

    # Hit Rate: RecBole Hit@K (binary hit per user)
    hr1 = _get_metric(result, "hit@1", "Hit@1")
    hr5 = _get_metric(result, "hit@5", "Hit@5")
    hr10 = _get_metric(result, "hit@10", "Hit@10")
    if hr1 == 0.0 and hr5 == 0.0 and hr10 == 0.0:
        hr1 = _get_metric(result, "recall@1", "Recall@1")
        hr5 = _get_metric(result, "recall@5", "Recall@5")
        hr10 = _get_metric(result, "recall@10", "Recall@10")
        if hr1 == 0.0 and hr5 == 0.0 and hr10 > 0.0:
            print(f"  ⚠️ RecBole에 hit@1/5/10 없음 (반환 키: {list(result.keys())}). recall로 대체.")

    # Recall: RecBole Recall@K (비율)
    rec1 = _get_metric(result, "recall@1", "Recall@1")
    rec5 = _get_metric(result, "recall@5", "Recall@5")
    rec10 = _get_metric(result, "recall@10", "Recall@10")

    mrr = _get_metric(result, "mrr@10", "MRR@10")
    ndcg10 = _get_metric(result, "ndcg@10", "NDCG@10")

    out = {
        "model": model_name,
        "test_result": result,
        "hit_rate@1": hr1,
        "hit_rate@5": hr5,
        "hit_rate@k": hr10,
        "hit_rate@10": hr10,
        "recall@1": rec1,
        "recall@5": rec5,
        "recall@k": rec10,
        "recall@10": rec10,
        "mrr": mrr,
        "ndcg@k": ndcg10,
        "ndcg@10": ndcg10,
    }
    result_file = results_dir / f"recbole_{model_name}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {result_file}")
    return out


def _get_model_class(model_name: str):
    """RecBole에 있는 모델만 반환. HGCF/CML/ContentLightGCN은 run_baselines 루프에서 별도 실행."""
    if model_name in ("HGCF", "CML", "ContentLightGCN", "ContentBPR", "ContentCML", "ContentHGCF"):
        return None  # HGCF/CML/Content* → 별도 스크립트
    from recbole.model.general_recommender import LightGCN, BPR
    if model_name == "LightGCN":
        return LightGCN
    if model_name == "BPR":
        return BPR
    raise ValueError(f"지원하지 않는 모델: {model_name}. 현재: LightGCN, BPR, HGCF, CML, ContentLightGCN, ContentBPR, ContentCML, ContentHGCF")


def _content_result_from_json(content_out: dict, model_name: str) -> dict:
    """Content* 학습 스크립트 JSON에서 지표 추출. top-level 우선, 없으면 test_result에서 fallback (분자 입력/프리컴퓨팅 수정 후에도 동일 키 유지)."""
    tr = content_out.get("test_result") or content_out
    def _v(key: str):
        return content_out.get(key) if content_out.get(key) is not None else tr.get(key)
    return {
        "model": model_name,
        "test_result": content_out.get("test_result", content_out),
        "hit_rate@1": _v("hit_rate@1"),
        "hit_rate@5": _v("hit_rate@5"),
        "hit_rate@k": _v("hit_rate@k"),
        "hit_rate@10": _v("hit_rate@10"),
        "recall@1": _v("recall@1"),
        "recall@5": _v("recall@5"),
        "recall@k": _v("recall@k"),
        "recall@10": _v("recall@10"),
        "mrr": _v("mrr"),
        "ndcg@k": _v("ndcg@k"),
        "ndcg@10": _v("ndcg@10"),
    }


def main():
    script_dir = Path(__file__).resolve().parent
    # 공정 비교: 베이스라인은 반드시 하이퍼볼릭과 동일한 데이터만 사용. 4만 개(43,111) datasets 우선.
    _checkpoints = (script_dir / ".." / "hyperbolic_model" / "results" / "checkpoints").resolve()
    _datasets = _checkpoints / "datasets"
    _datasets_refined = _checkpoints / "datasets_refined"
    _datasets_v2 = _checkpoints / "datasets_v2"
    _default_dataset_dir = str(_datasets)
    parser = argparse.ArgumentParser(
        description="베이스라인 실행: LightGCN, BPR, HGCF, CML, ContentLightGCN, ContentBPR, ContentCML, ContentHGCF. 공정 비교를 위해 반드시 하이퍼볼릭과 동일한 데이터(--dataset_dir) 사용."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=_default_dataset_dir,
        help="모든 모델 공통 데이터셋 경로 (기본: hyperbolic_model/results/checkpoints/datasets).",
    )
    parser.add_argument(
        "--recbole_data_path",
        type=str,
        default=None,
        help="[비교 시 비권장] RecBole .inter 상위 경로. 지정 시 하이퍼볼릭과 다른 데이터일 수 있음.",
    )
    parser.add_argument(
        "--use_unified_id",
        action="store_true",
        help="[비교 시 비권장] 통일 ID 데이터 사용. 지정 시 하이퍼볼릭과 다른 데이터임.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="odor",
        help="RecBole 데이터셋 이름",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["ContentLightGCN", "ContentBPR", "ContentCML", "ContentHGCF"],
        help="실행할 모델. 기본=분자 GNN 입력(Content*)만 비교대상. ID 기반 LightGCN/BPR/HGCF/CML은 필요 시 --models로 지정.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="결과 JSON 저장 디렉터리 (기본: recbole_baselines/results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (하이퍼볼릭과 동일, 기본: 42)",
    )
    parser.add_argument(
        "--precomputed_mol_embs",
        type=str,
        default=None,
        help="Content* 모델용: 프리컴퓨팅 .pt 경로 (기본: 미사용, GNN end-to-end). 지정 시에만 전달",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Content* 학습 시 z_recipe 붕괴·target_tensor 진단 로그 출력 (Val HR@10 고정 시 원인 확인용)",
    )
    parser.add_argument(
        "--no_precomputed",
        action="store_true",
        help="Content*: 프리컴퓨팅 비활성화, GNN end-to-end 학습 (성능 고정 시 프리컴퓨팅 원인 비교용)",
    )
    args = parser.parse_args()

    config_path = script_dir / "config" / "odor.yaml"
    effective_dataset_dir = None  # CML(run_cml_euclidean) 실행 시 필요

    if args.use_unified_id:
        data_path = script_dir / "data" / "recbole"
        if not (data_path / args.dataset_name / f"{args.dataset_name}.part1.inter").exists():
            print("ERROR: 통일 ID 데이터가 없습니다. 먼저 build_id_mapping_and_edges.py 를 실행하세요.")
            sys.exit(1)
        print("통일 ID 매핑 데이터 사용 (하이퍼볼릭과 다른 데이터일 수 있음):", data_path / args.dataset_name)
    elif args.recbole_data_path:
        data_path = Path(args.recbole_data_path)
        if not (data_path / args.dataset_name).exists():
            print(f"ERROR: {data_path / args.dataset_name} 디렉터리가 없습니다.")
            sys.exit(1)
        print("주의: --recbole_data_path 사용 중 → 하이퍼볼릭과 다른 데이터일 수 있음. 공정 비교 시 --dataset_dir 사용.")
    else:
        # 모든 모델이 사용하는 데이터셋: hyperbolic_model/results/checkpoints/datasets 고정
        dataset_dir = Path(args.dataset_dir).resolve()
        if not (dataset_dir / "train_combinations.json").exists() and (dataset_dir / "datasets" / "train_combinations.json").exists():
            dataset_dir = (dataset_dir / "datasets").resolve()
        if not (dataset_dir / "train_combinations.json").exists():
            print("ERROR: 공정 비교를 위해 하이퍼볼릭과 동일한 데이터(train/val/test_combinations.json)를 사용해야 합니다.")
            print(f"  현재 경로에 해당 파일이 없습니다: {dataset_dir}")
            print("  구비된 데이터가 다른 폴더에 있으면 반드시 --dataset_dir <그 폴더 경로> 로 지정한 뒤 실행하세요.")
            print(f"  예: python run_baselines.py --dataset_dir {_datasets}")
            sys.exit(1)
        effective_dataset_dir = dataset_dir
        # 사용 중인 데이터셋을 한 줄로 명확히 표시 (경로 + 스플릿 크기)
        train_s, val_s, test_s, seed_s = None, None, None, None
        try:
            with open(dataset_dir / "split_info.json", "r", encoding="utf-8") as f:
                si = json.load(f)
            train_s = si.get("train_size")
            val_s = si.get("val_size")
            test_s = si.get("test_original_size") or si.get("test_size")
            seed_s = si.get("random_seed")
        except Exception:
            pass
        if train_s is not None:
            print(f"[데이터셋] {dataset_dir}")
            print(f"  → Train={train_s:,}, Val={val_s:,}, Test={test_s:,}, seed={seed_s} (split_info.json 기준)")
        else:
            print(f"[데이터셋] {dataset_dir} (split_info.json 없음)")
        data_path, _ = run_prepare_if_needed(
            dataset_dir, args.dataset_name, Path(args.recbole_data_path) / args.dataset_name if args.recbole_data_path else None
        )

    results_dir = Path(args.results_dir) if args.results_dir else (script_dir / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    models_list = args.models
    print(f"\n실행할 모델 ({len(models_list)}개, 순서대로): {', '.join(models_list)}")
    print(f"결과 저장 경로: {results_dir}\n")

    # HGCF 곡률(c) 설정 (config/hgcf_curvature.yaml 또는 기본 0.4)
    hgcf_curvature = 0.4
    import re
    hgcf_yaml = script_dir / "config" / "hgcf_curvature.yaml"
    if hgcf_yaml.exists():
        try:
            raw = hgcf_yaml.read_text(encoding="utf-8")
            m = re.search(r"curvature:\s*([\d.]+)", raw)
            if m:
                hgcf_curvature = float(m.group(1))
        except Exception:
            pass

    _ID_ONLY_MODELS = ("LightGCN", "BPR", "HGCF", "CML")  # 분자/노트 없이 user_id·item_id만 사용 → 공정 비교 아님
    all_results = []
    for model_name in args.models:
        print(f"\n===== {model_name} =====")
        if model_name in _ID_ONLY_MODELS:
            print("  ⚠️  [비교실험 아님] 이 모델은 분자·노트 입력을 쓰지 않습니다 (user_id/item_id만 사용). 공정 비교는 Content* 모델만 해당합니다.")
        try:
            if model_name == "HGCF":
                from run_official_hgcf import run_official_hgcf
                recbole_inter_dir = data_path / args.dataset_name
                res = run_official_hgcf(
                    data_path=data_path,
                    dataset_name=args.dataset_name,
                    results_dir=results_dir,
                    seed=args.seed,
                    curvature=hgcf_curvature,
                    recbole_inter_dir=recbole_inter_dir,
                )
            elif model_name == "CML":
                # RecBole에 CML 없음 → run_cml_euclidean.py (동일 데이터·동일 지표)
                if effective_dataset_dir is None:
                    raise RuntimeError("CML 실행에는 --dataset_dir 필요 (train/val/test_combinations.json 경로)")
                cml_script = script_dir / "run_cml_euclidean.py"
                if not cml_script.exists():
                    raise FileNotFoundError(f"CML 스크립트 없음: {cml_script}")
                subprocess.run(
                    [
                        sys.executable,
                        str(cml_script),
                        "--dataset_dir",
                        str(effective_dataset_dir),
                        "--results_dir",
                        str(results_dir),
                        "--seed",
                        str(args.seed),
                    ],
                    cwd=str(script_dir),
                    check=True,
                )
                cml_json = results_dir / "recbole_CML_Euclidean.json"
                if not cml_json.exists():
                    raise FileNotFoundError(f"CML 결과 없음: {cml_json}")
                with open(cml_json, "r", encoding="utf-8") as f:
                    cml_out = json.load(f)
                res = {
                    "model": "CML",
                    "test_result": cml_out,
                    "hit_rate@1": cml_out.get("hit_rate@1"),
                    "hit_rate@5": cml_out.get("hit_rate@5"),
                    "hit_rate@k": cml_out.get("hit_rate@k"),
                    "hit_rate@10": cml_out.get("hit_rate@10"),
                    "recall@1": cml_out.get("recall@1"),
                    "recall@5": cml_out.get("recall@5"),
                    "recall@k": cml_out.get("recall@k"),
                    "recall@10": cml_out.get("recall@10"),
                    "mrr": cml_out.get("mrr"),
                    "ndcg@k": cml_out.get("ndcg@k"),
                    "ndcg@10": cml_out.get("ndcg@10"),
                }
                # 동일 키로 recbole_CML.json도 저장 (요약과 일치)
                result_file = results_dir / "recbole_CML.json"
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)
                print(f"결과 저장: {result_file}")
            elif model_name == "ContentLightGCN":
                # 분자 GNN + 노트 유클리드 베이스라인 (train_euclidean_content_baseline.py)
                if effective_dataset_dir is None:
                    raise RuntimeError("ContentLightGCN 실행에는 --dataset_dir 필요 (train/val/test_combinations.json 경로)")
                content_script = script_dir.parent / "hyperbolic_model" / "train_euclidean_content_baseline.py"
                if not content_script.exists():
                    raise FileNotFoundError(f"ContentLightGCN 스크립트 없음: {content_script}")
                content_cmd = [
                    sys.executable,
                    str(content_script),
                    "--load_dataset",
                    str(effective_dataset_dir),
                    "--results_dir",
                    str(results_dir),
                    "--seed",
                    str(args.seed),
                ]
                if getattr(args, "precomputed_mol_embs", None):
                    content_cmd += ["--precomputed_mol_embs", str(args.precomputed_mol_embs)]
                if getattr(args, "diagnose", False):
                    content_cmd += ["--diagnose"]
                if getattr(args, "no_precomputed", False):
                    content_cmd += ["--no_precomputed"]
                subprocess.run(content_cmd, cwd=str(content_script.parent), check=True)
                content_json = results_dir / "recbole_ContentLightGCN.json"
                if not content_json.exists():
                    raise FileNotFoundError(f"ContentLightGCN 결과 없음: {content_json}")
                with open(content_json, "r", encoding="utf-8") as f:
                    content_out = json.load(f)
                res = _content_result_from_json(content_out, "ContentLightGCN")
            elif model_name == "ContentBPR":
                # 분자 GNN + 노트, BPR 손실 (동일 스크립트 --model_name ContentBPR)
                if effective_dataset_dir is None:
                    raise RuntimeError("ContentBPR 실행에는 --dataset_dir 필요 (train/val/test_combinations.json 경로)")
                content_bpr_script = script_dir.parent / "hyperbolic_model" / "train_euclidean_content_baseline.py"
                if not content_bpr_script.exists():
                    raise FileNotFoundError(f"ContentBPR 스크립트 없음: {content_bpr_script}")
                content_bpr_cmd = [
                    sys.executable,
                    str(content_bpr_script),
                    "--load_dataset",
                    str(effective_dataset_dir),
                    "--results_dir",
                    str(results_dir),
                    "--seed",
                    str(args.seed),
                    "--model_name",
                    "ContentBPR",
                ]
                if getattr(args, "precomputed_mol_embs", None):
                    content_bpr_cmd += ["--precomputed_mol_embs", str(args.precomputed_mol_embs)]
                if getattr(args, "diagnose", False):
                    content_bpr_cmd += ["--diagnose"]
                if getattr(args, "no_precomputed", False):
                    content_bpr_cmd += ["--no_precomputed"]
                subprocess.run(content_bpr_cmd, cwd=str(content_bpr_script.parent), check=True)
                content_bpr_json = results_dir / "recbole_ContentBPR.json"
                if not content_bpr_json.exists():
                    raise FileNotFoundError(f"ContentBPR 결과 없음: {content_bpr_json}")
                with open(content_bpr_json, "r", encoding="utf-8") as f:
                    content_bpr_out = json.load(f)
                res = _content_result_from_json(content_bpr_out, "ContentBPR")
            elif model_name == "ContentCML":
                # 분자 GNN + 노트, CML margin loss (train_euclidean_content_cml.py)
                if effective_dataset_dir is None:
                    raise RuntimeError("ContentCML 실행에는 --dataset_dir 필요 (train/val/test_combinations.json 경로)")
                content_cml_script = script_dir.parent / "hyperbolic_model" / "train_euclidean_content_cml.py"
                if not content_cml_script.exists():
                    raise FileNotFoundError(f"ContentCML 스크립트 없음: {content_cml_script}")
                content_cml_cmd = [
                    sys.executable,
                    str(content_cml_script),
                    "--load_dataset",
                    str(effective_dataset_dir),
                    "--results_dir",
                    str(results_dir),
                    "--seed",
                    str(args.seed),
                ]
                if getattr(args, "precomputed_mol_embs", None):
                    content_cml_cmd += ["--precomputed_mol_embs", str(args.precomputed_mol_embs)]
                if getattr(args, "diagnose", False):
                    content_cml_cmd += ["--diagnose"]
                if getattr(args, "no_precomputed", False):
                    content_cml_cmd += ["--no_precomputed"]
                subprocess.run(content_cml_cmd, cwd=str(content_cml_script.parent), check=True)
                content_cml_json = results_dir / "recbole_ContentCML.json"
                if not content_cml_json.exists():
                    raise FileNotFoundError(f"ContentCML 결과 없음: {content_cml_json}")
                with open(content_cml_json, "r", encoding="utf-8") as f:
                    content_cml_out = json.load(f)
                res = _content_result_from_json(content_cml_out, "ContentCML")
            elif model_name == "ContentHGCF":
                # 분자 GNN + 노트, 포앵카레 HGCF 스타일 (train_hyperbolic_content_hgcf.py)
                if effective_dataset_dir is None:
                    raise RuntimeError("ContentHGCF 실행에는 --dataset_dir 필요 (train/val/test_combinations.json 경로)")
                content_hgcf_script = script_dir.parent / "hyperbolic_model" / "train_hyperbolic_content_hgcf.py"
                if not content_hgcf_script.exists():
                    raise FileNotFoundError(f"ContentHGCF 스크립트 없음: {content_hgcf_script}")
                content_hgcf_cmd = [
                    sys.executable,
                    str(content_hgcf_script),
                    "--load_dataset",
                    str(effective_dataset_dir),
                    "--results_dir",
                    str(results_dir),
                    "--seed",
                    str(args.seed),
                ]
                if getattr(args, "precomputed_mol_embs", None):
                    content_hgcf_cmd += ["--precomputed_mol_embs", str(args.precomputed_mol_embs)]
                if getattr(args, "diagnose", False):
                    content_hgcf_cmd += ["--diagnose"]
                if getattr(args, "no_precomputed", False):
                    content_hgcf_cmd += ["--no_precomputed"]
                subprocess.run(content_hgcf_cmd, cwd=str(content_hgcf_script.parent), check=True)
                content_hgcf_json = results_dir / "recbole_ContentHGCF.json"
                if not content_hgcf_json.exists():
                    raise FileNotFoundError(f"ContentHGCF 결과 없음: {content_hgcf_json}")
                with open(content_hgcf_json, "r", encoding="utf-8") as f:
                    content_hgcf_out = json.load(f)
                res = _content_result_from_json(content_hgcf_out, "ContentHGCF")
            else:
                res = run_recbole_model(
                    model_name=model_name,
                    data_path=data_path,
                    dataset_name=args.dataset_name,
                    config_path=config_path,
                    results_dir=results_dir,
                    seed=args.seed,
                )
            all_results.append(res)
        except Exception as e:
            print(f"{model_name} 실행 실패: {e}")
            all_results.append({
                "model": model_name,
                "error": str(e),
                "hit_rate@1": None,
                "hit_rate@5": None,
                "hit_rate@k": None,
                "recall@1": None,
                "recall@5": None,
                "recall@k": None,
                "mrr": None,
                "ndcg@k": None,
            })

    summary_path = results_dir / "recbole_baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n요약 저장: {summary_path}")
    print("지표: Hit Rate(HR) + Recall 둘 다 (하이퍼볼릭 val_metrics와 동일 키)")
    print("\n--- 베이스라인 지표 요약 (HR = Hit Rate, Recall = 비율) ---")
    for r in all_results:
        name = r.get("model", "?")
        if "error" in r:
            print(f"  {name}: 오류 - {r['error'][:60]}...")
            continue
        hr1 = r.get("hit_rate@1")
        hr5 = r.get("hit_rate@5")
        hr10 = r.get("hit_rate@k") or r.get("hit_rate@10")
        rec1 = r.get("recall@1")
        rec5 = r.get("recall@5")
        rec10 = r.get("recall@k") or r.get("recall@10")
        mrr = r.get("mrr")
        ndcg = r.get("ndcg@k") or r.get("ndcg@10")
        if hr1 is not None:
            _hr5 = hr5 if hr5 is not None else 0.0
            _hr10 = hr10 if hr10 is not None else 0.0
            _r1 = rec1 if rec1 is not None else 0.0
            _r5 = rec5 if rec5 is not None else 0.0
            _r10 = rec10 if rec10 is not None else 0.0
            _mrr = mrr if mrr is not None else 0.0
            _ndcg = ndcg if ndcg is not None else 0.0
            print(f"  {name}: HR@1={hr1:.4f} HR@5={_hr5:.4f} HR@10={_hr10:.4f}  |  Recall@1={_r1:.4f} Recall@5={_r5:.4f} Recall@10={_r10:.4f}  |  MRR={_mrr:.4f} NDCG@10={_ndcg:.4f}")
    return 0


if __name__ == "__main__":
    exit(main())
