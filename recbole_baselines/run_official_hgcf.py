#!/usr/bin/env python3
"""
공식 HGCF (https://github.com/layer6ai-labs/HGCF) 를 우리 데이터·지표로 실행합니다.
클론 → 데이터 복사 → data_generator 패치 → run.py 실행 → 출력 파싱 → 동일 지표 JSON 저장.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def get_script_dir() -> Path:
    return Path(__file__).resolve().parent


def ensure_hgcf_repo(script_dir: Path) -> Path:
    hgcf_repo = script_dir / "hgcf_repo"
    if not hgcf_repo.exists():
        print("공식 HGCF 레포 클론 중... (최초 1회, 네트워크 필요)")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/layer6ai-labs/HGCF.git", str(hgcf_repo)],
            cwd=script_dir,
            check=True,
        )
    return hgcf_repo


def ensure_odor_data(
    script_dir: Path,
    hgcf_repo: Path,
    data_src: Path | None = None,
) -> None:
    """HGCF용 train/test.pkl을 data_src에서 읽어 hgcf_repo/data/odor로 복사.
    data_src가 None이면 script_dir/data/odor_hgcf 사용 (기본/통일 ID 데이터)."""
    if data_src is None:
        data_src = script_dir / "data" / "odor_hgcf"
        if not data_src.exists() or not (data_src / "train.pkl").exists():
            subprocess.run(
                [sys.executable, str(script_dir / "prepare_hgcf_data.py")],
                cwd=script_dir,
                check=True,
            )
    data_dst = hgcf_repo / "data" / "odor"
    data_dst.mkdir(parents=True, exist_ok=True)
    for name in ["train.pkl", "test.pkl"]:
        src = data_src / name
        if src.exists():
            shutil.copy2(src, data_dst / name)


def apply_odor_patch(hgcf_repo: Path, script_dir: Path) -> None:
    dg_path = hgcf_repo / "utils" / "data_generator.py"
    content = dg_path.read_text(encoding="utf-8")
    if "dataset == 'odor'" in content:
        return
    from patches.hgcf_data_generator_odor import ODOR_BRANCH
    marker = " elif dataset.split('-')[0] in ['Amazon', 'yelp']:"
    if marker not in content:
        raise RuntimeError("HGCF data_generator.py 구조가 예상과 다릅니다.")
    content = content.replace(marker, ODOR_BRANCH + "\n" + marker, 1)
    dg_path.write_text(content, encoding="utf-8")
    print("data_generator.py에 odor 데이터셋 패치 적용됨.")


def apply_odor_patch_inline(hgcf_repo: Path, script_dir: Path) -> None:
    """patches 모듈 없이 인라인으로 패치 문자열 사용."""
    dg_path = hgcf_repo / "utils" / "data_generator.py"
    content = dg_path.read_text(encoding="utf-8")
    if "dataset == 'odor'" in content:
        return
    marker = " elif dataset.split('-')[0] in ['Amazon', 'yelp']:"
    if marker not in content:
        raise RuntimeError("HGCF data_generator.py 구조가 예상과 다릅니다.")
    # 원본 data_generator.py의 elif 블록은 12칸 들여쓰기 (ml-100k와 동일)
    patch = (
        "        elif dataset == 'odor':\n"
        "            pkl_path = os.path.join('./data/', dataset)\n"
        "            self.pkl_path = pkl_path\n"
        "            self.dataset = dataset\n"
        "            with open(os.path.join(pkl_path, 'train.pkl'), 'rb') as f:\n"
        "                self.train_dict = pkl.load(f)\n"
        "            with open(os.path.join(pkl_path, 'test.pkl'), 'rb') as f:\n"
        "                self.test_dict = pkl.load(f)\n"
        "            self.num_users = max(self.train_dict.keys()) + 1\n"
        "            all_items = set()\n"
        "            for items in list(self.train_dict.values()) + list(self.test_dict.values()):\n"
        "                all_items.update(items)\n"
        "            self.num_items = max(all_items) + 1 if all_items else 0\n"
        "            self.adj_train, _ = self.generate_adj()\n"
        "            if eval(norm_adj):\n"
        "                self.adj_train_norm = normalize(self.adj_train + sp.eye(self.adj_train.shape[0]))\n"
        "                self.adj_train_norm = sparse_mx_to_torch_sparse_tensor(self.adj_train_norm)\n"
        "            print('num_users %d, num_items %d' % (self.num_users, self.num_items))\n"
        "            print('adjacency matrix shape: ', self.adj_train.shape)\n"
        "            self.user_item_csr = self.generate_rating_matrix([*self.train_dict.values()], self.num_users, self.num_items)\n"
    )
    content = content.replace(marker, patch + "\n" + marker, 1)
    dg_path.write_text(content, encoding="utf-8")
    print("data_generator.py에 odor 데이터셋 패치 적용됨.")


# 베이스라인 요약과 동일한 지표 키 (hit_rate@1/5/10, recall@1/5/10, mrr, ndcg@k/10)
BASELINE_METRIC_KEYS = [
    "hit_rate@1", "hit_rate@5", "hit_rate@k", "hit_rate@10",
    "recall@1", "recall@5", "recall@k", "recall@10",
    "mrr", "ndcg@k", "ndcg@10",
]


def parse_hgcf_stdout(stdout: str) -> dict:
    """run.py 마지막 eval 출력 파싱. 형식: recall@1,5,10,20,50 (탭) / ndcg@1,5,10,20,50 (탭) / mrr (한 줄)."""
    lines = stdout.strip().split("\n")
    last_recall = None
    last_ndcg = None
    last_mrr = 0.0
    # state: 0=다음에 recall 5개, 1=다음에 ndcg 5개, 2=다음에 mrr 1개
    state = 0
    for line in lines:
        parts = line.strip().split("\t")
        try:
            nums = [float(x) for x in parts]
        except ValueError:
            continue
        if len(nums) == 5:
            if state == 0:
                last_recall = nums
                state = 1
            elif state == 1:
                last_ndcg = nums
                state = 2
            else:
                last_recall = nums
                last_ndcg = None
                state = 1
        elif len(nums) == 1 and state == 2:
            last_mrr = nums[0]
            state = 0
    if last_recall is None or last_ndcg is None:
        # 구 형식(4+4) fallback
        pairs = []
        recall_vals = None
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            try:
                nums = [float(x) for x in parts]
            except ValueError:
                continue
            if recall_vals is None:
                recall_vals = nums
            else:
                pairs.append((recall_vals, nums))
                recall_vals = None
        if pairs:
            last_recall = [0.0] + list(pairs[-1][0])
            last_ndcg = [0.0] + list(pairs[-1][1])
        else:
            last_recall = [0.0] * 5
            last_ndcg = [0.0] * 5
    # recall/ndcg: [@1, @5, @10, @20, @50] → 인덱스 0,1,2
    hr1 = last_recall[0] if last_recall and len(last_recall) >= 1 else 0.0
    hr5 = last_recall[1] if last_recall and len(last_recall) >= 2 else 0.0
    hr10 = last_recall[2] if last_recall and len(last_recall) >= 3 else 0.0
    ndcg10 = last_ndcg[2] if last_ndcg and len(last_ndcg) >= 3 else 0.0
    return {
        "hit_rate@1": hr1,
        "hit_rate@5": hr5,
        "hit_rate@k": hr10,
        "hit_rate@10": hr10,
        "recall@1": hr1,
        "recall@5": hr5,
        "recall@k": hr10,
        "recall@10": hr10,
        "mrr": last_mrr,
        "ndcg@k": ndcg10,
        "ndcg@10": ndcg10,
    }


def run_official_hgcf(
    data_path: Path,
    dataset_name: str,
    results_dir: Path,
    seed: int = 42,
    curvature: float = 0.4,
    epochs: int = 200,
    eval_freq: int = 20,
    recbole_inter_dir: Path | None = None,
) -> dict:
    """
    공식 HGCF 레포를 클론/패치한 뒤 실행하고, 우리 지표 형식으로 반환.
    - recbole_inter_dir 이 주어지면: 해당 경로의 .part1/.part3.inter 로 HGCF용 pkl 생성 후 사용 (제안 모델·다른 베이스라인과 동일 데이터).
    - recbole_inter_dir 이 없으면: script_dir/data/odor_hgcf (build_id_mapping 기반) 사용.
    """
    script_dir = get_script_dir()
    hgcf_repo = ensure_hgcf_repo(script_dir)
    data_src = None
    if recbole_inter_dir is not None:
        p1 = recbole_inter_dir / f"{dataset_name}.part1.inter"
        p3 = recbole_inter_dir / f"{dataset_name}.part3.inter"
        if p1.exists() and p3.exists():
            out_hgcf = script_dir / "data" / "odor_hgcf_from_recbole"
            out_hgcf.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    sys.executable,
                    str(script_dir / "prepare_hgcf_data.py"),
                    "--recbole_inter_dir",
                    str(recbole_inter_dir),
                    "--dataset_name",
                    dataset_name,
                    "--out_dir",
                    str(out_hgcf),
                ],
                cwd=script_dir,
                check=True,
            )
            data_src = out_hgcf
    ensure_odor_data(script_dir, hgcf_repo, data_src=data_src)
    apply_odor_patch_inline(hgcf_repo, script_dir)

    cmd = [
        sys.executable, "run.py",
        "--dataset", "odor",
        "--c", str(curvature),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--eval-freq", str(eval_freq),
    ]
    print("실행:", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(hgcf_repo),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        raise RuntimeError(f"HGCF run.py 실패 (exit {proc.returncode})\nstderr:\n{stderr[:2000]}")

    metrics = parse_hgcf_stdout(stdout)
    # 베이스라인 요약과 동일 키로 모두 포함 (hit_rate@1/5/10, recall@1/5/10, mrr, ndcg@k/10)
    out = {"model": "HGCF", "test_result": metrics}
    for k in BASELINE_METRIC_KEYS:
        out[k] = metrics.get(k, 0.0)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_file = results_dir / "recbole_HGCF.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {result_file}")
    return out


def _default_recbole_inter_dir(script_dir: Path, dataset_name: str = "odor") -> Path | None:
    """제안모델·run_baselines와 동일 경로: 정제 우선 datasets_refined → datasets_v2 → datasets."""
    _checkpoints = (script_dir / ".." / "hyperbolic_model" / "results" / "checkpoints").resolve()
    candidates = [
        (_checkpoints / "datasets_refined" / "recbole_data" / dataset_name).resolve(),
        (_checkpoints / "datasets_v2" / "recbole_data" / dataset_name).resolve(),
        (_checkpoints / "datasets" / "recbole_data" / dataset_name).resolve(),
        script_dir / "data" / "recbole" / dataset_name,
        (_checkpoints / "recbole_data" / dataset_name).resolve(),
    ]
    for d in candidates:
        if (d / f"{dataset_name}.part1.inter").exists() and (d / f"{dataset_name}.part3.inter").exists():
            return d
    return None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="공식 HGCF (GitHub) 실행 — 데이터·실험세팅은 제안모델과 동일 (RecBole part1/part3)")
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--recbole_inter_dir", type=str, default=None, help="RecBole .inter 디렉터리 (미지정 시 data/recbole/odor 등 자동 탐색)")
    p.add_argument("--dataset_name", type=str, default="odor")
    p.add_argument("--seed", type=int, default=42, help="하이퍼볼릭 RANDOM_SEED=42와 동일")
    p.add_argument("--curvature", type=float, default=0.4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--eval_freq", type=int, default=20)
    args = p.parse_args()
    script_dir = get_script_dir()
    results_dir = Path(args.results_dir) if args.results_dir else (script_dir / "results")
    recbole_inter_dir = Path(args.recbole_inter_dir).resolve() if args.recbole_inter_dir else _default_recbole_inter_dir(script_dir, args.dataset_name)
    if recbole_inter_dir is not None:
        print(f"데이터셋·실험세팅: 제안모델과 동일 (RecBole) — {recbole_inter_dir}", flush=True)
    run_official_hgcf(
        data_path=script_dir / "data",
        dataset_name=args.dataset_name,
        results_dir=results_dir,
        seed=args.seed,
        curvature=args.curvature,
        epochs=args.epochs,
        eval_freq=args.eval_freq,
        recbole_inter_dir=recbole_inter_dir,
    )
