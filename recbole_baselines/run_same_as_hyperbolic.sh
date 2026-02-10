#!/bin/bash
# 하이퍼볼릭 모델과 동일한 데이터셋·동일한 지표로 RecBole 베이스라인 실행
#
# 사용법:
#   ./run_same_as_hyperbolic.sh
#   ./run_same_as_hyperbolic.sh /path/to/hyperbolic/output_dir
#
# 데이터: train_combinations.json, val_combinations.json, test_combinations_original.json
# 지표: hit_rate@1, hit_rate@5, hit_rate@k, mrr, ndcg@k (하이퍼볼릭 val_metrics와 동일 키)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_DIR="${1:-$SCRIPT_DIR/../hyperbolic_model/results/checkpoints}"
# output_dir만 주어지면 내부 datasets 사용
if [ -f "$DATASET_DIR/datasets/train_combinations.json" ]; then
  DATASET_DIR="$DATASET_DIR/datasets"
fi

echo "데이터셋 경로: $DATASET_DIR"
python run_baselines.py --dataset_dir "$DATASET_DIR"
