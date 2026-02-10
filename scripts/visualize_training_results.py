"""
학습 결과 시각화 스크립트

학습 히스토리를 읽어서 그래프로 시각화합니다.
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
import numpy as np

def load_training_history(history_path):
    """학습 히스토리 로드"""
    if isinstance(history_path, str):
        history_path = Path(history_path)
    
    if not history_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {history_path}")
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history

def visualize_training_results(history_path="checkpoints/training_history.json", output_dir="checkpoints"):
    """학습 결과 시각화"""
    
    # 히스토리 로드
    history = load_training_history(history_path)
    if history is None or len(history) == 0:
        print("❌ 학습 히스토리가 비어있습니다.")
        return
    
    # 데이터 추출
    epochs = [h.get('epoch', i+1) for i, h in enumerate(history)]
    train_losses = [h.get('train_loss', 0) for h in history]
    val_hit_rate_1 = [h.get('val_hit_rate@1', h.get('val_hit_rate', 0)) for h in history]
    val_hit_rate_5 = [h.get('val_hit_rate@5', 0) for h in history]
    val_hit_rate_10 = [h.get('val_hit_rate@10', h.get('val_hit_rate', 0)) for h in history]
    val_mrr = [h.get('val_mrr', 0) for h in history]
    val_ndcg = [h.get('val_ndcg', 0) for h in history]
    val_diversity_std = [h.get('val_diversity_std', 0) for h in history]
    best_val_hit_rate = [h.get('best_val_hit_rate', 0) for h in history]
    
    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 한글 폰트 설정 (macOS)
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. Train Loss 그래프
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=6, label='Train Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Train Loss over Epochs', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / 'train_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Train Loss 그래프 저장: {output_dir / 'train_loss.png'}")
    
    # 2. Validation Hit Rate 그래프 (Top-1, Top-5, Top-10)
    fig, ax = plt.subplots(figsize=(10, 6))
    if any(val_hit_rate_1):
        ax.plot(epochs, val_hit_rate_1, 'r-o', linewidth=2, markersize=6, label='Hit Rate@1', alpha=0.8)
    if any(val_hit_rate_5):
        ax.plot(epochs, val_hit_rate_5, 'orange', marker='s', linewidth=2, markersize=6, label='Hit Rate@5', alpha=0.8)
    ax.plot(epochs, val_hit_rate_10, 'g-o', linewidth=2, markersize=6, label='Hit Rate@10', alpha=0.8)
    if best_val_hit_rate[0] > 0:
        ax.plot(epochs, best_val_hit_rate, 'r--', linewidth=2, alpha=0.5, label='Best Hit Rate@10')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Hit Rate', fontsize=12)
    ax.set_title('Validation Hit Rate (Top-1, Top-5, Top-10) over Epochs', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / 'val_hit_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Val Hit Rate 그래프 저장: {output_dir / 'val_hit_rate.png'}")
    
    # 3. Validation Metrics (MRR, NDCG) 그래프
    if any(val_mrr) or any(val_ndcg):
        fig, ax = plt.subplots(figsize=(10, 6))
        if any(val_mrr):
            ax.plot(epochs, val_mrr, 'purple', marker='o', linewidth=2, markersize=6, label='MRR')
        if any(val_ndcg):
            ax.plot(epochs, val_ndcg, 'orange', marker='s', linewidth=2, markersize=6, label='NDCG@10')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Validation Metrics (MRR, NDCG) over Epochs', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(output_dir / 'val_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Val Metrics 그래프 저장: {output_dir / 'val_metrics.png'}")
    
    # 4. 종합 대시보드 (2x2 subplot)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Train Loss
    axes[0, 0].plot(epochs, train_losses, 'b-o', linewidth=2, markersize=5)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Train Loss', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Val Hit Rate (Top-1, Top-5, Top-10)
    if any(val_hit_rate_1):
        axes[0, 1].plot(epochs, val_hit_rate_1, 'r-o', linewidth=2, markersize=4, label='Hit@1', alpha=0.7)
    if any(val_hit_rate_5):
        axes[0, 1].plot(epochs, val_hit_rate_5, 'orange', marker='s', linewidth=2, markersize=4, label='Hit@5', alpha=0.7)
    axes[0, 1].plot(epochs, val_hit_rate_10, 'g-o', linewidth=2, markersize=5, label='Hit@10', alpha=0.8)
    if best_val_hit_rate[0] > 0:
        axes[0, 1].plot(epochs, best_val_hit_rate, 'r--', linewidth=2, alpha=0.5, label='Best')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Hit Rate', fontsize=11)
    axes[0, 1].set_title('Validation Hit Rate (Top-1,5,10)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=9)
    
    # MRR & NDCG
    if any(val_mrr) or any(val_ndcg):
        if any(val_mrr):
            axes[1, 0].plot(epochs, val_mrr, 'purple', marker='o', linewidth=2, markersize=5, label='MRR')
        if any(val_ndcg):
            axes[1, 0].plot(epochs, val_ndcg, 'orange', marker='s', linewidth=2, markersize=5, label='NDCG@10')
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('Score', fontsize=11)
        axes[1, 0].set_title('Validation Metrics', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=9)
    else:
        axes[1, 0].text(0.5, 0.5, 'No MRR/NDCG data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Validation Metrics', fontsize=12, fontweight='bold')
    
    # Diversity
    if any(val_diversity_std):
        axes[1, 1].plot(epochs, val_diversity_std, 'brown', marker='^', linewidth=2, markersize=5)
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Diversity Std', fontsize=11)
        axes[1, 1].set_title('Recommendation Diversity', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Diversity data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Recommendation Diversity', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 종합 대시보드 저장: {output_dir / 'training_dashboard.png'}")
    
    # 5. Loss vs Hit Rate 비교 (dual y-axis)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss', color=color, fontsize=12)
    line1 = ax1.plot(epochs, train_losses, color=color, marker='o', linewidth=2, markersize=6, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Val Hit Rate', color=color, fontsize=12)
    lines = list(line1)
    labels = ['Train Loss']
    
    if any(val_hit_rate_1):
        line2 = ax2.plot(epochs, val_hit_rate_1, 'r', marker='o', linewidth=2, markersize=5, label='Hit Rate@1', alpha=0.7)
        lines.extend(line2)
        labels.append('Hit Rate@1')
    if any(val_hit_rate_5):
        line3 = ax2.plot(epochs, val_hit_rate_5, 'orange', marker='s', linewidth=2, markersize=5, label='Hit Rate@5', alpha=0.7)
        lines.extend(line3)
        labels.append('Hit Rate@5')
    line4 = ax2.plot(epochs, val_hit_rate_10, color=color, marker='s', linewidth=2, markersize=6, label='Hit Rate@10')
    lines.extend(line4)
    labels.append('Hit Rate@10')
    
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 범례 통합
    ax1.legend(lines, labels, loc='upper left', fontsize=10)
    
    plt.title('Train Loss vs Val Hit Rate (Top-1,5,10)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_vs_hitrate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Loss vs Hit Rate 그래프 저장: {output_dir / 'loss_vs_hitrate.png'}")
    
    # 통계 출력
    print("\n" + "=" * 70)
    print("📊 학습 결과 통계")
    print("=" * 70)
    print(f"총 에폭 수: {len(epochs)}")
    print(f"\nTrain Loss:")
    print(f"  초기: {train_losses[0]:.4f}")
    print(f"  최종: {train_losses[-1]:.4f}")
    print(f"  변화: {train_losses[-1] - train_losses[0]:.4f}")
    print(f"  최소: {min(train_losses):.4f} (Epoch {epochs[train_losses.index(min(train_losses))]})")
    
    print(f"\nVal Hit Rate@1:")
    if any(val_hit_rate_1):
        print(f"  초기: {val_hit_rate_1[0]:.4f}")
        print(f"  최종: {val_hit_rate_1[-1]:.4f}")
        print(f"  변화: {val_hit_rate_1[-1] - val_hit_rate_1[0]:.4f}")
        print(f"  최대: {max(val_hit_rate_1):.4f} (Epoch {epochs[val_hit_rate_1.index(max(val_hit_rate_1))]})")
    
    print(f"\nVal Hit Rate@5:")
    if any(val_hit_rate_5):
        print(f"  초기: {val_hit_rate_5[0]:.4f}")
        print(f"  최종: {val_hit_rate_5[-1]:.4f}")
        print(f"  변화: {val_hit_rate_5[-1] - val_hit_rate_5[0]:.4f}")
        print(f"  최대: {max(val_hit_rate_5):.4f} (Epoch {epochs[val_hit_rate_5.index(max(val_hit_rate_5))]})")
    
    print(f"\nVal Hit Rate@10:")
    print(f"  초기: {val_hit_rate_10[0]:.4f}")
    print(f"  최종: {val_hit_rate_10[-1]:.4f}")
    print(f"  변화: {val_hit_rate_10[-1] - val_hit_rate_10[0]:.4f}")
    print(f"  최대: {max(val_hit_rate_10):.4f} (Epoch {epochs[val_hit_rate_10.index(max(val_hit_rate_10))]})")
    
    if any(val_mrr):
        print(f"\nMRR:")
        print(f"  최종: {val_mrr[-1]:.4f}")
        print(f"  최대: {max(val_mrr):.4f} (Epoch {epochs[val_mrr.index(max(val_mrr))]})")
    
    if any(val_ndcg):
        print(f"\nNDCG@10:")
        print(f"  최종: {val_ndcg[-1]:.4f}")
        print(f"  최대: {max(val_ndcg):.4f} (Epoch {epochs[val_ndcg.index(max(val_ndcg))]})")
    
    print("=" * 70)
    print(f"\n✅ 모든 그래프가 저장되었습니다: {output_dir}")

if __name__ == "__main__":
    import sys
    
    # 명령줄 인자 처리
    if len(sys.argv) > 1:
        history_path = sys.argv[1]
    else:
        history_path = "checkpoints/training_history.json"
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "checkpoints"
    
    visualize_training_results(history_path, output_dir)
