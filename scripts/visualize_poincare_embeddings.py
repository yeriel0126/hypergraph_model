"""
Poincaré Disk 시각화 스크립트

하이퍼볼릭 공간(Poincaré Disk)에 Recipe와 Blender 임베딩을 2D로 시각화합니다.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import json
import sys

# 모델 import를 위한 경로 추가
model_dir = Path(__file__).parent.parent
sys.path.insert(0, str(model_dir))
sys.path.insert(0, str(model_dir / "model"))

try:
    from model.hierarchical_hyperbolic_hypergraph import HierarchicalFragranceHypergraph
    from model.hyperbolic_data_loader import load_data
    from torch.utils.data import DataLoader
    from model.hyperbolic_data_loader import HyperbolicRecipeDataset, collate_hyperbolic_recipes
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available.")
    sys.exit(1)

def project_to_2d(embeddings, method='pca'):
    """
    하이퍼볼릭 임베딩을 2D로 투영
    
    Args:
        embeddings: [N, dim] - 하이퍼볼릭 임베딩
        method: 'pca' or 'tsne' or 'first_two'
    
    Returns:
        coords_2d: [N, 2] - 2D 좌표
    """
    embeddings_np = embeddings.detach().cpu().numpy()
    
    if method == 'first_two':
        # 첫 두 차원 사용
        return embeddings_np[:, :2]
    elif method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(embeddings_np)
        return coords_2d
    elif method == 'tsne':
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            coords_2d = tsne.fit_transform(embeddings_np)
            return coords_2d
        except ImportError:
            print("t-SNE not available, using PCA instead")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(embeddings_np)
            return coords_2d
    else:
        raise ValueError(f"Unknown method: {method}")

def visualize_poincare_disk(
    model_path: str,
    vocab_path: str = None,
    data_path: str = None,
    output_path: str = "checkpoints/poincare_visualization.png",
    num_samples: int = 100,
    method: str = 'pca'
):
    """
    Poincaré Disk에 임베딩 시각화
    
    Args:
        model_path: 체크포인트 경로
        vocab_path: vocabulary 파일 경로
        data_path: 데이터 파일 경로
        output_path: 출력 이미지 경로
        num_samples: 시각화할 샘플 수
        method: 투영 방법 ('pca', 'tsne', 'first_two')
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("Poincaré Disk 시각화")
    print("=" * 70)
    
    # 모델 로드
    print(f"\n1. 모델 로드: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Vocabulary 로드
    if vocab_path:
        print(f"2. Vocabulary 로드: {vocab_path}")
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        num_blenders = vocab_data.get('blenders', {}).get('size', 100)
        vocab_size = vocab_data.get('notes', {}).get('size', 435)
    else:
        # Checkpoint에서 정보 추출
        args = checkpoint.get('args', {})
        num_blenders = args.get('num_blenders', 100)
        vocab_size = args.get('vocab_size', 435)
        vocab_data = None
    
    # 모델 생성
    model = HierarchicalFragranceHypergraph(
        node_dim=9,
        edge_dim=3,
        gnn_hidden_dim=128,
        gnn_output_dim=128,
        gnn_num_layers=3,
        vocab_size=vocab_size,
        note_embedding_dim=300,
        note_hyperbolic_dim=128,
        num_blenders=num_blenders,
        blender_dim=128,
        channel1_output_dim=128,
        channel2_output_dim=128,
        c=0.2,
        learnable_curvature=True,
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   ✓ 모델 로드 완료")
    
    # Blender 임베딩 추출
    print(f"\n3. Blender 임베딩 추출")
    with torch.no_grad():
        blender_embs = model.blender_anchors()  # [num_blenders, dim]
    print(f"   ✓ Blender 임베딩: {blender_embs.shape}")
    
    # Recipe 임베딩 추출 (샘플 데이터 사용)
    print(f"\n4. Recipe 임베딩 추출 (샘플 {num_samples}개)")
    recipe_embs_list = []
    
    if data_path and vocab_path:
        try:
            records, vocab_data = load_data(data_path, vocab_path)
            # 샘플 선택
            sample_records = records[:num_samples] if len(records) > num_samples else records
            
            dataset = HyperbolicRecipeDataset(
                records=sample_records,
                vocab_data=vocab_data,
                max_molecules=10,
                max_notes_per_molecule=20,
                max_blenders_per_molecule=10,
                mode="val"
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=False,
                collate_fn=collate_hyperbolic_recipes,
                num_workers=0
            )
            
            with torch.no_grad():
                for batch in dataloader:
                    note_indices = batch['note_indices'].to(device)
                    blender_indices = batch['blender_indices'].to(device)
                    molecule_mask = batch['molecule_mask'].to(device)
                    smiles_graphs = batch['smiles_graphs']
                    if hasattr(smiles_graphs, 'to'):
                        smiles_graphs = smiles_graphs.to(device)
                    else:
                        smiles_graphs.x = smiles_graphs.x.to(device)
                        smiles_graphs.edge_index = smiles_graphs.edge_index.to(device)
                    smiles_batch = batch['smiles_batch'].to(device)
                    
                    z_recipe = model(
                        smiles_graphs=smiles_graphs,
                        smiles_batch=smiles_batch,
                        note_indices=note_indices,
                        blender_indices=blender_indices,
                        molecule_mask=molecule_mask
                    )
                    recipe_embs_list.append(z_recipe.cpu())
            
            if recipe_embs_list:
                recipe_embs = torch.cat(recipe_embs_list, dim=0)
            else:
                # Fallback: 랜덤 샘플 생성
                recipe_embs = torch.randn(num_samples, 128) * 0.1
                print("   ⚠️  데이터 로드 실패, 랜덤 샘플 사용")
        except Exception as e:
            print(f"   ⚠️  데이터 로드 실패: {e}")
            # Fallback: 랜덤 샘플 생성
            recipe_embs = torch.randn(num_samples, 128) * 0.1
    else:
        # Fallback: 랜덤 샘플 생성
        recipe_embs = torch.randn(num_samples, 128) * 0.1
        print("   ⚠️  데이터 경로 없음, 랜덤 샘플 사용")
    
    print(f"   ✓ Recipe 임베딩: {recipe_embs.shape}")
    
    # 2D로 투영
    print(f"\n5. 2D 투영 (method: {method})")
    blender_2d = project_to_2d(blender_embs, method=method)
    recipe_2d = project_to_2d(recipe_embs, method=method)
    
    # Poincaré Disk 시각화
    print(f"\n6. 시각화 생성")
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Poincaré Disk 원 그리기
    circle = plt.Circle((0, 0), 1.0, fill=False, color='black', linewidth=2, linestyle='--', alpha=0.3)
    ax.add_patch(circle)
    
    # Blender 임베딩 시각화
    blender_norms = np.linalg.norm(blender_2d, axis=1)
    valid_blenders = blender_norms < 0.99  # Disk 내부만
    
    if valid_blenders.sum() > 0:
        ax.scatter(
            blender_2d[valid_blenders, 0],
            blender_2d[valid_blenders, 1],
            c='red',
            s=50,
            alpha=0.6,
            label=f'Blenders ({valid_blenders.sum()})',
            edgecolors='darkred',
            linewidths=0.5
        )
    
    # Recipe 임베딩 시각화
    recipe_norms = np.linalg.norm(recipe_2d, axis=1)
    valid_recipes = recipe_norms < 0.99
    
    if valid_recipes.sum() > 0:
        ax.scatter(
            recipe_2d[valid_recipes, 0],
            recipe_2d[valid_recipes, 1],
            c='blue',
            s=30,
            alpha=0.4,
            label=f'Recipes ({valid_recipes.sum()})',
            edgecolors='darkblue',
            linewidths=0.3
        )
    
    # 중심점 표시
    ax.scatter([0], [0], c='black', s=100, marker='x', linewidths=3, label='Origin', zorder=10)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title('Poincaré Disk: Recipe and Blender Embeddings', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # 통계 정보 추가
    stats_text = f"""
    Statistics:
    - Blenders: {valid_blenders.sum()}/{len(blender_2d)} (mean norm: {blender_norms.mean():.3f})
    - Recipes: {valid_recipes.sum()}/{len(recipe_2d)} (mean norm: {recipe_norms.mean():.3f})
    - Projection: {method.upper()}
    """
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 저장
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ 시각화 저장: {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize embeddings in Poincaré Disk")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab_path", type=str, default=None, help="Path to vocabulary file")
    parser.add_argument("--data_path", type=str, default=None, help="Path to data file")
    parser.add_argument("--output_path", type=str, default="checkpoints/poincare_visualization.png")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "tsne", "first_two"])
    
    args = parser.parse_args()
    
    visualize_poincare_disk(
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        data_path=args.data_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        method=args.method
    )
