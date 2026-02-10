"""
Data Loader for Hierarchical Hyperbolic Hypergraph Model

Handles variable-length molecule recipes with SMILES, Notes, and Blender information.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from ogb.utils import smiles2graph
import numpy as np

# RDKit imports for InChI support and molecular fingerprints
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. InChI conversion and fingerprints will not work.")


def smiles_to_morgan_fingerprint(
    smiles: str,
    n_bits: int = 2048,
    radius: int = 2,
) -> np.ndarray:
    """
    ECFP/Morgan fingerprint from SMILES. 실패 시 영벡터 반환.
    """
    if not RDKIT_AVAILABLE or not (smiles or "").strip():
        return np.zeros(n_bits, dtype=np.float32)
    s = (smiles or "").strip()
    if "." in s and not s.startswith("InChI="):
        s = s.split(".")[0].strip()
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        for i in range(n_bits):
            if fp.GetBit(i):
                arr[i] = 1.0
        return arr
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)


def normalize_smiles_key(s: str) -> str:
    """
    precomputed_mol_embs 조회 키 정규화. precompute_molecule_embeddings.py의 collect_unique_smiles와
    100% 동일해야 KeyError를 막을 수 있음 (strip 후, InChI가 아니면 '.' 기준 첫 조각).
    """
    s = (s or "").strip()
    if "." in s and not s.startswith("InChI="):
        s = s.split(".")[0].strip()
    return s


def molecule_to_graph_standalone(mol_string: str, cache: Optional[dict] = None) -> Data:
    """
    SMILES/InChI → PyG Data (프리컴퓨팅 등에서 재사용).
    HyperbolicRecipeDataset._molecule_to_graph와 동일 로직.
    """
    if cache is not None and mol_string in cache:
        return cache[mol_string]
    if "." in mol_string and not mol_string.strip().startswith("InChI="):
        mol_string = mol_string.split(".")[0].strip() or mol_string
    if not mol_string:
        out = Data(
            x=torch.zeros((1, 9), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 3), dtype=torch.float),
        )
        if cache is not None:
            cache[mol_string] = out
        return out
    is_inchi = mol_string.strip().startswith("InChI=")
    mol = None
    if is_inchi and RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromInchi(mol_string, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol, catchErrors=True)
                except Exception:
                    pass
        except Exception:
            pass
    if mol is None and RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(mol_string, sanitize=True)
        except Exception:
            pass
        if mol is None:
            try:
                mol = Chem.MolFromSmiles(mol_string, sanitize=False)
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    except Exception:
                        pass
            except Exception:
                pass
    if mol is not None and RDKIT_AVAILABLE:
        try:
            smiles_from_mol = Chem.MolToSmiles(mol, kekuleSmiles=False)
            g = smiles2graph(smiles_from_mol)
            if g is not None and g.get("node_feat") is not None and len(g["node_feat"]) > 0:
                x = torch.tensor(g["node_feat"], dtype=torch.float)
                edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
                edge_attr = torch.tensor(g["edge_feat"], dtype=torch.float) if "edge_feat" in g else None
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                if cache is not None:
                    cache[mol_string] = data
                return data
        except Exception:
            pass
    if not is_inchi:
        try:
            g = smiles2graph(mol_string)
            if g is not None and g.get("node_feat") is not None and len(g["node_feat"]) > 0:
                x = torch.tensor(g["node_feat"], dtype=torch.float)
                edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
                edge_attr = torch.tensor(g["edge_feat"], dtype=torch.float) if "edge_feat" in g else None
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                if cache is not None:
                    cache[mol_string] = data
                return data
        except Exception:
            pass
    dummy = Data(
        x=torch.zeros((1, 9), dtype=torch.float),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 3), dtype=torch.float),
    )
    if cache is not None:
        cache[mol_string] = dummy
    return dummy


class HyperbolicRecipeDataset(Dataset):
    """
    Dataset for hierarchical hyperbolic hypergraph model.
    
    Each sample contains:
    - Variable number of molecules (SMILES)
    - Notes for each molecule
    - Blender IDs associated with each molecule (for training)
    - Target blender ID (for ranking)
    """
    
    def __init__(
        self,
        records: List[Dict],
        vocab_data: Dict,
        max_molecules: int = 10,
        max_notes_per_molecule: int = 20,
        max_blenders_per_molecule: int = 10,
        mode: str = "train",
        molecule_dropout_rate: float = 0.15,  # ① 성분 마스킹: 15% 기본값 (10~20% 범위)
        precomputed_path: Optional[str] = None,  # 프리컴퓨팅된 분자 임베딩 .pt (GNN 스킵 시 사용)
        use_fingerprint: bool = False,  # True면 GNN 대신 ECFP/Morgan 지문 사용
        fp_dim: int = 2048,  # Morgan fingerprint 비트 수 (n_bits)
        fp_radius: int = 2,   # Morgan radius (ECFP4 = radius 2)
    ):
        """
        Args:
            records: List of molecule records with 'smiles', 'notes', 'blenders' fields
            precomputed_path: Path to precomputed_mol_embs.pt (smiles_to_emb, embed_dim) → 학습 시 GNN 생략
            use_fingerprint: If True, compute ECFP/Morgan fingerprints instead of using GNN
            fp_dim: Morgan fingerprint bit length (n_bits)
            fp_radius: Morgan radius (2 = ECFP4)
            vocab_data: Vocabulary data with 'notes' and 'blenders' mappings
            max_molecules: Maximum number of molecules per recipe
            max_notes_per_molecule: Maximum number of notes per molecule
            max_blenders_per_molecule: Maximum number of blenders per molecule
            mode: 'train', 'val', or 'test'
        """
        self.records = records
        self.vocab_data = vocab_data
        self.max_molecules = max_molecules
        self.max_notes_per_molecule = max_notes_per_molecule
        self.max_blenders_per_molecule = max_blenders_per_molecule
        self.mode = mode
        self.molecule_dropout_rate = molecule_dropout_rate if mode == "train" else 0.0  # 학습 시에만 적용
        self.precomputed_embeddings = None
        self._precomputed_dim = 128
        if precomputed_path and Path(precomputed_path).exists():
            data = torch.load(precomputed_path, map_location="cpu", weights_only=False)
            self.precomputed_embeddings = data.get("smiles_to_emb", data)
            self._precomputed_dim = data.get("embed_dim", 128)
        self.use_fingerprint = use_fingerprint
        self.fp_dim = fp_dim
        self.fp_radius = fp_radius

        # Build vocabularies
        # Try different possible keys for vocabulary mappings
        notes_dict = vocab_data.get('notes', {})
        self.note_to_idx = (
            notes_dict.get('to_idx') or 
            notes_dict.get('to_index') or 
            notes_dict.get('item_to_idx') or 
            {}
        )
        
        blenders_dict = vocab_data.get('blenders', {})
        self.blender_to_idx = (
            blenders_dict.get('to_idx') or 
            blenders_dict.get('to_index') or 
            blenders_dict.get('item_to_idx') or 
            {}
        )
        self.idx_to_blender = {v: k for k, v in self.blender_to_idx.items()}
        
        # Build recipe samples
        self.recipes = self._build_recipes()
        
        # SMILES to graph cache
        self._smiles_cache = {}
    
    def _build_recipes(self) -> List[Dict]:
        """
        Build recipe samples from records.
        
        For training: Create recipes from molecule combinations
        For inference: Use single molecules or combinations
        """
        recipes = []
        
        # Records can be either Dict records or recipe Dicts
        for item in self.records:
            if isinstance(item, dict) and 'molecules' in item:
                # Already a recipe dict from create_recipe_combinations
                # Ensure target_blender is properly set
                recipe = item.copy()
                if 'target_blender' in recipe:
                    # Convert numpy types to Python int if needed
                    target_blender = recipe['target_blender']
                    if hasattr(target_blender, 'item'):
                        recipe['target_blender'] = int(target_blender.item())
                    elif target_blender is not None:
                        recipe['target_blender'] = int(target_blender)
                recipes.append(recipe)
            else:
                # Single molecule record
                if not item.get('smiles') or not item.get('notes'):
                    continue
                
                # Extract target blender if available
                target_blender = None
                blenders = item.get('blenders', [])
                if blenders:
                    # Use first blender as target
                    blender = blenders[0]
                    if isinstance(blender, list):
                        blender_name = blender[0] if len(blender) > 0 else None
                    else:
                        blender_name = blender
                    
                    if blender_name:
                        target_blender = self.blender_to_idx.get(blender_name.lower(), None)
                
                recipe = {
                    'molecules': [item],  # Single molecule recipe
                    'target_blender': target_blender
                }
                recipes.append(recipe)
        
        return recipes
    
    def _molecule_to_graph(self, mol_string: str) -> Data:
        """
        Convert SMILES or InChI to PyTorch Geometric graph.
        Handles multi-molecule SMILES (separated by '.') by using the first molecule.
        
        Args:
            mol_string: SMILES string or InChI string
            
        Returns:
            Data: PyTorch Geometric graph (or dummy graph if conversion fails)
        """
        if mol_string in self._smiles_cache:
            return self._smiles_cache[mol_string]
        
        # Handle multi-molecule SMILES (separated by '.')
        # Use the first molecule if multiple are present
        if '.' in mol_string and not mol_string.strip().startswith('InChI='):
            mol_string = mol_string.split('.')[0].strip()
            if not mol_string:
                # Empty after splitting - return dummy
                dummy_data = Data(
                    x=torch.zeros((1, 9), dtype=torch.float),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, 3), dtype=torch.float)
                )
                self._smiles_cache[mol_string] = dummy_data
                return dummy_data
        
        # Check if input is InChI or SMILES
        is_inchi = mol_string.strip().startswith('InChI=')
        
        mol = None
        
        if is_inchi and RDKIT_AVAILABLE:
            # Try to parse InChI
            try:
                mol = Chem.MolFromInchi(mol_string, sanitize=False)
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol, catchErrors=True)
                    except Exception:
                        # If sanitization fails, try without it
                        pass
            except Exception:
                pass
        
        if mol is None and RDKIT_AVAILABLE:
            # Try SMILES parsing with various options
            # Option 1: Standard parsing
            try:
                mol = Chem.MolFromSmiles(mol_string, sanitize=True)
            except Exception:
                pass
            
            # Option 2: Without sanitization (for kekulization errors)
            if mol is None:
                try:
                    mol = Chem.MolFromSmiles(mol_string, sanitize=False)
                    if mol is not None:
                        # Try partial sanitization
                        try:
                            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                        except Exception:
                            pass
                except Exception:
                    pass
        
        # If RDKit parsing succeeded, convert to graph using ogb format
        if mol is not None and RDKIT_AVAILABLE:
            try:
                # Convert RDKit mol to SMILES and use smiles2graph
                smiles_from_mol = Chem.MolToSmiles(mol, kekuleSmiles=False)
                g = smiles2graph(smiles_from_mol)
                if g is not None and g.get('node_feat') is not None and len(g['node_feat']) > 0:
                    x = torch.tensor(g['node_feat'], dtype=torch.float)
                    edge_index = torch.tensor(g['edge_index'], dtype=torch.long)
                    edge_attr = torch.tensor(g['edge_feat'], dtype=torch.float) if 'edge_feat' in g else None
                    
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                    self._smiles_cache[mol_string] = data
                    return data
            except Exception:
                pass
        
        # If all parsing failed, try direct smiles2graph (for SMILES)
        if not is_inchi:
            try:
                g = smiles2graph(mol_string)
                if g is not None and g.get('node_feat') is not None and len(g['node_feat']) > 0:
                    x = torch.tensor(g['node_feat'], dtype=torch.float)
                    edge_index = torch.tensor(g['edge_index'], dtype=torch.long)
                    edge_attr = torch.tensor(g['edge_feat'], dtype=torch.float) if 'edge_feat' in g else None
                    
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                    self._smiles_cache[mol_string] = data
                    return data
            except Exception:
                pass
        
        # All parsing failed - return dummy graph
        # Create a dummy graph with single node (no edges)
        dummy_data = Data(
            x=torch.zeros((1, 9), dtype=torch.float),  # Single node with 9 features
            edge_index=torch.zeros((2, 0), dtype=torch.long),  # No edges
            edge_attr=torch.zeros((0, 3), dtype=torch.float)  # No edge attributes
        )
        self._smiles_cache[mol_string] = dummy_data
        return dummy_data
    
    def _smiles_to_graph(self, smiles: str) -> Data:
        """Convert SMILES to PyTorch Geometric graph (backward compatibility)."""
        return self._molecule_to_graph(smiles)
    
    def _notes_to_indices(self, notes: List[str]) -> torch.Tensor:
        """Convert note names to indices."""
        indices = []
        for note in notes[:self.max_notes_per_molecule]:
            idx = self.note_to_idx.get(note.lower(), 0)  # 0 for unknown
            indices.append(idx)
        
        # Pad to max_notes_per_molecule
        while len(indices) < self.max_notes_per_molecule:
            indices.append(0)  # Padding index
        
        return torch.tensor(indices, dtype=torch.long)
    
    def _blenders_to_indices(self, blenders: List) -> torch.Tensor:
        """Convert blender names to indices."""
        indices = []
        for blender in blenders[:self.max_blenders_per_molecule]:
            if isinstance(blender, list):
                blender_name = blender[0] if len(blender) > 0 else None
            else:
                blender_name = blender
            
            if blender_name:
                idx = self.blender_to_idx.get(blender_name.lower(), 0)
                if idx > 0:  # Only add valid blenders
                    indices.append(idx)
        
        # Pad to max_blenders_per_molecule
        while len(indices) < self.max_blenders_per_molecule:
            indices.append(0)  # Padding index
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.recipes)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a recipe sample."""
        recipe = self.recipes[idx]
        molecules = recipe['molecules']
        
        # Limit to max_molecules
        molecules = molecules[:self.max_molecules]
        
        # ① 성분 마스킹 (Molecule Dropout): 학습 시 10~20% 성분 무작위 제거
        if self.mode == "train" and self.molecule_dropout_rate > 0 and len(molecules) > 1:
            import random
            num_to_drop = max(1, int(len(molecules) * self.molecule_dropout_rate))
            num_to_drop = min(num_to_drop, len(molecules) - 1)  # 최소 1개는 남겨야 함
            
            if num_to_drop > 0:
                indices_to_keep = random.sample(range(len(molecules)), len(molecules) - num_to_drop)
                molecules = [molecules[i] for i in sorted(indices_to_keep)]
        
        # Prepare data structures
        smiles_graphs = []
        note_indices_list = []
        blender_indices_list = []
        
        for mol_record in molecules:
            # SMILES/InChI graph
            # Handle both SMILES and InChI formats
            mol_string = mol_record.get('smiles', '')
            if not mol_string:
                # Empty string - use dummy graph
                graph = Data(
                    x=torch.zeros((1, 9), dtype=torch.float),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, 3), dtype=torch.float)
                )
            else:
                # _molecule_to_graph handles both SMILES and InChI
                # Always returns a Data object (dummy graph if parsing fails)
                graph = self._molecule_to_graph(mol_string)
            smiles_graphs.append(graph)
            
            # Notes
            notes = mol_record.get('notes', [])
            note_indices = self._notes_to_indices(notes)
            note_indices_list.append(note_indices)
            
            # Blenders
            blenders = mol_record.get('blenders', [])
            blender_indices = self._blenders_to_indices(blenders)
            blender_indices_list.append(blender_indices)
        
        # Pad to max_molecules
        while len(smiles_graphs) < self.max_molecules:
            smiles_graphs.append(Data(
                x=torch.zeros((1, 9), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 3), dtype=torch.float)
            ))
            note_indices_list.append(torch.zeros(self.max_notes_per_molecule, dtype=torch.long))
            blender_indices_list.append(torch.zeros(self.max_blenders_per_molecule, dtype=torch.long))
        
        # Stack tensors
        note_indices_tensor = torch.stack(note_indices_list, dim=0)  # [max_molecules, max_notes]
        blender_indices_tensor = torch.stack(blender_indices_list, dim=0)  # [max_molecules, max_blenders]
        
        # Create molecule mask
        num_valid_molecules = len(molecules)
        molecule_mask = torch.zeros(self.max_molecules, dtype=torch.float)
        molecule_mask[:num_valid_molecules] = 1.0
        
        # For batch dimension: add batch_size=1
        note_indices_tensor = note_indices_tensor.unsqueeze(0)  # [1, max_molecules, max_notes]
        blender_indices_tensor = blender_indices_tensor.unsqueeze(0)  # [1, max_molecules, max_blenders]
        molecule_mask = molecule_mask.unsqueeze(0)  # [1, max_molecules]
        
        # 여러 정답 blender 지원
        target_blender = recipe.get('target_blender')  # 대표 blender (하위 호환성)
        target_blenders = recipe.get('target_blenders', [target_blender] if target_blender is not None else [])  # 모든 정답 blender
        target_group = recipe.get('target_group', 0)  # 계층 보조 loss용

        out = {
            'smiles_graphs': smiles_graphs,
            'note_indices': note_indices_tensor,
            'blender_indices': blender_indices_tensor,
            'molecule_mask': molecule_mask,
            'target_blender': target_blender,
            'target_blenders': target_blenders,
            'target_group': target_group
        }
        if self.precomputed_embeddings is not None:
            mol_embs_list = []
            for mol_record in molecules:
                s = normalize_smiles_key(mol_record.get("smiles") or "")
                emb = self.precomputed_embeddings.get(s)
                if emb is None:
                    emb = torch.zeros(self._precomputed_dim, dtype=torch.float)
                elif not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb, dtype=torch.float)
                mol_embs_list.append(emb)
            while len(mol_embs_list) < self.max_molecules:
                mol_embs_list.append(torch.zeros(self._precomputed_dim, dtype=torch.float))
            precomputed_mol_embs = torch.stack(mol_embs_list[:self.max_molecules], dim=0).unsqueeze(0)
            out['precomputed_mol_embs'] = precomputed_mol_embs
        if self.use_fingerprint:
            fp_list = []
            for mol_record in molecules:
                s = (mol_record.get("smiles") or "").strip()
                arr = smiles_to_morgan_fingerprint(s, n_bits=self.fp_dim, radius=self.fp_radius)
                fp_list.append(torch.tensor(arr, dtype=torch.float))
            while len(fp_list) < self.max_molecules:
                fp_list.append(torch.zeros(self.fp_dim, dtype=torch.float))
            mol_fingerprints = torch.stack(fp_list[:self.max_molecules], dim=0).unsqueeze(0)  # [1, max_molecules, fp_dim]
            out['mol_fingerprints'] = mol_fingerprints
        return out


def collate_hyperbolic_recipes(batch: List[Dict]) -> Dict:
    """
    Collate function for HyperbolicRecipeDataset.
    
    Handles variable-length molecule lists and creates batched graphs.
    """
    batch_size = len(batch)
    max_molecules = batch[0]['note_indices'].size(1)
    max_notes = batch[0]['note_indices'].size(2)
    max_blenders = batch[0]['blender_indices'].size(2)
    
    # Collect all graphs for batching
    all_graphs = []
    graph_batch_indices = []
    note_indices_list = []
    blender_indices_list = []
    molecule_mask_list = []
    target_blenders = []
    target_groups = []

    for i, sample in enumerate(batch):
        # Collect graphs
        for j, graph in enumerate(sample['smiles_graphs']):
            all_graphs.append(graph)
            # Batch index: sample_idx * max_molecules + mol_idx
            graph_batch_indices.append(i * max_molecules + j)
        
        # Collect other data (remove batch dimension added in __getitem__)
        note_indices_list.append(sample['note_indices'].squeeze(0))  # [max_molecules, max_notes]
        blender_indices_list.append(sample['blender_indices'].squeeze(0))  # [max_molecules, max_blenders]
        molecule_mask_list.append(sample['molecule_mask'].squeeze(0))  # [max_molecules]
        # 여러 정답 blender 지원
        if 'target_blenders' in sample and sample['target_blenders']:
            target_blenders.append(sample['target_blenders'])  # 리스트로 저장
        else:
            target_blender = sample.get('target_blender')
            target_blenders.append([target_blender] if target_blender is not None else [])
        target_groups.append(sample.get('target_group', 0))

    # Batch graphs
    batch_graphs = Batch.from_data_list(all_graphs)
    batch_indices = torch.tensor(graph_batch_indices, dtype=torch.long, device=batch_graphs.x.device)
    
    # Stack other tensors
    note_indices = torch.stack(note_indices_list, dim=0)  # [batch_size, max_molecules, max_notes]
    blender_indices = torch.stack(blender_indices_list, dim=0)  # [batch_size, max_molecules, max_blenders]
    molecule_mask = torch.stack(molecule_mask_list, dim=0)  # [batch_size, max_molecules]
    
    target_group_tensor = torch.tensor(target_groups, dtype=torch.long, device=note_indices.device)

    result = {
        'smiles_graphs': batch_graphs,
        'smiles_batch': batch_indices,
        'note_indices': note_indices,
        'blender_indices': blender_indices,
        'molecule_mask': molecule_mask,
        'target_blenders': target_blenders,
        'target_group': target_group_tensor
    }
    if 'precomputed_mol_embs' in batch[0]:
        result['precomputed_mol_embs'] = torch.cat([s['precomputed_mol_embs'] for s in batch], dim=0)
    if 'mol_fingerprints' in batch[0]:
        result['mol_fingerprints'] = torch.cat([s['mol_fingerprints'] for s in batch], dim=0)
    return result


def load_data(data_path: Optional[str] = None, vocab_path: Optional[str] = None) -> Tuple[List[Dict], Dict]:
    """
    Load cleaned data and vocabulary.
    
    Args:
        data_path: Path to cleaned_complete_data.json
        vocab_path: Path to vocabularies.json
    
    Returns:
        records: List of molecule records
        vocab_data: Vocabulary data
    """
    if data_path is None:
        data_path = Path(__file__).parent / "cleaned_data" / "cleaned_complete_data.json"
    else:
        data_path = Path(data_path)
    
    # Convert paths to Path objects and resolve relative paths
    data_path = Path(data_path)
    if not data_path.is_absolute():
        # If relative path, resolve from current working directory
        data_path = Path.cwd() / data_path
    data_path = data_path.resolve()
    
    if vocab_path is None:
        vocab_path = Path(__file__).parent.parent.parent / "feature_encoding" / "vocabularies.json"
    else:
        vocab_path = Path(vocab_path)
        if not vocab_path.is_absolute():
            vocab_path = Path.cwd() / vocab_path
        vocab_path = vocab_path.resolve()
    
    # Check if files exist
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        records = data.get('data', [])
    
    # Load vocabularies
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    return records, vocab_data

