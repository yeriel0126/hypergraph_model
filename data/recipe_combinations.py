"""
레시피 조합 생성: 단일 분자 레코드 → (분자 조합, target_blender) 학습용 데이터.

학습 로직과 완전히 독립. 데이터가 한 번 만들어지면 학습 시에는 JSON 로드만 하면 됨.
"""

from collections import Counter
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# hyperbolic_model이 sys.path에 있을 때 utils는 hyperbolic_model.utils
try:
    from utils import (
        detect_cheat_key_molecules,
        compute_molecule_frequency_weights,
        filter_duplicate_combinations,
        apply_cheat_key_masking,
        add_gaussian_noise_to_recipe,
    )
except ImportError:
    from ..utils import (
        detect_cheat_key_molecules,
        compute_molecule_frequency_weights,
        filter_duplicate_combinations,
        apply_cheat_key_masking,
        add_gaussian_noise_to_recipe,
    )


DEFAULT_TRAIN_SIZE = 60000
DEFAULT_RANDOM_SEED = 42


def build_group_vocab_and_blender_to_group(records: List[Dict], vocab_data: Dict) -> int:
    """
    레코드에서 blender [name, group] 쌍을 수집하여 그룹 vocab과 blender->group 매핑을 구축하고
    vocab_data에 저장. create_recipe_combinations에서 target_group 설정에 사용.
    """
    groups_data = vocab_data.get('groups', {})
    if isinstance(groups_data, dict) and 'vocab' in groups_data:
        group_vocab = groups_data['vocab']
        group_to_idx = {str(g).strip().lower(): i for i, g in enumerate(group_vocab)}
    elif isinstance(groups_data, dict) and 'to_idx' in groups_data:
        group_to_idx = {str(k).strip().lower(): int(v) for k, v in groups_data['to_idx'].items()}
    else:
        group_to_idx = {}
        group_vocab = []

    blender_group_pairs = []
    for record in records:
        blenders = record.get('blenders', [])
        if not blenders:
            continue
        for item in blenders:
            if isinstance(item, list) and len(item) >= 2:
                bname = str(item[0]).strip().lower()
                gname = str(item[1]).strip().lower() if item[1] else 'unknown'
                blender_group_pairs.append((bname, gname))
            elif isinstance(item, list) and len(item) == 1:
                blender_group_pairs.append((str(item[0]).strip().lower(), 'unknown'))

    if not blender_group_pairs:
        vocab_data['blender_to_group'] = {}
        vocab_data['groups'] = vocab_data.get('groups') or {'vocab': ['unknown'], 'to_idx': {'unknown': 0}, 'size': 1}
        return 1

    if not group_to_idx:
        unique_groups = sorted(set(g for _, g in blender_group_pairs))
        group_vocab = unique_groups
        group_to_idx = {g: i for i, g in enumerate(unique_groups)}
        vocab_data['groups'] = {'vocab': group_vocab, 'to_idx': group_to_idx, 'size': len(group_to_idx)}

    blender_groups = {}
    for bname, gname in blender_group_pairs:
        if bname not in blender_groups:
            blender_groups[bname] = Counter()
        blender_groups[bname][gname] += 1
    blender_to_group_idx = {}
    for bname, counter in blender_groups.items():
        most_common = counter.most_common(1)[0][0]
        blender_to_group_idx[bname] = group_to_idx.get(most_common, group_to_idx.get('unknown', 0))

    vocab_data['blender_to_group'] = blender_to_group_idx
    g = vocab_data['groups']
    num_groups = len(g.get('vocab', list(g.get('to_idx', {}))))
    g['size'] = num_groups
    print(f"  ✓ 그룹 vocab: {num_groups}개, blender→group 매핑: {len(blender_to_group_idx):,}개")
    return num_groups


def create_recipe_combinations(
    records: List[Dict],
    vocab_data: Dict,
    max_samples: int = DEFAULT_TRAIN_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
    use_augmentation: bool = True,
    molecule_idx_mapping: Optional[Dict[int, int]] = None,
    enable_cheat_key_detection: bool = True,
    enable_duplicate_filtering: bool = True,
    use_inverse_blender_sampling: bool = True
) -> List[Dict]:
    """
    Create recipe combinations with Gaussian distribution (μ=6, σ=2).

    Constraints:
    - Recipe length: 2~10
    - Blender cap: <= 10% of total or 1000 samples
    - Blender floor: >= 20 samples if possible
    - Near-duplicate removal within blender (>=80% overlap)
    """
    combinations = []

    molecule_blenders = {}
    blender_molecule_pools = {}
    molecule_indices = []

    blender_name_to_idx = {}
    if 'blenders' in vocab_data:
        blender_data = vocab_data['blenders']
        if isinstance(blender_data, list):
            for idx, blender_name in enumerate(blender_data):
                if isinstance(blender_name, str):
                    blender_name_to_idx[blender_name.lower()] = idx
        elif isinstance(blender_data, dict):
            if 'vocab' in blender_data:
                blender_list = blender_data['vocab']
                if isinstance(blender_list, list):
                    for idx, blender_name in enumerate(blender_list):
                        if isinstance(blender_name, str):
                            blender_name_to_idx[blender_name.lower()] = idx
            elif 'to_idx' in blender_data:
                blender_name_to_idx = {k.lower(): v for k, v in blender_data['to_idx'].items()}
            else:
                for idx, blender_name in enumerate(blender_data.keys()):
                    blender_name_to_idx[blender_name.lower()] = idx
        print(f"  ✓ blender vocabulary: {len(blender_name_to_idx):,}개")
        if len(blender_name_to_idx) > 0:
            sample_blenders = list(blender_name_to_idx.keys())[:3]
            print(f"     샘플: {sample_blenders}")
    else:
        print("  ⚠️  경고: vocabulary에 'blenders' 키가 없습니다!")

    blender_to_group = vocab_data.get('blender_to_group', {})
    blender_idx_to_group = {}
    for bname, bidx in blender_name_to_idx.items():
        blender_idx_to_group[bidx] = blender_to_group.get(bname, 0)
    num_groups = max(blender_idx_to_group.values(), default=0) + 1
    if vocab_data.get('groups') and isinstance(vocab_data['groups'].get('size'), int):
        num_groups = vocab_data['groups']['size']

    for idx, record in enumerate(records):
        if 'molecules' in record and record['molecules']:
            continue
        elif 'blenders' not in record or not record.get('blenders'):
            continue
        molecule_indices.append(idx)
        blenders_for_molecule = set()
        blenders = record.get('blenders', [])
        if isinstance(blenders, list):
            for blender_item in blenders:
                if isinstance(blender_item, list) and len(blender_item) > 0:
                    blender_name = blender_item[0].lower()
                elif isinstance(blender_item, str):
                    blender_name = blender_item.lower()
                else:
                    continue
                blender_idx = blender_name_to_idx.get(blender_name)
                if blender_idx is not None:
                    blenders_for_molecule.add(blender_idx)
                    if blender_idx not in blender_molecule_pools:
                        blender_molecule_pools[blender_idx] = set()
                    blender_molecule_pools[blender_idx].add(idx)
        if blenders_for_molecule:
            molecule_blenders[idx] = blenders_for_molecule

    if not molecule_indices:
        print("\n  ❌ 에러: 유효한 분자를 찾을 수 없습니다!")
        print(f"     - 총 레코드 수: {len(records)}")
        print(f"     - blender가 있는 레코드 수: {len([r for r in records if r.get('blenders')])}")
        print(f"     - blender vocabulary 매핑 수: {len(blender_name_to_idx)}")
        sample_records_with_blenders = [r for r in records[:10] if r.get('blenders')]
        if sample_records_with_blenders and len(blender_name_to_idx) > 0:
            test_record = sample_records_with_blenders[0]
            test_blenders = test_record.get('blenders', [])
            if test_blenders:
                test_blender = test_blenders[0]
                if isinstance(test_blender, list):
                    test_blender_name = test_blender[0].lower() if len(test_blender) > 0 else None
                else:
                    test_blender_name = test_blender.lower() if isinstance(test_blender, str) else None
                if test_blender_name:
                    matched = test_blender_name in blender_name_to_idx
                    print(f"     - 매칭 테스트: '{test_blender_name}' -> {matched}")
        return combinations

    print(f"  ✓ 유효한 분자: {len(molecule_indices):,}개")
    print(f"  ✓ blender-분자 매핑: {len(molecule_blenders):,}개")

    cheat_key_scores = {}
    if enable_cheat_key_detection:
        print(f"\n  🔍 치트키 성분 탐지 중...")
        cheat_key_scores = detect_cheat_key_molecules(
            records, molecule_blenders, blender_name_to_idx, cheat_threshold=0.8
        )
        high_cheat_count = sum(1 for score in cheat_key_scores.values() if score >= 0.7)
        if high_cheat_count > 0:
            print(f"     ⚠️  치트키 의심 성분: {high_cheat_count:,}개 (점수 >= 0.7)")
        else:
            print(f"     ✓ 치트키 성분 없음")

    num_blenders = len(blender_name_to_idx)
    molecule_weights = compute_molecule_frequency_weights(
        records, molecule_blenders, num_blenders
    )

    blender_sample_counts = {}
    for idx in molecule_indices:
        for blender_id in molecule_blenders.get(idx, set()):
            blender_sample_counts[blender_id] = blender_sample_counts.get(blender_id, 0) + 1
    max_count = max(blender_sample_counts.values()) if blender_sample_counts else 1
    blender_weights = {}
    for blender_id, count in blender_sample_counts.items():
        blender_weights[blender_id] = min(5.0, max_count / max(count, 1))
    blender_combination_counts = {}
    max_per_blender = min(1000, int(max_samples * 0.10))
    min_per_blender = 20
    max_attempts = max_samples * 50
    np.random.seed(seed)
    blender_combo_signatures = {}

    def sample_recipe_length() -> int:
        length = int(np.round(np.random.normal(loc=6.0, scale=2.0)))
        return int(np.clip(length, 2, 10))

    def is_near_duplicate(signature: Tuple[int, ...], blender_id: int) -> bool:
        existing = blender_combo_signatures.get(blender_id, [])
        if not existing:
            blender_combo_signatures[blender_id] = [signature]
            return False
        sig_set = set(signature)
        for prev in existing:
            prev_set = set(prev)
            union = len(sig_set | prev_set)
            if union == 0:
                continue
            if len(sig_set & prev_set) / union >= 0.8:
                return True
        existing.append(signature)
        return False

    def create_combination(num_molecules, selected_indices=None, forced_blender_id: Optional[int] = None):
        nonlocal combinations, blender_combination_counts
        if len(molecule_indices) < num_molecules:
            return False
        if selected_indices is None:
            selected_indices = np.random.choice(molecule_indices, size=num_molecules, replace=False)
        if molecule_idx_mapping is not None:
            original_selected_indices = [molecule_idx_mapping.get(idx, idx) for idx in selected_indices]
        else:
            original_selected_indices = selected_indices
        common_blenders = None
        for idx in selected_indices:
            blenders = molecule_blenders.get(idx, set())
            if common_blenders is None:
                common_blenders = blenders.copy()
            else:
                common_blenders = common_blenders & blenders
        target_blenders_list = None
        selected_blender = None
        if common_blenders and len(common_blenders) > 0:
            target_blenders_list = list(common_blenders)
            blender_list = list(common_blenders)
            weights = [blender_weights.get(bid, 1.0) for bid in blender_list]
            current_counts = [blender_combination_counts.get(bid, 0) for bid in blender_list]
            max_current = max(current_counts) if current_counts else 1
            for i, count in enumerate(current_counts):
                if max_current > 0:
                    weights[i] *= (1.0 + (max_current - count) / max(max_current, 1))
            total_weight = sum(weights)
            if total_weight > 0:
                probabilities = [w / total_weight for w in weights]
                selected_blender = np.random.choice(blender_list, p=probabilities)
            else:
                selected_blender = np.random.choice(blender_list)
        else:
            all_blenders_in_selection = set()
            for idx in selected_indices:
                all_blenders_in_selection.update(molecule_blenders.get(idx, set()))
            if all_blenders_in_selection:
                if use_inverse_blender_sampling:
                    blender_list = list(all_blenders_in_selection)
                    weights = [blender_weights.get(bid, 1.0) for bid in blender_list]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        probabilities = [w / total_weight for w in weights]
                        selected_blender = np.random.choice(blender_list, p=probabilities)
                        target_blenders_list = [selected_blender]
                    else:
                        selected_blender = np.random.choice(blender_list)
                        target_blenders_list = [selected_blender]
                else:
                    selected_blender = None
                    target_blenders_list = None
                combined_pool = set()
                for blender_id in all_blenders_in_selection:
                    combined_pool.update(blender_molecule_pools.get(blender_id, set()))
                all_in_combined_pool = all(idx in combined_pool for idx in selected_indices)
                if all_in_combined_pool and len(combined_pool) >= num_molecules:
                    blender_pool_sizes = {bid: len(blender_molecule_pools.get(bid, set())) for bid in all_blenders_in_selection}
                    selected_blender = max(blender_pool_sizes.items(), key=lambda x: x[1])[0]
                    target_blenders_list = [selected_blender]
                elif len(combined_pool) >= num_molecules:
                    pool_list = list(combined_pool)
                    selected_indices = np.random.choice(pool_list, size=num_molecules, replace=False)
                    blender_counts = {}
                    for idx in selected_indices:
                        for blender_id in molecule_blenders.get(idx, set()):
                            blender_counts[blender_id] = blender_counts.get(blender_id, 0) + 1
                    if blender_counts:
                        selected_blender = max(blender_counts.items(), key=lambda x: x[1])[0]
                        target_blenders_list = [selected_blender]
                    else:
                        return False
                else:
                    return False
            else:
                return False
        if selected_blender is None:
            return False
        if forced_blender_id is not None:
            selected_blender = forced_blender_id
            target_blenders_list = [forced_blender_id]
        blender_idx = int(selected_blender)
        if blender_idx >= len(blender_name_to_idx):
            return False
        if any(idx >= len(records) or idx < 0 for idx in selected_indices):
            return False
        if use_augmentation:
            selected_records = []
            for idx in selected_indices:
                record = records[idx]
                if enable_cheat_key_detection and cheat_key_scores:
                    record = apply_cheat_key_masking(record, cheat_key_scores, idx, masking_probability=0.3)
                selected_records.append(add_gaussian_noise_to_recipe(record, noise_level=0.01))
        else:
            selected_records = [records[idx].copy() for idx in selected_indices]
        if target_blenders_list is not None:
            valid_blenders = [int(b) for b in target_blenders_list if int(b) < len(blender_name_to_idx)]
        else:
            valid_blenders = [blender_idx] if blender_idx < len(blender_name_to_idx) else []
        if not valid_blenders:
            return False
        for bid in valid_blenders:
            if blender_combination_counts.get(bid, 0) >= max_per_blender:
                return False
        original_molecule_ids = [int(idx) for idx in original_selected_indices]
        signature = tuple(sorted(original_molecule_ids))
        for bid in valid_blenders:
            if is_near_duplicate(signature, bid):
                return False
        target_group = blender_idx_to_group.get(blender_idx, 0)
        recipe = {
            'molecules': selected_records,
            'target_blender': blender_idx,
            'target_blenders': valid_blenders,
            'target_group': target_group,
            'original_molecule_ids': original_molecule_ids,
            'recipe_len': len(selected_records)
        }
        combinations.append(recipe)
        if target_blenders_list is not None:
            for bid in target_blenders_list:
                blender_combination_counts[bid] = blender_combination_counts.get(bid, 0) + 1
        else:
            blender_combination_counts[selected_blender] = blender_combination_counts.get(selected_blender, 0) + 1
        return True

    _is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    pbar = tqdm(total=max_samples, desc="조합 생성", unit="개", ncols=100, mininterval=0.5, maxinterval=2.0, file=sys.stderr, dynamic_ncols=True, disable=not _is_tty)
    attempts = 0
    last_log = 0
    log_interval = max(1, max_samples // 20)
    while len(combinations) < max_samples and attempts < max_attempts:
        num_molecules = sample_recipe_length()
        if len(molecule_indices) < num_molecules:
            attempts += 1
            continue
        if create_combination(num_molecules):
            n = len(combinations)
            pbar.update(1)
            if not _is_tty and n >= last_log + log_interval:
                print(f"\r조합 생성: {n:,}/{max_samples:,} ({100*n/max_samples:.0f}%)", end="", file=sys.stderr)
                last_log = n
        attempts += 1
    pbar.close()
    if not _is_tty and last_log > 0:
        print(file=sys.stderr)

    for blender_id, pool in blender_molecule_pools.items():
        current_count = blender_combination_counts.get(blender_id, 0)
        if current_count >= min_per_blender or len(pool) < 2:
            continue
        needed = min_per_blender - current_count
        pool_list = list(pool)
        for _ in range(needed):
            num_molecules = sample_recipe_length()
            if len(pool_list) < num_molecules:
                num_molecules = min(len(pool_list), num_molecules)
            if num_molecules < 2:
                break
            selected_indices = np.random.choice(pool_list, size=num_molecules, replace=False)
            create_combination(num_molecules, selected_indices=selected_indices, forced_blender_id=blender_id)

    if enable_duplicate_filtering:
        print(f"\n  🧹 중복 조합 필터링 중...")
        combinations = filter_duplicate_combinations(combinations, similarity_threshold=0.85)
    np.random.shuffle(combinations)

    total_created = len(combinations)
    if total_created > 0:
        actual_very_simple = sum(1 for c in combinations if len(c.get('molecules', [])) == 2)
        actual_simple = sum(1 for c in combinations if len(c.get('molecules', [])) == 3)
        actual_medium_low = sum(1 for c in combinations if 4 <= len(c.get('molecules', [])) <= 5)
        actual_medium_high = sum(1 for c in combinations if 6 <= len(c.get('molecules', [])) <= 7)
        actual_complex = sum(1 for c in combinations if 8 <= len(c.get('molecules', [])) <= 9)
        actual_very_complex = sum(1 for c in combinations if len(c.get('molecules', [])) >= 10)
        print(f"\n  📊 생성된 조합 비율 확인 (정규분포 μ=6, σ=2):")
        print(f"     총 생성: {total_created:,}개")
        print(f"     2개: {actual_very_simple:,}개 ({actual_very_simple/total_created*100:.1f}%)")
        print(f"     3개: {actual_simple:,}개 ({actual_simple/total_created*100:.1f}%)")
        print(f"     4~5개: {actual_medium_low:,}개 ({actual_medium_low/total_created*100:.1f}%)")
        print(f"     6~7개: {actual_medium_high:,}개 ({actual_medium_high/total_created*100:.1f}%)")
        print(f"     8~9개: {actual_complex:,}개 ({actual_complex/total_created*100:.1f}%)")
        print(f"     10개+: {actual_very_complex:,}개 ({actual_very_complex/total_created*100:.1f}%)")
    return combinations
