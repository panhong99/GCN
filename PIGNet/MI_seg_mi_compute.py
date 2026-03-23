"""
MI Computation Module
계산 및 캐시 관리 전담
"""

import numpy as np
import pickle
import os
from tqdm.auto import trange


def _entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    p = counts / np.maximum(1, counts.sum())
    return float(-np.sum(p * np.log2(p + eps)))


def cal_mi_x_t_conditional(x: np.ndarray,
                           t: np.ndarray,
                           y: np.ndarray,
                           h_bins: int = 51,
                           ignore_label: int = -1):
    """
    Compute MI map I(X_i; T_j) for all (j,i) pairs for both SAME and DIFF modes.
    Returns mi_same, mi_diff, euc_map
    """
    N, H, W = t.shape
    P = H * W
    eps = 1e-12
    alpha = 1e-3

    x_flat = x.reshape(N, -1).astype(np.int32) + 1   # 0..50
    t_flat = t.reshape(N, -1).astype(np.int32) + 1   # 0..50
    y_flat = y.reshape(N, -1).astype(np.int32)

    max_x = h_bins
    max_t = h_bins

    mi_map_same_flat = np.zeros((P, P), dtype=np.float32)
    mi_map_diff_flat = np.zeros((P, P), dtype=np.float32)

    # Precompute grid distances once
    grid = np.indices((H, W)).reshape(2, -1).T
    h_diff = grid[:, 0:1] - grid[:, 0:1].T
    w_diff = grid[:, 1:2] - grid[:, 1:2].T
    euc_map = np.sqrt(h_diff**2 + w_diff**2).reshape(H, W, H, W)
    man_map = (np.abs(h_diff) + np.abs(w_diff)).reshape(H, W, H, W)

    for j_t in trange(P, desc="MI(X;T) conditional=same/diff", leave=False):
        t_vec_all = t_flat[:, j_t]
        y_j = y_flat[:, j_t]

        for i_x in range(P):
            y_i = y_flat[:, i_x]
            valid_base = (y_i != ignore_label) & (y_j != ignore_label)

            # SAME mode
            valid_same = valid_base & (y_i == y_j)
            if np.any(valid_same):
                x_vec = x_flat[valid_same, i_x]
                t_vec = t_vec_all[valid_same]

                counts_x = np.bincount(x_vec, minlength=max_x).astype(np.float64) + alpha
                counts_t = np.bincount(t_vec, minlength=max_t).astype(np.float64) + alpha

                joint = t_vec.astype(np.int64) * max_x + x_vec.astype(np.int64)
                counts_joint = np.bincount(joint, minlength=max_t * max_x).astype(np.float64) + alpha

                h_x = _entropy_from_counts(counts_x, eps)
                h_t = _entropy_from_counts(counts_t, eps)
                h_joint = _entropy_from_counts(counts_joint, eps)

                mi = h_t + h_x - h_joint
                mi_map_same_flat[j_t, i_x] = float(mi)

            # DIFF mode
            valid_diff = valid_base & (y_i != y_j)
            if np.any(valid_diff):
                x_vec = x_flat[valid_diff, i_x]
                t_vec = t_vec_all[valid_diff]

                counts_x = np.bincount(x_vec, minlength=max_x).astype(np.float64) + alpha
                counts_t = np.bincount(t_vec, minlength=max_t).astype(np.float64) + alpha

                joint = t_vec.astype(np.int64) * max_x + x_vec.astype(np.int64)
                counts_joint = np.bincount(joint, minlength=max_t * max_x).astype(np.float64) + alpha

                h_x = _entropy_from_counts(counts_x, eps)
                h_t = _entropy_from_counts(counts_t, eps)
                h_joint = _entropy_from_counts(counts_joint, eps)

                mi = h_t + h_x - h_joint
                mi_map_diff_flat[j_t, i_x] = float(mi)

    mi_map_same = mi_map_same_flat.reshape(H, W, H, W)
    mi_map_diff = mi_map_diff_flat.reshape(H, W, H, W)
    return mi_map_same, mi_map_diff, euc_map


def cal_seg_mi_t_y_conditional(t: np.ndarray,
                               y: np.ndarray,
                               h_bins_t: int = 51,
                               num_classes_y: int = 21,
                               ignore_label: int = -1):
    """
    Compute MI map I(T_i; Y_j) for all (i,j) pairs for both SAME and DIFF modes.
    Returns mi_same, mi_diff, euc_map
    """
    N, H, W = t.shape
    P = H * W
    eps = 1e-12
    alpha = 1e-3

    t_flat = t.reshape(N, -1).astype(np.int32) + 1   # 0..50
    y_flat = y.reshape(N, -1).astype(np.int32)       # 0..C-1 or -1(ignore)

    max_t = h_bins_t
    max_y = num_classes_y

    mi_map_same_flat = np.zeros((P, P), dtype=np.float32)
    mi_map_diff_flat = np.zeros((P, P), dtype=np.float32)

    grid = np.indices((H, W)).reshape(2, -1).T
    h_diff = grid[:, 0:1] - grid[:, 0:1].T
    w_diff = grid[:, 1:2] - grid[:, 1:2].T
    euc_map = np.sqrt(h_diff**2 + w_diff**2).reshape(H, W, H, W)
    man_map = (np.abs(h_diff) + np.abs(w_diff)).reshape(H, W, H, W)

    for j_y in trange(P, desc="MI(T;Y) conditional=same/diff", leave=False):
        y_j_all = y_flat[:, j_y]

        for i_t in range(P):
            y_i_all = y_flat[:, i_t]

            valid_base = (y_i_all != ignore_label) & (y_j_all != ignore_label)

            # SAME mode
            valid_same = valid_base & (y_i_all == y_j_all)
            if np.any(valid_same):
                t_vec = t_flat[valid_same, i_t]
                y_vec = y_j_all[valid_same]
                y_vec = np.clip(y_vec, 0, max_y - 1).astype(np.int32)

                counts_t = np.bincount(t_vec, minlength=max_t).astype(np.float64) + alpha
                counts_y = np.bincount(y_vec, minlength=max_y).astype(np.float64) + alpha

                joint = t_vec.astype(np.int64) * max_y + y_vec.astype(np.int64)
                counts_joint = np.bincount(joint, minlength=max_t * max_y).astype(np.float64) + alpha

                h_t = _entropy_from_counts(counts_t, eps)
                h_y = _entropy_from_counts(counts_y, eps)
                h_joint = _entropy_from_counts(counts_joint, eps)

                mi = h_t + h_y - h_joint
                mi_map_same_flat[i_t, j_y] = float(mi)

            # DIFF mode
            valid_diff = valid_base & (y_i_all != y_j_all)
            if np.any(valid_diff):
                t_vec = t_flat[valid_diff, i_t]
                y_vec = y_j_all[valid_diff]
                y_vec = np.clip(y_vec, 0, max_y - 1).astype(np.int32)

                counts_t = np.bincount(t_vec, minlength=max_t).astype(np.float64) + alpha
                counts_y = np.bincount(y_vec, minlength=max_y).astype(np.float64) + alpha

                joint = t_vec.astype(np.int64) * max_y + y_vec.astype(np.int64)
                counts_joint = np.bincount(joint, minlength=max_t * max_y).astype(np.float64) + alpha

                h_t = _entropy_from_counts(counts_t, eps)
                h_y = _entropy_from_counts(counts_y, eps)
                h_joint = _entropy_from_counts(counts_joint, eps)

                mi = h_t + h_y - h_joint
                mi_map_diff_flat[i_t, j_y] = float(mi)

    mi_map_same = mi_map_same_flat.reshape(H, W, H, W)
    mi_map_diff = mi_map_diff_flat.reshape(H, W, H, W)
    return mi_map_same, mi_map_diff, euc_map


def compute_and_cache_mi(seg_file_path, x_in, t_in, y_in, ignore_label=-1):
    """
    캐시 확인 후 MI 계산 또는 로드
    
    Returns:
        distance, mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff, ignore_label
    """
    cache_file = os.path.join(seg_file_path, 'mi_analysis_cache_same_diff_condmi.pkl')
    
    # 캐시 확인
    if os.path.exists(cache_file):
        print(f"Loading cached MI data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        distance = cache_data['distance']
        mi_xt_same = cache_data['mi_xt_same']
        mi_ty_same = cache_data['mi_ty_same']
        mi_xt_diff = cache_data['mi_xt_diff']
        mi_ty_diff = cache_data['mi_ty_diff']
        ignore_label = cache_data['ignore_label']
        
        print("✓ Cache loaded successfully!\n")
        return distance, mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff, ignore_label
    
    else:
        print("Cache not found. Computing MI values...\n")
        
        # Containers: per-layer flattened arrays for SAME and DIFF
        all_distance = []
        all_mi_xt_same, all_mi_ty_same = [], []
        all_mi_xt_diff, all_mi_ty_diff = [], []

        for layer_idx, t_layer in enumerate(t_in):
            print(f"Layer {layer_idx+1}/{len(t_in)} computing MI...", end=" ", flush=True)
            
            mi_xt_same, mi_xt_diff, euc_map = cal_mi_x_t_conditional(x_in, t_layer, y_in, ignore_label=ignore_label)
            mi_ty_same, mi_ty_diff, _ = cal_seg_mi_t_y_conditional(t_layer, y_in, ignore_label=ignore_label)

            all_distance.append(euc_map.flatten())
            all_mi_xt_same.append(mi_xt_same.flatten())
            all_mi_ty_same.append(mi_ty_same.flatten())
            all_mi_xt_diff.append(mi_xt_diff.flatten())
            all_mi_ty_diff.append(mi_ty_diff.flatten())
            
            print("done")

        distance = np.array(all_distance)
        mi_xt_same = np.array(all_mi_xt_same)
        mi_ty_same = np.array(all_mi_ty_same)
        mi_xt_diff = np.array(all_mi_xt_diff)
        mi_ty_diff = np.array(all_mi_ty_diff)

        # Save cache
        print(f"\nSaving computed data to {cache_file}...")
        cache_data = {
            'distance': distance,
            'mi_xt_same': mi_xt_same,
            'mi_ty_same': mi_ty_same,
            'mi_xt_diff': mi_xt_diff,
            'mi_ty_diff': mi_ty_diff,
            'ignore_label': ignore_label,
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print("✓ Cache saved successfully!\n")
        
        return distance, mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff, ignore_label
