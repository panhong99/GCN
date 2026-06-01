"""
JE Computation Module for Segmentation
- cal_je_x_t_conditional  : H(X_i, T_j) per pixel pair, SAME / DIFF class mode
- cal_seg_je_t_y_conditional : H(T_i, Y_j) per pixel pair, SAME / DIFF class mode
- compute_and_cache_je    : cache-aware wrapper for JE computation
- compute_kde_values      : KDE over the JE scatter data
"""

import numpy as np
import pickle
import os
from scipy.stats import gaussian_kde
from tqdm.auto import trange


# ─── helpers ──────────────────────────────────────────────────────────────────
def _entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    p = counts / np.maximum(1, counts.sum())
    return float(-np.sum(p * np.log2(p + eps)))


def _euc_map(H, W):
    grid = np.indices((H, W)).reshape(2, -1).T
    return np.sqrt(
        (grid[:, 0:1] - grid[:, 0:1].T) ** 2 +
        (grid[:, 1:2] - grid[:, 1:2].T) ** 2
    ).reshape(H, W, H, W)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JE Computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cal_je_x_t_conditional(x, t, y, h_bins=51, ignore_label=-1):
    """
    H(X_i, T_j) for all (j, i) pixel pairs, SAME and DIFF class modes.
    Returns joint_same (H,W,H,W), joint_diff (H,W,H,W), euc_map (H,W,H,W)
    """
    N, H, W = t.shape
    P = H * W
    alpha = 1e-3
    eps   = 1e-12

    x_flat = x.reshape(N, -1).astype(np.int32) + 1
    t_flat = t.reshape(N, -1).astype(np.int32) + 1
    y_flat = y.reshape(N, -1).astype(np.int32)

    joint_same = np.zeros((P, P), dtype=np.float32)
    joint_diff = np.zeros((P, P), dtype=np.float32)

    for j_t in trange(P, desc="H(X,T) same/diff", leave=False):
        t_j = t_flat[:, j_t]
        y_j = y_flat[:, j_t]
        for i_x in range(P):
            y_i   = y_flat[:, i_x]
            valid = (y_i != ignore_label) & (y_j != ignore_label)
            for cond, out in [(y_i == y_j, joint_same), (y_i != y_j, joint_diff)]:
                mask = valid & cond
                if not np.any(mask):
                    continue
                xv = x_flat[mask, i_x]
                tv = t_j[mask]
                cj = np.bincount(tv.astype(np.int64) * h_bins + xv.astype(np.int64),
                                 minlength=h_bins * h_bins).astype(np.float64) + alpha
                out[j_t, i_x] = _entropy_from_counts(cj, eps)

    return joint_same.reshape(H, W, H, W), joint_diff.reshape(H, W, H, W), _euc_map(H, W)


def cal_seg_je_t_y_conditional(t, y, h_bins_t=51, num_classes_y=21, ignore_label=-1):
    """
    H(T_i, Y_j) for all (i, j) pixel pairs, SAME and DIFF class modes.
    Returns joint_same (H,W,H,W), joint_diff (H,W,H,W), euc_map (H,W,H,W)
    """
    N, H, W = t.shape
    P = H * W
    alpha = 1e-3
    eps   = 1e-12

    t_flat = t.reshape(N, -1).astype(np.int32) + 1
    y_flat = y.reshape(N, -1).astype(np.int32)

    joint_same = np.zeros((P, P), dtype=np.float32)
    joint_diff = np.zeros((P, P), dtype=np.float32)

    for j_y in trange(P, desc="H(T,Y) same/diff", leave=False):
        y_j = y_flat[:, j_y]
        for i_t in range(P):
            y_i   = y_flat[:, i_t]
            valid = (y_i != ignore_label) & (y_j != ignore_label)
            for cond, out in [(y_i == y_j, joint_same), (y_i != y_j, joint_diff)]:
                mask = valid & cond
                if not np.any(mask):
                    continue
                tv = t_flat[mask, i_t]
                yv = np.clip(y_j[mask], 0, num_classes_y - 1).astype(np.int32)
                cj = np.bincount(tv.astype(np.int64) * num_classes_y + yv.astype(np.int64),
                                 minlength=h_bins_t * num_classes_y).astype(np.float64) + alpha
                out[i_t, j_y] = _entropy_from_counts(cj, eps)

    return joint_same.reshape(H, W, H, W), joint_diff.reshape(H, W, H, W), _euc_map(H, W)


def compute_and_cache_je(seg_file_path, x_in, t_in, y_in,
                         ignore_label=-1, calcul_type='joint'):
    """
    Load JE cache if it exists, otherwise compute and save.
    Returns distance, je_xt_same, je_ty_same, je_xt_diff, je_ty_diff, ignore_label
    """
    cache_file = os.path.join(seg_file_path, f'analysis_cache_same_diff_{calcul_type}.pkl')

    if os.path.exists(cache_file):
        print(f"Loading JE cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            d = pickle.load(f)
        return (d['distance'], d['je_xt_same'], d['je_ty_same'],
                d['je_xt_diff'],  d['je_ty_diff'],  d['ignore_label'])

    print("JE cache not found — computing...")
    all_dist, all_xt_s, all_ty_s, all_xt_d, all_ty_d = [], [], [], [], []

    for idx, t_layer in enumerate(t_in):
        print(f"  Layer {idx+1}/{len(t_in)}...", end=" ", flush=True)
        xt_s, xt_d, euc = cal_je_x_t_conditional(x_in, t_layer, y_in, ignore_label=ignore_label)
        ty_s, ty_d, _   = cal_seg_je_t_y_conditional(t_layer, y_in, ignore_label=ignore_label)
        all_dist.append(euc.flatten())
        all_xt_s.append(xt_s.flatten())
        all_ty_s.append(ty_s.flatten())
        all_xt_d.append(xt_d.flatten())
        all_ty_d.append(ty_d.flatten())
        print("done")

    distance   = np.array(all_dist)
    je_xt_same = np.array(all_xt_s)
    je_ty_same = np.array(all_ty_s)
    je_xt_diff = np.array(all_xt_d)
    je_ty_diff = np.array(all_ty_d)

    cache = dict(distance=distance, je_xt_same=je_xt_same, je_ty_same=je_ty_same,
                 je_xt_diff=je_xt_diff, je_ty_diff=je_ty_diff, ignore_label=ignore_label)
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    print(f"JE cache saved: {cache_file}\n")

    return distance, je_xt_same, je_ty_same, je_xt_diff, je_ty_diff, ignore_label


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KDE Computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_kde_values(je_xt_same, je_ty_same, je_xt_diff, je_ty_diff,
                       distance, threshold=1e-3):
    """
    Compute KDE for each layer (full + per distance bin).
    Only values above threshold are used.
    Returns kde_data dict.
    """
    num_layers = je_xt_same.shape[0]
    dist_bins  = np.arange(0, 50, 10)

    # Shared grid from all non-zero values
    all_vals = np.concatenate([
        je_xt_same[je_xt_same > threshold], je_ty_same[je_ty_same > threshold],
        je_xt_diff[je_xt_diff > threshold], je_ty_diff[je_ty_diff > threshold],
    ])
    if len(all_vals) == 0:
        all_vals = np.concatenate([je_xt_same.ravel(), je_ty_same.ravel(),
                                   je_xt_diff.ravel(), je_ty_diff.ravel()])

    v_min, v_max = all_vals.min(), all_vals.max()
    margin = (v_max - v_min) * 0.1
    xi = np.linspace(v_min - margin, v_max + margin, 100)
    yi = np.linspace(v_min - margin, v_max + margin, 100)
    Xi, Yi = np.meshgrid(xi, yi)
    grid_pts = np.vstack([Xi.ravel(), Yi.ravel()])

    kde_data = {'Xi': Xi, 'Yi': Yi}

    def _kde_or_zeros(x_filt, y_filt):
        if (len(x_filt) > 1 and np.std(x_filt) > 1e-8 and np.std(y_filt) > 1e-8):
            try:
                return gaussian_kde(np.vstack([x_filt, y_filt]), bw_method=0.3)(grid_pts).reshape(Xi.shape)
            except np.linalg.LinAlgError:
                pass
        return np.zeros_like(Xi)

    print(f"\n=== Computing KDE (threshold={threshold}) ===")
    for layer_idx in trange(num_layers, desc="Layer", leave=False):
        xs_full = je_xt_same[layer_idx]
        ys_full = je_ty_same[layer_idx]
        xd_full = je_xt_diff[layer_idx]
        yd_full = je_ty_diff[layer_idx]
        dist_l  = distance[layer_idx]

        mask_s = (xs_full > threshold) & (ys_full > threshold)
        mask_d = (xd_full > threshold) & (yd_full > threshold)
        xs, ys = xs_full[mask_s], ys_full[mask_s]
        xd, yd = xd_full[mask_d], yd_full[mask_d]

        kde_data[f'layer_{layer_idx}'] = {
            'Z_s': _kde_or_zeros(xs, ys), 'Z_d': _kde_or_zeros(xd, yd),
            'je_xt_same': xs, 'je_ty_same': ys,
            'je_xt_diff': xd, 'je_ty_diff': yd,
            'distance': dist_l,
            'n_points_s': len(xs), 'n_points_d': len(xd),
        }

        for bin_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
            bin_mask_s = mask_s & (dist_l >= b_min) & (dist_l < b_max)
            bin_mask_d = mask_d & (dist_l >= b_min) & (dist_l < b_max)
            xs_b, ys_b = xs_full[bin_mask_s], ys_full[bin_mask_s]
            xd_b, yd_b = xd_full[bin_mask_d], yd_full[bin_mask_d]
            kde_data[f'layer_{layer_idx}_bin_{bin_idx}'] = {
                'Z_s': _kde_or_zeros(xs_b, ys_b), 'Z_d': _kde_or_zeros(xd_b, yd_b),
                'je_xt_same': xs_b, 'je_ty_same': ys_b,
                'je_xt_diff': xd_b, 'je_ty_diff': yd_b,
                'n_points_s': len(xs_b), 'n_points_d': len(xd_b),
            }

    print("KDE computation done!\n")
    return kde_data
