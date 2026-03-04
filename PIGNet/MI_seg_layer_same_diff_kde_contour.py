import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from scipy.stats import gaussian_kde
import argparse
from tqdm.auto import trange

# ──────────────────────────────────────────────────────────────────
#  MI 계산 함수 (scatter.py와 동일)
# ──────────────────────────────────────────────────────────────────

def _entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    """엔트로피 계산"""
    p = counts / np.maximum(1, counts.sum())
    return float(-np.sum(p * np.log2(p + eps)))

def cal_mi_x_t_conditional(x: np.ndarray,
                           t: np.ndarray,
                           y: np.ndarray,
                           h_bins: int = 51,
                           ignore_label: int = 255):
    """
    Compute MI map I(X_i; T_j) for all (j,i) pairs for both SAME and DIFF modes.
    Returns mi_same, mi_diff, euc_map
    
    ✓ scatter.py와 동일한 로직
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

            # SAME mode: Y_i == Y_j
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
                mi_map_same_flat[j_t, i_x] = max(0.0, float(mi))

            # DIFF mode: Y_i != Y_j
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
                mi_map_diff_flat[j_t, i_x] = max(0.0, float(mi))

    mi_map_same = mi_map_same_flat.reshape(H, W, H, W)
    mi_map_diff = mi_map_diff_flat.reshape(H, W, H, W)
    return mi_map_same, mi_map_diff, euc_map


def cal_seg_mi_t_y_conditional(t: np.ndarray,
                               y: np.ndarray,
                               h_bins_t: int = 51,
                               num_classes_y: int = 21,
                               ignore_label: int = 255):
    """
    Compute MI map I(T_i; Y_j) for all (i,j) pairs for both SAME and DIFF modes.
    Returns mi_same, mi_diff, euc_map
    
    ✓ scatter.py와 동일한 로직
    """
    N, H, W = t.shape
    P = H * W
    eps = 1e-12
    alpha = 1e-3

    t_flat = t.reshape(N, -1).astype(np.int32) + 1   # 0..50
    y_flat = y.reshape(N, -1).astype(np.int32)       # 0..C-1 or 255(ignore)

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

            # SAME mode: Y_i == Y_j
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
                mi_map_same_flat[i_t, j_y] = max(0.0, float(mi))

            # DIFF mode: Y_i != Y_j
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
                mi_map_diff_flat[i_t, j_y] = max(0.0, float(mi))

    mi_map_same = mi_map_same_flat.reshape(H, W, H, W)
    mi_map_diff = mi_map_diff_flat.reshape(H, W, H, W)
    return mi_map_same, mi_map_diff, euc_map


# ──────────────────────────────────────────────────────────────────
#  Plotting 함수들 (KDE contour)
# ──────────────────────────────────────────────────────────────────

_MAX_KDE_PTS = 100_000  # KDE 계산용 최대 포인트


def _subample_if_needed(x, y, max_pts=_MAX_KDE_PTS):
    """포인트가 많으면 서브샘플링"""
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    if len(x) > max_pts:
        idx = np.random.choice(len(x), max_pts, replace=False)
        x, y = x[idx], y[idx]
    return x, y


def plot_scatter_same_diff(mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff,
                           distance, layer_idx, model_name, dataset_name):
    """
    KDE Contour: SAME과 DIFF를 별도의 contour plot으로 그림
    (scatter .py와 다른 점: scatter 대신 KDE contour 사용)
    """
    # 데이터 준비
    x_s, y_s = _subample_if_needed(mi_xt_same, mi_ty_same)
    x_d, y_d = _subample_if_needed(mi_xt_diff, mi_ty_diff)

    # SAME plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if len(x_s) > 1:
        sns.kdeplot(x=x_s, y=y_s, ax=ax, fill=True, levels=20, cmap='Reds', thresh=0)
        sns.kdeplot(x=x_s, y=y_s, ax=ax, fill=False, levels=10, color='darkred', linewidths=0.7)
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xlabel("I(X; T)", fontsize=12, fontweight='bold')
    ax.set_ylabel("I(T; Y)", fontsize=12, fontweight='bold')
    ax.set_title(f"Layer {layer_idx+1} - SAME Class KDE Contour", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_{dataset_name}_kde_layer{layer_idx+1}_SAME.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Layer {layer_idx+1} SAME saved")

    # DIFF plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if len(x_d) > 1:
        sns.kdeplot(x=x_d, y=y_d, ax=ax, fill=True, levels=20, cmap='Blues', thresh=0)
        sns.kdeplot(x=x_d, y=y_d, ax=ax, fill=False, levels=10, color='darkblue', linewidths=0.7)
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xlabel("I(X; T)", fontsize=12, fontweight='bold')
    ax.set_ylabel("I(T; Y)", fontsize=12, fontweight='bold')
    ax.set_title(f"Layer {layer_idx+1} - DIFF Class KDE Contour", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_{dataset_name}_kde_layer{layer_idx+1}_DIFF.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Layer {layer_idx+1} DIFF saved")


def plot_scatter_with_distance_bins(mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff,
                                    distance, layer_idx, model_name, dataset_name):
    """
    거리 구간별 KDE Contour (10 단위)
    """
    distance = np.asarray(distance).ravel()
    bins = np.arange(0, np.max(distance) + 11, 10)

    for b_min, b_max in zip(bins[:-1], bins[1:]):
        mask = (distance >= b_min) & (distance < b_max)
        if not mask.any():
            continue

        # 데이터 필터링 + 서브샘플링
        x_s, y_s = _subample_if_needed(
            np.asarray(mi_xt_same).ravel()[mask],
            np.asarray(mi_ty_same).ravel()[mask]
        )
        x_d, y_d = _subample_if_needed(
            np.asarray(mi_xt_diff).ravel()[mask],
            np.asarray(mi_ty_diff).ravel()[mask]
        )

        if len(x_s) < 2 and len(x_d) < 2:
            continue

        # 2x1 subplot (SAME, DIFF)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # SAME
        if len(x_s) > 1:
            sns.kdeplot(x=x_s, y=y_s, ax=axes[0], fill=True, levels=20, cmap='Reds', thresh=0)
            sns.kdeplot(x=x_s, y=y_s, ax=axes[0], fill=False, levels=10, color='darkred', linewidths=0.7)
        
        axes[0].set_xlim(0, 4)
        axes[0].set_ylim(0, 4)
        axes[0].set_xlabel("I(X; T)", fontsize=11, fontweight='bold')
        axes[0].set_ylabel("I(T; Y)", fontsize=11, fontweight='bold')
        axes[0].set_title(f"SAME - Distance [{b_min:.0f}–{b_max:.0f})", fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # DIFF
        if len(x_d) > 1:
            sns.kdeplot(x=x_d, y=y_d, ax=axes[1], fill=True, levels=20, cmap='Blues', thresh=0)
            sns.kdeplot(x=x_d, y=y_d, ax=axes[1], fill=False, levels=10, color='darkblue', linewidths=0.7)
        
        axes[1].set_xlim(0, 4)
        axes[1].set_ylim(0, 4)
        axes[1].set_xlabel("I(X; T)", fontsize=11, fontweight='bold')
        axes[1].set_ylabel("I(T; Y)", fontsize=11, fontweight='bold')
        axes[1].set_title(f"DIFF - Distance [{b_min:.0f}–{b_max:.0f})", fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"Layer {layer_idx+1} - KDE (dist {b_min:.0f}–{b_max:.0f})",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        fname = (f"{model_name}_{dataset_name}_kde_layer{layer_idx+1}"
                 f"_dist{int(b_min)}-{int(b_max)}.png")
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → Layer {layer_idx+1} dist [{b_min:.0f}–{b_max:.0f}) saved")


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',         type=str, default='pascal')
    parser.add_argument('--preprocess_type', type=str, default='layer')
    parser.add_argument('--model',           type=str, default='ASPP')
    args = parser.parse_args()

    seg_file_path = (f"/home/hail/pan/HDD/MI_dataset/{args.preprocess_type}_dataset"
                     f"/{args.dataset}/resnet101/pretrained/{args.model}/zoom/1")

    # ── Load ───────────────────────────────────────────────────────
    with open(os.path.join(seg_file_path, 'gt_labels.pkl'), 'rb') as f:
        y_in = pickle.load(f)

    ignore_label = 255
    if not args.dataset.lower().startswith('city'):
        y_in = np.where(y_in == -1, 0, y_in)

    with open(os.path.join(seg_file_path, 'layer_0.pkl'), 'rb') as f:
        x_in = pickle.load(f)

    vps = np.sum(y_in > 0, axis=(1, 2))
    print(f"\n=== GT Valid Points ===")
    print(f"N={y_in.shape[0]}  min={vps.min()}  max={vps.max()}  mean={vps.mean():.1f}")

    t_in = []
    for i in range(1, 5):
        with open(os.path.join(seg_file_path, f'layer_{i}.pkl'), 'rb') as f:
            t_in.append(pickle.load(f))

    # ── Compute MI (scatter.py와 동일) ──────────────────────────────
    all_dist = []
    all_mi_xt_same, all_mi_ty_same = [], []
    all_mi_xt_diff, all_mi_ty_diff = [], []

    for layer_idx, t_layer in enumerate(t_in):
        print(f"Layer {layer_idx+1}/4 computing MI...", end=" ")
        
        mi_xt_s, mi_xt_d, euc_map = cal_mi_x_t_conditional(
            x_in, t_layer, y_in, ignore_label=ignore_label)
        mi_ty_s, mi_ty_d, _ = cal_seg_mi_t_y_conditional(
            t_layer, y_in, ignore_label=ignore_label)

        all_dist.append(euc_map.flatten())
        all_mi_xt_same.append(mi_xt_s.flatten())
        all_mi_ty_same.append(mi_ty_s.flatten())
        all_mi_xt_diff.append(mi_xt_d.flatten())
        all_mi_ty_diff.append(mi_ty_d.flatten())
        
        print("done")

    distance   = np.array(all_dist)
    mi_xt_same = np.array(all_mi_xt_same)
    mi_ty_same = np.array(all_mi_ty_same)
    mi_xt_diff = np.array(all_mi_xt_diff)
    mi_ty_diff = np.array(all_mi_ty_diff)

    # ── Cache ──────────────────────────────────────────────────────
    cache_path = os.path.join(seg_file_path, 'mi_analysis_cache_same_diff_contour.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump({'distance': distance,
                     'mi_xt_same': mi_xt_same, 'mi_ty_same': mi_ty_same,
                     'mi_xt_diff': mi_xt_diff, 'mi_ty_diff': mi_ty_diff,
                     'ignore_label': ignore_label}, f)
    print(f"Cache saved → {cache_path}\n")

    # ── Plot ───────────────────────────────────────────────────────
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13,
                         'axes.titlesize': 14, 'legend.fontsize': 11,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11})

    print("=== KDE Contour Plots (SAME vs DIFF) ===")
    for li in range(distance.shape[0]):
        plot_scatter_same_diff(mi_xt_same[li], mi_ty_same[li],
                               mi_xt_diff[li], mi_ty_diff[li],
                               distance[li], li, args.model, args.dataset)

    print("\n=== Distance-Binned KDE Contour Plots ===")
    for li in range(distance.shape[0]):
        plot_scatter_with_distance_bins(mi_xt_same[li], mi_ty_same[li],
                                        mi_xt_diff[li], mi_ty_diff[li],
                                        distance[li], li, args.model, args.dataset)

    print("\n=== Done! ===")
