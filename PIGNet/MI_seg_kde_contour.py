import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from scipy.stats import gaussian_kde
import argparse
from tqdm.auto import trange
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

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
#  KDE 계산 함수 (분리)
# ──────────────────────────────────────────────────────────────────

def compute_kde_values(mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff, distance):
    """
    MI값들로부터 모든 layer의 KDE density values 계산 (distance bin별로 분리)
    
    Args:
        mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff: [num_layers, num_points]
        distance: [num_layers, num_points]
    
    Returns:
        kde_data: dictionary with KDE values for all layers and distance bins
    """
    num_layers = mi_xt_same.shape[0]
    kde_data = {}
    
    # Grid 생성 (모든 layer에서 동일)
    xi = np.linspace(0, 2, 100)
    yi = np.linspace(0, 2, 100)
    Xi, Yi = np.meshgrid(xi, yi)
    kde_data['Xi'] = Xi
    kde_data['Yi'] = Yi
    
    # Distance bin 정의
    dist_bins = np.arange(0, 50, 10)  # 0-10, 10-20, 20-30, 30-40
    
    print("\n=== Computing KDE Values (by distance bins) ===")
    for layer_idx in trange(num_layers, desc="Layer", leave=False):
        x_s_full = mi_xt_same[layer_idx]
        y_s_full = mi_ty_same[layer_idx]
        x_d_full = mi_xt_diff[layer_idx]
        y_d_full = mi_ty_diff[layer_idx]
        dist_layer = distance[layer_idx]
        
        # 전체 layer의 Z도 저장 (layer-wise plot용)
        if len(x_s_full) > 1:
            kde_s = gaussian_kde(np.vstack([x_s_full, y_s_full]), bw_method=0.3)
            Z_s_full = kde_s(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
        else:
            Z_s_full = np.zeros_like(Xi)
        
        if len(x_d_full) > 1:
            kde_d = gaussian_kde(np.vstack([x_d_full, y_d_full]), bw_method=0.3)
            Z_d_full = kde_d(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
        else:
            Z_d_full = np.zeros_like(Xi)
        
        kde_data[f'layer_{layer_idx}'] = {
            'Z_s': Z_s_full,
            'Z_d': Z_d_full,
            'distance': dist_layer,
            'n_points_s': len(x_s_full),
            'n_points_d': len(x_d_full),
            'mi_xt_same': x_s_full,
            'mi_ty_same': y_s_full,
            'mi_xt_diff': x_d_full,
            'mi_ty_diff': y_d_full,
        }
        
        # Distance bin별 KDE 계산
        for bin_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
            mask_s = (dist_layer >= b_min) & (dist_layer < b_max)
            mask_d = (dist_layer >= b_min) & (dist_layer < b_max)
            
            # SAME mode (bin별)
            x_s_bin = x_s_full[mask_s]
            y_s_bin = y_s_full[mask_s]
            if len(x_s_bin) > 1:
                kde_s_bin = gaussian_kde(np.vstack([x_s_bin, y_s_bin]), bw_method=0.3)
                Z_s_bin = kde_s_bin(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
            else:
                Z_s_bin = np.zeros_like(Xi)
            
            # DIFF mode (bin별)
            x_d_bin = x_d_full[mask_d]
            y_d_bin = y_d_full[mask_d]
            if len(x_d_bin) > 1:
                kde_d_bin = gaussian_kde(np.vstack([x_d_bin, y_d_bin]), bw_method=0.3)
                Z_d_bin = kde_d_bin(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
            else:
                Z_d_bin = np.zeros_like(Xi)
            
            kde_data[f'layer_{layer_idx}_bin_{bin_idx}'] = {
                'Z_s': Z_s_bin,
                'Z_d': Z_d_bin,
                'mi_xt_same': x_s_bin,
                'mi_ty_same': y_s_bin,
                'mi_xt_diff': x_d_bin,
                'mi_ty_diff': y_d_bin,
                'n_points_s': len(x_s_bin),
                'n_points_d': len(x_d_bin),
            }
    
    print("KDE computation done!\n")
    return kde_data

# ──────────────────────────────────────────────────────────────────
#  Plotting 함수들 (KDE contour)
# ──────────────────────────────────────────────────────────────────

def plot_scatter_same_diff(layer_idx, model_name, dataset_name, process_type, vmin, vmax, kde_data, median_same_x, median_same_y, median_diff_x, median_diff_y):
    """
    Cache에서 받은 KDE값을 이용해 SAME/DIFF plot 생성
    """
    Z_s = kde_data[f'layer_{layer_idx}']['Z_s']
    Z_d = kde_data[f'layer_{layer_idx}']['Z_d']
    Xi = kde_data['Xi']
    Yi = kde_data['Yi']
    n_s = kde_data[f'layer_{layer_idx}']['n_points_s']
    n_d = kde_data[f'layer_{layer_idx}']['n_points_d']
    
    print(f"\n  Layer {layer_idx+1}: SAME n={n_s}, DIFF n={n_d}")

    # density clip (핵심)
    Z_s_plot = np.clip(Z_s, vmin, vmax)
    Z_d_plot = np.clip(Z_d, vmin, vmax)

    # norm + levels 고정
    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 21)

    # Colormap 설정 (배경 흰색)
    cmap_s = plt.cm.get_cmap('Reds').copy()
    cmap_s.set_bad('white')
    cmap_d = plt.cm.get_cmap('Blues').copy()
    cmap_d.set_bad('white')

    # SAME plot
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')

    threshold = 1e-3
    Z_s_masked = np.ma.masked_less_equal(Z_s_plot, threshold)

    cf = ax.contourf(
        Xi, Yi,
        Z_s_masked,      # ← masked array 사용
        levels=levels,
        cmap=cmap_s,
        norm=norm,
    )
    
    # Median 포인트 표시
    ax.scatter(median_same_x, median_same_y, marker='*', s=150, c='lime', edgecolors='darkgreen', linewidth=2, zorder=5)
    
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('Density', fontsize=11)
    
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xlabel("I(X; T)", fontsize=12, fontweight='bold')
    ax.set_ylabel("I(T; Y)", fontsize=12, fontweight='bold')
    ax.set_title(f"Layer {layer_idx+1} - SAME Class KDE Contour", fontsize=13, fontweight='bold')
    # ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = f"{model_name}_{dataset_name}_{process_type}_kde_layer{layer_idx+1}_SAME.png"
    folder_path = f"./kde_imgs/{model_name}/{dataset_name}/{vmax}"
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ SAME saved")

    # DIFF plot
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')

    Z_d_masked = np.ma.masked_less_equal(Z_d_plot, threshold)

    cf = ax.contourf(
        Xi, Yi, 
        Z_d_masked,      # ← masked array 사용
        levels=levels, 
        cmap=cmap_d, 
        norm=norm
    )
    
    # Median 포인트 표시
    ax.scatter(median_diff_x, median_diff_y, marker='*', s=150, c='lime', edgecolors='darkgreen', linewidth=2, zorder=5)

    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('Density', fontsize=11)
    
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xlabel("I(X; T)", fontsize=12, fontweight='bold')
    ax.set_ylabel("I(T; Y)", fontsize=12, fontweight='bold')
    ax.set_title(f"Layer {layer_idx+1} - DIFF Class KDE Contour", fontsize=13, fontweight='bold')
    # ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = f"{model_name}_{dataset_name}_{process_type}_kde_layer{layer_idx+1}_{vmax}_DIFF.png"
    folder_path = f"./kde_imgs/{model_name}/{dataset_name}/{vmax}"
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')            
    plt.close()
    print(f"    ✓ DIFF saved")

def plot_scatter_with_distance_bins(layer_idx, model_name, dataset_name, process_type, vmin, vmax, kde_data):
    """
    Distance bin별로 계산된 KDE값을 이용해 거리 구간별 plot 생성 (SAME, DIFF 각각 개별 plot)
    """
    Xi = kde_data['Xi']
    Yi = kde_data['Yi']
    
    dist_bins = np.arange(0, 50, 10)  # 0-10, 10-20, 20-30, 30-40

    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 21)

    # Colormap 설정 (배경 흰색)
    cmap_s = plt.cm.get_cmap('Reds').copy()
    cmap_s.set_bad('white')
    cmap_d = plt.cm.get_cmap('Blues').copy()
    cmap_d.set_bad('white')
    
    threshold = 1e-3

    print(f"\n  Layer {layer_idx+1} (Distance-binned):")
    for bin_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
        cache_key = f'layer_{layer_idx}_bin_{bin_idx}'
        
        if cache_key not in kde_data:
            print(f"    dist [{b_min:.0f}–{b_max:.0f}): no data")
            continue
        
        bin_data = kde_data[cache_key]
        Z_s = bin_data['Z_s']
        Z_d = bin_data['Z_d']
        mi_xt_s = bin_data['mi_xt_same']
        mi_ty_s = bin_data['mi_ty_same']
        mi_xt_d = bin_data['mi_xt_diff']
        mi_ty_d = bin_data['mi_ty_diff']

        print(f"    dist [{b_min:.0f}–{b_max:.0f}): ", end="")

        # density clip
        Z_s_plot = np.clip(Z_s, vmin, vmax)
        Z_d_plot = np.clip(Z_d, vmin, vmax)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SAME 개별 plot
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        ax.set_facecolor('white')

        Z_s_masked = np.ma.masked_less_equal(Z_s_plot, threshold)
        cf_s = ax.contourf(Xi, Yi, Z_s_masked, levels=levels, cmap=cmap_s, norm=norm)
        
        # Median 포인트 표시 (같은 데이터에서 계산)
        if len(mi_xt_s) > 0:
            median_x = np.median(mi_xt_s)
            median_y = np.median(mi_ty_s)
            ax.scatter(median_x, median_y, marker='*', s=150, c='lime', edgecolors='darkgreen', linewidth=2, zorder=5)
        
        cbar_s = plt.colorbar(cf_s, ax=ax)
        cbar_s.set_label('Density', fontsize=11)
        
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xlabel("I(X; T)", fontsize=12, fontweight='bold')
        ax.set_ylabel("I(T; Y)", fontsize=12, fontweight='bold')
        ax.set_title(f"Layer {layer_idx+1} - SAME - Distance [{b_min:.0f}–{b_max:.0f})", 
                     fontsize=13, fontweight='bold')

        plt.tight_layout()
        fname = (f"{model_name}_{dataset_name}_{process_type}_kde_layer{layer_idx+1}"
                 f"_dist{int(b_min)}-{int(b_max)}_SAME_{vmax}.png")
        
        folder_path = f"./kde_imgs/{model_name}/{dataset_name}/{vmax}"
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')        
        plt.close()
        print("SAME saved, ", end="")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # DIFF 개별 plot
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        ax.set_facecolor('white')

        Z_d_masked = np.ma.masked_less_equal(Z_d_plot, threshold)
        cf_d = ax.contourf(Xi, Yi, Z_d_masked, levels=levels, cmap=cmap_d, norm=norm)
        
        # Median 포인트 표시 (같은 데이터에서 계산)
        if len(mi_xt_d) > 0:
            median_x = np.median(mi_xt_d)
            median_y = np.median(mi_ty_d)
            ax.scatter(median_x, median_y, marker='*', s=150, c='lime', edgecolors='darkgreen', linewidth=2, zorder=5)
        
        cbar_d = plt.colorbar(cf_d, ax=ax)
        cbar_d.set_label('Density', fontsize=11)
        
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xlabel("I(X; T)", fontsize=12, fontweight='bold')
        ax.set_ylabel("I(T; Y)", fontsize=12, fontweight='bold')
        ax.set_title(f"Layer {layer_idx+1} - DIFF - Distance [{b_min:.0f}–{b_max:.0f})", 
                     fontsize=13, fontweight='bold')

        plt.tight_layout()
        fname = (f"{model_name}_{dataset_name}_{process_type}_kde_layer{layer_idx+1}"
                 f"_dist{int(b_min)}-{int(b_max)}_DIFF_{vmax}.png")
        
        folder_path = f"./kde_imgs/{model_name}/{dataset_name}/{vmax}"
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')        
        plt.close()
        print("DIFF saved")


def plot_kde_matrix_same(model_name, dataset_name, vmin, vmax, kde_data, process_type):
    """
    Matrix plot: Layer (y축) x Distance (x축) for SAME mode
    4x4 grid (4 layers, 4 distance bins) - bin별 KDE 사용
    """
    Xi = kde_data['Xi']
    Yi = kde_data['Yi']
    layers = 4
    dist_bins = np.arange(0, 50, 10)  # 0-10, 10-20, 20-30, 30-40
    num_dist = len(dist_bins) - 1
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 21)
    cmap_s = plt.cm.get_cmap('Reds').copy()
    cmap_s.set_bad('white')
    threshold = 1e-3
    
    fig, axes = plt.subplots(layers, num_dist, figsize=(20, 18), facecolor='white')
    
    for layer_idx in range(layers):
        for dist_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
            ax = axes[layer_idx, dist_idx]
            ax.set_facecolor('white')
            
            cache_key = f'layer_{layer_idx}_bin_{dist_idx}'
            if cache_key not in kde_data:
                ax.text(1, 1, 'No data', ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(0, 2)
                ax.set_ylim(0, 2)
                continue
            
            bin_data = kde_data[cache_key]
            Z_s = bin_data['Z_s']
            mi_xt_s = bin_data['mi_xt_same']
            mi_ty_s = bin_data['mi_ty_same']
            
            Z_s_plot = np.clip(Z_s, vmin, vmax)
            Z_masked = np.ma.masked_less_equal(Z_s_plot, threshold)
            cf = ax.contourf(Xi, Yi, Z_masked, levels=levels, cmap=cmap_s, norm=norm)
            
            # Median 포인트 표시 (bin 데이터에서)
            if len(mi_xt_s) > 0:
                median_x = np.median(mi_xt_s)
                median_y = np.median(mi_ty_s)
                ax.scatter(median_x, median_y, marker='*', s=150, c='lime', edgecolors='darkgreen', linewidth=1.5, zorder=5)
            
            # Ticks 제거
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)
            
            # Y축 레이블 (좌측만)
            if dist_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx+1}", fontsize=11, fontweight='bold')
            
            # X축 레이블 (상단만)
            if layer_idx == 0:
                ax.set_title(f"Dist [{b_min:.0f}-{b_max:.0f})", fontsize=11, fontweight='bold')
    
    plt.suptitle(f"{model_name}_{dataset_name}_KDE Matrix - SAME Mode", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    folder_path = f"./kde_imgs/{model_name}/{dataset_name}/{vmax}"
    os.makedirs(folder_path, exist_ok=True)
    fname = f"{model_name}_{dataset_name}_{process_type}_kde_matrix_SAME_{vmax}.png"
    plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ SAME matrix plot saved: {fname}")


def plot_kde_matrix_diff(model_name, dataset_name, vmin, vmax, kde_data, process_type):
    """
    Matrix plot: Layer (y축) x Distance (x축) for DIFF mode
    4x4 grid (4 layers, 4 distance bins) - bin별 KDE 사용
    """
    Xi = kde_data['Xi']
    Yi = kde_data['Yi']
    layers = 4
    dist_bins = np.arange(0, 50, 10)  # 0-10, 10-20, 20-30, 30-40
    num_dist = len(dist_bins) - 1
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 21)
    cmap_d = plt.cm.get_cmap('Blues').copy()
    cmap_d.set_bad('white')
    threshold = 1e-3
    
    fig, axes = plt.subplots(layers, num_dist, figsize=(20, 18), facecolor='white')
    
    for layer_idx in range(layers):
        for dist_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
            ax = axes[layer_idx, dist_idx]
            ax.set_facecolor('white')
            
            cache_key = f'layer_{layer_idx}_bin_{dist_idx}'
            if cache_key not in kde_data:
                ax.text(1, 1, 'No data', ha='center', va='center', fontsize=12)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xlim(0, 2)
                ax.set_ylim(0, 2)
                continue
            
            bin_data = kde_data[cache_key]
            Z_d = bin_data['Z_d']
            mi_xt_d = bin_data['mi_xt_diff']
            mi_ty_d = bin_data['mi_ty_diff']
            
            Z_d_plot = np.clip(Z_d, vmin, vmax)
            Z_masked = np.ma.masked_less_equal(Z_d_plot, threshold)
            cf = ax.contourf(Xi, Yi, Z_masked, levels=levels, cmap=cmap_d, norm=norm)
            
            # Median 포인트 표시 (bin 데이터에서)
            if len(mi_xt_d) > 0:
                median_x = np.median(mi_xt_d)
                median_y = np.median(mi_ty_d)
                ax.scatter(median_x, median_y, marker='*', s=150, c='lime', edgecolors='darkgreen', linewidth=1.5, zorder=5)
            
            # Tick 레이블 제거 (marks는 유지)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)
            
            # Y축 레이블 (좌측만)
            if dist_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx+1}", fontsize=11, fontweight='bold')
            
            # X축 레이블 (상단만)
            if layer_idx == 0:
                ax.set_title(f"Dist [{b_min:.0f}-{b_max:.0f})", fontsize=11, fontweight='bold')
    
    plt.suptitle(f"{model_name}_{dataset_name}_KDE Matrix - DIFF Mode", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    folder_path = f"./kde_imgs/{model_name}/{dataset_name}/{vmax}"
    os.makedirs(folder_path, exist_ok=True)
    fname = f"{model_name}_{dataset_name}_{process_type}_kde_matrix_DIFF_{vmax}.png"
    plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ DIFF matrix plot saved: {fname}")

# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',         type=str, default='cityscape')
    parser.add_argument('--preprocess_type', type=str, default='layer', help='pixel or layer')
    parser.add_argument('--model',           type=str, default='ASPP')
    parser.add_argument('--vmin',            type=int, default=0)
    parser.add_argument('--vmax',            type=int, default=25)
    args = parser.parse_args()

    seg_file_path = (f"/home/hail/pan/HDD/MI_dataset/{args.preprocess_type}_dataset"
                     f"/{args.dataset}/resnet101/pretrained/{args.model}/zoom/1")

    # ── Cache 확인 ────────────────────────────────────────────────
    cache_path = os.path.join(seg_file_path, 'mi_analysis_cache_same_diff_contour.pkl')
    
    if os.path.exists(cache_path):
        print(f"Loading cached MI data from {cache_path}...")
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        distance = cache_data['distance']
        mi_xt_same = cache_data['mi_xt_same']
        mi_ty_same = cache_data['mi_ty_same']
        mi_xt_diff = cache_data['mi_xt_diff']
        mi_ty_diff = cache_data['mi_ty_diff']
        ignore_label = cache_data['ignore_label']
        
        median_same_x = np.median(mi_xt_same)
        median_same_y = np.median(mi_ty_same)

        median_diff_x = np.median(mi_xt_diff)
        median_diff_y = np.median(mi_ty_diff)
        
        print("Cache loaded successfully!\n")
    else:
        # ── Load ───────────────────────────────────────────────────────
        print("Cache not found. Computing MI values...")
        
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

        # ── Cache 저장 ──────────────────────────────────────────────────
        print(f"\nSaving computed data to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump({'distance': distance,
                         'mi_xt_same': mi_xt_same, 'mi_ty_same': mi_ty_same,
                         'mi_xt_diff': mi_xt_diff, 'mi_ty_diff': mi_ty_diff,
                         'ignore_label': ignore_label}, f)
        print("Cache saved successfully!\n")

    # ── Plot ───────────────────────────────────────────────────────
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13,
                         'axes.titlesize': 14, 'legend.fontsize': 11,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11})

    # ── KDE Cache 확인 ────────────────────────────────────────────────
    kde_cache_path = os.path.join(seg_file_path, 'kde_cache_contour.pkl')
    
    # 필요한 key들 정의
    required_keys = ['Xi', 'Yi']
    for layer_idx in range(distance.shape[0]):
        required_keys.append(f'layer_{layer_idx}')
        for bin_idx in range(4):  # 0-10, 10-20, 20-30, 30-40 = 4개 bin
            required_keys.append(f'layer_{layer_idx}_bin_{bin_idx}')
    
    # 캐시 파일 존재하면 로드
    kde_cache_valid = False
    if os.path.exists(kde_cache_path):
        print(f"Loading cached KDE data from {kde_cache_path}...")
        with open(kde_cache_path, 'rb') as f:
            kde_data = pickle.load(f)
        
        # 필요한 key들이 모두 있는지 확인
        if all(key in kde_data for key in required_keys):
            print("✓ All required keys found in cache!")
            kde_cache_valid = True
        else:
            print("⚠ Some keys missing in cache. Recomputing KDE values...")
            missing_keys = [k for k in required_keys if k not in kde_data]
            print(f"  Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"  Missing keys: {missing_keys}")
    
    # 캐시가 유효하지 않으면 KDE 계산
    if not kde_cache_valid:
        print("\nComputing KDE values...")
        kde_data = compute_kde_values(mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff, distance)
        
        # ── KDE Cache 저장 ──────────────────────────────────────────────────
        print(f"\nSaving KDE data to {kde_cache_path}...")
        with open(kde_cache_path, 'wb') as f:
            pickle.dump(kde_data, f)
        print("KDE cache saved successfully!\n")
    else:
        print("KDE cache loaded successfully!\n")

    print("=== KDE Contour Plots (SAME vs DIFF) ===")
    for li in range(distance.shape[0]):
        plot_scatter_same_diff(li, args.model, args.dataset, args.preprocess_type, args.vmin, args.vmax, kde_data, median_same_x, median_same_y, median_diff_x, median_diff_y)

    print("\n=== Distance-Binned KDE Contour Plots ===")
    for li in range(distance.shape[0]):
        plot_scatter_with_distance_bins(li, args.model, args.dataset, args.preprocess_type, args.vmin, args.vmax, kde_data)

    print("\n=== KDE Matrix Plots ===")
    plot_kde_matrix_same(args.model, args.dataset, args.vmin, args.vmax, kde_data, process_type=args.preprocess_type)
    plot_kde_matrix_diff(args.model, args.dataset, args.vmin, args.vmax, kde_data, process_type=args.preprocess_type)

    print("\n=== Done! ===")
