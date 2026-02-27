import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import cv2
from PIL import Image
import torch
import numpy as np
from tqdm.auto import trange
from scipy.interpolate import griddata
import argparse


def _mode_per_position(y_in: np.ndarray, ignore_label: int = 255) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-position mode label across batch.
    Returns:
        mode_flat: (H*W,) int labels (undefined positions set to ignore_label)
        valid_pos: (H*W,) bool positions with at least 1 non-ignore sample
    """
    N, H, W = y_in.shape
    y_flat = y_in.reshape(N, -1).astype(np.int32)
    P = y_flat.shape[1]
    mode = np.full(P, ignore_label, dtype=np.int32)
    valid = np.zeros(P, dtype=bool)

    for p in range(P):
        col = y_flat[:, p]
        if ignore_label is not None:
            col = col[col != ignore_label]
        if col.size == 0:
            continue
        counts = np.bincount(col)
        mode[p] = int(counts.argmax())
        valid[p] = True
    return mode, valid


def _entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    p = counts / np.maximum(1, counts.sum())
    return float(-np.sum(p * np.log2(p + eps)))


def cal_mi_x_t_conditional(x: np.ndarray,
                           t: np.ndarray,
                           y: np.ndarray,
                           h_bins: int = 51,
                           ignore_label: int = 255,
                           mode: str = "same"):
    """
    Compute MI map I(X_i; T_j) for all (j,i) pairs, but estimate probabilities
    using ONLY samples n where (Y_i == Y_j) [mode='same'] or (Y_i != Y_j) [mode='diff'].
    """
    assert mode in ("same", "diff")
    N, H, W = t.shape
    P = H * W
    eps = 1e-12
    alpha = 1e-3

    x_flat = x.reshape(N, -1).astype(np.int32) + 1
    t_flat = t.reshape(N, -1).astype(np.int32) + 1
    y_flat = y.reshape(N, -1).astype(np.int32)

    max_x = h_bins
    max_t = h_bins

    mi_map_flat = np.zeros((P, P), dtype=np.float32)

    grid = np.indices((H, W)).reshape(2, -1).T
    h_diff = grid[:, 0:1] - grid[:, 0:1].T
    w_diff = grid[:, 1:2] - grid[:, 1:2].T
    euc_map = np.sqrt(h_diff**2 + w_diff**2).reshape(H, W, H, W)
    man_map = (np.abs(h_diff) + np.abs(w_diff)).reshape(H, W, H, W)

    for j_t in trange(P, desc=f"MI(X;T) conditional={mode}", leave=False):
        t_vec_all = t_flat[:, j_t]
        y_j = y_flat[:, j_t]

        for i_x in range(P):
            y_i = y_flat[:, i_x]
            valid = (y_i != ignore_label) & (y_j != ignore_label)

            if mode == "same":
                valid = valid & (y_i == y_j)
            else:
                valid = valid & (y_i != y_j)

            if not np.any(valid):
                mi_map_flat[j_t, i_x] = 0.0
                continue

            x_vec = x_flat[valid, i_x]
            t_vec = t_vec_all[valid]

            counts_x = np.bincount(x_vec, minlength=max_x).astype(np.float64) + alpha
            counts_t = np.bincount(t_vec, minlength=max_t).astype(np.float64) + alpha

            joint = t_vec.astype(np.int64) * max_x + x_vec.astype(np.int64)
            counts_joint = np.bincount(joint, minlength=max_t * max_x).astype(np.float64) + alpha

            h_x = _entropy_from_counts(counts_x, eps)
            h_t = _entropy_from_counts(counts_t, eps)
            h_joint = _entropy_from_counts(counts_joint, eps)

            mi = h_t + h_x - h_joint
            mi_map_flat[j_t, i_x] = max(0.0, float(mi))

    mi_map = mi_map_flat.reshape(H, W, H, W)
    return mi_map, euc_map, man_map


def cal_seg_mi_t_y_conditional(t: np.ndarray,
                               y: np.ndarray,
                               h_bins_t: int = 51,
                               num_classes_y: int = 21,
                               ignore_label: int = 255,
                               mode: str = "same"):
    """
    Compute MI map I(T_i; Y_j) for all (i,j) pairs, but estimate probabilities using ONLY samples n where:
      - Y_i and Y_j are both valid (!= ignore_label)
      - and (Y_i == Y_j) if mode='same' else (Y_i != Y_j)
    """
    assert mode in ("same", "diff")
    N, H, W = t.shape
    P = H * W
    eps = 1e-12
    alpha = 1e-3

    t_flat = t.reshape(N, -1).astype(np.int32) + 1
    y_flat = y.reshape(N, -1).astype(np.int32)

    max_t = h_bins_t
    max_y = num_classes_y

    mi_map_flat = np.zeros((P, P), dtype=np.float32)

    grid = np.indices((H, W)).reshape(2, -1).T
    h_diff = grid[:, 0:1] - grid[:, 0:1].T
    w_diff = grid[:, 1:2] - grid[:, 1:2].T
    euc_map = np.sqrt(h_diff**2 + w_diff**2).reshape(H, W, H, W)
    man_map = (np.abs(h_diff) + np.abs(w_diff)).reshape(H, W, H, W)

    for j_y in trange(P, desc=f"MI(T;Y) conditional={mode}", leave=False):
        y_j_all = y_flat[:, j_y]

        for i_t in range(P):
            y_i_all = y_flat[:, i_t]

            valid = (y_i_all != ignore_label) & (y_j_all != ignore_label)
            if mode == "same":
                valid = valid & (y_i_all == y_j_all)
            else:
                valid = valid & (y_i_all != y_j_all)

            if not np.any(valid):
                mi_map_flat[i_t, j_y] = 0.0
                continue

            t_vec = t_flat[valid, i_t]
            y_vec = y_j_all[valid]

            y_vec = np.clip(y_vec, 0, max_y - 1).astype(np.int32)

            counts_t = np.bincount(t_vec, minlength=max_t).astype(np.float64) + alpha
            counts_y = np.bincount(y_vec, minlength=max_y).astype(np.float64) + alpha

            joint = t_vec.astype(np.int64) * max_y + y_vec.astype(np.int64)
            counts_joint = np.bincount(joint, minlength=max_t * max_y).astype(np.float64) + alpha

            h_t = _entropy_from_counts(counts_t, eps)
            h_y = _entropy_from_counts(counts_y, eps)
            h_joint = _entropy_from_counts(counts_joint, eps)

            mi = h_t + h_y - h_joint
            mi_map_flat[i_t, j_y] = max(0.0, float(mi))

    mi_map = mi_map_flat.reshape(H, W, H, W)
    return mi_map, euc_map, man_map


def _interpolate_with_smoothing(points, values, grid_resolution=150):
    """
    Interpolate scattered points to a regular grid with smoothing.
    Uses cubic interpolation and applies Gaussian smoothing for better contours.
    """
    from scipy.ndimage import gaussian_filter
    
    xi = np.linspace(0, 4, grid_resolution)
    yi = np.linspace(0, 4, grid_resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Use cubic interpolation for smoother results
    Zi = griddata(points, values, (Xi, Yi), method='cubic', fill_value=np.nanmean(values))
    
    # Apply Gaussian smoothing to fill NaN and smooth the field
    mask = ~np.isnan(Zi)
    if np.sum(mask) > 0:
        Zi_smooth = gaussian_filter(np.nan_to_num(Zi, nan=np.nanmean(values)), sigma=1.5)
    else:
        Zi_smooth = Zi
    
    return Xi, Yi, Zi_smooth


def plot_contour_same_diff(mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff, 
                           distance, layer_idx, model_name, dataset_name,
                           grid_resolution=150, min_points=5):
    """
    Plot filled contour map for SAME and DIFF separately in Information Plane.
    Uses cubic interpolation and Gaussian smoothing for smooth, beautiful contours.
    
    Args:
        mi_xt_same, mi_ty_same: SAME mode MI values (flattened)
        mi_xt_diff, mi_ty_diff: DIFF mode MI values (flattened)
        distance: Euclidean distance values (flattened)
        layer_idx: Layer index
        model_name: Model name for file saving
        dataset_name: Dataset name for file saving
        grid_resolution: Resolution of interpolation grid (default 150 for smooth contours)
        min_points: Minimum number of points required for interpolation
    """
    
    # Plot SAME
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare points for SAME
    points_same = np.column_stack([mi_xt_same, mi_ty_same])
    values_same = distance
    
    # Check if we have enough points for interpolation
    if len(points_same) >= min_points:
        try:
            # Interpolate with smoothing
            Xi, Yi, Zi_same = _interpolate_with_smoothing(points_same, values_same, grid_resolution)
            
            # Create filled contour plot for SAME
            contour_levels = np.linspace(np.min(values_same), np.max(values_same), 20)
            cf_same = ax.contourf(Xi, Yi, Zi_same, levels=contour_levels, cmap='Reds', alpha=0.85)
            cs_same = ax.contour(Xi, Yi, Zi_same, levels=contour_levels[::2], colors='darkred', linewidths=0.7, alpha=0.5)
            
            cbar_same = plt.colorbar(cf_same, ax=ax)
            cbar_same.set_label('Euclidean Distance', fontsize=11)
            
            # Overlay actual data points
            scatter_same = ax.scatter(mi_xt_same, mi_ty_same, c=distance, cmap='Reds', 
                                     s=30, alpha=0.5, edgecolors='darkred', linewidth=0.5)
        except Exception as e:
            print(f"Warning: Interpolation failed for SAME at layer {layer_idx+1}: {str(e)[:50]}")
            sc_same = ax.scatter(mi_xt_same, mi_ty_same, c=distance, cmap='Reds', s=30, alpha=0.6, edgecolors='darkred', linewidth=0.5)
            cbar_same = plt.colorbar(sc_same, ax=ax)
            cbar_same.set_label('Euclidean Distance', fontsize=11)
    else:
        # Not enough points
        sc_same = ax.scatter(mi_xt_same, mi_ty_same, c=distance, cmap='Reds', s=30, alpha=0.6, edgecolors='darkred', linewidth=0.5)
        cbar_same = plt.colorbar(sc_same, ax=ax)
        cbar_same.set_label('Euclidean Distance', fontsize=11)
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xlabel("I(X; T)", fontsize=12, fontweight='bold')
    ax.set_ylabel("I(T; Y)", fontsize=12, fontweight='bold')
    ax.set_title(f"Layer {layer_idx+1} - SAME Class Conditional Information Plane", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_{dataset_name}_contour_layer{layer_idx+1}_SAME.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Layer {layer_idx+1} SAME contour plot saved.")
    
    # Plot DIFF
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare points for DIFF
    points_diff = np.column_stack([mi_xt_diff, mi_ty_diff])
    values_diff = distance
    
    # Check if we have enough points for interpolation
    if len(points_diff) >= min_points:
        try:
            # Interpolate with smoothing
            Xi, Yi, Zi_diff = _interpolate_with_smoothing(points_diff, values_diff, grid_resolution)
            
            # Create filled contour plot for DIFF
            contour_levels_diff = np.linspace(np.min(values_diff), np.max(values_diff), 20)
            cf_diff = ax.contourf(Xi, Yi, Zi_diff, levels=contour_levels_diff, cmap='Blues', alpha=0.85)
            cs_diff = ax.contour(Xi, Yi, Zi_diff, levels=contour_levels_diff[::2], colors='darkblue', linewidths=0.7, alpha=0.5)
            
            cbar_diff = plt.colorbar(cf_diff, ax=ax)
            cbar_diff.set_label('Euclidean Distance', fontsize=11)
            
            # Overlay actual data points
            scatter_diff = ax.scatter(mi_xt_diff, mi_ty_diff, c=distance, cmap='Blues', 
                                     s=30, alpha=0.5, edgecolors='darkblue', linewidth=0.5)
        except Exception as e:
            print(f"Warning: Interpolation failed for DIFF at layer {layer_idx+1}: {str(e)[:50]}")
            sc_diff = ax.scatter(mi_xt_diff, mi_ty_diff, c=distance, cmap='Blues', s=30, alpha=0.6, edgecolors='darkblue', linewidth=0.5)
            cbar_diff = plt.colorbar(sc_diff, ax=ax)
            cbar_diff.set_label('Euclidean Distance', fontsize=11)
    else:
        # Not enough points
        sc_diff = ax.scatter(mi_xt_diff, mi_ty_diff, c=distance, cmap='Blues', s=30, alpha=0.6, edgecolors='darkblue', linewidth=0.5)
        cbar_diff = plt.colorbar(sc_diff, ax=ax)
        cbar_diff.set_label('Euclidean Distance', fontsize=11)
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xlabel("I(X; T)", fontsize=12, fontweight='bold')
    ax.set_ylabel("I(T; Y)", fontsize=12, fontweight='bold')
    ax.set_title(f"Layer {layer_idx+1} - DIFF Class Conditional Information Plane", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_{dataset_name}_contour_layer{layer_idx+1}_DIFF.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Layer {layer_idx+1} DIFF contour plot saved.")


def plot_contour_with_distance_bins(mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff, 
                                     distance, layer_idx, model_name, dataset_name,
                                     grid_resolution=150, min_points=5):
    """
    Plot contour maps with distance binning (10-unit intervals).
    Uses cubic interpolation and Gaussian smoothing for smooth contours.
    """
    from scipy.ndimage import gaussian_filter
    
    max_distance = np.max(distance) + 1
    distance_bins = np.arange(0, max_distance + 10, 10)
    
    for bin_idx in range(len(distance_bins) - 1):
        bin_min = distance_bins[bin_idx]
        bin_max = distance_bins[bin_idx + 1]
        
        mask_same = (distance >= bin_min) & (distance < bin_max)
        mask_diff = (distance >= bin_min) & (distance < bin_max)
        
        if not np.any(mask_same) and not np.any(mask_diff):
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot SAME
        if np.any(mask_same):
            points_same = np.column_stack([mi_xt_same[mask_same], mi_ty_same[mask_same]])
            values_same = distance[mask_same]
            
            # Check if we have enough points for interpolation
            if len(points_same) >= min_points:
                try:
                    # Interpolate with smoothing
                    Xi, Yi, Zi_same = _interpolate_with_smoothing(points_same, values_same, grid_resolution)
                    
                    contour_levels = np.linspace(bin_min, bin_max, 15)
                    cf_same = axes[0].contourf(Xi, Yi, Zi_same, levels=contour_levels, cmap='Reds', alpha=0.85)
                    cs_same = axes[0].contour(Xi, Yi, Zi_same, levels=contour_levels[::2], colors='darkred', linewidths=0.7, alpha=0.5)
                    
                    cbar_same = plt.colorbar(cf_same, ax=axes[0])
                    cbar_same.set_label('Euclidean Distance', fontsize=10)
                    
                    # Overlay actual data points
                    axes[0].scatter(mi_xt_same[mask_same], mi_ty_same[mask_same], 
                                   c=distance[mask_same], cmap='Reds', s=20, alpha=0.4, edgecolors='darkred', linewidth=0.3)
                except Exception as e:
                    print(f"  Warning: Interpolation failed for SAME bin [{bin_min}-{bin_max}), using scatter: {str(e)[:40]}")
                    sc_same = axes[0].scatter(mi_xt_same[mask_same], mi_ty_same[mask_same], 
                                             c=distance[mask_same], cmap='Reds', s=20, alpha=0.6, edgecolors='darkred', linewidth=0.5)
                    cbar_same = plt.colorbar(sc_same, ax=axes[0])
                    cbar_same.set_label('Euclidean Distance', fontsize=10)
            else:
                # Use scatter plot if insufficient points
                sc_same = axes[0].scatter(mi_xt_same[mask_same], mi_ty_same[mask_same], 
                                         c=distance[mask_same], cmap='Reds', s=20, alpha=0.6, edgecolors='darkred', linewidth=0.5)
                cbar_same = plt.colorbar(sc_same, ax=axes[0])
                cbar_same.set_label('Euclidean Distance', fontsize=10)
        
        axes[0].set_xlim(0, 4)
        axes[0].set_ylim(0, 4)
        axes[0].set_xlabel("I(X; T)", fontsize=11, fontweight='bold')
        axes[0].set_ylabel("I(T; Y)", fontsize=11, fontweight='bold')
        axes[0].set_title(f"SAME Class - Distance [{bin_min}-{bin_max})", fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Plot DIFF
        if np.any(mask_diff):
            points_diff = np.column_stack([mi_xt_diff[mask_diff], mi_ty_diff[mask_diff]])
            values_diff = distance[mask_diff]
            
            if len(points_diff) >= min_points:
                try:
                    Xi, Yi, Zi_diff = _interpolate_with_smoothing(points_diff, values_diff, grid_resolution)
                    
                    contour_levels_diff = np.linspace(bin_min, bin_max, 15)
                    cf_diff = axes[1].contourf(Xi, Yi, Zi_diff, levels=contour_levels_diff, cmap='Blues', alpha=0.85)
                    cs_diff = axes[1].contour(Xi, Yi, Zi_diff, levels=contour_levels_diff[::2], colors='darkblue', linewidths=0.7, alpha=0.5)
                    
                    cbar_diff = plt.colorbar(cf_diff, ax=axes[1])
                    cbar_diff.set_label('Euclidean Distance', fontsize=10)
                    
                    # Overlay actual data points
                    axes[1].scatter(mi_xt_diff[mask_diff], mi_ty_diff[mask_diff], 
                                   c=distance[mask_diff], cmap='Blues', s=20, alpha=0.4, edgecolors='darkblue', linewidth=0.3)
                except Exception as e:
                    print(f"  Warning: Interpolation failed for DIFF bin [{bin_min}-{bin_max}), using scatter: {str(e)[:40]}")
                    sc_diff = axes[1].scatter(mi_xt_diff[mask_diff], mi_ty_diff[mask_diff], 
                                             c=distance[mask_diff], cmap='Blues', s=20, alpha=0.6, edgecolors='darkblue', linewidth=0.5)
                    cbar_diff = plt.colorbar(sc_diff, ax=axes[1])
                    cbar_diff.set_label('Euclidean Distance', fontsize=10)
            else:
                sc_diff = axes[1].scatter(mi_xt_diff[mask_diff], mi_ty_diff[mask_diff], 
                                         c=distance[mask_diff], cmap='Blues', s=20, alpha=0.6, edgecolors='darkblue', linewidth=0.5)
                cbar_diff = plt.colorbar(sc_diff, ax=axes[1])
                cbar_diff.set_label('Euclidean Distance', fontsize=10)
        
        axes[1].set_xlim(0, 4)
        axes[1].set_ylim(0, 4)
        axes[1].set_xlabel("I(X; T)", fontsize=11, fontweight='bold')
        axes[1].set_ylabel("I(T; Y)", fontsize=11, fontweight='bold')
        axes[1].set_title(f"DIFF Class - Distance [{bin_min}-{bin_max})", fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f"Layer {layer_idx+1} - Contour Plot Comparison (Distance Bin: {bin_min}-{bin_max})", 
                     fontsize=13, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(f"{model_name}_{dataset_name}_contour_layer{layer_idx+1}_dist{int(bin_min)}-{int(bin_max)}.png",
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Layer {layer_idx+1} distance [{bin_min}-{bin_max}) plot saved.")


if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='pascal', help='pascal or cityscape')
    argparser.add_argument('--preprocess_type', type=str, default='layer', help='layer or pixel')
    argparser.add_argument('--model', type=str, default='ASPP', help='ASPP or PIGNet_GSPonly')
    args = argparser.parse_args()
    
    # Load data
    seg_file_path = f"/home/hail/pan/HDD/MI_dataset/{args.preprocess_type}_dataset/{args.dataset}/resnet101/pretrained/{args.model}/zoom/1"
    
    with open(os.path.join(seg_file_path, 'gt_labels.pkl'), 'rb') as f:
        y_in = pickle.load(f)
    
    if args.dataset.lower().startswith('city'):
        ignore_label = 255
    else:
        ignore_label = 255
        y_in = np.where(y_in == -1, 0, y_in)

    with open(os.path.join(seg_file_path, 'layer_0.pkl'), 'rb') as f:
        x_in = pickle.load(f)
    
    valid_per_sample = np.sum(y_in > 0, axis=(1, 2))
    print(f"\n=== GT Valid Points Statistics ===")
    print(f"Total samples: {y_in.shape[0]}")
    print(f"Valid points per sample - Min: {valid_per_sample.min()}, Max: {valid_per_sample.max()}, Mean: {valid_per_sample.mean():.1f}")
 
    t_in = []
    for i in range(1, 5):
        with open(os.path.join(seg_file_path, f'layer_{i}.pkl'), 'rb') as f:
            t_in.append(pickle.load(f))

    # Compute conditional MI
    all_distance = []
    all_mi_xt_same, all_mi_ty_same = [], []
    all_mi_xt_diff, all_mi_ty_diff = [], []

    for layer_idx, t_layer in enumerate(t_in):
        mi_xt_same, euc_map, _ = cal_mi_x_t_conditional(x_in, t_layer, y_in, ignore_label=ignore_label, mode="same")
        mi_ty_same, _, _ = cal_seg_mi_t_y_conditional(t_layer, y_in, ignore_label=ignore_label, mode="same")

        mi_xt_diff, _, _ = cal_mi_x_t_conditional(x_in, t_layer, y_in, ignore_label=ignore_label, mode="diff")
        mi_ty_diff, _, _ = cal_seg_mi_t_y_conditional(t_layer, y_in, ignore_label=ignore_label, mode="diff")

        all_distance.append(euc_map.flatten())
        all_mi_xt_same.append(mi_xt_same.flatten())
        all_mi_ty_same.append(mi_ty_same.flatten())
        all_mi_xt_diff.append(mi_xt_diff.flatten())
        all_mi_ty_diff.append(mi_ty_diff.flatten())

    distance = np.array(all_distance)
    mi_xt_same = np.array(all_mi_xt_same)
    mi_ty_same = np.array(all_mi_ty_same)
    mi_xt_diff = np.array(all_mi_xt_diff)
    mi_ty_diff = np.array(all_mi_ty_diff)

    # Save cache
    cache_file = os.path.join(seg_file_path, 'mi_analysis_cache_same_diff_contour.pkl')
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
    print("Cache saved successfully!")

    # Plot styling
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11

    # Plot contour: SAME vs DIFF (main)
    print("\n=== Generating Contour Plots (SAME vs DIFF) ===")
    for layer_idx in range(distance.shape[0]):
        plot_contour_same_diff(mi_xt_same[layer_idx], mi_ty_same[layer_idx],
                              mi_xt_diff[layer_idx], mi_ty_diff[layer_idx],
                              distance[layer_idx], layer_idx, args.model, args.dataset)

    # Plot contour with distance binning
    print("\n=== Generating Distance-Binned Contour Plots ===")
    for layer_idx in range(distance.shape[0]):
        plot_contour_with_distance_bins(mi_xt_same[layer_idx], mi_ty_same[layer_idx],
                                       mi_xt_diff[layer_idx], mi_ty_diff[layer_idx],
                                       distance[layer_idx], layer_idx, args.model, args.dataset)

    print("\n=== All plots generated successfully! ===")
