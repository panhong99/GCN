import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.stats import gaussian_kde
import argparse
from tqdm.auto import trange

def _entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    """Calculate entropy from count histogram."""
    p = counts / np.maximum(1, counts.sum())
    return float(-np.sum(p * np.log2(p + eps)))


def cal_mi_x_t(x, t):
    """
    Calculate pairwise Mutual Information I(X_i; T_j) for all pixel pairs (i,j).
    Optimized vectorized version for classification task.
    """
    N, H, W = t.shape
    num_pixels = H * W
    eps = 1e-12
    
    x_flat = x.reshape(N, -1)
    t_flat = t.reshape(N, -1)
    
    # Pre-calculate H(X) and H(T) for all positions
    h_x_all = np.zeros(num_pixels)
    h_t_all = np.zeros(num_pixels)
    
    for i in range(num_pixels):
        _, counts_x = np.unique(x_flat[:, i], return_counts=True)
        _, counts_t = np.unique(t_flat[:, i], return_counts=True)
        h_x_all[i] = _entropy_from_counts(counts_x, eps)
        h_t_all[i] = _entropy_from_counts(counts_t, eps)
    
    # Calculate MI for all pixel pairs
    mi_map_flat = np.zeros((num_pixels, num_pixels))
    
    for i_t in trange(num_pixels, desc="MI(X;T) - Classification", leave=False):
        t_vec = t_flat[:, i_t]
        
        for i_x in range(num_pixels):
            x_vec = x_flat[:, i_x]
            
            # Joint entropy using vstack + 2D unique
            stack_tx = np.vstack((t_vec, x_vec))
            _, counts_tx = np.unique(stack_tx, axis=1, return_counts=True)
            h_joint = _entropy_from_counts(counts_tx, eps)
            
            # MI = H(T) + H(X) - H(T, X)
            mi_val = h_t_all[i_t] + h_x_all[i_x] - h_joint
            mi_map_flat[i_t, i_x] = max(0.0, float(mi_val))
    
    # Compute Euclidean distance map
    grid = np.indices((H, W)).reshape(2, -1).T
    h_diff = grid[:, 0:1] - grid[:, 0:1].T
    w_diff = grid[:, 1:2] - grid[:, 1:2].T
    euc_map = np.sqrt(h_diff**2 + w_diff**2).reshape(H, W, H, W)
    
    mi_map = mi_map_flat.reshape(H, W, H, W)
    
    return mi_map, euc_map


def cal_mi_t_y(t, y, eps=1e-12):
    """
    Calculate Mutual Information I(T; Y) per pixel.
    Y is global target (class label).
    """
    N, H, W = t.shape
    mi_map = np.zeros((H, W))
    
    # 1. Calculate H(Y) once
    y_counts = np.bincount(y.astype(np.int32))
    h_y = _entropy_from_counts(y_counts, eps)
    
    # 2. Pre-calculate H(T) for all positions
    t_flat = t.reshape(N, -1).astype(np.int32)
    num_pixels = H * W
    h_t_all = np.zeros(num_pixels)
    
    for i in range(num_pixels):
        _, counts_t = np.unique(t_flat[:, i], return_counts=True)
        h_t_all[i] = _entropy_from_counts(counts_t, eps)
    
    # 3. Calculate H(T, Y) and MI for each pixel
    offset_base = np.max(y) + 1
    
    for h in range(H):
        for w in range(W):
            t_pixel = t[:, h, w].astype(np.int32)
            
            # Scalarization: 1D encoding으로 (t_pixel, y) 쌍을 유니크하게 표현
            joint_encoded = t_pixel * offset_base + y.astype(np.int32)
            _, counts_joint = np.unique(joint_encoded, return_counts=True)
            h_joint = _entropy_from_counts(counts_joint, eps)
            
            # MI = H(T) + H(Y) - H(T, Y)
            pixel_idx = h * W + w
            mi_map[h, w] = h_t_all[pixel_idx] + h_y - h_joint
    
    return mi_map

def compute_kde_values(all_layers_data):
    """
    Compute KDE for all layers and store in a dictionary.
    Returns: dict with layer_idx as key, containing KDE values and original data.
    """
    kde_data = {}
    
    for layer_data in all_layers_data:
        layer_idx = layer_data['layer_idx']
        mi_xt = layer_data['mi_xt'].flatten()
        mi_ty = layer_data['mi_ty'].flatten()
        
        print(f"Computing KDE for Layer {layer_idx}...", end=" ")
        
        # Data validation & jitter adjustment
        eps = 1e-8
        
        xt_std = np.std(mi_xt)
        ty_std = np.std(mi_ty)
        
        if xt_std < eps:
            mi_xt = mi_xt + np.random.normal(0, eps * 10, mi_xt.shape)
            print(f"[mi_xt jitter added]", end=" ")
        if ty_std < eps:
            mi_ty = mi_ty + np.random.normal(0, eps * 10, mi_ty.shape)
            print(f"[mi_ty jitter added]", end=" ")
        
        # Stack for KDE input
        points = np.vstack([mi_xt, mi_ty])
        
        # Compute KDE with covariance factor adjustment
        try:
            kde = gaussian_kde(points)
        except np.linalg.LinAlgError:
            print(f"[KDE fatal error - using histogram instead]", end=" ")
            kde = None
        
        # Evaluate on grid
        x_min, x_max = mi_xt.min(), mi_xt.max()
        y_min, y_max = mi_ty.min(), mi_ty.max()
        
        # Prevent zero-range grids
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range < eps:
            x_range = 1.0
        if y_range < eps:
            y_range = 1.0
        
        x_grid = np.linspace(x_min - x_range * 0.1, x_max + x_range * 0.1, 100)
        y_grid = np.linspace(y_min - y_range * 0.1, y_max + y_range * 0.1, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        positions = np.vstack([X.ravel(), Y.ravel()])
        
        if kde is not None:
            Z = kde(positions).reshape(X.shape)
        else:
            Z = np.zeros_like(X)
            for i in range(len(mi_xt)):
                x_idx = np.argmin(np.abs(x_grid - mi_xt[i]))
                y_idx = np.argmin(np.abs(y_grid - mi_ty[i]))
                Z[y_idx, x_idx] += 1
            Z = Z / (np.max(Z) + eps)
        
        # Store KDE data including original MI values for later processing
        kde_data[layer_idx] = {
            'X': X, 'Y': Y, 'Z': Z,
            'mi_xt': mi_xt, 'mi_ty': mi_ty,
            'x_grid': x_grid, 'y_grid': y_grid,
            'n_points': len(mi_xt),
        }
        
        print("done")
    
    return kde_data

def plot_kde_contour(layer_idx, model_name, dataset_name, kde_data):
    """
    Plot KDE contour for a single layer.
    """
    layer_data = kde_data[layer_idx]
    X = layer_data['X']
    Y = layer_data['Y']
    Z = layer_data['Z']
    mi_xt = layer_data['mi_xt']
    mi_ty = layer_data['mi_ty']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Contour plot with filled levels
    cmap_s = plt.cm.get_cmap('Reds').copy()
    cmap_s.set_bad('white')

    contour_filled = ax.contourf(X, Y, Z, levels=15, cmap=cmap_s, alpha=0.8)
    # contour_lines = ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    # # Scatter points
    # scatter = ax.scatter(mi_xt, mi_ty, c='red', s=5, alpha=0.3, edgecolors='none', label='Data Points')
    
    # Colorbar
    cbar = plt.colorbar(contour_filled, ax=ax)
    cbar.set_label('KDE Density', fontsize=12)
    
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    ax.set_xlabel("I(X; T)", fontsize=13, fontweight='bold')
    ax.set_ylabel("I(T; Y)", fontsize=13, fontweight='bold')
    ax.set_title(f"Layer {layer_idx} - KDE Contour (Classification)", fontsize=14, fontweight='bold')
    # ax.grid(True, alpha=0.3, linestyle='--')
    # ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_{dataset_name}_kde_contour_layer{layer_idx}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Layer {layer_idx} KDE contour plot saved.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR-10', 
                       help='Dataset name (e.g., Imagenet, CIFAR-10, CIFAR-100)')
    parser.add_argument('--model', type=str, default='vit', 
                       help='Model name (e.g., vit, Resnet, PIGNet_GSPonly_classification)')
    args = parser.parse_args()
    
    # Define paths and parameters first
    data_path = f'/home/hail/pan/HDD/MI_dataset/{args.dataset}/layer_dataset/resnet101/pretrained/{args.model}/zoom/1'
    num = 7 if args.model == "PIGNet_GSPonly_classification" else 4

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. Load or Compute MI
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    mi_cache_file = os.path.join(data_path, 'mi_analysis_cache_classification.pkl')
    mi_cache_valid = False
    
    if os.path.exists(mi_cache_file):
        print(f"Loading cached MI data from {mi_cache_file}...")
        with open(mi_cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Check if all layers are present in cache
        expected_layers = num - 1  # layers 1 to num-1
        cached_layer_count = len([d for d in cached_data if isinstance(d, dict) and 'layer_idx' in d])
        
        if cached_layer_count == expected_layers:
            all_layers_data = cached_data
            mi_cache_valid = True
            print(f"✓ All {expected_layers} layers found in MI cache!")
        else:
            print(f"⚠ MI cache has {cached_layer_count} layers, expected {expected_layers}. Recomputing...")
    
    if not mi_cache_valid:
        print("Computing MI values...")
        
        # Load raw data
        with open(os.path.join(data_path, 'gt_labels.pkl'), 'rb') as f:
            y_in = pickle.load(f)
        print(f"Loaded labels: {y_in.shape}")
        
        with open(os.path.join(data_path, 'layer_0.pkl'), 'rb') as f:
            x_in = pickle.load(f)
        print(f"Loaded input features: {x_in.shape}")
        
        t_in = []
        for i in range(1, num):
            with open(os.path.join(data_path, f'layer_{i}.pkl'), 'rb') as f:
                t_layer = pickle.load(f)
                t_in.append(t_layer)
                print(f"Loaded layer {i}: {t_layer.shape}")
        
        H_dim, W_dim = x_in.shape[1], x_in.shape[2]
        all_layers_data = []
        
        # Compute MI for each layer
        for layer_idx, t_layer in enumerate(t_in):
            print(f"\n--- Computing MI for Layer {layer_idx+1} ---")
            
            print("Computing I(X; T)...")
            mi_xt, euc_map = cal_mi_x_t(x_in, t_layer)
            
            print("Computing I(T; Y)...")
            mi_ty = cal_mi_t_y(t_layer, y_in)
            
            # Flatten for scatter plot
            num_pixels = H_dim * W_dim
            
            layer_mi_xt = []
            layer_mi_ty = []
            layer_distance = []
            
            # For each reference pixel in T (ht, wt)
            for ht in range(H_dim):
                for wt in range(W_dim):
                    mi_xt_ref = mi_xt[ht, wt, :, :].flatten()
                    mi_ty_val = mi_ty[ht, wt]
                    mi_ty_ref = np.full_like(mi_xt_ref, mi_ty_val)
                    dist_ref = euc_map[ht, wt, :, :].flatten()
                    
                    layer_mi_xt.extend(mi_xt_ref.tolist())
                    layer_mi_ty.extend(mi_ty_ref.tolist())
                    layer_distance.extend(dist_ref.tolist())
            
            layer_mi_xt = np.array(layer_mi_xt)
            layer_mi_ty = np.array(layer_mi_ty)
            layer_distance = np.array(layer_distance)
            
            all_layers_data.append({
                'layer_idx': layer_idx + 1,
                'mi_xt': layer_mi_xt,
                'mi_ty': layer_mi_ty,
                'distance': layer_distance,
            })
            
            print(f"Layer {layer_idx+1} MI statistics:")
            print(f"  Data points: {len(layer_mi_xt)}")
            print(f"  I(X;T) range: [{layer_mi_xt.min():.4f}, {layer_mi_xt.max():.4f}]")
            print(f"  I(T;Y) range: [{layer_mi_ty.min():.4f}, {layer_mi_ty.max():.4f}]")
            print(f"  Distance range: [{layer_distance.min():.2f}, {layer_distance.max():.2f}]")
        
        # Save MI cache
        print(f"\nSaving MI cache to {mi_cache_file}...")
        with open(mi_cache_file, 'wb') as f:
            pickle.dump(all_layers_data, f)
        print("MI cache saved successfully!\n")
    else:
        print("MI cache loaded successfully!\n")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. Load or Compute KDE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    kde_cache_file = os.path.join(data_path, 'kde_cache_classification.pkl')
    
    # Check if KDE cache has all required layers
    expected_layers = [d['layer_idx'] for d in all_layers_data]
    required_keys = expected_layers[:]  # layer_idx values
    
    kde_cache_valid = False
    if os.path.exists(kde_cache_file):
        print(f"Loading cached KDE data from {kde_cache_file}...")
        with open(kde_cache_file, 'rb') as f:
            kde_data = pickle.load(f)
        
        # Check if all required layers are present and have necessary fields
        all_keys_present = True
        for layer_idx in required_keys:
            if layer_idx not in kde_data:
                all_keys_present = False
                break
            layer_kde = kde_data[layer_idx]
            if not all(key in layer_kde for key in ['X', 'Y', 'Z', 'mi_xt', 'mi_ty']):
                all_keys_present = False
                break
        
        if all_keys_present:
            print(f"✓ All required KDE data for {len(required_keys)} layers found!")
            kde_cache_valid = True
        else:
            print(f"⚠ Some KDE layers or fields missing in cache. Recomputing...")
    
    if not kde_cache_valid:
        print("\nComputing KDE values...")
        kde_data = compute_kde_values(all_layers_data)
        
        # Save KDE cache
        print(f"\nSaving KDE cache to {kde_cache_file}...")
        with open(kde_cache_file, 'wb') as f:
            pickle.dump(kde_data, f)
        print("KDE cache saved successfully!\n")
    else:
        print("KDE cache loaded successfully!\n")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Plot
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13,
                         'axes.titlesize': 14, 'legend.fontsize': 11,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11})
    
    print("=== KDE Contour Plots (Classification) ===")
    for layer_data in all_layers_data:
        layer_idx = layer_data['layer_idx']
        plot_kde_contour(layer_idx, args.model, args.dataset, kde_data)
    
    print("\n=== Done! ===")


