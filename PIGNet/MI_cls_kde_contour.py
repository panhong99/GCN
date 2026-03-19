import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.stats import gaussian_kde
import argparse
from tqdm.auto import trange
from matplotlib.colors import Normalize

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

def plot_kde_contour(layer_idx, model_name, dataset_name, process_type, kde_data, group_name=None):
    """
    Plot KDE contour for a single layer.
    """
    layer_data = kde_data[layer_idx]
    X = layer_data['X']
    Y = layer_data['Y']
    Z = layer_data['Z']
    mi_xt = layer_data['mi_xt']
    mi_ty = layer_data['mi_ty']
    
    vmin=-2
    vmax=2

    Z = np.clip(Z, vmin, vmax)
    fig, ax = plt.subplots(figsize=(10, 8))
        
    # Normalize와 levels 설정
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 21)

    # Contour plot with filled levels
    cmap_s = plt.cm.get_cmap('Reds').copy()
    cmap_s.set_bad('white')

    # Threshold 적용하여 masked array 생성
    threshold = 2e-1
    Z_masked = np.ma.masked_less_equal(Z, threshold)

    contour_filled = ax.contourf(
        X, Y, 
        Z_masked, 
        levels=levels, 
        cmap=cmap_s, 
        norm=norm
    )
        
    # Colorbar
    cbar = plt.colorbar(contour_filled, ax=ax)
    cbar.set_label('KDE Density', fontsize=12)
    
    # ax.set_xlim(0,2)
    # ax.set_ylim(0,2)

    ax.set_xlabel("I(X; T)", fontsize=13, fontweight='bold')
    ax.set_ylabel("I(T; Y)", fontsize=13, fontweight='bold')
    
    # Title with group info if available
    if group_name:
        title = f"Layer {layer_idx} - {group_name} - KDE Contour (Classification)"
    else:
        title = f"Layer {layer_idx} - KDE Contour (Classification)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Filename with group info if available
    if group_name:
        fname = f"{model_name}_{dataset_name}_{process_type}_kde_contour_{group_name}_layer{layer_idx}.png"
    else:
        fname = f"{model_name}_{dataset_name}_{process_type}_kde_contour_layer{layer_idx}.png"
    
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Layer {layer_idx} KDE contour plot saved.")


def plot_kde_contour_all_layers_classification(all_layers_data, model_name, dataset_name, kde_data, num, process_type, group_name=None):
    """
    Plot KDE contours for all layers in a single figure with subplots.
    
    Args:
        all_layers_data: List of dictionaries containing layer MI data
        model_name: Name of the model
        dataset_name: Name of the dataset
        kde_data: Dictionary with KDE information for each layer
        num: Total number of layers
        process_type: Process type (e.g., 'layer', 'pixel')
        group_name: Optional group name (e.g., 'Backbone', 'GSP') for file naming
    """
    num_layers = len(all_layers_data)
    
    # Determine subplot dimensions (prefer wider layout)
    ncols = max(4, num - 1)
    nrows = (num_layers + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4*nrows))
    
    # Flatten axes array for easier iteration
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]
    
    # KDE parameters (shared across all subplots)
    vmin = 0
    vmax = 2
    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 21)
    threshold = 2e-1
    cmap_s = plt.cm.get_cmap('Reds').copy()
    cmap_s.set_bad('white')
    
    # Track contour objects for colorbar
    contour_objs = []
    
    for idx, layer_data in enumerate(all_layers_data):
        ax = axes_flat[idx]
        layer_idx = layer_data['layer_idx']
        
        # Get KDE data
        kde_layer = kde_data[layer_idx]
        X = kde_layer['X']
        Y = kde_layer['Y']
        Z = np.clip(kde_layer['Z'], vmin, vmax)
        Z_masked = np.ma.masked_less_equal(Z, threshold)
        
        # Plot contour
        contour = ax.contourf(X, Y, Z_masked, levels=levels, cmap=cmap_s, norm=norm)
        contour_objs.append(contour)
        
        # ax.set_xlim(0, 2)
        # ax.set_ylim(0, 2)
        ax.set_xlabel("I(X; T)", fontsize=11, fontweight='bold')
        ax.set_ylabel("I(T; Y)", fontsize=11, fontweight='bold')
        ax.set_title(f"Layer {layer_idx}", fontsize=12, fontweight='bold')
        
        # Remove tick labels to show clean subplot
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    # Remove empty subplots
    for idx in range(num_layers, len(axes_flat)):
        fig.delaxes(axes_flat[idx])
    
    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour_objs[0], cax=cbar_ax, norm=norm)
    cbar.set_label('KDE Density', fontsize=11, fontweight='bold')
    
    # Overall title with model and dataset info
    fig.suptitle(f"{model_name} - {dataset_name} - KDE Contour (Classification)", 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    # Include group name and process type in filename if provided
    if group_name:
        fname = f"{model_name}_{dataset_name}_{process_type}_kde_contour_all_layers_{group_name}.png"
    else:
        fname = f"{model_name}_{dataset_name}_{process_type}_kde_contour_all_layers.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"All layers combined KDE contour plot saved as: {fname}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR-10', 
                       help='Dataset name (e.g., Imagenet, CIFAR-10, CIFAR-100)')
    parser.add_argument('--model', type=str, default='PIGNet_GSPonly_classification', 
                       help='Model name (e.g., vit, Resnet, PIGNet_GSPonly_classification)')
    parser.add_argument('--process_type', type=str, default='layer', 
                       help='Preprocessing type (e.g., pixel or layer)')
    parser.add_argument('--cluster_num', type=int, default=200,
                       help='Number of clusters used in MI data generation (e.g., 50, 100, 200)')
    args = parser.parse_args()
    
    if args.model == "Resnet" or args.model == "vit":
        layer_num = 5
    elif args.model == "PIGNet_GSPonly_classification":
        backbonenum, gsp_layer_num = 4, 5
    
    # Define paths and parameters first
    data_path = f'/home/hail/pan/HDD/MI_dataset/{args.dataset}/{args.process_type}_dataset/resnet101/pretrained/{args.model}/zoom/1'

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. Load or Compute MI
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    mi_cache_file = os.path.join(data_path, f'mi_analysis_cache_classification_cluster{args.cluster_num}.pkl')
    backbone_cache_file = os.path.join(data_path, f'mi_analysis_cache_backbone_classification_cluster{args.cluster_num}.pkl')
    gsp_cache_file = os.path.join(data_path, f'mi_analysis_cache_gsp_classification_cluster{args.cluster_num}.pkl')
    mi_cache_valid = False
    
    if os.path.exists(mi_cache_file):
        print(f"Loading cached MI data from {mi_cache_file}...")
        with open(mi_cache_file, 'rb') as f:
            all_layers_data = pickle.load(f)
        mi_cache_valid = True
        print("MI cache loaded successfully!\n")

    elif (args.model == "PIGNet_GSPonly_classification" and 
          os.path.exists(backbone_cache_file) and os.path.exists(gsp_cache_file)):
        print(f"Loading cached MI data from: {backbone_cache_file} and {gsp_cache_file}")
        with open(backbone_cache_file, 'rb') as f:
            backbone_layers_data = pickle.load(f)

        with open(gsp_cache_file, 'rb') as f:
            gsp_layers_data = pickle.load(f)
        # Combine for plotting later
        all_layers_data = backbone_layers_data + gsp_layers_data
        mi_cache_valid = True
        print("MI cache loaded successfully!\n")
    
    if not mi_cache_valid:
        print(f"Loading classification MI data from: {data_path}")
        
        if args.model != "PIGNet_GSPonly_classification":        
            with open(os.path.join(data_path, f'y_labels_{args.cluster_num}.pkl'), 'rb') as f:
                y_in = pickle.load(f)
            print(f"Loaded labels: {y_in.shape}")
            
            with open(os.path.join(data_path, f'layer_0_{args.cluster_num}.pkl'), 'rb') as f:
                x_in = pickle.load(f)
            print(f"Loaded input features: {x_in.shape}")
            
            t_in = []
            for i in range(1, layer_num):
                with open(os.path.join(data_path, f'layer_{i}_{args.cluster_num}.pkl'), 'rb') as f:
                    t_layer = pickle.load(f)
                    t_in.append(t_layer)
                    print(f"Loaded layer {i}: {t_layer.shape}")

            H_dim, W_dim = x_in.shape[1], x_in.shape[2]
                    
        else: # PIGNet_GSPonly_classification
            with open(os.path.join(data_path, f'y_labels_{args.cluster_num}.pkl'), 'rb') as f:
                y_in = pickle.load(f)
            print(f"Loaded labels: {y_in.shape}")
            
            with open(os.path.join(data_path, f'backbone_layer_0_{args.cluster_num}.pkl'), 'rb') as f:
                backbone_x_in = pickle.load(f)
            print(f"Loaded input features: {backbone_x_in.shape}")
            
            with open(os.path.join(data_path, f'gsp_layer_0_{args.cluster_num}.pkl'), 'rb') as f:
                gsp_x_in = pickle.load(f)
            print(f"Loaded input features: {gsp_x_in.shape}")
            
            backbone_t_in = []
            gsp_t_in = []

            for i in range(1, backbonenum):
                with open(os.path.join(data_path, f'backbone_layer_{i}_{args.cluster_num}.pkl'), 'rb') as f:
                    t_layer = pickle.load(f)
                    backbone_t_in.append(t_layer)
                    print(f"Loaded layer {i}: {t_layer.shape}")

            for i in range(1, gsp_layer_num):
                with open(os.path.join(data_path, f'gsp_layer_{i}_{args.cluster_num}.pkl'), 'rb') as f:
                    t_layer = pickle.load(f)
                    gsp_t_in.append(t_layer)
                    print(f"Loaded layer {i}: {t_layer.shape}")

            H_dim, W_dim = backbone_x_in.shape[1], backbone_x_in.shape[2]

        print(f"\n=== Computing MI for Classification ===")
        print(f"Feature map dimensions: {H_dim} x {W_dim}")
        
        if args.model != "PIGNet_GSPonly_classification":
            # Standard computation for other models
            print(f"Number of layers: {len(t_in)}")
            all_layers_data = []
                
            # Compute MI for each layer
            for layer_idx, t_layer in enumerate(t_in):
                print(f"\n--- Layer {layer_idx+1} ---")
                
                # Compute I(X; T)
                print("Computing I(X; T)...")
                mi_xt, euc_map = cal_mi_x_t(x_in, t_layer)
                
                # Compute I(T; Y)
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
                
                print(f"Layer {layer_idx+1} statistics:")
                print(f"  Data points: {len(layer_mi_xt)}")
                print(f"  I(X;T) range: [{layer_mi_xt.min():.4f}, {layer_mi_xt.max():.4f}]")
                print(f"  I(T;Y) range: [{layer_mi_ty.min():.4f}, {layer_mi_ty.max():.4f}]")
                print(f"  Distance range: [{layer_distance.min():.2f}, {layer_distance.max():.2f}]")

            # Save cache
            print(f"\nSaving computed data to {mi_cache_file}...")
            with open(mi_cache_file, 'wb') as f:
                pickle.dump(all_layers_data, f)
            print("Cache saved successfully!")
        
        else:  # PIGNet_GSPonly_classification
            backbone_layers_data = []
            gsp_layers_data = []

            print(f"Number of Backbone layers: {len(backbone_t_in[:backbonenum])}")
            print(f"Number of GSP layers: {len(gsp_t_in)}")
            
            # Process Backbone layers
            print(f"\n=== Computing MI for Backbone Layers ===")
            for layer_idx, t_layer in enumerate(backbone_t_in):
                print(f"\n--- Backbone Layer {layer_idx+1} ---")
                
                # Compute I(X; T)
                print("Computing I(X; T)...")
                mi_xt, euc_map = cal_mi_x_t(backbone_x_in, t_layer)
                
                # Compute I(T; Y)
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
                
                backbone_layers_data.append({
                    'layer_idx': layer_idx + 1,
                    'mi_xt': layer_mi_xt,
                    'mi_ty': layer_mi_ty,
                    'distance': layer_distance,
                    'group': 'Backbone',
                })
                
                print(f"Backbone Layer {layer_idx+1} statistics:")
                print(f"  Data points: {len(layer_mi_xt)}")
                print(f"  I(X;T) range: [{layer_mi_xt.min():.4f}, {layer_mi_xt.max():.4f}]")
                print(f"  I(T;Y) range: [{layer_mi_ty.min():.4f}, {layer_mi_ty.max():.4f}]")
                print(f"  Distance range: [{layer_distance.min():.2f}, {layer_distance.max():.2f}]")
            
            # Process GSP layers
            print(f"\n=== Computing MI for GSP Layers ===")
            for layer_idx, t_layer in enumerate(gsp_t_in):
                print(f"\n--- GSP Layer {layer_idx+1} ---")
                
                # Compute I(X; T)
                print("Computing I(X; T)...")
                mi_xt, euc_map = cal_mi_x_t(gsp_x_in, t_layer)
                
                # Compute I(T; Y)
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
                
                gsp_layers_data.append({
                    'layer_idx': layer_idx + 1,
                    'mi_xt': layer_mi_xt,
                    'mi_ty': layer_mi_ty,
                    'distance': layer_distance,
                    'group': 'GSP',
                })
                
                print(f"GSP Layer {layer_idx+1} statistics:")
                print(f"  Data points: {len(layer_mi_xt)}")
                print(f"  I(X;T) range: [{layer_mi_xt.min():.4f}, {layer_mi_xt.max():.4f}]")
                print(f"  I(T;Y) range: [{layer_mi_ty.min():.4f}, {layer_mi_ty.max():.4f}]")
                print(f"  Distance range: [{layer_distance.min():.2f}, {layer_distance.max():.2f}]")
            
            # Save cache
            print(f"\nSaving computed data to {mi_cache_file}...")

            with open(backbone_cache_file, 'wb') as f:
                pickle.dump(backbone_layers_data, f)
            print("Cache saved successfully!")

            with open(gsp_cache_file, 'wb') as f:
                pickle.dump(gsp_layers_data, f)
            print("Cache saved successfully!")
    

    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. Load or Compute KDE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    kde_cache_file = os.path.join(data_path, f'kde_cache_classification_cluster{args.cluster_num}.pkl')
    kde_backbone_cache_file = os.path.join(data_path, f'kde_cache_backbone_classification_cluster{args.cluster_num}.pkl')
    kde_gsp_cache_file = os.path.join(data_path, f'kde_cache_gsp_classification_cluster{args.cluster_num}.pkl')
    
    kde_cache_valid = False
    
    # For standard models: single KDE cache file
    if args.model != "PIGNet_GSPonly_classification":
        # Check if KDE cache has all required layers
        expected_layers = [d['layer_idx'] for d in all_layers_data]
        required_keys = expected_layers[:]  # layer_idx values
        
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

    else:  # PIGNet_GSPonly_classification: separate backbone and gsp KDE caches
        # Separate all_layers_data into backbone and gsp
        backbone_layers_data = [d for d in all_layers_data if d.get('group') == 'Backbone']
        gsp_layers_data = [d for d in all_layers_data if d.get('group') == 'GSP']
        
        # Try to load both backbone and gsp KDE caches
        backbone_kde_valid = False
        gsp_kde_valid = False
        
        # Check backbone KDE cache
        if os.path.exists(kde_backbone_cache_file):
            print(f"Loading cached backbone KDE data from {kde_backbone_cache_file}...")
            with open(kde_backbone_cache_file, 'rb') as f:
                backbone_kde_data = pickle.load(f)
            
            backbone_expected = [d['layer_idx'] for d in backbone_layers_data]
            all_keys_present = True
            for layer_idx in backbone_expected:
                if layer_idx not in backbone_kde_data:
                    all_keys_present = False
                    break
                layer_kde = backbone_kde_data[layer_idx]
                if not all(key in layer_kde for key in ['X', 'Y', 'Z', 'mi_xt', 'mi_ty']):
                    all_keys_present = False
                    break
            
            if all_keys_present:
                print(f"✓ Backbone KDE data for {len(backbone_expected)} layers found!")
                backbone_kde_valid = True
            else:
                print(f"⚠ Backbone KDE layers missing in cache. Recomputing...")
        
        # Check GSP KDE cache
        if os.path.exists(kde_gsp_cache_file):
            print(f"Loading cached GSP KDE data from {kde_gsp_cache_file}...")
            with open(kde_gsp_cache_file, 'rb') as f:
                gsp_kde_data = pickle.load(f)
            
            gsp_expected = [d['layer_idx'] for d in gsp_layers_data]
            all_keys_present = True
            for layer_idx in gsp_expected:
                if layer_idx not in gsp_kde_data:
                    all_keys_present = False
                    break
                layer_kde = gsp_kde_data[layer_idx]
                if not all(key in layer_kde for key in ['X', 'Y', 'Z', 'mi_xt', 'mi_ty']):
                    all_keys_present = False
                    break
            
            if all_keys_present:
                print(f"✓ GSP KDE data for {len(gsp_expected)} layers found!")
                gsp_kde_valid = True
            else:
                print(f"⚠ GSP KDE layers missing in cache. Recomputing...")
        
        # Compute backbone KDE if needed
        if not backbone_kde_valid:
            print("\nComputing Backbone KDE values...")
            backbone_kde_data = compute_kde_values(backbone_layers_data)
            
            print(f"Saving Backbone KDE cache to {kde_backbone_cache_file}...")
            with open(kde_backbone_cache_file, 'wb') as f:
                pickle.dump(backbone_kde_data, f)
            print("Backbone KDE cache saved successfully!")
        else:
            print("Backbone KDE cache loaded successfully!")
        
        # Compute GSP KDE if needed
        if not gsp_kde_valid:
            print("\nComputing GSP KDE values...")
            gsp_kde_data = compute_kde_values(gsp_layers_data)
            
            print(f"Saving GSP KDE cache to {kde_gsp_cache_file}...")
            with open(kde_gsp_cache_file, 'wb') as f:
                pickle.dump(gsp_kde_data, f)
            print("GSP KDE cache saved successfully!")
        else:
            print("GSP KDE cache loaded successfully!")
        
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Plot
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13,
                         'axes.titlesize': 14, 'legend.fontsize': 11,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11})
    
    print("=== KDE Contour Plots (Classification) ===")
    
    if args.model != "PIGNet_GSPonly_classification":
        # Standard models: plot all layers together
        for layer_data in all_layers_data:
            layer_idx = layer_data['layer_idx']
            group_name = layer_data.get('group', None)
            
            if group_name:
                print(f"{group_name} - Layer {layer_idx}:")
            else:
                print(f"Layer {layer_idx}:")
            
            plot_kde_contour(layer_idx, args.model, args.dataset, args.process_type, kde_data, group_name)
        
        # Generate combined subplot plot for all layers
        print(f"\n=== Generating Combined Subplot Plot ===")
        total_layers = len(all_layers_data)
        plot_kde_contour_all_layers_classification(all_layers_data, args.model, args.dataset, kde_data, total_layers, args.process_type, group_name=None)
        
        print("\n=== Done! ===")
    
    else:  # PIGNet_GSPonly_classification: plot backbone and GSP separately
        # Separate all_layers_data into backbone and gsp
        backbone_layers_data = [d for d in all_layers_data if d.get('group') == 'Backbone']
        gsp_layers_data = [d for d in all_layers_data if d.get('group') == 'GSP']
        
        # Plot backbone layers
        print("\n=== Backbone Layers ===")
        for layer_data in backbone_layers_data:
            layer_idx = layer_data['layer_idx']
            print(f"Backbone - Layer {layer_idx}:")
            plot_kde_contour(layer_idx, args.model, args.dataset, args.process_type, backbone_kde_data, 'Backbone')
        
        print(f"\n--- Generating Backbone Combined Subplot Plot ---")
        plot_kde_contour_all_layers_classification(backbone_layers_data, args.model, args.dataset, backbone_kde_data, len(backbone_layers_data), args.process_type, group_name='Backbone')
        
        # Plot GSP layers
        print("\n=== GSP Layers ===")
        for layer_data in gsp_layers_data:
            layer_idx = layer_data['layer_idx']
            print(f"GSP - Layer {layer_idx}:")
            plot_kde_contour(layer_idx, args.model, args.dataset, args.process_type, gsp_kde_data, 'GSP')
        
        print(f"\n--- Generating GSP Combined Subplot Plot ---")
        plot_kde_contour_all_layers_classification(gsp_layers_data, args.model, args.dataset, gsp_kde_data, len(gsp_layers_data), args.process_type, group_name='GSP')
        
        print("\n=== Done! ===")


