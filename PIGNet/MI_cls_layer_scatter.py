import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import cv2
from PIL import Image
import torch
import numpy as np
from tqdm.auto import trange
import argparse


def _entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    """Calculate entropy from count histogram."""
    p = counts / np.maximum(1, counts.sum())
    return float(-np.sum(p * np.log2(p + eps)))


def cal_mi_x_t(x, t):
    """
    Calculate pairwise Mutual Information I(X_i; T_j) for all pixel pairs (i,j).
    Optimized vectorized version for classification task (same as utils_cls_MI.py concept).
    
    ★ offset_base가 불필요한 이유 (cal_mi_t_y와의 차이):
    - vstack으로 2D 배열 생성: (2, N) 형태
    - axis=1 따라 unique하면 각 열 (t_val, x_val)이 자동으로 유니크
    - 따라서 1D encoding이 필요 없음
    
    반면 cal_mi_t_y에서:
    - 모든 샘플이 동일한 y값이므로 t_pixel만 변함
    - 1D로 encoding해야 하므로 offset_base 필수
    
    Args:
        x: (N, H, W) - Input feature maps
        t: (N, H, W) - Layer feature maps
        
    Returns:
        mi_map: (H, W, H, W) - MI[ht,wt,hx,wx] = I(X[hx,wx]; T[ht,wt])
        euc_map: (H, W, H, W) - Euclidean distances
    """
    N, H, W = t.shape
    num_pixels = H * W
    eps = 1e-12
    
    x_flat = x.reshape(N, -1)  # (N, H*W)
    t_flat = t.reshape(N, -1)  # (N, H*W)
    
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
            
            # Joint entropy using vstack + 2D unique (no offset_base needed)
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
    Y is global target (class label), so it does not have spatial dimensions.
    Optimized vectorized version following utils_cls_MI.py algorithm.
    
    ★ offset_base가 필요한 이유 (cal_mi_x_t와의 차이):
    
    cal_mi_x_t: vstack + 2D unique 사용
    - 각 (t_val, x_val)이 2D 배열의 각 열로 표현 → 자동으로 유니크
    - offset_base 불필요
    
    cal_mi_t_y: 1D encoding 사용 (Y 기준)
    - Y는 단일 클래스값, t_pixel은 0-49 범위의 이산화된 값
    - joint_encoded = t_pixel * offset_base + y로 1D 스칼라 생성
    - offset_base = max(Y) + 1로 해야 (T, Y) 쌍 모두 유니크:
      * t=0, y=0 → 0*1000+0 = 0
      * t=1, y=0 → 1*1000+0 = 1000       [Y 같으면 T로 구분]
      * t=0, y=1 → 0*1000+1 = 1          [T 같으면 Y로 구분]
      * t=49, y=999 → 49*1000+999 = 49999 [모든 쌍이 유니크]
    
    Args:
        t: (N, H, W) - Layer feature maps
        y: (N,) - Global class labels
        eps: Small value for numerical stability
        
    Returns:
        mi_map: (H, W) - MI[h,w] = I(T[h,w]; Y)
    """
    N, H, W = t.shape
    mi_map = np.zeros((H, W))
    
    # 1. Calculate H(Y) once (constant for all pixels)
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
    # offset_base: Y 기준으로 1D encoding에서 (t_pixel, y) 쌍을 유니크하게 구분
    # Y를 기준이므로: offset_base = max(Y) + 1
    # 예시: offset_base = 1000이면
    #   (t=0, y=0) → 0*1000+0 = 0
    #   (t=1, y=0) → 1*1000+0 = 1000     [Y 값이 같으면 T 값으로 구분]
    #   (t=0, y=1) → 0*1000+1 = 1        [T 값이 같으면 Y 값으로 구분]
    #   (t=49, y=999) → 49*1000+999 = 49999 [모든 쌍이 유니크]
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

def plot_scatter_classification(mi_xt, mi_ty, distance, layer_idx, model_name, dataset_name):
    """
    Plot scatter map in Information Plane for classification task.
    X-axis: I(X; T), Y-axis: I(T; Y), Color: Euclidean Distance
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(mi_xt, mi_ty, c=distance, cmap='viridis', 
                         s=20, alpha=0.6, edgecolors='none')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Euclidean Distance', fontsize=12)
    
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    ax.set_xlabel("I(X; T)", fontsize=13, fontweight='bold')
    ax.set_ylabel("I(T; Y)", fontsize=13, fontweight='bold')
    ax.set_title(f"Layer {layer_idx+1} - Information Plane (Classification)", 
                 fontsize=14, fontweight='bold')
    # ax.grid(True, alpha=0.3, linestyle='--')
        
    plt.tight_layout()
    plt.savefig(f"{model_name}_{dataset_name}_scatter_layer{layer_idx+1}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Layer {layer_idx+1} scatter plot saved.")

if __name__ == "__main__":  # Classification Task
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='CIFAR-10', 
                          help='Dataset name (e.g., Imagenet, CIFAR-100)')
    argparser.add_argument('--model', type=str, default='Resnet', 
                          help='Model name (e.g., Resnet, PIGNet_GSPonly_classification, vit)')
    args = argparser.parse_args()
    
    num = 7 if args.model == "PIGNet_GSPonly_classification" else 4
    
    # Setup data path
    data_path = f'/home/hail/pan/HDD/MI_dataset/{args.dataset}/layer_dataset/resnet101/pretrained/{args.model}/zoom/1'
    mi_cache_file = os.path.join(data_path, 'mi_analysis_cache_classification.pkl')
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Check MI Cache
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if os.path.exists(mi_cache_file):
        print(f"Loading cached MI data from: {mi_cache_file}")
        with open(mi_cache_file, 'rb') as f:
            all_layers_data = pickle.load(f)
        print("MI cache loaded successfully!\n")
    else:
        print(f"Loading classification MI data from: {data_path}")
        
        with open(os.path.join(data_path, 'y_labels.pkl'), 'rb') as f:
            y_in = pickle.load(f)  # (N,) - class labels
        print(f"Loaded labels: {y_in.shape}")
        
        with open(os.path.join(data_path, 'layer_0.pkl'), 'rb') as f:
            x_in = pickle.load(f)  # (N, H, W) - input features
        print(f"Loaded input features: {x_in.shape}")
        
        t_in = []
        for i in range(1, num):
            with open(os.path.join(data_path, f'layer_{i}.pkl'), 'rb') as f:
                t_layer = pickle.load(f)
                t_in.append(t_layer)  # (N, H, W) - layer features
                print(f"Loaded layer {i}: {t_layer.shape}")
        
        H_dim, W_dim = x_in.shape[1], x_in.shape[2]
        num_layers = len(t_in)
        
        # Initialize storage for each layer
        all_layers_data = []
        
        print(f"\n=== Computing MI for Classification ===")
        print(f"Number of layers: {num_layers}")
        print(f"Feature map dimensions: {H_dim} x {W_dim}")
        
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
            # mi_xt: (H, W, H, W) -> flatten to (H*W*H*W,)
            # mi_ty: (H, W) -> replicate for each reference pixel
            num_pixels = H_dim * W_dim
            
            layer_mi_xt = []
            layer_mi_ty = []
            layer_distance = []
            
            # For each reference pixel in T (ht, wt)
            for ht in range(H_dim):
                for wt in range(W_dim):
                    # Get all MI values I(X[hx,wx]; T[ht,wt]) for all (hx, wx)
                    mi_xt_ref = mi_xt[ht, wt, :, :].flatten()  # (H*W,)
                    
                    # Get I(T[ht,wt]; Y) - constant for all comparison pixels
                    mi_ty_val = mi_ty[ht, wt]
                    mi_ty_ref = np.full_like(mi_xt_ref, mi_ty_val)  # (H*W,)
                    
                    # Get euclidean distances from (ht, wt) to all (hx, wx)
                    dist_ref = euc_map[ht, wt, :, :].flatten()  # (H*W,)
                    
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
    
    # Plot styling
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    
    print("\n=== Generating Scatter Plots ===")
    
    # Generate plots for each layer
    for layer_data in all_layers_data:
        layer_idx = layer_data['layer_idx']
        mi_xt = layer_data['mi_xt']
        mi_ty = layer_data['mi_ty']
        distance = layer_data['distance']
        
        print(f"\nLayer {layer_idx}:")
        
        # Overall scatter plot
        print(f"  Generating overall scatter plot...")
        plot_scatter_classification(mi_xt, mi_ty, distance, 
                                   layer_idx - 1, args.model, args.dataset)
        
        # # Distance-binned plots
        # print(f"  Generating distance-binned scatter plots...")
        # plot_scatter_with_distance_bins(mi_xt, mi_ty, distance, 
        #                                 layer_idx - 1, args.model, args.dataset)
    
    print("\n=== All plots generated successfully! ===")
