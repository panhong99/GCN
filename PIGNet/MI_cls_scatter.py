import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pickle
import cv2
from PIL import Image
import torch
import numpy as np
from tqdm.auto import trange
import argparse

plt.rcParams['font.family'] = 'Arial'

COLOR_MAP = {'PIGNet_GSPonly': '#D81B60', 'ResNet': '#5C6BC0', 'ViT': '#FF7043'}


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
    # mi_map_flat = np.zeros((num_pixels, num_pixels))
    joint_map_flat = np.zeros((num_pixels, num_pixels))

    
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
            # mi_map_flat[i_t, i_x] = float(mi_val)
            joint_map_flat[i_t, i_x] = float(h_joint)
    
    # Compute Euclidean distance map
    grid = np.indices((H, W)).reshape(2, -1).T
    h_diff = grid[:, 0:1] - grid[:, 0:1].T
    w_diff = grid[:, 1:2] - grid[:, 1:2].T
    euc_map = np.sqrt(h_diff**2 + w_diff**2).reshape(H, W, H, W)
    
    joint_map = joint_map_flat.reshape(H, W, H, W)
    
    return joint_map, euc_map, # mi_map

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
    # mi_map = np.zeros((H, W))
    joint_map = np.zeros((H, W))
    
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
            # mi_map[h, w] = h_t_all[pixel_idx] + h_y - h_joint
            joint_map[h, w] = h_joint

    return joint_map

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 비활성화: 개별 layer scatter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# def plot_scatter_classification(mi_xt, mi_ty, distance, layer_idx, model_name, dataset_name, process_type, group_name=None, cluster_num=None, calcul_type=None):
#     fig, ax = plt.subplots(figsize=(10, 8))
#     scatter = ax.scatter(mi_xt, mi_ty, c=distance, cmap='viridis', s=20, alpha=0.6, edgecolors='none')
#     plt.colorbar(scatter, ax=ax)
#     ax.set_xlabel("H(X,T)", fontsize=13); ax.set_ylabel("H(T,Y)", fontsize=13)
#     ct = f"_{calcul_type}" if calcul_type else ""
#     if group_name:
#         fname = f"{model_name}_{dataset_name}_{process_type}{ct}_scatter_{group_name}_layer{layer_idx}.png"
#     else:
#         fname = f"{model_name}_{dataset_name}_{process_type}{ct}_scatter_layer{layer_idx}.png"
#     plt.tight_layout(); plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 활성화: Scatter Matrix (1 × num_layers)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_scatter_all_layers_classification(all_layers_data, model_name, dataset_name, num, process_type, group_name=None, cluster_num=None, calcul_type=None):
    """
    Plot scatter maps for all layers in a single figure (1 × num_layers matrix).
    CLS에는 distance 없음 → H(X,T) 값으로 색상 표현
    """
    num_layers = len(all_layers_data)

    model_color = next(
        (color for key, color in COLOR_MAP.items() if key.lower() in model_name.lower()),
        '#888888'
    )
    model_cmap = mcolors.LinearSegmentedColormap.from_list(
        'model_cmap', ['#ffffff', model_color]
    )

    ncols = max(4, num - 1)
    nrows = (num_layers + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), facecolor='white')

    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    last_sc = None

    for idx, layer_data in enumerate(all_layers_data):
        ax = axes_flat[idx]
        ax.set_facecolor('white')
        layer_idx = layer_data['layer_idx']
        ct = calcul_type if calcul_type else 'joint'
        mi_xt = np.asarray(layer_data[f'{ct}_xt'], dtype=float)
        mi_ty = np.asarray(layer_data[f'{ct}_ty'], dtype=float)
        layer_group = layer_data.get('group', None)

        sc = ax.scatter(mi_xt, mi_ty, c=mi_xt, cmap=model_cmap,
                        s=20, alpha=0.6, edgecolors='none')
        last_sc = sc

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

        if idx == 0:
            ax.set_ylabel("H(T,Y)", fontsize=18)
        ax.set_xlabel("H(X,T)", fontsize=18)

        title = f"{layer_group} Layer {layer_idx}" if layer_group else f"Layer {layer_idx}"
        ax.set_title(title, fontsize=18)

    for idx in range(num_layers, len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    if last_sc is not None:
        valid_axes = [ax for ax in axes_flat[:num_layers] if ax.get_visible()]
        top    = valid_axes[0].get_position().y1
        bottom = valid_axes[-1].get_position().y0
        cbar_ax = fig.add_axes([0.90, bottom, 0.03, top - bottom])
        cbar = fig.colorbar(last_sc, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=18)

    ct = f"_{calcul_type}" if calcul_type else ""
    if group_name:
        if cluster_num:
            fname = f"{model_name}_{dataset_name}_{process_type}{ct}_scatter_all_layers_{group_name}_cluster{cluster_num}.png"
        else:
            fname = f"{model_name}_{dataset_name}_{process_type}{ct}_scatter_all_layers_{group_name}.png"
    else:
        if cluster_num:
            fname = f"{model_name}_{dataset_name}_{process_type}{ct}_scatter_all_layers_cluster{cluster_num}.png"
        else:
            fname = f"{model_name}_{dataset_name}_{process_type}{ct}_scatter_all_layers.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"All layers scatter matrix saved: {fname}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 비활성화: ratio boxplot / lineplot (barplot 작업 예정)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# def plot_ratio_boxplot_classification(all_layers_data, model_name, dataset_name, process_type,
#                                       calcul_type='MI', median_alpha=0.8):
#     if not all_layers_data:
#         return
#     n_layers = len(all_layers_data)
#     layer_labels = [f"L{d['layer_idx']}" for d in all_layers_data]
#     x_positions = np.arange(1, n_layers + 1)
#     ratios = []
#     median_values = []
#     for layer_data in all_layers_data:
#         mi_xt = np.asarray(layer_data[f'{calcul_type}_xt'], dtype=float)
#         mi_ty = np.asarray(layer_data[f'{calcul_type}_ty'], dtype=float)
#         valid_mask = np.isfinite(mi_xt) & np.isfinite(mi_ty) & (mi_ty != 0)
#         ratio = mi_xt[valid_mask] / mi_ty[valid_mask]
#         ratios.append(ratio)
#         median_values.append(np.median(ratio) if len(ratio) > 0 else np.nan)
#     fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
#     bp = ax.boxplot(ratios, positions=x_positions, patch_artist=True, showfliers=False, widths=0.5, labels=layer_labels)
#     plt.tight_layout()
#     folder_path = f"./{model_name}/{dataset_name}/{process_type}/ratio"
#     os.makedirs(folder_path, exist_ok=True)
#     fname = f"{model_name}_{dataset_name}_{process_type}_{calcul_type}_ratio_boxplot.png"
#     plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')
#     plt.close()


# def plot_ratio_lineplot_all_models(models_data, dataset_name, process_type, calcul_type='MI'):
#     model_list = list(models_data.keys())
#     n_models = len(model_list)
#     color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     model_colors = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(model_list)}
#     first_layers = models_data[model_list[0]]['all_layers_data']
#     n_layers = len(first_layers)
#     layer_labels = [f"L{d['layer_idx']}" for d in first_layers]
#     x_base = np.arange(1, n_layers + 1)
#     box_width = 0.12
#     total_span = box_width * n_models * 1.5
#     offsets = np.linspace(-total_span / 2, total_span / 2, n_models)
#     folder_path = f"./ALL_MODELS/{dataset_name}/{process_type}/ratio"
#     os.makedirs(folder_path, exist_ok=True)
#     fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
#     plt.tight_layout()
#     fname = f"ALL_{dataset_name}_{process_type}_{calcul_type}_ratio_lineplot.png"
#     plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')
#     plt.close()


if __name__ == "__main__":  # Classification Task
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='CIFAR-10', 
                          help='Dataset name (e.g., Imagenet, CIFAR-100)')
    argparser.add_argument('--process_type', type=str, default='pixel', 
                          help='Process type (e.g., layer, pixel)')
    argparser.add_argument('--model', type=str, default='PIGNet_GSPonly_classification', 
                          help='Model name (e.g., Resnet, PIGNet_GSPonly_classification, vit)')
    argparser.add_argument('--cluster_num', type=int, default=50,
                          help='Number of clusters used in MI data generation (e.g., 50, 100)')
    argparser.add_argument('--calcul_type', type=str, default='joint',
                          help='Type of calculation for scatter plot (e.g., mi, joint)')
    args = argparser.parse_args()
    
    if args.model == "Resnet" or args.model == "vit":
        layer_num = 5
    elif args.model == "PIGNet_GSPonly_classification":
        backbonenum, gsp_layer_num = 4,5 # backbone 3 + GSP block 5    

    # Setup data path
    data_path = f'/home/hail/pan/HDD/MI_dataset/{args.dataset}/{args.process_type}_dataset/resnet101/pretrained/{args.model}/zoom/1'
    
    mi_cache_file = os.path.join(data_path, f'{args.calcul_type}_mi_analysis_cache_classification_cluster{args.cluster_num}.pkl')
    backbone_cache_file = os.path.join(data_path, f'{args.calcul_type}_mi_analysis_cache_backbone_classification_cluster{args.cluster_num}.pkl')
    gsp_cache_file = os.path.join(data_path, f'{args.calcul_type}_mi_analysis_cache_gsp_classification_cluster{args.cluster_num}.pkl')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Check MI Cache
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if os.path.exists(mi_cache_file):
        print(f"Loading cached MI data from: {mi_cache_file}")
        with open(mi_cache_file, 'rb') as f:
            all_layers_data = pickle.load(f)
        print("MI cache loaded successfully!\n")

    elif (os.path.exists(backbone_cache_file) and os.path.exists(gsp_cache_file)):
        print(f"Loading cached MI data from: {backbone_cache_file} and {gsp_cache_file}")
        with open(backbone_cache_file, 'rb') as f:
            backbone_layers_data = pickle.load(f)
        with open(gsp_cache_file, 'rb') as f:
            gsp_layers_data = pickle.load(f)
        print("MI cache loaded successfully!\n")

    else:
        print(f"Loading classification MI data from: {data_path}")

        if args.model != "PIGNet_GSPonly_classification":        
            with open(os.path.join(data_path, f'y_labels_{args.cluster_num}.pkl'), 'rb') as f:
                y_in = pickle.load(f)  # (N,) - class labels
            print(f"Loaded labels: {y_in.shape}")
            
            with open(os.path.join(data_path, f'layer_0_{args.cluster_num}.pkl'), 'rb') as f:
                x_in = pickle.load(f)  # (N, H, W) - input features
            print(f"Loaded input features: {x_in.shape}")
            
            t_in = []
            for i in range(1, layer_num):
                with open(os.path.join(data_path, f'layer_{i}_{args.cluster_num}.pkl'), 'rb') as f:
                    t_layer = pickle.load(f)
                    t_in.append(t_layer)  # (N, H, W) - layer features
                    print(f"Loaded layer {i}: {t_layer.shape}")

            H_dim, W_dim = x_in.shape[1], x_in.shape[2]
                    
        else: # PIGNet_GSPonly_classification
            with open(os.path.join(data_path, f'y_labels_{args.cluster_num}.pkl'), 'rb') as f:
                y_in = pickle.load(f)  # (N,) - class labels
            print(f"Loaded labels: {y_in.shape}")
            
            with open(os.path.join(data_path, f'backbone_layer_0_{args.cluster_num}.pkl'), 'rb') as f:
                backbone_x_in = pickle.load(f)  # (N, H, W) - input features
            print(f"Loaded input features: {backbone_x_in.shape}")
            
            with open(os.path.join(data_path, f'gsp_layer_0_{args.cluster_num}.pkl'), 'rb') as f:
                gsp_x_in = pickle.load(f)  # (N, H, W) - input features
            print(f"Loaded input features: {gsp_x_in.shape}")
            
            backbone_t_in = []
            gsp_t_in = []

            for i in range(1, backbonenum):
                with open(os.path.join(data_path, f'backbone_layer_{i}_{args.cluster_num}.pkl'), 'rb') as f:
                    t_layer = pickle.load(f)
                    backbone_t_in.append(t_layer)  # (N, H, W) - layer features
                    print(f"Loaded layer {i}: {t_layer.shape}")

            for i in range(1, gsp_layer_num):
                with open(os.path.join(data_path, f'gsp_layer_{i}_{args.cluster_num}.pkl'), 'rb') as f:
                    t_layer = pickle.load(f)
                    gsp_t_in.append(t_layer)  # (N, H, W) - layer features
                    print(f"Loaded layer {i}: {t_layer.shape}")

            H_dim, W_dim = backbone_x_in.shape[1], backbone_x_in.shape[2]

        # Initialize storage for each layer

        
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
                    f'{args.calcul_type}_xt': layer_mi_xt,
                    f'{args.calcul_type}_ty': layer_mi_ty,
                    'distance': layer_distance,
                })
                
                print(f"Layer {layer_idx+1} statistics:")
                print(f"  Data points: {len(layer_mi_xt)}")
                print(f"  {args.calcul_type.upper()} range: [{layer_mi_xt.min():.4f}, {layer_mi_xt.max():.4f}]")
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
            print(f"Number of GSP layers: {len(backbone_t_in[backbonenum:])}")
            
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
                    f'{args.calcul_type}_xt': layer_mi_xt,
                    f'{args.calcul_type}_ty': layer_mi_ty,
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
                    f'{args.calcul_type}_xt': layer_mi_xt,
                    f'{args.calcul_type}_ty': layer_mi_ty,
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
    
    # Plot styling
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    
    print("\n=== Generating Scatter Plots ===")
    
    # Generate individual plots for each layer
    if args.model != "PIGNet_GSPonly_classification":
        for layer_data in all_layers_data:
            layer_idx = layer_data['layer_idx']
            mi_xt = layer_data[f'{args.calcul_type}_xt']
            mi_ty = layer_data[f'{args.calcul_type}_ty']
            distance = layer_data['distance']
            group_name = layer_data.get('group', None)
            
            if group_name:
                print(f"\n{group_name} - Layer {layer_idx}:")
                print(f"  Generating individual scatter plot...")
                # plot_scatter_classification(mi_xt, mi_ty, distance,
                #                         layer_idx, args.model, args.dataset, args.process_type, group_name, args.cluster_num, calcul_type=args.calcul_type)
            else:
                print(f"\nLayer {layer_idx}:")
                print(f"  Generating individual scatter plot...")
                # plot_scatter_classification(mi_xt, mi_ty, distance,
                #                         layer_idx, args.model, args.dataset, args.process_type, cluster_num=args.cluster_num, calcul_type=args.calcul_type)

        # Generate combined subplot plot for all layers
        print(f"\n=== Generating Combined Subplot Plot ===")
        total_layers = len(all_layers_data)
        plot_scatter_all_layers_classification(all_layers_data, args.model, args.dataset, total_layers, args.process_type, group_name=None, cluster_num=args.cluster_num, calcul_type=args.calcul_type)
        
        # Generate ratio boxplot
        print(f"\n=== Generating Ratio Boxplot ===")
        # plot_ratio_boxplot_classification(all_layers_data, args.model, args.dataset, args.process_type, calcul_type=args.calcul_type)
        
        print("\n=== All plots generated successfully! ===")
        
    else:
        for all_layers_data in [backbone_layers_data, gsp_layers_data]:
            group_label = [d.get('group') for d in all_layers_data][0] if all_layers_data else None
            
            for layer_data in all_layers_data:
                layer_idx = layer_data['layer_idx']
                mi_xt = layer_data[f'{args.calcul_type}_xt']
                mi_ty = layer_data[f'{args.calcul_type}_ty']
                distance = layer_data['distance']
                group_name = layer_data.get('group', None)
                
                if group_name:
                    print(f"\n{group_name} - Layer {layer_idx}:")
                    print(f"  Generating individual scatter plot...")
                    # plot_scatter_classification(mi_xt, mi_ty, distance,
                    #                         layer_idx, args.model, args.dataset, args.process_type, group_name, args.cluster_num, calcul_type=args.calcul_type)
                else:
                    print(f"\nLayer {layer_idx}:")
                    print(f"  Generating individual scatter plot...")
                    # plot_scatter_classification(mi_xt, mi_ty, distance,
                    #                         layer_idx, args.model, args.dataset, args.process_type, cluster_num=args.cluster_num, calcul_type=args.calcul_type)

            # Generate combined subplot plot for this group
            print(f"\n=== Generating Combined Subplot Plot ===")
            total_layers = len(all_layers_data)
            plot_scatter_all_layers_classification(all_layers_data, args.model, args.dataset, total_layers, args.process_type, group_name=group_label, cluster_num=args.cluster_num, calcul_type=args.calcul_type)
            
            # Generate ratio boxplot for this group
            print(f"\n=== Generating Ratio Boxplot for {group_label} ===")
            # plot_ratio_boxplot_classification(all_layers_data, args.model, args.dataset, args.process_type, calcul_type=args.calcul_type)
            
            print(f"\n=== All plots for {group_label} generated successfully! ===")
