import numpy as np
import pickle
import numpy as np
from tqdm.auto import trange
from scipy.stats import gaussian_kde

def entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    """Calculate entropy from count histogram."""
    p = counts / np.maximum(1, counts.sum())
    return float(-np.sum(p * np.log2(p + eps)))

def cal_je_x_t(x, t):
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
        h_x_all[i] = entropy_from_counts(counts_x, eps)
        h_t_all[i] = entropy_from_counts(counts_t, eps)
    
    joint_map_flat = np.zeros((num_pixels, num_pixels))

    
    for i_t in trange(num_pixels, desc="MI(X;T) - Classification", leave=False):
        t_vec = t_flat[:, i_t]
        
        for i_x in range(num_pixels):
            x_vec = x_flat[:, i_x]
            
            # Joint entropy using vstack + 2D unique (no offset_base needed)
            stack_tx = np.vstack((t_vec, x_vec))
            _, counts_tx = np.unique(stack_tx, axis=1, return_counts=True)
            h_joint = entropy_from_counts(counts_tx, eps)
            joint_map_flat[i_t, i_x] = float(h_joint)
    
    # Compute Euclidean distance map
    grid = np.indices((H, W)).reshape(2, -1).T
    h_diff = grid[:, 0:1] - grid[:, 0:1].T
    w_diff = grid[:, 1:2] - grid[:, 1:2].T
    euc_map = np.sqrt(h_diff**2 + w_diff**2).reshape(H, W, H, W)
    
    joint_map = joint_map_flat.reshape(H, W, H, W)
    
    return joint_map, euc_map, # mi_map

def cal_je_t_y(t, y, eps=1e-12):
    N, H, W = t.shape
    # mi_map = np.zeros((H, W))
    joint_map = np.zeros((H, W))
    
    # 1. Calculate H(Y) once (constant for all pixels)
    y_counts = np.bincount(y.astype(np.int32))
    h_y = entropy_from_counts(y_counts, eps)
    
    # 2. Pre-calculate H(T) for all positions
    t_flat = t.reshape(N, -1).astype(np.int32)
    num_pixels = H * W
    h_t_all = np.zeros(num_pixels)
    
    for i in range(num_pixels):
        _, counts_t = np.unique(t_flat[:, i], return_counts=True)
        h_t_all[i] = entropy_from_counts(counts_t, eps)
    
    offset_base = np.max(y) + 1
    
    for h in range(H):
        for w in range(W):
            t_pixel = t[:, h, w].astype(np.int32)
            
            # Scalarization: 1D encoding으로 (t_pixel, y) 쌍을 유니크하게 표현
            joint_encoded = t_pixel * offset_base + y.astype(np.int32)
            _, counts_joint = np.unique(joint_encoded, return_counts=True)
            h_joint = entropy_from_counts(counts_joint, eps)
            joint_map[h, w] = h_joint

    return joint_map

def calcul_JE(args, x_in, t_in, y_in, H_dim, W_dim, backbone_x_in=None, backbone_t_in=None, gsp_x_in=None, gsp_t_in=None,
                 je_cache_file=None, backbone_cache_file=None, gsp_cache_file=None,
                 backbonenum=None, gsp_layer_num=None):

    print(f"\n=== Computing JE for Classification ===")
    print(f"Feature map dimensions: {H_dim} x {W_dim}")

    if args.model != "PIGNet_GSPonly_classification":
        # Standard computation for other models
        print(f"Number of layers: {len(t_in)}")
        all_layers_data = []
            
        # Compute JE for each layer
        for layer_idx, t_layer in enumerate(t_in):
            print(f"\n--- Layer {layer_idx+1} ---")
            
            # Compute I(X; T)
            print("Computing I(X; T)...")
            mi_xt, euc_map = cal_je_x_t(x_in, t_layer)
            
            # Compute I(T; Y)
            print("Computing I(T; Y)...")
            mi_ty = cal_je_t_y(t_layer, y_in)
            
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
        print(f"\nSaving computed data to {je_cache_file}...")
        with open(je_cache_file, 'wb') as f:
            pickle.dump(all_layers_data, f)
        print("Cache saved successfully!")
    
    else:  # PIGNet_GSPonly_classification
        backbone_layers_data = []
        gsp_layers_data = []

        print(f"Number of Backbone layers: {len(backbone_t_in[:backbonenum])}")
        print(f"Number of GSP layers: {len(backbone_t_in[backbonenum:])}")
        
        # Process Backbone layers
        print(f"\n=== Computing JE for Backbone Layers ===")
        for layer_idx, t_layer in enumerate(backbone_t_in):
            print(f"\n--- Backbone Layer {layer_idx+1} ---")
            
            # Compute I(X; T)
            print("Computing I(X; T)...")
            mi_xt, euc_map = cal_je_x_t(backbone_x_in, t_layer)
            
            # Compute I(T; Y)
            print("Computing I(T; Y)...")
            mi_ty = cal_je_t_y(t_layer, y_in)
            
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
            mi_xt, euc_map = cal_je_x_t(gsp_x_in, t_layer)
            
            # Compute I(T; Y)
            print("Computing I(T; Y)...")
            mi_ty = cal_je_t_y(t_layer, y_in)
            
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
        print(f"\nSaving computed data to {je_cache_file}...")

        with open(backbone_cache_file, 'wb') as f:
            pickle.dump(backbone_layers_data, f)
        print("Cache saved successfully!")

        with open(gsp_cache_file, 'wb') as f:
            pickle.dump(gsp_layers_data, f)
        print("Cache saved successfully!")
    
    return 0;

def calcul_JE_kde(all_layers_data, calcul_type='joint'):
    """
    Compute KDE for all layers and store in a dictionary.
    Returns: dict with layer_idx as key, containing KDE values and original data.
    """
    kde_data = {}

    for layer_data in all_layers_data:
        layer_idx = layer_data['layer_idx']
        joint_xt = layer_data[f'{calcul_type}_xt'].flatten()
        joint_ty = layer_data[f'{calcul_type}_ty'].flatten()
        
        print(f"Computing KDE for Layer {layer_idx}...", end=" ")
        
        # Data validation & jitter adjustment
        eps = 1e-8
        
        xt_std = np.std(joint_xt)
        ty_std = np.std(joint_ty)
        
        if xt_std < eps:
            joint_xt = joint_xt + np.random.normal(0, eps * 10, joint_xt.shape)
            print(f"[joint_xt jitter added]", end=" ")
        if ty_std < eps:
            joint_ty = joint_ty + np.random.normal(0, eps * 10, joint_ty.shape)
            print(f"[joint_ty jitter added]", end=" ")
        
        # Stack for KDE input
        points = np.vstack([joint_xt, joint_ty])
        
        # Compute KDE with covariance factor adjustment
        try:
            kde = gaussian_kde(points)
        except np.linalg.LinAlgError:
            print(f"[KDE fatal error - using histogram instead]", end=" ")
            kde = None
        
        # Evaluate on grid
        x_min, x_max = joint_xt.min(), joint_xt.max()
        y_min, y_max = joint_ty.min(), joint_ty.max()
        
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
            for i in range(len(joint_xt)):
                x_idx = np.argmin(np.abs(x_grid - joint_xt[i]))
                y_idx = np.argmin(np.abs(y_grid - joint_ty[i]))
                Z[y_idx, x_idx] += 1
            Z = Z / (np.max(Z) + eps)
        
        kde_data[layer_idx] = {
            'X': X, 'Y': Y, 'Z': Z,
            f'{calcul_type}_xt': joint_xt,
            f'{calcul_type}_ty': joint_ty,
            'x_grid': x_grid, 'y_grid': y_grid,
            'n_points': len(joint_xt),
        }
        
        print("done")
    
    return kde_data