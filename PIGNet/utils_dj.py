import numpy as np
import matplotlib.pyplot as plt

def cal_mi_x_t(x, t):
    '''
    Calculate Pairwise Mutual Information and Spatial Distances.
    Comparisons are made between every pixel of T and every pixel of X.
    
    Args:
        x: (N, H, W) - Discrete Target maps
        t: (N, H, W) - Discrete Feature maps
        
    Returns:
        mi_map:  (H, W, H, W) - MI between T[ht,wt] and X[hx,wx]
        euc_map: (H, W, H, W) - Spatial Euclidean dist between (ht,wt) and (hx,wx)
        man_map: (H, W, H, W) - Spatial Manhattan dist between (ht,wt) and (hx,wx)
    '''
    N, H, W = t.shape
    
    # Initialize 4D output tensors
    # Shape: (T_height, T_width, X_height, X_width)
    mi_map = np.zeros((H, W, H, W))
    euc_map = np.zeros((H, W, H, W))
    man_map = np.zeros((H, W, H, W))
    
    eps = 1e-12
    
    # --- Pre-calculation for optimization ---
    
    # 1. Flatten spatial dims for easier iteration logic internally
    # But we will write back to the 4D array.
    x_flat = x.reshape(N, -1)     # (N, H*W)
    t_flat = t.reshape(N, -1)     # (N, H*W)
    num_pixels = H * W
    
    # 2. Pre-calculate Entropy H(X) and H(T) for all positions
    # This reduces redundant calculations inside the nested loop.
    
    # H(X) for every position
    h_x_all = np.zeros(num_pixels)
    for i in range(num_pixels):
        _, counts = np.unique(x_flat[:, i], return_counts=True)
        p = counts / N
        h_x_all[i] = -np.sum(p * np.log2(p + eps))
        
    # H(T) for every position
    h_t_all = np.zeros(num_pixels)
    for i in range(num_pixels):
        _, counts = np.unique(t_flat[:, i], return_counts=True)
        p = counts / N
        h_t_all[i] = -np.sum(p * np.log2(p + eps))

    # --- Main Nested Loop (All-to-All) ---
    # We iterate using linear indices (0 to HW-1) and convert to (h, w) for saving
    
    for i_t in range(num_pixels): # Index for T
        # Convert linear index to (ht, wt)
        ht, wt = divmod(i_t, W)
        
        t_vec = t_flat[:, i_t]
        h_t_val = h_t_all[i_t]
        
        for i_x in range(num_pixels): # Index for X
            # Convert linear index to (hx, wx)
            hx, wx = divmod(i_x, W)
            
            x_vec = x_flat[:, i_x]
            
            # --- [Metric 1] Mutual Information ---
            # MI(T_i, X_j) = H(T_i) + H(X_j) - H(T_i, X_j)
            
            # Joint Entropy H(T_i, X_j)
            stack_tx = np.vstack((t_vec, x_vec))
            _, counts_tx = np.unique(stack_tx, axis=1, return_counts=True)
            p_tx = counts_tx / N
            h_tx = -np.sum(p_tx * np.log2(p_tx + eps))
            
            mi_val = h_t_val + h_x_all[i_x] - h_tx
            mi_map[ht, wt, hx, wx] = mi_val
            
            # --- [Metric 2 & 3] Spatial Distance ---
            # Distance between coordinate (ht, wt) and (hx, wx)
            # This is constant regardless of sample data (N)
            
            diff_h = ht - hx
            diff_w = wt - wx
            
            # Euclidean
            euc_map[ht, wt, hx, wx] = np.sqrt(diff_h**2 + diff_w**2)
            
            # Manhattan
            man_map[ht, wt, hx, wx] = np.abs(diff_h) + np.abs(diff_w)
            
    return mi_map, euc_map, man_map

def cal_mi_t_y(t, y):
    '''
    Calculate Mutual Information I(T; Y) per pixel.
    Y is a global target vector (e.g., Image Class Label), so it does not have spatial dimensions.
    
    Args:
        t: (num_samples, height, width) - Discrete Feature maps
        y: (num_samples,) - Discrete Global Target labels
        
    Returns:
        mi_map: (height, width) - MI map between each pixel of T and Y
    '''
    N, H, W = t.shape
    mi_map = np.zeros((H, W))
    eps = 1e-12
    
    # 1. Calculate H(Y) (Computed once, as Y is constant for all pixels)
    _, counts_y = np.unique(y, return_counts=True)
    p_y = counts_y / N
    h_y = -np.sum(p_y * np.log2(p_y + eps))
    
    # Offset base for scalarization (optimization trick)
    # Using a multiplier larger than max(y) ensures unique (t, y) combinations
    offset_base = np.max(y) + 1
    
    # 2. Iterate over each pixel position
    for h in range(H):
        for w in range(W):
            t_pixel = t[:, h, w] # Shape: (N,)
            
            # --- H(T) ---
            _, counts_t = np.unique(t_pixel, return_counts=True)
            p_t = counts_t / N
            h_t = -np.sum(p_t * np.log2(p_t + eps))
            
            # --- H(T, Y) ---
            # Combine t and y into a single unique integer array for speed
            # (Instead of vstack, we use scalar math: t * base + y)
            # Scalrization for fast joint distribution
            joint_vec = t_pixel.astype(np.int64) * offset_base + y
            
            _, counts_ty = np.unique(joint_vec, return_counts=True)
            p_ty = counts_ty / N
            h_ty = -np.sum(p_ty * np.log2(p_ty + eps))
            
            # --- MI = H(T) + H(Y) - H(T, Y) ---
            mi_map[h, w] = h_t + h_y - h_ty
            
    return mi_map

if __name__ == "__main__":
    # 데이터 생성 (이미 0~4로 이산화된 상태라고 가정)
    N_sample = 500
    N_layer = 5
    H_dim, W_dim = 8, 8
    
    t_in = np.random.randint(0, 50, (N_sample, N_layer, H_dim, W_dim)) 
    y_in = np.random.randint(0, 30, (N_sample, H_dim, W_dim))
    
    # 이거 y는 원래 사이즈에 맞게 label 되어있을텐데 그걸 같은 클래스에 있는거랑 다른 클래스에 있는거랑 구분해서 
    # MI가 얼마나 크게 되고 얼마나 자근지 확인해야함
    # 또한 distance도 같이 한번 비교해봐야할 수 있음
    # 그래야 같은 클래스에 있을때 전반적으로 MI가 높은지 확인하고 
    # 그 중에서도 CNN계열은 distance가 가까울수록 MI가 더 높은지 확인 가능 (locality)
    # 반면에 GSP?를 쓰는 경우에는 둘다 가깝든 멀든 좋게 나타나는지 확인해볼 수 있음
    
    


if __name__ == "classification":
    # 데이터 생성 (이미 0~4로 이산화된 상태라고 가정)
    N_sample = 500
    N_layer = 5
    H_dim, W_dim = 8, 8
    
    # x input shape is now (N, H, W)
    t_in = np.random.randint(0, 100, (N_sample, N_layer, H_dim, W_dim)) 
    y_in = np.random.randint(0, 10, (N_sample,))  
    
    x_in = t_in[:, 0, :, :]  # Using layer 0 as X input
    
        
    distance_s = []
    mi_s = []
    for layer_idx in range(1, N_layer):
        t_layer = t_in[:, layer_idx, :, :]
        mi_result, euc_result, man_result = cal_mi_x_t(x_in, t_layer)
        mi_t_y_layer = cal_mi_t_y(t_layer, y_in)
    
        # distance 기반으로 어떻게 달라지는지 확인해야함
        # hx, wx = 0, 0
        distance_ = []
        mi_ = []
        for hx in range(H_dim):
            for wx in range(W_dim):
                mi_flat = mi_result[:,:,hx,wx].flatten()
                euc_flat = euc_result[:,:,hx,wx].flatten()
                distance_.extend(euc_flat.tolist())
                mi_.extend(mi_flat.tolist())
        distance_s.append(distance_)
        mi_s.append(mi_)
    distance_s = np.array(distance_s)
    mi_s = np.array(mi_s)
    
    # 각 레이어에서 distance에 따른 MI가 어떻게 달라지는지 보여줘야함 
    
    for layer_idx in range(distance_s.shape[0]):
        plt.figure()
        plt.scatter(distance_s[layer_idx], mi_s[layer_idx], alpha=0.1)
        plt.title(f"Layer {layer_idx+1} MI vs Euclidean Distance")
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Mutual Information I(X; T_layer)")
        plt.grid(True)
        
        # save figure
        plt.savefig(f"layer_{layer_idx+1}_mi_vs_distance.png")