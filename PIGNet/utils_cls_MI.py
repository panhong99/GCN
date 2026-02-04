import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import cv2
from PIL import Image
import torch
import numpy as np

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
    # h_x_all -> 64 x 64
    h_x_all = np.zeros(num_pixels)

    for i in range(num_pixels):
        # 한 개 픽셀값에 대한 kmeans를 진행을 하고 vq된 값들을 확률분포로 변환
        _, counts = np.unique(x_flat[:, i], return_counts=True)
        
        # 특정 픽셀위치의 확률 분포
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
        # divmod: 몫과 나머지를 동시에 반환
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
            
            # MI Calculation
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

if __name__ == "__main__": # clssification
    # cls
    cls_file_path = f"/home/hail/pan/HDD/MI_dataset/imagenet/resnet101/pretrained/PIGNet_GSPonly_classification/zoom/1"
    
    with open(os.path.join(cls_file_path, 'gt_labels.pkl'), 'rb') as f:
        y_in = pickle.load(f)

    with open(os.path.join(cls_file_path, 'layer_0.pkl'), 'rb') as f:
        x_in = pickle.load(f)

    t_in = []
    for i in range(1, 5):
        with open(os.path.join(cls_file_path, f'layer_{i}.pkl'), 'rb') as f:
            t_in.append(pickle.load(f))        

    H_dim, W_dim = x_in.shape[1], x_in.shape[2]
    distance_s = []
    mi_s = []
    mi_y = [] 

    for t_layer in t_in:
        mi_result, euc_result, man_result = cal_mi_x_t(x_in, t_layer)
        mi_t_y = cal_mi_t_y(t_layer, y_in)
        
        distance_ = []
        mi_ = []
        mi_ty = []

        for hx in range(H_dim):
            for wx in range(W_dim):
                # 기준 픽셀 (hx, wx)와 모든 다른 픽셀 (ht, wt) 간의 I(X;T)와 거리
                mi_flat = mi_result[:,:,hx,wx].flatten()  # (H*W,) - 각 원소는 (ht,wt)와 (hx,wx) 간의 I(X;T)
                euc_flat = euc_result[:,:,hx,wx].flatten()  # (H*W,) - 거리
                
                # 각 타겟 픽셀 (ht, wt)의 I(T;Y) 값들
                # mi_t_y는 (H, W) 형태이므로, 모든 (ht, wt) 위치의 I(T;Y)를 가져옴
                mi_t_y_flat = mi_t_y.flatten()  # (H*W,) - 각 원소는 (ht,wt) 위치의 I(T;Y)
                
                distance_.extend(euc_flat.tolist())
                mi_.extend(mi_flat.tolist())
                mi_ty.extend(mi_t_y_flat.tolist())
                
        distance_s.append(distance_)
        mi_s.append(mi_)
        mi_y.append(mi_ty)
        
    distance_s = np.array(distance_s)
    mi_s = np.array(mi_s)
    mi_y = np.array(mi_y)    
    
    # graph distance, I(X;T)
    for layer_idx in range(distance_s.shape[0]):
        distance_data = distance_s[layer_idx]
        mi_data = mi_s[layer_idx]    
        plt.figure(figsize=(12, 7))
        plt.scatter(distance_data, mi_data, alpha=0.5, s=10)
        plt.title(f"Layer {layer_idx+1} MI vs Euclidean Distance", fontsize=14, fontweight='bold')
        plt.xlabel("Euclidean Distance", fontsize=12)
        plt.ylabel("Mutual Information I(X; T_layer)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"layer_{layer_idx+1}_mi_vs_distance.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    for layer_idx in range(distance_s.shape[0]):
        plt.figure(figsize=(12, 7))
        
        scatter = plt.scatter(mi_s[layer_idx], mi_y[layer_idx], 
                            c=distance_s[layer_idx], cmap='coolwarm', 
                            alpha=0.5, s=20, edgecolors='none')
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Euclidean Distance', fontsize=10)
        
        plt.xlabel("I(X; T)", fontsize=11, fontweight='bold')
        plt.ylabel("I(T; Y)", fontsize=11, fontweight='bold')
        plt.title(f"Layer {layer_idx+1} - Information Plane", fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 통계 정보 추가
        plt.text(0.02, 0.98, 
               f"I(X;T): [{mi_s[layer_idx].min():.3f}, {mi_s[layer_idx].max():.3f}]\n"
               f"I(T;Y): {mi_y[layer_idx][0]:.3f} (constant)\n"
               f"Distance: [{distance_s[layer_idx].min():.2f}, {distance_s[layer_idx].max():.2f}]",
               transform=plt.gca().transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"information_plane_layer_{layer_idx+1}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # 각 레이어별 개별 고해상도 그림
    for layer_idx in range(distance_s.shape[0]):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(mi_s[layer_idx], mi_y[layer_idx], 
                            c=distance_s[layer_idx], cmap='coolwarm', 
                            alpha=0.6, s=30, edgecolors='none')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Euclidean Distance', fontsize=12)
        
        ax.set_xlabel("I(X; T)", fontsize=13, fontweight='bold')
        ax.set_ylabel("I(T; Y)", fontsize=13, fontweight='bold')
        ax.set_title(f"Layer {layer_idx+1} - Information Plane (Classification)", 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 범례 추가
        plt.text(0.02, 0.98, 
               f"Data Points: {len(mi_s[layer_idx])}\n"
               f"I(X;T) range: [{mi_s[layer_idx].min():.4f}, {mi_s[layer_idx].max():.4f}]\n"
               f"I(T;Y): {mi_y[layer_idx][0]:.4f} (constant for all pixels)\n"
               f"Distance range: [{distance_s[layer_idx].min():.2f}, {distance_s[layer_idx].max():.2f}]",
               transform=plt.gca().transAxes, fontsize=11,
               verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f"layer_{layer_idx+1}_information_plane.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot: x축 I(X;T), y축 I(T;Y), z축 거리 - Contour plot
    for layer_idx in range(distance_s.shape[0]):
        # 데이터 준비
        mi_xt_data = mi_s[layer_idx]
        mi_ty_data = mi_y[layer_idx]
        dist_data = distance_s[layer_idx]
        
        # I(X;T) vs I(T;Y) 그리드 생성
        bins = 50
        xt_edges = np.linspace(mi_xt_data.min(), mi_xt_data.max(), bins + 1)
        ty_edges = np.linspace(mi_ty_data.min(), mi_ty_data.max(), bins + 1)
        
        # 각 그리드 셀의 평균 거리 계산
        xt_indices = np.digitize(mi_xt_data, xt_edges) - 1
        ty_indices = np.digitize(mi_ty_data, ty_edges) - 1
        
        # 인덱스 범위 체크
        xt_indices = np.clip(xt_indices, 0, bins - 1)
        ty_indices = np.clip(ty_indices, 0, bins - 1)
        
        # 각 셀의 평균 거리 계산
        dist_grid = np.zeros((bins, bins))
        count_grid = np.zeros((bins, bins))
        
        for i in range(len(mi_xt_data)):
            x_idx = xt_indices[i]
            y_idx = ty_indices[i]
            dist_grid[y_idx, x_idx] += dist_data[i]
            count_grid[y_idx, x_idx] += 1
        
        # 0으로 나누기 방지
        with np.errstate(divide='ignore', invalid='ignore'):
            dist_grid = np.divide(dist_grid, count_grid, out=np.zeros_like(dist_grid), where=count_grid!=0)
        
        # 그리드 중심 좌표
        xt_centers = (xt_edges[:-1] + xt_edges[1:]) / 2
        ty_centers = (ty_edges[:-1] + ty_edges[1:]) / 2
        X, Y = np.meshgrid(xt_centers, ty_centers)
        
        # Contour plot 옵션 1: 데이터 밀도 (어떤 조합이 많이 나타나는지)
        # 2D 히스토그램으로 데이터 밀도 계산
        H_density, _, _ = np.histogram2d(mi_xt_data, mi_ty_data, bins=[xt_edges, ty_edges])
        H_density = H_density.T  # transpose for correct orientation
        
        plt.figure(figsize=(10, 6))
        contour = plt.contourf(X, Y, H_density, levels=20, cmap='viridis')
        plt.colorbar(contour, label='Data Point Density')
        plt.contour(X, Y, H_density, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        plt.title(f"Layer {layer_idx+1}: I(X;T) vs I(T;Y) (Contour: Data Density)")
        plt.xlabel("I(X; T)")
        plt.ylabel("I(T; Y)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"layer_{layer_idx+1}_ixt_vs_ity_contour_density.png", dpi=100, bbox_inches='tight')
        plt.close()
        
        # Contour plot 옵션 2: 평균 거리 (거리가 어떻게 분포하는지)
        plt.figure(figsize=(10, 6))
        contour = plt.contourf(X, Y, dist_grid, levels=20, cmap='viridis')
        plt.colorbar(contour, label='Average Euclidean Distance')
        plt.contour(X, Y, dist_grid, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        plt.title(f"Layer {layer_idx+1}: I(X;T) vs I(T;Y) (Contour: Average Distance)")
        plt.xlabel("I(X; T)")
        plt.ylabel("I(T; Y)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"layer_{layer_idx+1}_ixt_vs_ity_contour_distance.png", dpi=100, bbox_inches='tight')
        plt.close()












