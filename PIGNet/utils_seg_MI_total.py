import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import cv2
from PIL import Image
import torch
import numpy as np
from tqdm.auto import trange

def cal_mi_x_t(x, t, y):
    '''
    Calculate Pairwise Mutual Information and Spatial Distances (Optimized).
    Computed on all points (no mask).
    
    Args:
        x: (N, H, W) - Discrete Target maps
        t: (N, H, W) - Discrete Feature maps
        y: (N, H, W) - GT maps (not used for masking, for compatibility)
        
    Returns:
        mi_map:  (H, W, H, W) - MI between T[ht,wt] and X[hx,wx]
        euc_map: (H, W, H, W) - Spatial Euclidean dist
        man_map: (H, W, H, W) - Spatial Manhattan dist
        h_t_all: entropy of T
    '''
    N, H, W = t.shape
    mi_map = np.zeros((H, W, H, W))
    eps = 1e-12
    
    x_flat = x.reshape(N, -1)     # (N, H*W)
    t_flat = t.reshape(N, -1)     # (N, H*W)
    num_pixels = H * W
    
    # ===== 1. 거리 맵 사전 계산 (최적화) =====
    h_coords = np.arange(num_pixels) // W
    w_coords = np.arange(num_pixels) % W
    
    h_diff = h_coords[:, np.newaxis] - h_coords[np.newaxis, :]
    w_diff = w_coords[:, np.newaxis] - w_coords[np.newaxis, :]
    
    euc_map_flat = np.sqrt(h_diff**2 + w_diff**2)
    man_map_flat = np.abs(h_diff) + np.abs(w_diff)
    
    # Reshape to 4D
    euc_map = euc_map_flat.reshape(H, W, H, W)
    man_map = man_map_flat.reshape(H, W, H, W)
    
    # ===== 2. Entropy 사전 계산 (모든 데이터) =====
    h_x_all = np.zeros(num_pixels)
    h_t_all = np.zeros(num_pixels)
    
    for i in trange(num_pixels, desc="Computing Entropies", leave=False):
        # 모든 데이터 사용 (mask 없음)
        _, counts_x = np.unique(x_flat[:, i], return_counts=True)
        p_x = counts_x / N
        h_x_all[i] = -np.sum(p_x * np.log2(p_x + eps))
        
        _, counts_t = np.unique(t_flat[:, i], return_counts=True)
        p_t = counts_t / N
        h_t_all[i] = -np.sum(p_t * np.log2(p_t + eps))
    
    # ===== 3. MI 계산 (모든 데이터) =====
    for i_t in trange(num_pixels, desc="Computing MI", leave=False):
        t_vec = t_flat[:, i_t]
        h_t_val = h_t_all[i_t]
        ht, wt = divmod(i_t, W)
        
        for i_x in range(num_pixels):
            x_vec = x_flat[:, i_x]
            hx, wx = divmod(i_x, W)
            
            # Joint entropy 계산
            joint_data = np.column_stack([t_vec, x_vec])
            _, counts_tx = np.unique(joint_data, axis=0, return_counts=True)
            p_tx = counts_tx / N
            h_tx = -np.sum(p_tx * np.log2(p_tx + eps))
            
            # MI
            mi_val = h_t_val + h_x_all[i_x] - h_tx
            mi_map[ht, wt, hx, wx] = mi_val
    
    return mi_map, euc_map, man_map, h_t_all

def cal_seg_mi_t_y(t, y, h_t_all):
    """
    Calculate MI(T; Y) without recomputing H(T).
    Computed on all points (no mask).
    
    Args:
        t: (N, H, W) - Discrete Feature maps
        y: (N, H, W) - Discrete Target maps (GT)
        h_t_all: (num_pixels,) - Pre-computed H(T) from cal_mi_x_t
    
    Returns:
        mi_map: (H, W, H, W) - MI between T[t_pos] and Y[y_pos]
        euc_map: (H, W, H, W) - Euclidean distance
        man_map: (H, W, H, W) - Manhattan distance
    """
    N, H, W = t.shape
    num_pixels = H * W
    eps = 1e-12
    
    # Flatten
    t_flat = t.reshape(N, -1)  # (N, num_pixels)
    y_flat = y.reshape(N, -1)  # (N, num_pixels)
    
    # Initialize outputs
    mi_map = np.zeros((H, W, H, W))
    euc_map = np.zeros((H, W, H, W))
    man_map = np.zeros((H, W, H, W))
    
    # ===== Pre-compute distance maps (최적화) =====
    h_coords = np.arange(num_pixels) // W
    w_coords = np.arange(num_pixels) % W
    
    h_diff = h_coords[:, np.newaxis] - h_coords[np.newaxis, :]
    w_diff = w_coords[:, np.newaxis] - w_coords[np.newaxis, :]
    
    euc_map_flat = np.sqrt(h_diff**2 + w_diff**2)
    man_map_flat = np.abs(h_diff) + np.abs(w_diff)
    
    euc_map = euc_map_flat.reshape(H, W, H, W)
    man_map = man_map_flat.reshape(H, W, H, W)
    
    # ===== Pre-compute H(Y) for each position (모든 데이터) =====
    h_y_all = np.zeros(num_pixels)
    for i in trange(num_pixels, desc="Computing H(Y)", leave=False):
        # 모든 데이터 사용 (mask 없음)
        _, counts = np.unique(y_flat[:, i], return_counts=True)
        p = counts / N
        h_y_all[i] = -np.sum(p * np.log2(p + eps))
    
    # ===== Main computation (use pre-computed h_t_all, 모든 데이터) =====
    for i_ref in trange(num_pixels, desc="Computing MI(T;Y)", leave=False):
        h_ref, w_ref = divmod(i_ref, W)
        y_ref_vec = y_flat[:, i_ref]
        h_y_ref = h_y_all[i_ref]
        
        # Compute MI for all comparison points
        for i_comp in range(num_pixels):
            h_comp, w_comp = divmod(i_comp, W)
            t_comp_vec = t_flat[:, i_comp]
            
            # Use pre-computed H(T_comp)
            h_t_comp = h_t_all[i_comp]
            
            # H(T_comp, Y_ref) - joint entropy
            joint_data = np.column_stack([t_comp_vec, y_ref_vec])
            _, counts_joint = np.unique(joint_data, axis=0, return_counts=True)
            p_joint = counts_joint / N
            h_joint = -np.sum(p_joint * np.log2(p_joint + eps))
            
            # MI(T_comp; Y_ref) = H(T) + H(Y) - H(T, Y)
            mi_val = h_t_comp + h_y_ref - h_joint
            mi_map[h_comp, w_comp, h_ref, w_ref] = max(0, mi_val)

    return mi_map, euc_map, man_map

def resize_gt(gt_masks, target_size=33):

    B, H, W = gt_masks.shape    
    gt_resized = np.zeros((B, target_size, target_size), dtype=np.uint8)
    scale_h, scale_w = H / target_size, W / target_size
    
    for b in range(B):
        for i in range(target_size):
            for j in range(target_size):

                # 1. 지정된 범위 내 영역 추출
                r_s, r_e = int(i * scale_h), int((i + 1) * scale_h)
                c_s, c_e = int(j * scale_w), int((j + 1) * scale_w)
                region = gt_masks[b, r_s:r_e, c_s:c_e].flatten()
                
                # 2. 255를 0으로 변경
                region = np.where(region == 255, 0, region).astype(int)
                
                # 3. 영역 내에서 가장 많이 나타나는 클래스 선택 (최빈값)
                unique, counts = np.unique(region, return_counts=True)
                gt_resized[b, i, j] = unique[np.argmax(counts)]
                    
    return gt_resized

# Segmentation 코드 실행하려면 아래 주석 해제하고 classification 코드 주석 처리
if __name__ == "__main__":  # segmentation
# if False:  # segmentation (set to True to run segmentation code)
    folder_name = ["PIGNet_GSPonly", "ASPP"]
    
    # seg
    seg_file_path = f"/home/hail/pan/HDD/MI_dataset/pascal/total_dataset/resnet101/pretrained/PIGNet_GSPonly/zoom/1"
    seg_datas = os.listdir(seg_file_path)
    
    with open(os.path.join(seg_file_path, 'gt_labels.pkl'), 'rb') as f:
        y_in = pickle.load(f)

    with open(os.path.join(seg_file_path, 'layer_0.pkl'), 'rb') as f:
        x_in = pickle.load(f)

    t_in = []
    for i in range(1, 5):
        with open(os.path.join(seg_file_path, f'layer_{i}.pkl'), 'rb') as f:
            t_in.append(pickle.load(f))        
    
    H_dim, W_dim = x_in.shape[1], x_in.shape[2]
    
    # 모든 레이어 데이터를 저장할 리스트
    all_distance_x_t = []
    all_mi_x_t = []
    all_distance_t_y = []
    all_mi_t_y = []
    class_mi_t_y = []

    for layer_idx, t_layer in enumerate(t_in):

        # I(X;T) - returns h_t_all as well (computed on valid points Y > 0)
        mi_result_xt, euc_result_xt, man_result_xt, h_t_all = cal_mi_x_t(x_in, t_layer, y_in)
        
        # I(T;Y) - computed on valid points Y > 0
        mi_result_ty, euc_result_ty, man_result_ty = cal_seg_mi_t_y(t_layer, y_in, h_t_all)
        
        # 각 레이어의 데이터 (초기화)
        distance_x_t = []
        mi_x_t = []
        distance_t_y = []
        mi_t_y = []
        
        # Debug: 유효한 포인트 개수 확인
        valid_count_ty = np.sum(~np.isnan(mi_result_ty))
        total_pixels = H_dim * W_dim * H_dim * W_dim
        print(f"Layer {layer_idx}: Valid MI(T;Y) points: {valid_count_ty}/{total_pixels} ({100*valid_count_ty/total_pixels:.1f}%)")
        
        for hx in range(H_dim):
            for wx in range(W_dim):
                mi_flat_xt = mi_result_xt[:,:,hx,wx].flatten()
                euc_flat_xt = euc_result_xt[:,:,hx,wx].flatten()
                mi_flat_ty = mi_result_ty[:,:,hx,wx].flatten()
                euc_flat_ty = euc_result_ty[:,:,hx,wx].flatten()
                
                distance_x_t.extend(euc_flat_xt.tolist())
                distance_t_y.extend(euc_flat_ty.tolist())
                mi_x_t.extend(mi_flat_xt.tolist())
                mi_t_y.extend(mi_flat_ty.tolist())

        # 각 레이어의 데이터를 저장
        all_distance_x_t.append(distance_x_t)
        all_distance_t_y.append(distance_t_y)
        all_mi_x_t.append(mi_x_t)
        all_mi_t_y.append(mi_t_y)

    # NumPy 배열로 변환
    distance_x_t = np.array(all_distance_x_t, dtype=object)
    distance_t_y = np.array(all_distance_t_y, dtype=object)
    mi_x_t = np.array(all_mi_x_t, dtype=object)
    mi_t_y = np.array(all_mi_t_y, dtype=object)
    
    # 계산 결과를 pickle로 저장
    cache_file = os.path.join(seg_file_path, 'mi_analysis_cache.pkl')
    print(f"Saving computed data to {cache_file}...")
    cache_data = {
        'distance_x_t': distance_x_t,
        'distance_t_y': distance_t_y,
        'mi_x_t': mi_x_t,
        'mi_t_y': mi_t_y
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print("Cache saved successfully!")
    
    # 폰트 사이즈 설정
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 17
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
        
    for layer_idx in range(distance_t_y.shape[0]):
        plt.figure(figsize=(12, 7))
        
        scatter = plt.scatter(mi_x_t[layer_idx], mi_t_y[layer_idx], 
                            c=distance_t_y[layer_idx], cmap='coolwarm', 
                            alpha=0.5, s=20, edgecolors='none')
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Euclidean Distance', fontsize=10)
        
        plt.xlabel("I(X; T)", fontsize=11, fontweight='bold')
        plt.ylabel("I(T; Y)", fontsize=11, fontweight='bold')
        plt.xlim(0,4)
        plt.ylim(0,4)
        plt.title(f"Layer {layer_idx+1} - Information Plane_", fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)        
        plt.tight_layout()
        plt.savefig(f"total_information_plane_layer_{layer_idx+1}_GSP.png", dpi=150, bbox_inches='tight')
        plt.close()










