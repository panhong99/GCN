import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import cv2
from PIL import Image
import torch
import numpy as np

def cal_mi_same_diff(x_in, t_feat, y_gt):
    '''
    Information Plane을 위한 Same/Diff 주머니 기반 MI 계산
    
    Args:
        x_in: (N, H, W) - 입력 이미지 (Discrete/VQ)
        t_feat: (N, H, W) - 레이어 특징값 (Discrete/VQ)
        y_gt: (N, H, W) - 정답 레이블 (Ground Truth)
        
    Returns:
        results: Same/Diff 각각의 I(X;T)와 I(T;Y) 값
    '''
    N, H, W = y_gt.shape
    eps = 1e-12

    # 데이터를 모을 주머니 (Pair 수집)
    # (기준점의 값, 비교대상점의 값) 형태로 저장
    pair_xt_same, pair_ty_same = [], []
    pair_xt_diff, pair_ty_diff = [], []

    # 1. 모든 이미지와 모든 픽셀을 순회 (기준점 설정)
    for n in range(N):
        for r in range(H):
            for c in range(W):
                # 기준점의 정보
                ref_y = y_gt[n, r, c]   # 기준점의 정답 클래스
                ref_t = t_feat[n, r, c] # 기준점의 특징값
                ref_x = x_in[n, r, c]   # 기준점의 입력값

                # 이미지 내의 모든 다른 픽셀들과 비교
                # (비효율적일 경우 랜덤 샘플링 가능)
                for hr in range(H):
                    for wc in range(W):
                        comp_y = y_gt[n, hr, wc]
                        comp_t = t_feat[n, hr, wc]
                        comp_x = x_in[n, hr, wc]

                        # --- 핵심: Same vs Diff 판정 ---
                        if ref_y == comp_y: # 같은 클래스라면 (Same 주머니)
                            # I(X;T)용 페어: 기준점X와 비교점T
                            pair_xt_same.append([ref_x, comp_t])
                            # I(T;Y)용 페어: 기준점T와 비교점Y
                            pair_ty_same.append([ref_t, comp_y])
                        else: # 다른 클래스라면 (Diff 주머니)
                            pair_xt_diff.append([ref_x, comp_t])
                            pair_ty_diff.append([ref_t, comp_y])

    # 2. 모인 주머니 데이터를 기반으로 MI 계산 (함수화 추천)
    def compute_mi_from_pairs(pairs):
        pairs = np.array(pairs)
        if len(pairs) == 0: return 0
        
        # H(A), H(B), H(A,B) 계산 로직
        # 1D 변수 확률
        _, counts_a = np.unique(pairs[:, 0], return_counts=True)
        _, counts_b = np.unique(pairs[:, 1], return_counts=True)

        # Joint 확률
        _, counts_ab = np.unique(pairs, axis=0, return_counts=True)
        
        pa = counts_a / len(pairs)
        pb = counts_b / len(pairs)
        pab = counts_ab / len(pairs)
        
        h_a = -np.sum(pa * np.log2(pa + eps))
        h_b = -np.sum(pb * np.log2(pb + eps))
        h_ab = -np.sum(pab * np.log2(pab + eps))
        
        return h_a + h_b - h_ab

    # 최종 MI 점수 도출
    results = {
        "same": {
            "I_XT": compute_mi_from_pairs(pair_xt_same),
            "I_TY": compute_mi_from_pairs(pair_ty_same)
        },
        "diff": {
            "I_XT": compute_mi_from_pairs(pair_xt_diff),
            "I_TY": compute_mi_from_pairs(pair_ty_diff)
        }
    }

    return results

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

def resize_gt(gt_masks, target_size=33, num_classes=21, ignore_index=255):

    B, H, W = gt_masks.shape    
    gt_resized = np.zeros((B, target_size, target_size), dtype=np.uint8)
    scale_h, scale_w = H / target_size, W / target_size
    
    for b in range(B):
        for i in range(target_size):
            for j in range(target_size):

                # 1. 15x15(혹은 해당 스케일) 영역 추출
                r_s, r_e = int(i * scale_h), int((i + 1) * scale_h)
                c_s, c_e = int(j * scale_w), int((j + 1) * scale_w)
                region = gt_masks[b, r_s:r_e, c_s:c_e].flatten()
                
                # 2. 패딩(255) 제거
                valid = region[region != ignore_index]
                if len(valid) == 0:
                    gt_resized[b, i, j] = ignore_index
                    continue

                # 3. 객체 픽셀(1~20)만 필터링
                obj_pixels = valid[valid > 0]
                
                if len(obj_pixels) > 0:
                    # [핵심] 영역 내에 객체가 하나라도 있다면 배경(0)은 무시하고 
                    # 객체들 중에서 가장 많이 나타난 클래스를 선택
                    counts = np.bincount(obj_pixels.astype(int), minlength=num_classes)
                    gt_resized[b, i, j] = np.argmax(counts)
                else:
                    # 객체가 하나도 없을 때만 배경(0)으로 지정
                    gt_resized[b, i, j] = 0
                    
    return gt_resized

# Segmentation 코드 실행하려면 아래 주석 해제하고 classification 코드 주석 처리
if __name__ == "__main__":  # segmentation
# if False:  # segmentation (set to True to run segmentation code)
    folder_name = ["PIGNet_GSPonly", "ASPP"]
    
    # seg
    seg_file_path = f"/home/hail/pan/HDD/MI_dataset/pascal/resnet101/pretrained/ASPP/zoom/0.1"
    seg_datas = os.listdir(seg_file_path)
    
    with open(os.path.join(seg_file_path, 'gt_labels.pkl'), 'rb') as f:
        y_in = pickle.load(f)

    with open(os.path.join(seg_file_path, 'layer_0.pkl'), 'rb') as f:
        x_in = pickle.load(f)

    # Pascal: 20 classes + 1 background = 21, ignore_index=255
    y_in = resize_gt(y_in, target_size=x_in.shape[1], num_classes=21, ignore_index=255)

    t_in = []
    for i in range(1, 5):
        with open(os.path.join(seg_file_path, f'layer_{i}.pkl'), 'rb') as f:
            t_in.append(pickle.load(f))        

    # 이거 y는 원래 사이즈에 맞게 label 되어있을텐데 그걸 같은 클래스에 있는거랑 다른 클래스에 있는거랑 구분해서 
    # MI가 얼마나 크게 되고 얼마나 자근지 확인해야함
    # 또한 distance도 같이 한번 비교해봐야할 수 있음
    # 그래야 같은 클래스에 있을때 전반적으로 MI가 높은지 확인하고 
    # 그 중에서도 CNN계열은 distance가 가까울수록 MI가 더 높은지 확인 가능 (locality)
    # 반면에 GSP?를 쓰는 경우에는 둘다 가깝든 멀든 좋게 나타나는지 확인해볼 수 있음
    
    H_dim, W_dim = x_in.shape[1], x_in.shape[2]
    
    # I(X;T)
    distance_l = []
    mi_l = []

    # I(T;Y)
    distance_m = []
    mi_m = []

    for layer_idx, t_layer in enumerate(t_in):

        # I(X;T)
        mi_result, euc_result, man_result = cal_mi_x_t(x_in, t_layer)
        
        # I(T;Y)
        mi_t_y_result = cal_mi_same_diff(x_in, t_layer, y_in)
    
        # I(X;T) 관련 데이터
        distance_x_t = []
        mi_x_t = []
        
        # 동일한 cls
        distance_t_y_same = []
        mi_t_y_same = []
        
        # 다른 cls
        distance_t_y_diff = []
        mi_t_y_diff = []
        
        for hx in range(H_dim):
            for wx in range(W_dim):
                mi_flat_x_t = mi_result[:,:,hx,wx].flatten()
                euc_flat_x_t = euc_result[:,:,hx,wx].flatten()
                distance_x_t.extend(euc_flat_x_t.tolist())
                mi_x_t.extend(mi_flat_x_t.tolist())

        y_class = y_in[0]  # (H_dim, W_dim)
        
        # 4D boolean mask: ht,wt 위치 클래스 == hx,wx 위치 클래스
        same_class_mask = (y_class[:, :, np.newaxis, np.newaxis] == 
                          y_class[np.newaxis, np.newaxis, :, :])  # (H_dim, W_dim, H_dim, W_dim)
        
        # 4D 배열을 1D로 flatten
        mi_flat = mi_result_ty.reshape(-1)
        dist_flat = euc_result_ty.reshape(-1)
        same_mask_flat = same_class_mask.reshape(-1)
        
        # boolean indexing으로 같은/다른 클래스 분리
        distance_t_y_same = dist_flat[same_mask_flat].tolist()
        mi_t_y_same = mi_flat[same_mask_flat].tolist()

        distance_t_y_diff = dist_flat[~same_mask_flat].tolist()
        mi_t_y_diff = mi_flat[~same_mask_flat].tolist()
                
        # I(X;T) 데이터 저장
        distance_l.append(distance_x_t)
        mi_l.append(mi_x_t)

        # 같은/다른 클래스 데이터 저장
        distance_m.append([distance_t_y_same, distance_t_y_diff])
        mi_m.append([mi_t_y_same, mi_t_y_diff])

    distance_l = np.array(distance_l)
    mi_l = np.array(mi_l)
    
    # I(X;T) vs Distance 그래프
    for layer_idx in range(distance_l.shape[0]):
        plt.figure(figsize=(10, 6))
        plt.scatter(distance_l[layer_idx], mi_l[layer_idx], alpha=0.1, s=5)
        plt.title(f"Seg Layer {layer_idx+1}: I(X; T) vs Distance")
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Mutual Information I(X; T)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"seg_layer_{layer_idx+1}_x_t_vs_distance.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    # I(T;Y) vs Distance 그래프 (같은/다른 클래스 구분)
    for layer_idx in range(len(distance_m)):
        distance_t_y_same, distance_t_y_diff = distance_m[layer_idx]
        mi_t_y_same, mi_t_y_diff = mi_m[layer_idx]
        
        plt.figure(figsize=(12, 6))
        
        # Different Class (파란색 계열 colormap)
        if distance_t_y_diff:
            plt.scatter(distance_t_y_diff, mi_t_y_diff, c='blue', alpha=0.3, s=5, label='Different Class', zorder=1)
        if distance_t_y_same:
            plt.scatter(distance_t_y_same, mi_t_y_same, c='red', alpha=0.5, s=5, label='Same Class', zorder=2)
        
        plt.title(f"Seg Layer {layer_idx+1}: I(T; Y) vs Distance (Class Distinction)")
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Mutual Information I(T; Y)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"seg_layer_{layer_idx+1}_t_y_vs_distance.png", dpi=100, bbox_inches='tight')
        plt.close()
    
# if __name__ == "__main__": # clssification

if False:
    # cls
    cls_file_path = f"/home/hail/pan/HDD/MI_dataset/CIFAR-10/resnet101/pretrained/PIGNet_GSPonly_classification/zoom/1"
    
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
                mi_flat = mi_result[:,:,hx,wx].flatten()
                euc_flat = euc_result[:,:,hx,wx].flatten()
                mi_t_y_flat = mi_t_y.flatten()
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
        plt.figure()
        plt.scatter(distance_s[layer_idx], mi_s[layer_idx], alpha=0.1)
        plt.title(f"Layer {layer_idx+1} MI vs Euclidean Distance")
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Mutual Information I(X; T_layer)")
        plt.grid(True)
        plt.savefig(f"layer_{layer_idx+1}_mi_vs_distance.png")
        plt.close()
    
    # Plot: x축 I(X;T), y축 I(T;Y), 색상 거리
    for layer_idx in range(distance_s.shape[0]):
        plt.figure()
        scatter = plt.scatter(mi_s[layer_idx], mi_y[layer_idx], 
                             c=distance_s[layer_idx], cmap='viridis', alpha=0.3)
        plt.colorbar(scatter, label='Euclidean Distance')
        plt.title(f"Layer {layer_idx+1} I(X;T) vs I(T;Y)")
        plt.xlabel("I(X; T)")
        plt.ylabel("I(T; Y)")
        plt.grid(True)
        plt.savefig(f"layer_{layer_idx+1}_ixt_vs_ity.png")
        plt.close()












