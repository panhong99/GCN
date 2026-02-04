import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import cv2
from PIL import Image
import torch
import numpy as np
from tqdm.auto import trange

def cal_mi_x_t(x, t):
    N, H, W = t.shape
    num_pixels = H * W
    eps = 1e-12
    
    # 1. 데이터를 평탄화하고 정수형으로 변환 (인덱싱을 위함)
    x_flat = x.reshape(N, -1).astype(np.int32) # (N, num_pixels)
    t_flat = t.reshape(N, -1).astype(np.int32) # (N, num_pixels)
    
    # 2. 값의 범위를 파악 (K-Means bin 수인 50 내외 가정)
    max_x = x_flat.max() + 1
    max_t = t_flat.max() + 1
    
    # 3. 개별 엔트로피 계산 (벡터화)
    def get_entropy(data, num_bins):
        # bincount를 사용하여 각 픽셀 위치의 확률 분포를 한 번에 계산
        # data: (N, num_pixels)
        counts = np.array([np.bincount(data[:, i], minlength=num_bins) for i in range(num_pixels)])
        p = counts / N
        return -np.sum(p * np.log2(p + eps), axis=1)

    h_x_all = get_entropy(x_flat, max_x) # (num_pixels,)
    h_t_all = get_entropy(t_flat, max_t) # (num_pixels,)

    # 4. MI 계산 (핵심 최적화 부분)
    mi_map_flat = np.zeros((num_pixels, num_pixels))
    
    # Joint Entropy를 위한 정수 인코딩: (t, x) 쌍을 하나의 숫자로 변환
    # pair_val = t * max_x + x
    for i_t in trange(num_pixels, desc="Optimized MI", leave=False):
        t_vec = t_flat[:, i_t]
        
        # 모든 i_x에 대해 한 번에 Joint 계산 시도
        # (N, 1) * max_x + (N, num_pixels) -> (N, num_pixels)
        joint_encoded = t_vec[:, np.newaxis] * max_x + x_flat
        
        for i_x in range(num_pixels):
            # bincount가 unique보다 훨씬 빠름
            counts_tx = np.bincount(joint_encoded[:, i_x], minlength=max_t * max_x)
            p_tx = counts_tx / N
            h_tx = -np.sum(p_tx * np.log2(p_tx + eps))
            
            mi_map_flat[i_t, i_x] = h_t_all[i_t] + h_x_all[i_x] - h_tx

    # 5. 거리 맵 계산 (기존과 동일하되 최적화)
    grid = np.indices((H, W)).reshape(2, -1).T
    h_diff = grid[:, 0:1] - grid[:, 0:1].T
    w_diff = grid[:, 1:2] - grid[:, 1:2].T
    
    euc_map = np.sqrt(h_diff**2 + w_diff**2).reshape(H, W, H, W)
    man_map = (np.abs(h_diff) + np.abs(w_diff)).reshape(H, W, H, W)
    mi_map = mi_map_flat.reshape(H, W, H, W)
    
    return mi_map, euc_map, man_map, h_t_all

def cal_seg_mi_t_y(t, y, h_t_all):
    N, H, W = t.shape
    num_pixels = H * W
    eps = 1e-12
    
    # 1. 데이터를 정수형으로 평탄화
    t_flat = t.reshape(N, -1).astype(np.int32)
    y_flat = y.reshape(N, -1).astype(np.int32)
    
    # 값의 최대 범위 파악 (인코딩을 위함)
    max_t = t_flat.max() + 1
    max_y = y_flat.max() + 1
    
    # 2. H(Y) 사전 계산 (bincount 활용)
    h_y_all = np.zeros(num_pixels)
    for i in range(num_pixels):
        y_vec = y_flat[:, i]
        # 원본 코드의 조건(y > 0) 유지 (valid한 부분만)
        # y_valid = y_vec[y_vec > 0]
        # if len(y_valid) > 0:
        counts = np.bincount(y_vec)
        p = counts[counts > 0] / len(y_vec)
        h_y_all[i] = -np.sum(p * np.log2(p + eps))

    # 3. MI(T; Y) 계산 (핵심 최적화)
    mi_map_flat = np.zeros((num_pixels, num_pixels))
    
    # i_ref(Y의 위치)를 기준으로 고정하고 i_comp(T의 위치)를 순회
    for i_ref in trange(num_pixels, desc="Optimized MI(T;Y)", leave=False):
        y_vec = y_flat[:, i_ref]
        h_y_ref = h_y_all[i_ref]
        
        # Joint 계산을 위한 인코딩: (T, Y) 쌍을 하나의 정수로
        # (N, num_pixels) 구조로 한 번에 인코딩 가능
        # t_flat: (N, num_pixels), y_vec: (N,)
        joint_encoded = t_flat * max_y + y_vec[:, np.newaxis]
        
        for i_comp in range(num_pixels):
            # i_comp 위치의 T와 i_ref 위치의 Y 사이의 Joint Entropy
            counts_joint = np.bincount(joint_encoded[:, i_comp], minlength=max_t * max_y)
            p_joint = counts_joint / N
            h_joint = -np.sum(p_joint * np.log2(p_joint + eps))
            
            # MI(T_comp; Y_ref)
            mi_val = h_t_all[i_comp] + h_y_ref - h_joint
            mi_map_flat[i_comp, i_ref] = max(0, mi_val)

    # 4. 거리 맵 계산 (동일)
    grid = np.indices((H, W)).reshape(2, -1).T
    h_diff = grid[:, 0:1] - grid[:, 0:1].T
    w_diff = grid[:, 1:2] - grid[:, 1:2].T
    
    euc_map = np.sqrt(h_diff**2 + w_diff**2).reshape(H, W, H, W)
    man_map = (np.abs(h_diff) + np.abs(w_diff)).reshape(H, W, H, W)
    mi_map = mi_map_flat.reshape(H, W, H, W)

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
    seg_file_path = f"/home/hail/pan/HDD/MI_dataset/pascal/pixel_dataset/resnet101/pretrained/PIGNet_GSPonly/zoom/1"
    seg_datas = os.listdir(seg_file_path)
    
    with open(os.path.join(seg_file_path, 'gt_labels.pkl'), 'rb') as f:
        y_in = pickle.load(f)
        y_in = resize_gt(y_in, target_size=33)

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

    for layer_idx, t_layer in enumerate(t_in):

        # I(X;T) - returns h_t_all as well (computed on valid points Y > 0)
        mi_result_xt, euc_result_xt, man_result_xt, h_t_all = cal_mi_x_t(x_in, t_layer)
        
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
                # euc_flat_ty = euc_result_ty[:,:,hx,wx].flatten()
                
                distance_x_t.extend(euc_flat_xt.tolist())
                # distance_t_y.extend(euc_flat_ty.tolist())
                mi_x_t.extend(mi_flat_xt.tolist())
                mi_t_y.extend(mi_flat_ty.tolist())

        # 각 레이어의 데이터를 저장
        all_distance_x_t.append(distance_x_t)
        # all_distance_t_y.append(distance_t_y)
        all_mi_x_t.append(mi_x_t)
        all_mi_t_y.append(mi_t_y)

    # NumPy 배열로 변환
    distance_x_t = np.array(all_distance_x_t, dtype=object)
    # distance_t_y = np.array(all_distance_t_y, dtype=object)
    mi_x_t = np.array(all_mi_x_t, dtype=object)
    mi_t_y = np.array(all_mi_t_y, dtype=object)
    
    # 계산 결과를 pickle로 저장
    cache_file = os.path.join(seg_file_path, 'mi_analysis_cache.pkl')
    print(f"Saving computed data to {cache_file}...")
    cache_data = {
        'distance_x_t': distance_x_t,
        # 'distance_t_y': distance_t_y,
        'mi_x_t': mi_x_t,
        'mi_t_y': mi_t_y,
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
        
    for layer_idx in range(distance_x_t.shape[0]):
        plt.figure(figsize=(12, 7))
        
        # 1. 유효한 데이터 추출 (NaN 제거)
        mx = mi_x_t[layer_idx]
        my = mi_t_y[layer_idx]
        dist = distance_x_t[layer_idx]
        
        # mask = ~np.isnan(my)
        # mx, my, dist = mx[mask], my[mask], dist[mask]

        # 2. 데이터 샘플링 (10만 개 초과 시 무작위 추출)
        # 100만 개를 다 그리는 것보다 10만 개만 그려도 경향성은 완벽히 유지됩니다.
        # if len(mx) > 100000:
        #     idx = np.random.choice(len(mx), 100000, replace=False)
        #     mx, my, dist = mx[idx], my[idx], dist[idx]

        # 3. Scatter 최적화
        # rasterized=True: 수많은 점을 벡터가 아닌 비트맵으로 렌더링하여 저장 속도 향상
        scatter = plt.scatter(mx, my, 
                            c=dist, cmap='coolwarm', 
                            alpha=0.4, s=15, # 점 크기를 살짝 줄여 겹침 방지
                            edgecolors='none',
                            rasterized=True) 
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Euclidean Distance', fontsize=10)
        
        plt.xlabel("I(X; T)", fontsize=11, fontweight='bold')
        plt.ylabel("I(T; Y)", fontsize=11, fontweight='bold')
        plt.title(f"Layer {layer_idx+1} - Information Plane", fontsize=12, fontweight='bold')
        
        # 축 범위 고정 (레이어 간 비교를 위해)
        plt.xlim(0, max(mx.max(), 1.0) * 1.1)
        plt.ylim(0, max(my.max(), 1.0) * 1.1)
        
        plt.grid(True, alpha=0.3)        
        plt.tight_layout()
        
        # 저장 속도 향상을 위해 dpi 조정 및 불필요한 여백 제거
        plt.savefig(f"GSP_pixel_layer_{layer_idx+1}.png", 
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Layer {layer_idx+1} plot saved.")










