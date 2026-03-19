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

def cal_mi_x_t(x, t):
    N, H, W = t.shape
    num_pixels = H * W
    eps = 1e-12
    
    # 1. 데이터를 평탄화하고 정수형으로 변환 (인덱싱을 위함)
    x_flat = x.reshape(N, -1).astype(np.int32) # (N, num_pixels)
    t_flat = t.reshape(N, -1).astype(np.int32) # (N, num_pixels)
    
    # 2. 전체 데이터 사용 (+1 shift로 -1도 포함: -1→0, 0-49→1-50)
    x_flat = x_flat + 1
    t_flat = t_flat + 1
    
    max_x = 51  # 0(invalid) + 1-50(valid)
    max_t = 51  # 0(invalid) + 1-50(valid)
    
    # 3. 개별 엔트로피 계산 (벡터화) - 전체 데이터 사용
    def get_entropy_all(data, num_bins):
        """모든 데이터(invalid 포함)를 사용하여 엔트로피 계산"""
        h = np.zeros(num_pixels)
        for i in range(num_pixels):
            data_i = data[:, i]
            counts = np.bincount(data_i, minlength=num_bins)
            p = counts / len(data_i)
            h[i] = -np.sum(p * np.log2(p + eps))
        return h

    h_x_all = get_entropy_all(x_flat, max_x)
    h_t_all = get_entropy_all(t_flat, max_t)

    # 4. MI 계산 (벡터화된 접근) - 전체 데이터 사용
    mi_map_flat = np.zeros((num_pixels, num_pixels))
    
    for i_t in trange(num_pixels, desc="Optimized MI", leave=False):
        t_vec = t_flat[:, i_t]
        
        # 모든 i_x에 대해 한 번에 처리
        # x_flat: (N, num_pixels)
        
        # joint encoding (전체 데이터)
        joint_encoded = t_vec[:, np.newaxis] * max_x + x_flat
        
        for i_x in range(num_pixels):
            x_vec = x_flat[:, i_x]
            # 전체 데이터로 joint 엔트로피 계산
            joint_all = joint_encoded[:, i_x]
            counts_tx = np.bincount(joint_all, minlength=max_t * max_x)
            p_tx = counts_tx / N
            h_tx = -np.sum(p_tx * np.log2(p_tx + eps))
            
            mi_map_flat[i_t, i_x] = h_t_all[i_t] + h_x_all[i_x] - h_tx

    # 5. 거리 맵 계산
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
    
    # 2. 데이터 범위 설정
    # t: -1→0, 0-49→1-50 (범위: 0-50, +1 shift)
    # y: 이미 0-20 범위 (shift 없음)
    t_flat = t_flat + 1
    
    max_t = 51  # 0(invalid) + 1-50(valid)
    max_y = 21  # 0-20 범위
    
    # 3. H(Y) 사전 계산 (전체 데이터)
    h_y_all = np.zeros(num_pixels)
    for i in range(num_pixels):
        y_vec = y_flat[:, i]
        counts = np.bincount(y_vec, minlength=max_y)
        p = counts / len(y_vec)
        h_y_all[i] = -np.sum(p * np.log2(p + eps))

    # 4. MI(T; Y) 계산 (벡터화) - 전체 데이터 사용
    mi_map_flat = np.zeros((num_pixels, num_pixels))
    
    for i_ref in trange(num_pixels, desc="Optimized MI(T;Y)", leave=False):
        y_vec = y_flat[:, i_ref]
        h_y_ref = h_y_all[i_ref]
        
        for i_comp in range(num_pixels):
            t_vec = t_flat[:, i_comp]
            
            # 전체 데이터로 joint 엔트로피 계산
            joint_encoded = t_vec * max_y + y_vec
            counts_joint = np.bincount(joint_encoded, minlength=max_t * max_y)
            p_joint = counts_joint / N
            h_joint = -np.sum(p_joint * np.log2(p_joint + eps))
            
            # MI(T_comp; Y_ref)
            mi_val = h_t_all[i_comp] + h_y_ref - h_joint
            mi_map_flat[i_comp, i_ref] = max(0, mi_val)

    # 5. 거리 맵 계산
    grid = np.indices((H, W)).reshape(2, -1).T
    h_diff = grid[:, 0:1] - grid[:, 0:1].T
    w_diff = grid[:, 1:2] - grid[:, 1:2].T
    
    euc_map = np.sqrt(h_diff**2 + w_diff**2).reshape(H, W, H, W)
    man_map = (np.abs(h_diff) + np.abs(w_diff)).reshape(H, W, H, W)
    mi_map = mi_map_flat.reshape(H, W, H, W)

    return mi_map, euc_map, man_map

# Segmentation 코드 실행하려면 아래 주석 해제하고 classification 코드 주석 처리
if __name__ == "__main__":  # segmentation
# if False:  # segmentation (set to True to run segmentation code)
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='pascal', help='pascal or cityscape')
    argparser.add_argument('--preprocess_type', type=str, default='layer', help='layer or pixel')
    argparser.add_argument('--model', type=str, default='ASPP', help='ASPP or PIGNet_GSPonly')
    args = argparser.parse_args()
    
    # seg
    seg_file_path = f"/home/hail/pan/HDD/MI_dataset/{args.preprocess_type}_dataset/{args.dataset}/resnet101/pretrained/{args.model}/zoom/1"
    seg_datas = os.listdir(seg_file_path)
    
    with open(os.path.join(seg_file_path, 'gt_labels.pkl'), 'rb') as f:
        y_in = pickle.load(f)
    
    # -1을 0으로 변환 (invalid: 0, valid classes: 1-20)
    y_in = np.where(y_in == -1, 0, y_in)

    with open(os.path.join(seg_file_path, 'layer_0.pkl'), 'rb') as f:
        x_in = pickle.load(f)
    
    # Debug: GT 유효 포인트 분포 확인
    valid_per_sample = np.sum(y_in > 0, axis=(1, 2))
    print(f"\n=== GT Valid Points Statistics ===")
    print(f"Total samples: {y_in.shape[0]}")
    print(f"Valid points per sample - Min: {valid_per_sample.min()}, Max: {valid_per_sample.max()}, Mean: {valid_per_sample.mean():.1f}")
 
    t_in = []
    for i in range(1, 5):
        with open(os.path.join(seg_file_path, f'layer_{i}.pkl'), 'rb') as f:
            t_in.append(pickle.load(f))        
    
    H_dim, W_dim = x_in.shape[1], x_in.shape[2]
    
    # 모든 레이어 데이터를 저장할 리스트
    all_distance_x_t = []
    all_distance_t_y = []
    all_mi_x_t = []
    all_mi_t_y = []

    for layer_idx, t_layer in enumerate(t_in):

        mi_result_xt, euc_result_xt, _, h_t_all = cal_mi_x_t(x_in, t_layer)
        mi_result_ty, euc_result_ty, _ = cal_seg_mi_t_y(t_layer, y_in, h_t_all)
        
        # I(X;T) 데이터 추출
        mi_x_t_flat = mi_result_xt.flatten()
        dist_x_t_flat = euc_result_xt.flatten()
        
        # I(T;Y) 데이터 추출
        mi_t_y_flat = mi_result_ty.flatten()
        dist_t_y_flat = euc_result_ty.flatten()

        # 5. 결과 리스트에 추가 (여전히 numpy 배열 상태 유지)
        all_distance_x_t.append(dist_x_t_flat)
        all_distance_t_y.append(dist_t_y_flat)
        all_mi_x_t.append(mi_x_t_flat)
        all_mi_t_y.append(mi_t_y_flat)

    distance_x_t = np.array(all_distance_x_t)
    distance_t_y = np.array(all_distance_t_y)
    mi_x_t = np.array(all_mi_x_t)
    mi_t_y = np.array(all_mi_t_y)
    
    # 계산 결과를 pickle로 저장
    cache_file = os.path.join(seg_file_path, 'mi_analysis_cache.pkl')
    print(f"Saving computed data to {cache_file}...")
    cache_data = {
        'distance_x_t': distance_x_t,
        'distance_t_y': distance_t_y,
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
        
    for layer_idx in range(distance_t_y.shape[0]):
        plt.figure(figsize=(12, 7))
        
        # 1. 유효한 데이터 추출 (NaN 제거)
        mx = mi_x_t[layer_idx]
        my = mi_t_y[layer_idx]
        dist = distance_t_y[layer_idx]
        
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
        plt.xlim(0, 4)
        plt.ylim(0, 4)
        
        plt.grid(True, alpha=0.3)        
        plt.tight_layout()
        
        # 저장 속도 향상을 위해 dpi 조정 및 불필요한 여백 제거
        plt.savefig(f"{args.model}_{args.dataset}_{layer_idx+1}.png", 
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Layer {layer_idx+1} plot saved.")
    
    # =============== 거리 구간별 플롯 ===============
    # 거리 구간 정의 (0-9, 10-19, ..., 50+)
    distance_ranges = [
        (0, 9, "0-9"),
        (10, 19, "10-19"),
        (20, 29, "20-29"),
        (30, 39, "30-39"),
        (40, 49, "40-49"),
        (50, float('inf'), "50+")
    ]
    
    for layer_idx in range(distance_t_y.shape[0]):
        mx = mi_x_t[layer_idx]
        my = mi_t_y[layer_idx]
        dist = distance_t_y[layer_idx]
        
        print(f"\n=== Layer {layer_idx+1} - Distance Range Summary ===")
        
        for dist_min, dist_max, dist_label in distance_ranges:
            # 거리 구간에 해당하는 데이터만 필터링
            mask = (dist >= dist_min) & (dist <= dist_max)
            mx_filtered = mx[mask]
            my_filtered = my[mask]
            
            num_points = len(mx_filtered)
            print(f"Distance [{dist_label:5s}]: {num_points:7d} points", end="")
            
            # 데이터가 있는 경우에만 플롯 생성
            if num_points > 0:
                plt.figure(figsize=(12, 7))
                
                plt.scatter(mx_filtered, my_filtered, 
                           alpha=0.6, s=20, 
                           color='red',
                           edgecolors='none',
                           rasterized=True)
                
                plt.xlabel("I(X; T)", fontsize=11, fontweight='bold')
                plt.ylabel("I(T; Y)", fontsize=11, fontweight='bold')
                plt.title(f"Layer {layer_idx+1} - Distance Range [{dist_label}] - Information Plane", 
                         fontsize=12, fontweight='bold')
                
                # 축 범위 설정
                plt.xlim(0, 4)
                plt.ylim(0, 4)
                
                plt.grid(True, alpha=0.3)        
                plt.tight_layout()
                
                # 파일명: model_dataset_layer_distance_range.png
                plt.savefig(f"{args.model}_{args.dataset}_{layer_idx+1}_{dist_label}.png", 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                print(" -> saved")
                
                # ===== 각 구간별 최빈값, 중앙값 추출 (주석 처리) =====
                # from scipy import stats
                # 
                # # 최빈값 (mode) 계산 - I(X;T)
                # mi_x_t_mode_result = stats.mode(np.round(mx_filtered, 1), keepdims=True)
                # mi_x_t_mode = mi_x_t_mode_result.mode[0] if len(mi_x_t_mode_result.mode) > 0 else np.nan
                # 
                # # 최빈값 (mode) 계산 - I(T;Y)
                # mi_t_y_mode_result = stats.mode(np.round(my_filtered, 1), keepdims=True)
                # mi_t_y_mode = mi_t_y_mode_result.mode[0] if len(mi_t_y_mode_result.mode) > 0 else np.nan
                # 
                # # 중앙값 (median) 계산
                # mi_x_t_median = np.median(mx_filtered)
                # mi_t_y_median = np.median(my_filtered)
                # 
                # print(f"Layer {layer_idx+1} - Distance [{dist_label}]:")
                # print(f"  I(X;T) - Mode: {mi_x_t_mode:.3f}, Median: {mi_x_t_median:.3f}, Count: {num_points}")
                # print(f"  I(T;Y) - Mode: {mi_t_y_mode:.3f}, Median: {mi_t_y_median:.3f}")
            else:
                print(" -> skipped (no data)")