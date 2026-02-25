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


def _mode_per_position(y_in: np.ndarray, ignore_label: int = 255) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-position mode label across batch.
    Returns:
        mode_flat: (H*W,) int labels (undefined positions set to ignore_label)
        valid_pos: (H*W,) bool positions with at least 1 non-ignore sample
    """
    N, H, W = y_in.shape
    y_flat = y_in.reshape(N, -1).astype(np.int32)
    P = y_flat.shape[1]
    mode = np.full(P, ignore_label, dtype=np.int32)
    valid = np.zeros(P, dtype=bool)

    for p in range(P):
        col = y_flat[:, p]
        if ignore_label is not None:
            col = col[col != ignore_label]
        if col.size == 0:
            continue
        counts = np.bincount(col)
        mode[p] = int(counts.argmax())
        valid[p] = True
    return mode, valid


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


def _entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    p = counts / np.maximum(1, counts.sum())
    return float(-np.sum(p * np.log2(p + eps)))


def cal_mi_x_t_conditional(x: np.ndarray,
                           t: np.ndarray,
                           y: np.ndarray,
                           h_bins: int = 51,
                           ignore_label: int = 255,
                           mode: str = "same"):
    """
    Compute MI map I(X_i; T_j) for all (j,i) pairs, but estimate probabilities
    using ONLY samples n where (Y_i == Y_j) [mode='same'] or (Y_i != Y_j) [mode='diff'].

    Notes:
    - x, t are expected to contain cluster ids in [-1..49] (invalid=-1). We +1 shift internally.
    - y is expected to contain dataset labels; ignore pixels are removed using ignore_label.
    - MI is computed with Laplace smoothing to avoid negative/unstable estimates.
    """
    assert mode in ("same", "diff")
    N, H, W = t.shape
    P = H * W
    eps = 1e-12
    alpha = 1e-3  # mild smoothing

    x_flat = x.reshape(N, -1).astype(np.int32) + 1   # 0..50
    t_flat = t.reshape(N, -1).astype(np.int32) + 1   # 0..50
    y_flat = y.reshape(N, -1).astype(np.int32)

    max_x = h_bins
    max_t = h_bins

    mi_map_flat = np.zeros((P, P), dtype=np.float32)

    # Precompute grid distances once (same as original)
    grid = np.indices((H, W)).reshape(2, -1).T
    h_diff = grid[:, 0:1] - grid[:, 0:1].T
    w_diff = grid[:, 1:2] - grid[:, 1:2].T
    euc_map = np.sqrt(h_diff**2 + w_diff**2).reshape(H, W, H, W)
    man_map = (np.abs(h_diff) + np.abs(w_diff)).reshape(H, W, H, W)

    for j_t in trange(P, desc=f"MI(X;T) conditional={mode}", leave=False):
        t_vec_all = t_flat[:, j_t]
        y_j = y_flat[:, j_t]

        # For each i_x, build subset mask based on y_i relation to y_j (depends on i_x)
        for i_x in range(P):
            y_i = y_flat[:, i_x]
            # valid labels for both positions
            valid = (y_i != ignore_label) & (y_j != ignore_label)

            if mode == "same":
                valid = valid & (y_i == y_j)
            else:
                valid = valid & (y_i != y_j)

            if not np.any(valid):
                mi_map_flat[j_t, i_x] = 0.0
                continue

            x_vec = x_flat[valid, i_x]
            t_vec = t_vec_all[valid]

            # Histograms with Laplace smoothing
            counts_x = np.bincount(x_vec, minlength=max_x).astype(np.float64) + alpha
            counts_t = np.bincount(t_vec, minlength=max_t).astype(np.float64) + alpha

            # Joint
            joint = t_vec.astype(np.int64) * max_x + x_vec.astype(np.int64)
            counts_joint = np.bincount(joint, minlength=max_t * max_x).astype(np.float64) + alpha

            h_x = _entropy_from_counts(counts_x, eps)
            h_t = _entropy_from_counts(counts_t, eps)
            h_joint = _entropy_from_counts(counts_joint, eps)

            mi = h_t + h_x - h_joint
            mi_map_flat[j_t, i_x] = max(0.0, float(mi))

    mi_map = mi_map_flat.reshape(H, W, H, W)
    return mi_map, euc_map, man_map


def cal_seg_mi_t_y_conditional(t: np.ndarray,
                               y: np.ndarray,
                               h_bins_t: int = 51,
                               num_classes_y: int = 21,
                               ignore_label: int = 255,
                               mode: str = "same"):
    """
    Compute MI map I(T_i; Y_j) for all (i,j) pairs, but estimate probabilities using ONLY samples n where:
      - Y_i and Y_j are both valid (!= ignore_label)
      - and (Y_i == Y_j) if mode='same' else (Y_i != Y_j)
    """
    assert mode in ("same", "diff")
    N, H, W = t.shape
    P = H * W
    eps = 1e-12
    alpha = 1e-3

    t_flat = t.reshape(N, -1).astype(np.int32) + 1   # 0..50
    y_flat = y.reshape(N, -1).astype(np.int32)       # 0..C-1 or 255(ignore)

    max_t = h_bins_t
    max_y = num_classes_y

    mi_map_flat = np.zeros((P, P), dtype=np.float32)

    grid = np.indices((H, W)).reshape(2, -1).T
    h_diff = grid[:, 0:1] - grid[:, 0:1].T
    w_diff = grid[:, 1:2] - grid[:, 1:2].T
    euc_map = np.sqrt(h_diff**2 + w_diff**2).reshape(H, W, H, W)
    man_map = (np.abs(h_diff) + np.abs(w_diff)).reshape(H, W, H, W)

    for j_y in trange(P, desc=f"MI(T;Y) conditional={mode}", leave=False):
        y_j_all = y_flat[:, j_y]

        for i_t in range(P):
            y_i_all = y_flat[:, i_t]

            valid = (y_i_all != ignore_label) & (y_j_all != ignore_label)
            if mode == "same":
                valid = valid & (y_i_all == y_j_all)
            else:
                valid = valid & (y_i_all != y_j_all)

            if not np.any(valid):
                mi_map_flat[i_t, j_y] = 0.0
                continue

            t_vec = t_flat[valid, i_t]
            y_vec = y_j_all[valid]

            # Clip y into [0..max_y-1] just in case (ignore already removed)
            y_vec = np.clip(y_vec, 0, max_y - 1).astype(np.int32)

            counts_t = np.bincount(t_vec, minlength=max_t).astype(np.float64) + alpha
            counts_y = np.bincount(y_vec, minlength=max_y).astype(np.float64) + alpha

            joint = t_vec.astype(np.int64) * max_y + y_vec.astype(np.int64)
            counts_joint = np.bincount(joint, minlength=max_t * max_y).astype(np.float64) + alpha

            h_t = _entropy_from_counts(counts_t, eps)
            h_y = _entropy_from_counts(counts_y, eps)
            h_joint = _entropy_from_counts(counts_joint, eps)

            mi = h_t + h_y - h_joint
            mi_map_flat[i_t, j_y] = max(0.0, float(mi))

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
    
    # Dataset-specific label handling
    # pascal: background=0, invalid often stored as -1
    # cityscape: ignore_label=255 (background is NOT a special class)
    if args.dataset.lower().startswith('city'):
        ignore_label = 255
        # keep 255 as ignore
    else:
        ignore_label = 255  # keep interface consistent; pascal invalid will be remapped
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

    # =========================
    # Conditional SAME/DIFF MI
    # =========================

    # For Pascal we keep background(0) as valid class.
    # For Cityscapes we remove only ignore_label=255.
    # (No Y>0 filtering is applied.)

    # Containers: per-layer flattened arrays for SAME and DIFF
    all_distance = []
    all_mi_xt_same, all_mi_ty_same = [], []
    all_mi_xt_diff, all_mi_ty_diff = [], []

    for layer_idx, t_layer in enumerate(t_in):
        # SAME
        mi_xt_same, euc_map, _ = cal_mi_x_t_conditional(x_in, t_layer, y_in, ignore_label=ignore_label, mode="same")
        mi_ty_same, _, _ = cal_seg_mi_t_y_conditional(t_layer, y_in, ignore_label=ignore_label, mode="same")

        # DIFF
        mi_xt_diff, _, _ = cal_mi_x_t_conditional(x_in, t_layer, y_in, ignore_label=ignore_label, mode="diff")
        mi_ty_diff, _, _ = cal_seg_mi_t_y_conditional(t_layer, y_in, ignore_label=ignore_label, mode="diff")

        # Flatten
        all_distance.append(euc_map.flatten())
        all_mi_xt_same.append(mi_xt_same.flatten())
        all_mi_ty_same.append(mi_ty_same.flatten())
        all_mi_xt_diff.append(mi_xt_diff.flatten())
        all_mi_ty_diff.append(mi_ty_diff.flatten())

    distance = np.array(all_distance)
    mi_xt_same = np.array(all_mi_xt_same)
    mi_ty_same = np.array(all_mi_ty_same)
    mi_xt_diff = np.array(all_mi_xt_diff)
    mi_ty_diff = np.array(all_mi_ty_diff)

    # Save cache
    cache_file = os.path.join(seg_file_path, 'mi_analysis_cache_same_diff_condmi.pkl')
    print(f"Saving computed data to {cache_file}...")
    cache_data = {
        'distance': distance,
        'mi_xt_same': mi_xt_same,
        'mi_ty_same': mi_ty_same,
        'mi_xt_diff': mi_xt_diff,
        'mi_ty_diff': mi_ty_diff,
        'ignore_label': ignore_label,
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print("Cache saved successfully!")

    # Plot styling
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 17
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # Plot per layer: SAME vs DIFF in one figure (distance as colormap intensity, alpha different)
    for layer_idx in range(distance.shape[0]):
        plt.figure(figsize=(12, 7))

        # SAME
        sc_same = plt.scatter(mi_xt_same[layer_idx], mi_ty_same[layer_idx],
                              c=distance[layer_idx], cmap='Reds',
                              alpha=0.35, s=12,
                              edgecolors='none', rasterized=True,
                              label='Same (conditional MI)')
        # DIFF
        sc_diff = plt.scatter(mi_xt_diff[layer_idx], mi_ty_diff[layer_idx],
                              c=distance[layer_idx], cmap='Blues',
                              alpha=0.07, s=12,
                              edgecolors='none', rasterized=True,
                              label='Diff (conditional MI)')

        cbar1 = plt.colorbar(sc_same)
        cbar1.set_label('Euclidean Distance (Same)', fontsize=10)
        cbar2 = plt.colorbar(sc_diff)
        cbar2.set_label('Euclidean Distance (Diff)', fontsize=10)

        plt.legend(frameon=False, loc='upper right')
        plt.xlabel("I(X; T)", fontsize=11, fontweight='bold')
        plt.ylabel("I(T; Y)", fontsize=11, fontweight='bold')
        plt.title(f"Layer {layer_idx+1} - Conditional SAME/DIFF Information Plane", fontsize=12, fontweight='bold')

        plt.xlim(0, 4)
        plt.ylim(0, 4)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{args.model}_{args.dataset}_condMI_layer{layer_idx+1}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Layer {layer_idx+1} plot saved (conditional SAME/DIFF).")

    # Plot per layer with distance bins (10-unit intervals)
    max_distance = np.max(distance) + 1
    distance_bins = np.arange(0, max_distance + 10, 10)
    bin_labels = [f"{int(distance_bins[i])}-{int(distance_bins[i+1])}" for i in range(len(distance_bins)-1)]

    for layer_idx in range(distance.shape[0]):
        for bin_idx in range(len(distance_bins) - 1):
            bin_min = distance_bins[bin_idx]
            bin_max = distance_bins[bin_idx + 1]

            # Create mask for current distance range
            mask_same = (distance[layer_idx] >= bin_min) & (distance[layer_idx] < bin_max)
            mask_diff = (distance[layer_idx] >= bin_min) & (distance[layer_idx] < bin_max)

            if not np.any(mask_same) and not np.any(mask_diff):
                continue

            plt.figure(figsize=(12, 7))

            # SAME
            if np.any(mask_same):
                plt.scatter(mi_xt_same[layer_idx][mask_same], mi_ty_same[layer_idx][mask_same],
                           alpha=0.5, s=20,
                           edgecolors='none', rasterized=True,
                           color='red', label='Same (conditional MI)')

            # DIFF
            if np.any(mask_diff):
                plt.scatter(mi_xt_diff[layer_idx][mask_diff], mi_ty_diff[layer_idx][mask_diff],
                           alpha=0.15, s=20,
                           edgecolors='none', rasterized=True,
                           color='blue', label='Diff (conditional MI)')

            plt.legend(frameon=False, loc='upper right')
            plt.xlabel("I(X; T)", fontsize=11, fontweight='bold')
            plt.ylabel("I(T; Y)", fontsize=11, fontweight='bold')
            plt.title(f"Layer {layer_idx+1} - Distance [{bin_min}-{bin_max}) Conditional SAME/DIFF Information Plane",
                     fontsize=12, fontweight='bold')

            plt.xlim(0, 4)
            plt.ylim(0, 4)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{args.model}_{args.dataset}_condMI_layer{layer_idx+1}_dist{int(bin_min)}-{int(bin_max)}.png",
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Layer {layer_idx+1} distance [{bin_min}-{bin_max}) plot saved.")
