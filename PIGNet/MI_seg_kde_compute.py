"""
KDE Computation Module
MI 계산 및 KDE 계산 전담
"""

import numpy as np
import pickle
import os
from scipy.stats import gaussian_kde
from tqdm.auto import trange

def compute_kde_values(mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff, distance, threshold=1e-3):
    """
    threshold 이상인 값들만 KDE 계산
    """
    num_layers = mi_xt_same.shape[0]
    kde_data = {}
    
    # Grid 생성 (필터링된 데이터의 범위로)
    # threshold를 넘는 값들만으로 grid 범위 결정
    all_values = np.concatenate([
        mi_xt_same[mi_xt_same > threshold],
        mi_ty_same[mi_ty_same > threshold],
        mi_xt_diff[mi_xt_diff > threshold],
        mi_ty_diff[mi_ty_diff > threshold]
    ])
    
    if len(all_values) == 0:
        print("⚠️ Warning: No values above threshold!")
        all_values = np.concatenate([mi_xt_same.ravel(), mi_ty_same.ravel(), 
                                      mi_xt_diff.ravel(), mi_ty_diff.ravel()])
    
    v_min, v_max = all_values.min(), all_values.max()
    margin = (v_max - v_min) * 0.1
    
    xi = np.linspace(v_min - margin, v_max + margin, 100)
    yi = np.linspace(v_min - margin, v_max + margin, 100)
    Xi, Yi = np.meshgrid(xi, yi)
    kde_data['Xi'] = Xi
    kde_data['Yi'] = Yi
    
    dist_bins = np.arange(0, 50, 10)
    
    print(f"\n=== Computing KDE (threshold={threshold}) ===")
    for layer_idx in trange(num_layers, desc="Layer", leave=False):
        x_s_full = mi_xt_same[layer_idx]
        y_s_full = mi_ty_same[layer_idx]
        x_d_full = mi_xt_diff[layer_idx]
        y_d_full = mi_ty_diff[layer_idx]
        dist_layer = distance[layer_idx]
        
        # ✅ threshold 적용 - SAME mode
        mask_s_full = (x_s_full > threshold) & (y_s_full > threshold)
        x_s_filtered = x_s_full[mask_s_full]
        y_s_filtered = y_s_full[mask_s_full]
        
        if len(x_s_filtered) > 1 and np.std(x_s_filtered) > 1e-8 and np.std(y_s_filtered) > 1e-8:
            try:
                kde_s = gaussian_kde(np.vstack([x_s_filtered, y_s_filtered]), bw_method=0.3)
                Z_s_full = kde_s(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
            except np.linalg.LinAlgError:
                Z_s_full = np.zeros_like(Xi)
        else:
            Z_s_full = np.zeros_like(Xi)
        
        # ✅ threshold 적용 - DIFF mode
        mask_d_full = (x_d_full > threshold) & (y_d_full > threshold)
        x_d_filtered = x_d_full[mask_d_full]
        y_d_filtered = y_d_full[mask_d_full]
        
        if len(x_d_filtered) > 1 and np.std(x_d_filtered) > 1e-8 and np.std(y_d_filtered) > 1e-8:
            try:
                kde_d = gaussian_kde(np.vstack([x_d_filtered, y_d_filtered]), bw_method=0.3)
                Z_d_full = kde_d(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
            except np.linalg.LinAlgError:
                Z_d_full = np.zeros_like(Xi)
        else:
            Z_d_full = np.zeros_like(Xi)
        
        kde_data[f'layer_{layer_idx}'] = {
            'Z_s': Z_s_full,
            'Z_d': Z_d_full,
            'distance': dist_layer,
            'n_points_s': len(x_s_filtered),  # 필터링된 개수 표시
            'n_points_d': len(x_d_filtered),
            'mi_xt_same': x_s_filtered,  # 필터링된 데이터 저장
            'mi_ty_same': y_s_filtered,
            'mi_xt_diff': x_d_filtered,
            'mi_ty_diff': y_d_filtered,
        }
        
        # Distance bin별 KDE 계산에서도 threshold 적용
        for bin_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
            # ✅ bin mask + threshold mask 결합
            mask_s = mask_s_full & (dist_layer >= b_min) & (dist_layer < b_max)
            mask_d = mask_d_full & (dist_layer >= b_min) & (dist_layer < b_max)
            
            x_s_bin = x_s_full[mask_s]
            y_s_bin = y_s_full[mask_s]
            
            # SAME mode (bin별)
            if len(x_s_bin) > 1 and np.std(x_s_bin) > 1e-8 and np.std(y_s_bin) > 1e-8:
                try:
                    kde_s_bin = gaussian_kde(np.vstack([x_s_bin, y_s_bin]), bw_method=0.3)
                    Z_s_bin = kde_s_bin(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
                except np.linalg.LinAlgError:
                    Z_s_bin = np.zeros_like(Xi)
            else:
                Z_s_bin = np.zeros_like(Xi)
            
            # DIFF mode (bin별)
            x_d_bin = x_d_full[mask_d]
            y_d_bin = y_d_full[mask_d]
            if len(x_d_bin) > 1 and np.std(x_d_bin) > 1e-8 and np.std(y_d_bin) > 1e-8:
                try:
                    kde_d_bin = gaussian_kde(np.vstack([x_d_bin, y_d_bin]), bw_method=0.3)
                    Z_d_bin = kde_d_bin(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
                except np.linalg.LinAlgError:
                    Z_d_bin = np.zeros_like(Xi)
            else:
                Z_d_bin = np.zeros_like(Xi)
            
            kde_data[f'layer_{layer_idx}_bin_{bin_idx}'] = {
                'Z_s': Z_s_bin,
                'Z_d': Z_d_bin,
                'mi_xt_same': x_s_bin,
                'mi_ty_same': y_s_bin,
                'mi_xt_diff': x_d_bin,
                'mi_ty_diff': y_d_bin,
                'n_points_s': len(x_s_bin),
                'n_points_d': len(x_d_bin),
            }
    
    print("KDE computation done!\n")
    return kde_data
