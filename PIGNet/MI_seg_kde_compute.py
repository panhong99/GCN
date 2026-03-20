"""
KDE Computation Module
MI 계산 및 KDE 계산 전담
"""

import numpy as np
import pickle
import os
from scipy.stats import gaussian_kde
from tqdm.auto import trange

def compute_kde_values(mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff, distance):
    """
    MI값들로부터 모든 layer의 KDE density values 계산 (distance bin별로 분리)
    
    Args:
        mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff: [num_layers, num_points]
        distance: [num_layers, num_points]
    
    Returns:
        kde_data: dictionary with KDE values for all layers and distance bins
    """
    num_layers = mi_xt_same.shape[0]
    kde_data = {}
    
    # Grid 생성 (모든 layer에서 동일)
    xi = np.linspace(0, 2, 100)
    yi = np.linspace(0, 2, 100)
    Xi, Yi = np.meshgrid(xi, yi)
    kde_data['Xi'] = Xi
    kde_data['Yi'] = Yi
    
    # Distance bin 정의
    dist_bins = np.arange(0, 50, 10)  # 0-10, 10-20, 20-30, 30-40
    
    print("\n=== Computing KDE Values (by distance bins) ===")
    for layer_idx in trange(num_layers, desc="Layer", leave=False):
        x_s_full = mi_xt_same[layer_idx]
        y_s_full = mi_ty_same[layer_idx]
        x_d_full = mi_xt_diff[layer_idx]
        y_d_full = mi_ty_diff[layer_idx]
        dist_layer = distance[layer_idx]
        
        # 전체 layer의 Z도 저장 (layer-wise plot용)
        if len(x_s_full) > 1 and np.std(x_s_full) > 1e-8 and np.std(y_s_full) > 1e-8:
            try:
                kde_s = gaussian_kde(np.vstack([x_s_full, y_s_full]), bw_method=0.3)
                Z_s_full = kde_s(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
            except np.linalg.LinAlgError:
                Z_s_full = np.zeros_like(Xi)
        else:
            Z_s_full = np.zeros_like(Xi)
        
        if len(x_d_full) > 1 and np.std(x_d_full) > 1e-8 and np.std(y_d_full) > 1e-8:
            try:
                kde_d = gaussian_kde(np.vstack([x_d_full, y_d_full]), bw_method=0.3)
                Z_d_full = kde_d(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
            except np.linalg.LinAlgError:
                Z_d_full = np.zeros_like(Xi)
        else:
            Z_d_full = np.zeros_like(Xi)
        
        kde_data[f'layer_{layer_idx}'] = {
            'Z_s': Z_s_full,
            'Z_d': Z_d_full,
            'distance': dist_layer,
            'n_points_s': len(x_s_full),
            'n_points_d': len(x_d_full),
            'mi_xt_same': x_s_full,
            'mi_ty_same': y_s_full,
            'mi_xt_diff': x_d_full,
            'mi_ty_diff': y_d_full,
        }
        
        # Distance bin별 KDE 계산
        for bin_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
            mask_s = (dist_layer >= b_min) & (dist_layer < b_max)
            mask_d = (dist_layer >= b_min) & (dist_layer < b_max)
            
            # SAME mode (bin별)
            x_s_bin = x_s_full[mask_s]
            y_s_bin = y_s_full[mask_s]
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
