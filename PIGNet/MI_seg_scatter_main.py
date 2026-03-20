"""
MI Scatter Plot - Main Entry Point
데이터 로드 및 계산/plot 모듈 호출
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse
from MI_seg_compute import compute_and_cache_mi
from GCN.PIGNet.MI_seg_scatter_plot import (plot_scatter_same_diff, 
                         plot_scatter_with_distance_bins,
                         plot_scatter_matrix_same,
                         plot_scatter_matrix_diff)


if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='pascal', help='pascal or cityscape')
    argparser.add_argument('--preprocess_type', type=str, default='pixel', help='layer or pixel')
    argparser.add_argument('--model', type=str, default='PIGNet_GSPonly', help='ASPP or PIGNet_GSPonly')
    args = argparser.parse_args()
    
    # ============ Data Loading ============
    print("=" * 50)
    print("Loading data...")
    print("=" * 50)
    
    seg_file_path = f"/home/hail/pan/HDD/MI_dataset/{args.preprocess_type}_dataset/{args.dataset}/resnet101/pretrained/{args.model}/zoom/1"
    
    with open(os.path.join(seg_file_path, 'gt_labels.pkl'), 'rb') as f:
        y_in = pickle.load(f)
    
    with open(os.path.join(seg_file_path, 'layer_0.pkl'), 'rb') as f:
        x_in = pickle.load(f)
    
    t_in = []
    for i in range(1, 5):
        with open(os.path.join(seg_file_path, f'layer_{i}.pkl'), 'rb') as f:
            t_in.append(pickle.load(f))
    
    print(f"Data loaded: x_in={x_in.shape}, y_in={y_in.shape}, t_in={len(t_in)}")
    print()
    
    # ============ MI Computation ============
    print("=" * 50)
    print("Computing MI values...")
    print("=" * 50)
    
    ignore_label = -1
    distance, mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff, ignore_label = \
        compute_and_cache_mi(seg_file_path, x_in, t_in, y_in, ignore_label)
    
    # ============ Plot Setup ============
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 17
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    # ============ Plotting ============
    print("\n" + "=" * 50)
    print("Generating plots...")
    print("=" * 50)
    
    # Plot per layer: SAME and DIFF separately
    print("\n=== Generating Separate SAME/DIFF Scatter Plots ===")
    for layer_idx in range(distance.shape[0]):
        plot_scatter_same_diff(mi_xt_same[layer_idx], mi_ty_same[layer_idx],
                              mi_xt_diff[layer_idx], mi_ty_diff[layer_idx],
                              distance[layer_idx], layer_idx, args.model, args.dataset, args.preprocess_type)

    # Plot per layer with distance bins
    print("\n=== Generating Distance-Binned Scatter Plots ===")
    for layer_idx in range(distance.shape[0]):
        plot_scatter_with_distance_bins(mi_xt_same[layer_idx], mi_ty_same[layer_idx],
                                       mi_xt_diff[layer_idx], mi_ty_diff[layer_idx],
                                       distance[layer_idx], layer_idx, args.model, args.dataset, args.preprocess_type)

    # Plot matrix: Layer x Distance bins
    print("\n=== Generating Scatter Matrix Plots ===")
    plot_scatter_matrix_same(mi_xt_same, mi_ty_same, distance, 
                            args.model, args.dataset, args.preprocess_type)
    plot_scatter_matrix_diff(mi_xt_diff, mi_ty_diff, distance, 
                            args.model, args.dataset, args.preprocess_type)

    print("\n" + "=" * 50)
    print("✓ All plots generated successfully!")
    print("=" * 50)
