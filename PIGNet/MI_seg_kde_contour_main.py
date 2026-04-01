"""
KDE Contour Main Entry Point
MI cache 로드 → KDE 계산 → Plot 생성
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse
from MI_seg_kde_compute import compute_kde_values
from MI_seg_kde_plot import (plot_scatter_same_diff, plot_scatter_with_distance_bins,
                             plot_kde_matrix_same, plot_kde_matrix_diff,
                             plot_ratio_boxplot_distance_bins)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',         type=str, default='cityscape', help='pascal or cityscape')
    parser.add_argument('--preprocess_type', type=str, default='pixel', help='pixel or layer')
    parser.add_argument('--model',           type=str, default='ASPP')
    parser.add_argument('--vmin',            type=int, default=0)
    parser.add_argument('--vmax',            type=int, default=25)
    parser.add_argument('--valid_pascal', action='store_true', default=True, 
                    help='if specified, use valid_0; otherwise use invalid_0')
    parser.add_argument('--calcul_type', type=str, default='joint', help='MI or joint')
    args = parser.parse_args()

    seg_file_path = (f"/home/hail/pan/HDD/MI_dataset/{args.preprocess_type}_dataset"
                     f"/{args.dataset}/resnet101/pretrained/{args.model}/zoom/1")

    if args.dataset != "pascal":
        cache_path = os.path.join(seg_file_path, 'analysis_cache_same_diff_joint.pkl')
    else:
        valid_dir = 'valid_0' if args.valid_pascal else 'invalid_0'
        cache_path = os.path.join(seg_file_path, f'{valid_dir}/analysis_cache_same_diff_joint.pkl')
    
    if not os.path.exists(cache_path):
        print("\n" + "="*60)
        print("❌ MI Cache Not Found!")
        print("="*60)
        print(f"\nCache file expected at:")
        print(f"  {cache_path}")
        print("\nPlease run MI computation first using:")
        print("  python MI_seg_compute_main.py \\")
        print(f"    --dataset {args.dataset} \\")
        print(f"    --preprocess_type {args.preprocess_type} \\")
        print(f"    --model {args.model}")
        print("\n" + "="*60)
        exit(1)

    print(f"Loading cached MI data from {cache_path}...")
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    
    distance = cache_data['distance']
    mi_xt_same = cache_data['mi_xt_same']
    mi_ty_same = cache_data['mi_ty_same']
    mi_xt_diff = cache_data['mi_xt_diff']
    mi_ty_diff = cache_data['mi_ty_diff']
    ignore_label = cache_data['ignore_label']
    
    median_same_x = np.median(mi_xt_same)
    median_same_y = np.median(mi_ty_same)

    median_diff_x = np.median(mi_xt_diff)
    median_diff_y = np.median(mi_ty_diff)
    
    print("✓ MI cache loaded successfully!\n")

    # ── Plot 설정 ───────────────────────────────────────────────────
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13,
                         'axes.titlesize': 14, 'legend.fontsize': 11,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11})

    # ── KDE Cache 확인 ────────────────────────────────────────────────
    kde_cache_path = os.path.join(seg_file_path, 'kde_cache_contour.pkl')
    
    # 필요한 key들 정의
    required_keys = ['Xi', 'Yi']
    for layer_idx in range(distance.shape[0]):
        required_keys.append(f'layer_{layer_idx}')
        for bin_idx in range(4):  # 0-10, 10-20, 20-30, 30-40 = 4개 bin
            required_keys.append(f'layer_{layer_idx}_bin_{bin_idx}')
    
    # 캐시 파일 존재하면 로드
    kde_cache_valid = False
    if os.path.exists(kde_cache_path):
        print(f"Loading cached KDE data from {kde_cache_path}...")
        with open(kde_cache_path, 'rb') as f:
            kde_data = pickle.load(f)
        
        # 필요한 key들이 모두 있는지 확인
        if all(key in kde_data for key in required_keys):
            print("✓ All required keys found in cache!")
            kde_cache_valid = True
        else:
            print("⚠ Some keys missing in cache. Recomputing KDE values...")
            missing_keys = [k for k in required_keys if k not in kde_data]
            print(f"  Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"  Missing keys: {missing_keys}")
    
    # 캐시가 유효하지 않으면 KDE 계산
    if not kde_cache_valid:
        print("\nComputing KDE values...")
        kde_data = compute_kde_values(mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff, distance)
        
        # ── KDE Cache 저장 ──────────────────────────────────────────────────
        print(f"\nSaving KDE data to {kde_cache_path}...")
        with open(kde_cache_path, 'wb') as f:
            pickle.dump(kde_data, f)
        print("✓ KDE cache saved successfully!\n")
    else:
        print("✓ KDE cache loaded successfully!\n")

    # # ── Plot ───────────────────────────────────────────────────────
    # print("=== KDE Contour Plots (SAME vs DIFF) ===")
    # for li in range(distance.shape[0]):
    #     plot_scatter_same_diff(li, args.model, args.dataset, args.preprocess_type, 
    #                            args.vmin, args.vmax, kde_data, 
    #                            median_same_x, median_same_y, median_diff_x, median_diff_y,
    #                            args.valid_pascal, args.calcul_type)

    # print("\n=== Distance-Binned KDE Contour Plots ===")
    # for li in range(distance.shape[0]):
    #     plot_scatter_with_distance_bins(li, args.model, args.dataset, args.preprocess_type, 
    #                                     args.vmin, args.vmax, kde_data,
    #                                     args.valid_pascal, args.calcul_type)

    # print("\n=== KDE Matrix Plots ===")
    # plot_kde_matrix_same(args.model, args.dataset, args.vmin, args.vmax, kde_data, 
    #                      process_type=args.preprocess_type, valid_pascal=args.valid_pascal, calcul_type=args.calcul_type)
    # plot_kde_matrix_diff(args.model, args.dataset, args.vmin, args.vmax, kde_data, 
    #                      process_type=args.preprocess_type, valid_pascal=args.valid_pascal, calcul_type=args.calcul_type)

    print("\n=== Ratio Boxplot ===")
    plot_ratio_boxplot_distance_bins(mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff, distance,
                                     args.model, args.dataset, args.preprocess_type,
                                     valid_pascal=args.valid_pascal,
                                     calcul_type=args.calcul_type)

    print("\n=== Done! ===")
