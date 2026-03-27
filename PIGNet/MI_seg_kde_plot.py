"""
KDE Plotting Module
모든 KDE 기반 시각화 함수
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import Normalize

def plot_scatter_same_diff(layer_idx, model_name, dataset_name, process_type, vmin, vmax, kde_data, median_same_x, median_same_y, median_diff_x, median_diff_y, valid_pascal=False, calcul_type='MI'):
    """
    Cache에서 받은 KDE값을 이용해 SAME/DIFF plot 생성
    """
    valid_str = 'valid' if valid_pascal else 'invalid'
    Z_s = kde_data[f'layer_{layer_idx}']['Z_s']
    Z_d = kde_data[f'layer_{layer_idx}']['Z_d']
    Xi = kde_data['Xi']
    Yi = kde_data['Yi']
    n_s = kde_data[f'layer_{layer_idx}']['n_points_s']
    n_d = kde_data[f'layer_{layer_idx}']['n_points_d']
    
    print(f"\n  Layer {layer_idx+1}: SAME n={n_s}, DIFF n={n_d}")

    # density clip (핵심)
    # Z_s_plot = np.clip(Z_s, vmin, vmax)
    # Z_d_plot = np.clip(Z_d, vmin, vmax)
    Z_s_plot = Z_s
    Z_d_plot = Z_d

    # norm + levels 고정
    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 21)

    # Colormap 설정 (배경 흰색)
    cmap_s = plt.cm.get_cmap('Reds').copy()
    cmap_s.set_bad('white')
    cmap_d = plt.cm.get_cmap('Blues').copy()
    cmap_d.set_bad('white')

    # SAME plot
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')

    threshold = 1e-3
    Z_s_masked = np.ma.masked_less_equal(Z_s_plot, threshold)

    cf = ax.contourf(
        Xi, Yi,
        Z_s_masked,      # ← masked array 사용
        levels=levels,
        cmap=cmap_s,
        norm=norm,
    )
    
    # Median 포인트 표시
    ax.scatter(median_same_x, median_same_y, marker='*', s=150, c='lime', edgecolors='darkgreen', linewidth=2, zorder=5)
    
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('Density', fontsize=11)
    
    ax.set_xlabel("I(X; T)", fontsize=12, fontweight='bold')
    ax.set_ylabel("I(T; Y)", fontsize=12, fontweight='bold')
    ax.set_title(f"Layer {layer_idx+1} - SAME Class KDE Contour", fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    fname = f"{model_name}_{dataset_name}_{process_type}_{valid_str}_{calcul_type}_kde_layer{layer_idx+1}_SAME.png"
    folder_path = f"./{model_name}/{dataset_name}/{process_type}/all"
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ SAME saved")

    # DIFF plot
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')

    Z_d_masked = np.ma.masked_less_equal(Z_d_plot, threshold)

    cf = ax.contourf(
        Xi, Yi, 
        Z_d_masked,      # ← masked array 사용
        levels=levels, 
        cmap=cmap_d, 
        norm=norm
    )
    
    # Median 포인트 표시
    ax.scatter(median_diff_x, median_diff_y, marker='*', s=150, c='lime', edgecolors='darkgreen', linewidth=2, zorder=5)

    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('Density', fontsize=11)
    
    ax.set_xlabel("I(X; T)", fontsize=12, fontweight='bold')
    ax.set_ylabel("I(T; Y)", fontsize=12, fontweight='bold')
    ax.set_title(f"Layer {layer_idx+1} - DIFF Class KDE Contour", fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    fname = f"{model_name}_{dataset_name}_{process_type}_{valid_str}_{calcul_type}_kde_layer{layer_idx+1}_DIFF.png"
    folder_path = f"./{model_name}/{dataset_name}/{process_type}/all"
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')            
    plt.close()
    print(f"    ✓ DIFF saved")


def plot_scatter_with_distance_bins(layer_idx, model_name, dataset_name, process_type, vmin, vmax, kde_data, valid_pascal=False, calcul_type='MI'):
    """
    Distance bin별로 계산된 KDE값을 이용해 거리 구간별 plot 생성 (SAME, DIFF 각각 개별 plot)
    """
    valid_str = 'valid' if valid_pascal else 'invalid'
    Xi = kde_data['Xi']
    Yi = kde_data['Yi']
    
    dist_bins = np.arange(0, 50, 10)  # 0-10, 10-20, 20-30, 30-40

    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 21)

    # Colormap 설정 (배경 흰색)
    cmap_s = plt.cm.get_cmap('Reds').copy()
    cmap_s.set_bad('white')
    cmap_d = plt.cm.get_cmap('Blues').copy()
    cmap_d.set_bad('white')
    
    threshold = 1e-3

    print(f"\n  Layer {layer_idx+1} (Distance-binned):")
    for bin_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
        cache_key = f'layer_{layer_idx}_bin_{bin_idx}'
        
        if cache_key not in kde_data:
            print(f"    dist [{b_min:.0f}–{b_max:.0f}): no data")
            continue
        
        bin_data = kde_data[cache_key]
        Z_s = bin_data['Z_s']
        Z_d = bin_data['Z_d']
        mi_xt_s = bin_data['mi_xt_same']
        mi_ty_s = bin_data['mi_ty_same']
        mi_xt_d = bin_data['mi_xt_diff']
        mi_ty_d = bin_data['mi_ty_diff']

        print(f"    dist [{b_min:.0f}–{b_max:.0f}): ", end="")

        # density clip
        # Z_s_plot = np.clip(Z_s, vmin, vmax)
        # Z_d_plot = np.clip(Z_d, vmin, vmax)
        Z_s_plot = Z_s
        Z_d_plot = Z_d

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SAME 개별 plot
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        ax.set_facecolor('white')

        Z_s_masked = np.ma.masked_less_equal(Z_s_plot, threshold)
        cf_s = ax.contourf(Xi, Yi, Z_s_masked, levels=levels, cmap=cmap_s, norm=norm)
        
        # Median 포인트 표시 (같은 데이터에서 계산)
        if len(mi_xt_s) > 0:
            median_x = np.median(mi_xt_s)
            median_y = np.median(mi_ty_s)
            ax.scatter(median_x, median_y, marker='*', s=150, c='lime', edgecolors='darkgreen', linewidth=2, zorder=5)
        
        cbar_s = plt.colorbar(cf_s, ax=ax)
        cbar_s.set_label('Density', fontsize=11)
        
        ax.set_xlabel("I(X; T)", fontsize=12, fontweight='bold')
        ax.set_ylabel("I(T; Y)", fontsize=12, fontweight='bold')
        ax.set_title(f"Layer {layer_idx+1} - SAME - Distance [{b_min:.0f}–{b_max:.0f})", 
                     fontsize=13, fontweight='bold')

        plt.tight_layout()
        fname = (f"{model_name}_{dataset_name}_{process_type}_{valid_str}_{calcul_type}_kde_layer{layer_idx+1}"
                 f"_dist{int(b_min)}-{int(b_max)}_SAME.png")
        
        folder_path = f"./{model_name}/{dataset_name}/{process_type}/distance"
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')        
        plt.close()
        print("SAME saved, ", end="")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # DIFF 개별 plot
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        ax.set_facecolor('white')

        Z_d_masked = np.ma.masked_less_equal(Z_d_plot, threshold)
        cf_d = ax.contourf(Xi, Yi, Z_d_masked, levels=levels, cmap=cmap_d, norm=norm)
        
        # Median 포인트 표시 (같은 데이터에서 계산)
        if len(mi_xt_d) > 0:
            median_x = np.median(mi_xt_d)
            median_y = np.median(mi_ty_d)
            ax.scatter(median_x, median_y, marker='*', s=150, c='lime', edgecolors='darkgreen', linewidth=2, zorder=5)
        
        cbar_d = plt.colorbar(cf_d, ax=ax)
        cbar_d.set_label('Density', fontsize=11)
        
        ax.set_xlabel("I(X; T)", fontsize=12, fontweight='bold')
        ax.set_ylabel("I(T; Y)", fontsize=12, fontweight='bold')
        ax.set_title(f"Layer {layer_idx+1} - DIFF - Distance [{b_min:.0f}–{b_max:.0f})", 
                     fontsize=13, fontweight='bold')

        plt.tight_layout()
        fname = (f"{model_name}_{dataset_name}_{process_type}_{valid_str}_{calcul_type}_kde_layer{layer_idx+1}"
                 f"_dist{int(b_min)}-{int(b_max)}_DIFF.png")
        
        folder_path = f"./{model_name}/{dataset_name}/{process_type}/distance"
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')        
        plt.close()
        print("DIFF saved")


def plot_kde_matrix_same(model_name, dataset_name, vmin, vmax, kde_data, process_type, valid_pascal=False, calcul_type='MI'):
    """
    Matrix plot: Layer (y축) x Distance (x축) for SAME mode
    4x4 grid (4 layers, 4 distance bins) - bin별 KDE 사용
    """
    valid_str = 'valid' if valid_pascal else 'invalid'
    Xi = kde_data['Xi']
    Yi = kde_data['Yi']
    layers = 4
    dist_bins = np.arange(0, 50, 10)  # 0-10, 10-20, 20-30, 30-40
    num_dist = len(dist_bins) - 1
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 21)
    cmap_s = plt.cm.get_cmap('Reds').copy()
    cmap_s.set_bad('white')
    threshold = 1e-3
    
    fig, axes = plt.subplots(layers, num_dist, figsize=(20, 18), facecolor='white')
    
    for layer_idx in range(layers):
        for dist_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
            ax = axes[layer_idx, dist_idx]
            ax.set_facecolor('white')
            
            cache_key = f'layer_{layer_idx}_bin_{dist_idx}'
            if cache_key not in kde_data:
                ax.text(1, 1, 'No data', ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            bin_data = kde_data[cache_key]
            Z_s = bin_data['Z_s']
            mi_xt_s = bin_data['mi_xt_same']
            mi_ty_s = bin_data['mi_ty_same']
            
            # Z_s_plot = np.clip(Z_s, vmin, vmax)
            Z_s_plot = Z_s
            Z_masked = np.ma.masked_less_equal(Z_s_plot, threshold)
            cf = ax.contourf(Xi, Yi, Z_masked, levels=levels, cmap=cmap_s, norm=norm)
            
            # Median 포인트 표시 (bin 데이터에서)
            if len(mi_xt_s) > 0:
                median_x = np.median(mi_xt_s)
                median_y = np.median(mi_ty_s)
                ax.scatter(median_x, median_y, marker='*', s=150, c='lime', edgecolors='darkgreen', linewidth=1.5, zorder=5)
            
            # Ticks 제거
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            # Y축 레이블 (좌측만)
            if dist_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx+1}", fontsize=11, fontweight='bold')
            
            # X축 레이블 (상단만)
            if layer_idx == 0:
                ax.set_title(f"Dist [{b_min:.0f}-{b_max:.0f})", fontsize=11, fontweight='bold')
    
    plt.suptitle(f"{model_name}_{dataset_name}_KDE Matrix - SAME Mode", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    folder_path = f"./{model_name}/{dataset_name}/{process_type}/all"
    os.makedirs(folder_path, exist_ok=True)
    fname = f"{model_name}_{dataset_name}_{process_type}_{valid_str}_{calcul_type}_kde_matrix_SAME.png"
    plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ SAME matrix plot saved: {fname}")


def plot_ratio_boxplot_distance_bins(mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff, distance,
                                    model_name, dataset_name, process_type,
                                    valid_pascal=False, calcul_type='MI', median_alpha=0.8):
    """
    Distance bin별로 subplot 생성 (1, 4)
    각 subplot에서 x축=Layer(1-4), y축=ratio boxplot 표시
    + 각 layer의 median을 point로 표시하고 line으로 연결
    
    Parameters:
    -----------
    mi_xt_same, mi_ty_same: shape (n_layers, n_samples)
    mi_xt_diff, mi_ty_diff: shape (n_layers, n_samples)
    distance: shape (n_layers, n_samples)
    median_alpha: float, median point의 투명도 (0~1)
    """
    valid_str = 'valid' if valid_pascal else 'invalid'
    
    dist_bins = np.arange(0, 50, 10)  # 0-10, 10-20, 20-30, 30-40
    num_dist = len(dist_bins) - 1      # 4
    n_layers = mi_xt_same.shape[0]     # 4
    
    eps = 1e-8
    layer_labels = [f"L{i+1}" for i in range(n_layers)]
    x_positions = np.arange(1, n_layers + 1)  # x축 위치: 1, 2, 3, 4
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SAME 그래프
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig_s, axes_s = plt.subplots(1, num_dist, figsize=(20, 5), facecolor='white')
    fig_s.suptitle(f"{model_name} {dataset_name} — SAME: I(X;T)/I(T;Y) Ratio",
                   fontsize=14, fontweight='bold')
    
    for dist_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
        ax = axes_s[dist_idx]
        ax.set_facecolor('white')
        
        same_ratios = []      # list of arrays (layer별)
        median_values = []    # median point 저장
        
        for layer_idx in range(n_layers):
            # 해당 layer의 distance bin 범위 내 데이터 추출
            mask = (distance[layer_idx] >= b_min) & (distance[layer_idx] < b_max)
            
            mi_xt = np.asarray(mi_xt_same[layer_idx][mask], dtype=float)
            mi_ty = np.asarray(mi_ty_same[layer_idx][mask], dtype=float)
            
            ratio = mi_xt / (mi_ty + eps)
            same_ratios.append(ratio)
            
            # Median 값 저장
            if len(ratio) > 0:
                median_values.append(np.median(ratio))
            else:
                median_values.append(np.nan)
        
        # Boxplot 그리기
        bp = ax.boxplot(same_ratios, positions=x_positions, patch_artist=True, 
                        showfliers=False, widths=0.6, labels=layer_labels)
        
        for patch in bp['boxes']:
            patch.set_facecolor('coral')
            patch.set_alpha(0.7)
        
        for median in bp['medians']:
            median.set_color('salmon')
            median.set_linewidth(1.5)
        
        # Median point 표시 및 line 연결
        median_values = np.array(median_values)
        valid_mask = ~np.isnan(median_values)
        valid_x = x_positions[valid_mask]
        valid_y = median_values[valid_mask]
        
        if len(valid_x) > 0:
            # Point 표시 (테두리 없음)
            ax.scatter(valid_x, valid_y, s=100, c='darkred', marker='o', 
                      zorder=5, alpha=median_alpha)
            # Line 연결
            if len(valid_x) > 1:
                ax.plot(valid_x, valid_y, 'r-', linewidth=2, alpha=0.6, zorder=4)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(layer_labels)
        ax.set_xlabel("Layer", fontsize=11)
        if dist_idx == 0:
            ax.set_ylabel("I(X;T) / I(T;Y)", fontsize=11)
        ax.set_title(f"Distance [{b_min:.0f}–{b_max:.0f})", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    folder_path = f"./{model_name}/{dataset_name}/{process_type}/ratio"
    os.makedirs(folder_path, exist_ok=True)
    fname = f"{model_name}_{dataset_name}_{process_type}_{valid_str}_{calcul_type}_ratio_boxplot_SAME.png"
    plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ SAME boxplot saved: {fname}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DIFF 그래프
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig_d, axes_d = plt.subplots(1, num_dist, figsize=(20, 5), facecolor='white')
    fig_d.suptitle(f"{model_name} {dataset_name} — DIFF: I(X;T)/I(T;Y) Ratio",
                   fontsize=14, fontweight='bold')
    
    for dist_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
        ax = axes_d[dist_idx]
        ax.set_facecolor('white')
        
        diff_ratios = []      # list of arrays (layer별)
        median_values = []    # median point 저장
        
        for layer_idx in range(n_layers):
            # 해당 layer의 distance bin 범위 내 데이터 추출
            mask = (distance[layer_idx] >= b_min) & (distance[layer_idx] < b_max)
            
            mi_xt = np.asarray(mi_xt_diff[layer_idx][mask], dtype=float)
            mi_ty = np.asarray(mi_ty_diff[layer_idx][mask], dtype=float)
            
            ratio = mi_xt / (mi_ty + eps)
            diff_ratios.append(ratio)
            
            # Median 값 저장
            if len(ratio) > 0:
                median_values.append(np.median(ratio))
            else:
                median_values.append(np.nan)
        
        # Boxplot 그리기
        bp = ax.boxplot(diff_ratios, positions=x_positions, patch_artist=True, 
                        showfliers=False, widths=0.6, labels=layer_labels)
        
        for patch in bp['boxes']:
            patch.set_facecolor('azure')
            patch.set_alpha(0.7)
        
        for median in bp['medians']:
            median.set_color('navy')
            median.set_linewidth(1.5)
        
        # Median point 표시 및 line 연결
        median_values = np.array(median_values)
        valid_mask = ~np.isnan(median_values)
        valid_x = x_positions[valid_mask]
        valid_y = median_values[valid_mask]
        
        if len(valid_x) > 0:
            # Point 표시 (테두리 없음)
            ax.scatter(valid_x, valid_y, s=100, c='darkblue', marker='o', 
                      zorder=5, alpha=median_alpha)
            # Line 연결
            if len(valid_x) > 1:
                ax.plot(valid_x, valid_y, 'b-', linewidth=2, alpha=0.6, zorder=4)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(layer_labels)
        ax.set_xlabel("Layer", fontsize=11)
        if dist_idx == 0:
            ax.set_ylabel("I(X;T) / I(T;Y)", fontsize=11)
        ax.set_title(f"Distance [{b_min:.0f}–{b_max:.0f})", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fname = f"{model_name}_{dataset_name}_{process_type}_{valid_str}_{calcul_type}_ratio_boxplot_DIFF.png"
    plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ DIFF boxplot saved: {fname}")

def plot_kde_matrix_diff(model_name, dataset_name, vmin, vmax, kde_data, process_type, valid_pascal=False, calcul_type='MI'):
    """
    Matrix plot: Layer (y축) x Distance (x축) for DIFF mode
    4x4 grid (4 layers, 4 distance bins) - bin별 KDE 사용
    """
    valid_str = 'valid' if valid_pascal else 'invalid'
    Xi = kde_data['Xi']
    Yi = kde_data['Yi']
    layers = 4
    dist_bins = np.arange(0, 50, 10)  # 0-10, 10-20, 20-30, 30-40
    num_dist = len(dist_bins) - 1
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 21)
    cmap_d = plt.cm.get_cmap('Blues').copy()
    cmap_d.set_bad('white')
    threshold = 1e-3
    
    fig, axes = plt.subplots(layers, num_dist, figsize=(20, 18), facecolor='white')
    
    for layer_idx in range(layers):
        for dist_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
            ax = axes[layer_idx, dist_idx]
            ax.set_facecolor('white')
            
            cache_key = f'layer_{layer_idx}_bin_{dist_idx}'
            if cache_key not in kde_data:
                ax.text(1, 1, 'No data', ha='center', va='center', fontsize=12)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                continue
            
            bin_data = kde_data[cache_key]
            Z_d = bin_data['Z_d']
            mi_xt_d = bin_data['mi_xt_diff']
            mi_ty_d = bin_data['mi_ty_diff']
            
            # Z_d_plot = np.clip(Z_d, vmin, vmax)
            Z_d_plot = Z_d
            Z_masked = np.ma.masked_less_equal(Z_d_plot, threshold)
            cf = ax.contourf(Xi, Yi, Z_masked, levels=levels, cmap=cmap_d, norm=norm)
            
            # Median 포인트 표시 (bin 데이터에서)
            if len(mi_xt_d) > 0:
                median_x = np.median(mi_xt_d)
                median_y = np.median(mi_ty_d)
                ax.scatter(median_x, median_y, marker='*', s=150, c='lime', edgecolors='darkgreen', linewidth=1.5, zorder=5)
            
            # Tick 레이블 제거 (marks는 유지)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            # Y축 레이블 (좌측만)
            if dist_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx+1}", fontsize=11, fontweight='bold')
            
            # X축 레이블 (상단만)
            if layer_idx == 0:
                ax.set_title(f"Dist [{b_min:.0f}-{b_max:.0f})", fontsize=11, fontweight='bold')
    
    plt.suptitle(f"{model_name}_{dataset_name}_KDE Matrix - DIFF Mode", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    folder_path = f"./{model_name}/{dataset_name}/{process_type}/all"
    os.makedirs(folder_path, exist_ok=True)
    fname = f"{model_name}_{dataset_name}_{process_type}_{valid_str}_{calcul_type}_kde_matrix_DIFF.png"
    plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ DIFF matrix plot saved: {fname}")
