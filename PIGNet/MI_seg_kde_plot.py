"""
KDE Plotting Module
모든 KDE 기반 시각화 함수
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'regular'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 활성화: 전체 모델 bar plot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_ratio_barplot_all_models(models_data_pascal, models_data_cityscape, process_type,
                                  calcul_type='MI'):
    FS = 35

    import matplotlib.patches as mpatches

    color_map  = {'PIGNet_GSPonly': '#D81B60', 'ASPP': '#1E88E5', 'Mask2Former': '#FFC107'}
    marker_map = {'SAME': 'o', 'DIFF': '^'}
    alpha_map  = {'PIGNet_GSPonly': 0.55, 'ASPP': 0.45, 'Mask2Former': 0.45}
    draw_order = ['Mask2Former', 'ASPP', 'PIGNet_GSPonly']
    dist_bins  = np.arange(0, 50, 10)
    num_dist   = len(dist_bins) - 1

    def get_model_list(models_data):
        return [m for m in draw_order if m in models_data] + \
               [m for m in models_data if m not in draw_order]

    model_list_p = get_model_list(models_data_pascal)
    model_list_c = get_model_list(models_data_cityscape)
    all_models   = list(dict.fromkeys(model_list_p + model_list_c))
    model_colors = {m: color_map.get(m, '#000000') for m in all_models}

    n_layers     = models_data_pascal[model_list_p[0]]['mi_xt_same'].shape[0]
    layer_labels = [f"L{i+1}" for i in range(n_layers)]
    x_base       = np.arange(1, n_layers + 1)

    folder_path = f"./ALL_MODELS/combined/{process_type}/ratio"
    os.makedirs(folder_path, exist_ok=True)

    # 컬럼 설정: (models_data, model_list, mode, ylim)
    col_configs = [
        (models_data_pascal,    model_list_p, 'SAME', (0, 15)),
        (models_data_pascal,    model_list_p, 'DIFF', (0, 15)),
        (models_data_cityscape, model_list_c, 'SAME', (0, 10)),
        (models_data_cityscape, model_list_c, 'DIFF', (0, 10)),
    ]

    _, axes = plt.subplots(num_dist, 4, figsize=(32, 22), facecolor='white')

    for col_idx, (models_data, model_list, mode, ylim) in enumerate(col_configs):
        xt_key = 'mi_xt_same' if mode == 'SAME' else 'mi_xt_diff'
        ty_key = 'mi_ty_same' if mode == 'SAME' else 'mi_ty_diff'
        marker = marker_map[mode]

        for dist_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
            ax = axes[dist_idx, col_idx]
            ax.set_facecolor('white')

            for model_name in model_list:
                data     = models_data[model_name]
                distance = data['distance']
                mi_xt    = data[xt_key]
                mi_ty    = data[ty_key]
                color    = model_colors[model_name]
                alpha    = alpha_map.get(model_name, 0.6)

                mean_values = []
                err_vals    = []

                for layer_idx in range(n_layers):
                    mask = (distance[layer_idx] >= b_min) & (distance[layer_idx] < b_max)
                    xt = np.asarray(mi_xt[layer_idx][mask], dtype=float)
                    ty = np.asarray(mi_ty[layer_idx][mask], dtype=float)

                    valid = np.isfinite(xt) & np.isfinite(ty) & (ty != 0)
                    ratio = xt[valid] / ty[valid]
                    # ratio =  ty[valid]/xt[valid]

                    if len(ratio) > 0:
                        mean_values.append(np.mean(ratio))
                        err_vals.append(np.std(ratio, ddof=1))
                    else:
                        mean_values.append(np.nan)
                        err_vals.append(0)

                mean_values = np.array(mean_values)
                err_vals    = np.array(err_vals)
                valid_mask  = ~np.isnan(mean_values)
                vx = x_base[valid_mask]
                vy = mean_values[valid_mask]
                ve = err_vals[valid_mask]

                if len(vx) == 0:
                    continue

                ax.bar(vx, vy, width=0.6, color=color, alpha=alpha,
                       zorder=3, edgecolor='none')
                ax.plot(vx, vy, '-', color=color, linewidth=2,
                        alpha=min(alpha + 0.15, 1.0), zorder=5)
                ax.plot(vx, vy, marker, color=color, markersize=7,
                        markeredgecolor='white', markeredgewidth=0.8,
                        alpha=min(alpha + 0.15, 1.0), zorder=6)
                ax.errorbar(vx, vy, yerr=ve, fmt='none',
                            ecolor=color, elinewidth=1.2,
                            capsize=4, capthick=1.2,
                            alpha=min(alpha + 0.15, 1.0), zorder=4)

            ax.set_xticks(x_base)
            ax.set_xticklabels(layer_labels, fontsize=FS)
            ax.tick_params(axis='y', labelsize=FS)
            ax.set_xlim(x_base[0] - 0.5, x_base[-1] + 0.5)
            ax.set_ylim(*ylim)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)

            if dist_idx == num_dist - 1:
                ax.set_xlabel("Layer", fontsize=FS)
            if col_idx == 0:
                ax.set_ylabel("H(X,T) / H(T,Y)", fontsize=FS)
            ax.set_title(f"{mode} - Distance [{b_min:.0f}–{b_max:.0f})", fontsize=FS)

            # Legend: Pascal SAME (col 0), 첫 번째 행에만
            if col_idx == 0 and dist_idx == 0:
                legend_order = ['PIGNet_GSPonly', 'ASPP', 'Mask2Former']
                handles = []
                for name in legend_order:
                    if name in model_colors:
                        handles.append(
                            mpatches.Patch(color=model_colors[name],
                                           alpha=alpha_map.get(name, 0.6),
                                           label=name))
                ax.legend(handles=handles, fontsize=FS, loc='upper left', frameon=False)

    plt.tight_layout()

    fname = f"ALL_combined_{process_type}_{calcul_type}_ratio_errorbar.pdf"
    plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ combined errorbar plot saved: {fname}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━══
# 활성화: KDE Matrix
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_kde_matrix_same(model_name, dataset_name, vmin, vmax, kde_data, process_type, valid_pascal=False, calcul_type='MI'):
    """
    Matrix plot: Layer (y축) x Distance (x축) for SAME mode
    4x4 grid (4 layers, 4 distance bins) - bin별 KDE 사용
    전체 figure 오른쪽에 하나의 colorbar 표시
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

    fig, axes = plt.subplots(layers, num_dist, figsize=(22, 18), facecolor='white')

    last_cf = None

    for layer_idx in range(layers):
        for dist_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
            ax = axes[layer_idx, dist_idx]
            ax.set_facecolor('white')

            cache_key = f'layer_{layer_idx}_bin_{dist_idx}'
            if cache_key not in kde_data:
                ax.text(1, 1, 'No data', ha='center', va='center', fontsize=25)
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
            last_cf = cf

            if len(mi_xt_s) > 0:
                median_x = np.median(mi_xt_s)
                median_y = np.median(mi_ty_s)
                ax.scatter(median_x, median_y, marker='*', s=150, c='lime', edgecolors='darkgreen', linewidth=1.5, zorder=5)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)

            if dist_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx+1}", fontsize=18)

            if layer_idx == 0:
                ax.set_title(f"Dist [{b_min:.0f}-{b_max:.0f})", fontsize=18)

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    if last_cf is not None:
        top    = axes[0, 0].get_position().y1
        bottom = axes[-1, 0].get_position().y0
        cbar_ax = fig.add_axes([0.90, bottom, 0.03, top - bottom])
        cbar = fig.colorbar(last_cf, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=18)

    folder_path = f"./{model_name}/{dataset_name}/{process_type}/all"
    os.makedirs(folder_path, exist_ok=True)
    fname = f"{model_name}_{dataset_name}_{process_type}_{valid_str}_{calcul_type}_kde_matrix_SAME.png"
    plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ SAME matrix plot saved: {fname}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 활성화: KDE Matrix
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_kde_matrix_diff(model_name, dataset_name, vmin, vmax, kde_data, process_type, valid_pascal=False, calcul_type='MI'):
    """
    Matrix plot: Layer (y축) x Distance (x축) for DIFF mode
    4x4 grid (4 layers, 4 distance bins) - bin별 KDE 사용
    전체 figure 오른쪽에 하나의 colorbar 표시
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

    fig, axes = plt.subplots(layers, num_dist, figsize=(22, 18), facecolor='white')

    last_cf = None

    for layer_idx in range(layers):
        for dist_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
            ax = axes[layer_idx, dist_idx]
            ax.set_facecolor('white')

            cache_key = f'layer_{layer_idx}_bin_{dist_idx}'
            if cache_key not in kde_data:
                ax.text(1, 1, 'No data', ha='center', va='center', fontsize=25)
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
            last_cf = cf

            if len(mi_xt_d) > 0:
                median_x = np.median(mi_xt_d)
                median_y = np.median(mi_ty_d)
                ax.scatter(median_x, median_y, marker='*', s=150, c='lime', edgecolors='darkgreen', linewidth=1.5, zorder=5)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)

            if dist_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx+1}", fontsize=18)

            if layer_idx == 0:
                ax.set_title(f"Dist [{b_min:.0f}-{b_max:.0f})", fontsize=18)

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    if last_cf is not None:
        top    = axes[0, 0].get_position().y1
        bottom = axes[-1, 0].get_position().y0
        cbar_ax = fig.add_axes([0.90, bottom, 0.03, top - bottom])
        cbar = fig.colorbar(last_cf, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=18)

    folder_path = f"./{model_name}/{dataset_name}/{process_type}/all"
    os.makedirs(folder_path, exist_ok=True)
    fname = f"{model_name}_{dataset_name}_{process_type}_{valid_str}_{calcul_type}_kde_matrix_DIFF.png"
    plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ DIFF matrix plot saved: {fname}")
