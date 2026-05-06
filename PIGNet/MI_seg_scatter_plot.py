"""
MI Plotting Module
모든 plot 함수 전담
"""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'regular'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 활성화: Scatter Matrix
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_scatter_matrix_same(mi_xt_same, mi_ty_same, distance,
                             model_name, dataset_name, process_type,
                             valid_pascal=False, calcul_type='MI'):
    """
    Matrix plot: Layer (y축) x Distance (x축) for SAME mode
    4x4 grid (4 layers, 4 distance bins)
    전체 figure 오른쪽에 하나의 colorbar 표시
    """
    from matplotlib.colors import LinearSegmentedColormap
    color_map = {'PIGNet_GSPonly': '#D81B60', 'ASPP': '#1E88E5', 'Mask2Former': '#FFC107'}
    model_color = color_map.get(model_name, '#000000')
    cmap = LinearSegmentedColormap.from_list(model_name, ['#ffffff', model_color])

    valid_str = 'valid' if valid_pascal else 'invalid'
    layers = mi_xt_same.shape[0]
    dist_bins = np.arange(0, 50, 10)
    num_dist = len(dist_bins) - 1
    folder_path = f"./{model_name}/{dataset_name}/{process_type}/all"
    os.makedirs(folder_path, exist_ok=True)

    fig, axes = plt.subplots(layers, num_dist, figsize=(20, 18), facecolor='white')

    last_sc = None

    for layer_idx in range(layers):
        for dist_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
            ax = axes[layer_idx, dist_idx]
            ax.set_facecolor('white')

            mask = (distance[layer_idx] >= b_min) & (distance[layer_idx] < b_max)

            if not np.any(mask):
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=25,
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            dist_bin = distance[layer_idx][mask]
            sc = ax.scatter(mi_xt_same[layer_idx][mask],
                            mi_ty_same[layer_idx][mask],
                            c=dist_bin, cmap=cmap,
                            s=30, alpha=0.6, edgecolors='none', marker='o')
            last_sc = sc

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)

            if dist_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx+1}", fontsize=25)

            if layer_idx == 0:
                ax.set_title(f"Dist [{b_min:.0f}-{b_max:.0f})", fontsize=25)

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    if last_sc is not None:
        top    = axes[0, 0].get_position().y1
        bottom = axes[-1, 0].get_position().y0
        cbar_ax = fig.add_axes([0.90, bottom, 0.03, top - bottom])
        cbar = fig.colorbar(last_sc, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=18)

    fname = f"{model_name}_{dataset_name}_{process_type}_{valid_str}_{calcul_type}_scatter_matrix_SAME.png"
    plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ SAME scatter matrix plot saved: {fname}")


def plot_scatter_matrix_diff(mi_xt_diff, mi_ty_diff, distance,
                             model_name, dataset_name, process_type,
                             valid_pascal=False, calcul_type='MI'):
    """
    Matrix plot: Layer (y축) x Distance (x축) for DIFF mode
    4x4 grid (4 layers, 4 distance bins)
    전체 figure 오른쪽에 하나의 colorbar 표시
    """
    from matplotlib.colors import LinearSegmentedColormap
    color_map = {'PIGNet_GSPonly': '#D81B60', 'ASPP': '#1E88E5', 'Mask2Former': '#FFC107'}
    model_color = color_map.get(model_name, '#000000')
    cmap = LinearSegmentedColormap.from_list(model_name, ['#ffffff', model_color])

    valid_str = 'valid' if valid_pascal else 'invalid'
    layers = mi_xt_diff.shape[0]
    dist_bins = np.arange(0, 50, 10)
    num_dist = len(dist_bins) - 1
    folder_path = f"./{model_name}/{dataset_name}/{process_type}/all"
    os.makedirs(folder_path, exist_ok=True)

    fig, axes = plt.subplots(layers, num_dist, figsize=(20, 18), facecolor='white')

    last_sc = None

    for layer_idx in range(layers):
        for dist_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
            ax = axes[layer_idx, dist_idx]
            ax.set_facecolor('white')

            mask = (distance[layer_idx] >= b_min) & (distance[layer_idx] < b_max)

            if not np.any(mask):
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=25,
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            dist_bin = distance[layer_idx][mask]
            sc = ax.scatter(mi_xt_diff[layer_idx][mask],
                            mi_ty_diff[layer_idx][mask],
                            c=dist_bin, cmap=cmap,
                            s=30, alpha=0.6, edgecolors='none', marker='^')
            last_sc = sc

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)

            if dist_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx+1}", fontsize=25)

            if layer_idx == 0:
                ax.set_title(f"Dist [{b_min:.0f}-{b_max:.0f})", fontsize=25)

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    if last_sc is not None:
        top    = axes[0, 0].get_position().y1
        bottom = axes[-1, 0].get_position().y0
        cbar_ax = fig.add_axes([0.90, bottom, 0.03, top - bottom])
        cbar = fig.colorbar(last_sc, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=25)

    fname = f"{model_name}_{dataset_name}_{process_type}_{valid_str}_{calcul_type}_scatter_matrix_DIFF.png"
    plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ DIFF scatter matrix plot saved: {fname}")
