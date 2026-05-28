"""
MI Plotting Module
모든 plot 함수 전담
"""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'regular'

# ─── Font size constants ──────────────────────────────────────────────────────
FS_TITLE = 50   # subplot titles  ("Dist [0-10)", …)
FS_LABEL = 50   # axis labels     ("Block 1", …)
FS_CBAR  = 50   # colorbar tick labels
# ─────────────────────────────────────────────────────────────────────────────

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
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=FS_LABEL,
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
                ax.set_ylabel(f"Block {layer_idx+1}", fontsize=FS_LABEL)

            if layer_idx == 0:
                ax.set_title(f"Dist [{b_min:.0f}-{b_max:.0f})", fontsize=FS_TITLE)

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    if last_sc is not None:
        top    = axes[0, 0].get_position().y1
        bottom = axes[-1, 0].get_position().y0
        cbar_ax = fig.add_axes([0.90, bottom, 0.03, top - bottom])
        cbar = fig.colorbar(last_sc, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=FS_CBAR)

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
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=FS_LABEL,
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
                ax.set_ylabel(f"Block {layer_idx+1}", fontsize=FS_LABEL)

            if layer_idx == 0:
                ax.set_title(f"Dist [{b_min:.0f}-{b_max:.0f})", fontsize=FS_TITLE)

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    if last_sc is not None:
        top    = axes[0, 0].get_position().y1
        bottom = axes[-1, 0].get_position().y0
        cbar_ax = fig.add_axes([0.90, bottom, 0.03, top - bottom])
        cbar = fig.colorbar(last_sc, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=FS_CBAR)

    fname = f"{model_name}_{dataset_name}_{process_type}_{valid_str}_{calcul_type}_scatter_matrix_DIFF.png"
    plt.savefig(os.path.join(folder_path, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ DIFF scatter matrix plot saved: {fname}")


def plot_scatter_matrix_combined(datasets_list, model_name, process_type,
                                  valid_pascal=False, calcul_type='MI', fname=None):
    """
    2 × num_datasets combined scatter matrix.
    Top row = SAME, Bottom row = DIFF. One column per dataset.
    Single shared colorbar on the far right.

    datasets_list: list of dicts with keys:
        'name'          : str  (display name, e.g. 'Pascal VOC')
        'mi_xt_same'    : (num_layers, N)
        'mi_ty_same'    : (num_layers, N)
        'distance_same' : (num_layers, N)
        'mi_xt_diff'    : (num_layers, N)
        'mi_ty_diff'    : (num_layers, N)
        'distance_diff' : (num_layers, N)
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import matplotlib.cm as cm
    import string

    color_map_colors = {'PIGNet_GSPonly': '#D81B60', 'ASPP': '#1E88E5', 'Mask2Former': '#FFC107'}
    model_color = color_map_colors.get(model_name, '#000000')
    cmap = LinearSegmentedColormap.from_list(model_name, ['#ffffff', model_color])

    dist_bins  = np.arange(0, 50, 10)
    num_dist   = len(dist_bins) - 1
    num_layers = datasets_list[0]['mi_xt_same'].shape[0]
    num_ds     = len(datasets_list)

    # ── global colour normalisation (shared colorbar) ────────────────────
    all_dists = np.concatenate([
        np.concatenate([ds['distance_same'].ravel(), ds['distance_diff'].ravel()])
        for ds in datasets_list
    ])
    vmin, vmax = float(np.nanmin(all_dists)), float(np.nanmax(all_dists))
    norm = Normalize(vmin=vmin, vmax=vmax)

    cell   = 5          # inches per scatter subplot
    fig_w  = cell * num_dist * num_ds + 1.5
    fig_h  = cell * num_layers * 2 + 1.5
    fig    = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    # outer GridSpec: 2 rows × (num_ds scatter blocks + 1 cbar)
    outer_gs = gridspec.GridSpec(
        2, num_ds + 1,
        figure=fig,
        width_ratios=[num_dist] * num_ds + [0.10],
        hspace=0.30,
        wspace=0.08,
    )

    modes = [
        ('same', 'mi_xt_same', 'mi_ty_same', 'distance_same'),
        ('diff', 'mi_xt_diff', 'mi_ty_diff', 'distance_diff'),
    ]

    col_axes = {c: [] for c in range(num_ds)}   # all axes per outer column
    last_sc  = None

    for row_idx, (mode_name, xt_key, ty_key, dist_key) in enumerate(modes):
        is_top = (row_idx == 0)
        for col_idx, ds in enumerate(datasets_list):
            is_left = (col_idx == 0)

            mi_xt    = ds[xt_key]
            mi_ty    = ds[ty_key]
            distance = ds[dist_key]

            inner_gs = gridspec.GridSpecFromSubplotSpec(
                num_layers, num_dist,
                subplot_spec=outer_gs[row_idx, col_idx],
                hspace=0.12,
                wspace=0.08,
            )

            for layer_idx in range(num_layers):
                for dist_idx, (b_min, b_max) in enumerate(
                        zip(dist_bins[:-1], dist_bins[1:])):
                    ax = fig.add_subplot(inner_gs[layer_idx, dist_idx])
                    ax.set_facecolor('white')
                    col_axes[col_idx].append(ax)

                    mask = ((distance[layer_idx] >= b_min) &
                            (distance[layer_idx] < b_max))

                    if not np.any(mask):
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                                fontsize=FS_LABEL, transform=ax.transAxes)
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        sc = ax.scatter(
                            mi_xt[layer_idx][mask],
                            mi_ty[layer_idx][mask],
                            c=distance[layer_idx][mask],
                            cmap=cmap, norm=norm,
                            s=30, alpha=0.6, edgecolors='none',
                            marker='o' if mode_name == 'same' else '^',
                        )
                        last_sc = sc

                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(False)

                    if is_left and dist_idx == 0:
                        ax.set_ylabel(f"Block {layer_idx+1}", fontsize=FS_LABEL)

                    if is_top and layer_idx == 0:
                        ax.set_title(f"Dist [{b_min:.0f}-{b_max:.0f})",
                                     fontsize=FS_TITLE)

    plt.tight_layout(rect=[0, 0.06, 0.97, 1])

    # ── dataset name labels below each column (positions after tight_layout) ──
    for col_idx, ds in enumerate(datasets_list):
        axs     = col_axes[col_idx]
        x_left  = min(a.get_position().x0 for a in axs)
        x_right = max(a.get_position().x1 for a in axs)
        y_bot   = min(a.get_position().y0 for a in axs)
        label   = f"({string.ascii_lowercase[col_idx]}) {ds['name']}"
        fig.text((x_left + x_right) / 2, y_bot - 0.03, label,
                 ha='center', va='top', fontsize=FS_LABEL,
                 fontweight='bold', transform=fig.transFigure)

    # ── single shared colorbar ────────────────────────────────────────────
    cbar_ax = fig.add_subplot(outer_gs[:, num_ds])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=FS_CBAR)

    valid_str = 'valid' if valid_pascal else 'invalid'
    if fname is None:
        ct_str = f"_{calcul_type}" if calcul_type else ""
        ds_str = "_".join(d['name'] for d in datasets_list)
        fname  = f"{model_name}_{ds_str}_{process_type}{ct_str}_{valid_str}_scatter_matrix_combined.png"

    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Combined scatter matrix saved: {fname}")
