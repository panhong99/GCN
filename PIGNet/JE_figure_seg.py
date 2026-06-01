"""
Figure Module for Segmentation IB Experiment
- plot_scatter_matrix        : scatter (Layer × Distance bin), SAME or DIFF
- plot_scatter_matrix_combined : multi-dataset combined scatter (SAME top / DIFF bottom)
- plot_kde_matrix            : KDE contour (Layer × Distance bin), SAME or DIFF
- plot_ratio_barplot_all_models : mean±SD ratio barplot across all models
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
import os
import string

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'regular'

# ─── Constants ────────────────────────────────────────────────────────────────
FS_TITLE = 50
FS_LABEL = 50
FS_CBAR  = 50
FS_BAR   = 35

FIGURE_DIR = 'seg_final_figures'

COLOR_MAP = {'PIGNet_GSPonly': '#D81B60', 'ASPP': '#1E88E5', 'Mask2Former': '#FFC107'}
ALPHA_MAP = {'PIGNet_GSPonly': 0.55,      'ASPP': 0.45,      'Mask2Former': 0.45}

DIST_BINS  = np.arange(0, 50, 10)
NUM_DIST   = len(DIST_BINS) - 1
DRAW_ORDER = ['Mask2Former', 'ASPP', 'PIGNet_GSPonly']
# ─────────────────────────────────────────────────────────────────────────────


def _model_cmap(model_name):
    return LinearSegmentedColormap.from_list(
        model_name, ['#ffffff', COLOR_MAP.get(model_name, '#000000')])


def _save(fig, folder, fname):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Scatter matrix — single dataset, SAME or DIFF
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_scatter_matrix(mode, je_xt, je_ty, distance,
                        model_name, dataset_name,
                        valid_pascal=False, calcul_type='joint'):
    """
    mode: 'SAME' (circle marker) or 'DIFF' (triangle marker)
    je_xt, je_ty, distance: (num_layers, N)
    """
    assert mode in ('SAME', 'DIFF')
    marker    = 'o' if mode == 'SAME' else '^'
    valid_str = 'valid' if valid_pascal else 'invalid'
    layers    = je_xt.shape[0]
    cmap      = _model_cmap(model_name)

    fig, axes = plt.subplots(layers, NUM_DIST, figsize=(20, 18), facecolor='white')
    last_sc   = None

    for layer_idx in range(layers):
        for dist_idx, (b_min, b_max) in enumerate(zip(DIST_BINS[:-1], DIST_BINS[1:])):
            ax   = axes[layer_idx, dist_idx]
            ax.set_facecolor('white')
            mask = (distance[layer_idx] >= b_min) & (distance[layer_idx] < b_max)

            if not np.any(mask):
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        fontsize=FS_LABEL, transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            sc = ax.scatter(je_xt[layer_idx][mask], je_ty[layer_idx][mask],
                            c=distance[layer_idx][mask], cmap=cmap,
                            s=30, alpha=0.6, edgecolors='none', marker=marker)
            last_sc = sc
            ax.set_xticklabels([]); ax.set_yticklabels([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)
            if dist_idx == 0:
                ax.set_ylabel(f"Block {layer_idx+1}", fontsize=FS_LABEL)
            if layer_idx == 0:
                ax.set_title(f"Dist [{b_min:.0f}-{b_max:.0f})", fontsize=FS_TITLE)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    if last_sc is not None:
        top, bottom = axes[0, 0].get_position().y1, axes[-1, 0].get_position().y0
        cbar = fig.colorbar(last_sc, cax=fig.add_axes([0.90, bottom, 0.03, top - bottom]))
        cbar.ax.tick_params(labelsize=FS_CBAR)

    folder = os.path.join(FIGURE_DIR, model_name, dataset_name, 'all')
    fname  = f"{model_name}_{dataset_name}_{valid_str}_{calcul_type}_scatter_{mode}.png"
    _save(fig, folder, fname)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━══
# Scatter matrix — multi-dataset combined (SAME top row / DIFF bottom row)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_scatter_matrix_combined(datasets_list, model_name, 
                                 valid_pascal=False, calcul_type='joint', fname=None):
    """
    2 × num_datasets combined scatter.  Top row = SAME, bottom row = DIFF.
    datasets_list: list of dicts with keys:
        'name', 'je_xt_same', 'je_ty_same', 'distance_same',
                'je_xt_diff', 'je_ty_diff', 'distance_diff'
    """
    cmap       = _model_cmap(model_name)
    num_layers = datasets_list[0]['je_xt_same'].shape[0]
    num_ds     = len(datasets_list)

    all_dists = np.concatenate([
        np.concatenate([ds['distance_same'].ravel(), ds['distance_diff'].ravel()])
        for ds in datasets_list
    ])
    norm = Normalize(vmin=float(np.nanmin(all_dists)), vmax=float(np.nanmax(all_dists)))

    cell  = 5
    fig   = plt.figure(figsize=(cell * NUM_DIST * num_ds + 1.5,
                                cell * num_layers * 2 + 1.5), facecolor='white')
    outer = gridspec.GridSpec(2, num_ds + 1, figure=fig,
                              width_ratios=[NUM_DIST] * num_ds + [0.10],
                              hspace=0.30, wspace=0.08)

    MODES = [
        ('same', 'je_xt_same', 'je_ty_same', 'distance_same'),
        ('diff', 'je_xt_diff', 'je_ty_diff', 'distance_diff'),
    ]
    col_axes = {c: [] for c in range(num_ds)}
    last_sc  = None

    for row_idx, (mode_name, xt_key, ty_key, dist_key) in enumerate(MODES):
        for col_idx, ds in enumerate(datasets_list):
            inner = gridspec.GridSpecFromSubplotSpec(
                num_layers, NUM_DIST,
                subplot_spec=outer[row_idx, col_idx],
                hspace=0.12, wspace=0.08)

            for layer_idx in range(num_layers):
                for dist_idx, (b_min, b_max) in enumerate(zip(DIST_BINS[:-1], DIST_BINS[1:])):
                    ax = fig.add_subplot(inner[layer_idx, dist_idx])
                    ax.set_facecolor('white')
                    col_axes[col_idx].append(ax)

                    mask = ((ds[dist_key][layer_idx] >= b_min) &
                            (ds[dist_key][layer_idx] <  b_max))
                    if not np.any(mask):
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                                fontsize=FS_LABEL, transform=ax.transAxes)
                        ax.set_xticks([]); ax.set_yticks([])
                    else:
                        sc = ax.scatter(
                            ds[xt_key][layer_idx][mask], ds[ty_key][layer_idx][mask],
                            c=ds[dist_key][layer_idx][mask],
                            cmap=cmap, norm=norm, s=30, alpha=0.6, edgecolors='none',
                            marker='o' if mode_name == 'same' else '^')
                        last_sc = sc

                    ax.set_xticklabels([]); ax.set_yticklabels([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(False)
                    if col_idx == 0 and dist_idx == 0:
                        ax.set_ylabel(f"Block {layer_idx+1}", fontsize=FS_LABEL)
                    if row_idx == 0 and layer_idx == 0:
                        ax.set_title(f"Dist [{b_min:.0f}-{b_max:.0f})", fontsize=FS_TITLE)

    plt.tight_layout(rect=[0, 0.06, 0.97, 1])

    for col_idx, ds in enumerate(datasets_list):
        axs    = col_axes[col_idx]
        x_mid  = (min(a.get_position().x0 for a in axs) +
                  max(a.get_position().x1 for a in axs)) / 2
        y_bot  = min(a.get_position().y0 for a in axs)
        fig.text(x_mid, y_bot - 0.03,
                 f"({string.ascii_lowercase[col_idx]}) {ds['name']}",
                 ha='center', va='top', fontsize=FS_LABEL,
                 fontweight='bold', transform=fig.transFigure)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=fig.add_subplot(outer[:, num_ds])).ax.tick_params(labelsize=FS_CBAR)

    valid_str = 'valid' if valid_pascal else 'invalid'
    if fname is None:
        ct_str = f"_{calcul_type}" if calcul_type else ""
        ds_str = "_".join(d['name'] for d in datasets_list)
        fname  = f"{model_name}_{ds_str}_{valid_str}_{calcul_type}_scatter_combined.png"
    _save(fig, FIGURE_DIR, fname)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━══
# KDE matrix — single dataset, SAME or DIFF
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_kde_matrix(mode, model_name, dataset_name, vmin, vmax, kde_data,
                    process_type, valid_pascal=False, calcul_type='joint'):
    """
    mode: 'SAME' (Reds colormap) or 'DIFF' (Blues colormap)
    """
    assert mode in ('SAME', 'DIFF')
    z_key  = 'Z_s'        if mode == 'SAME' else 'Z_d'
    xt_key = 'je_xt_same' if mode == 'SAME' else 'je_xt_diff'
    ty_key = 'je_ty_same' if mode == 'SAME' else 'je_ty_diff'
    cmap   = plt.cm.get_cmap('Reds' if mode == 'SAME' else 'Blues').copy()
    cmap.set_bad('white')

    valid_str = 'valid' if valid_pascal else 'invalid'
    Xi, Yi    = kde_data['Xi'], kde_data['Yi']
    layers    = 4
    norm      = Normalize(vmin=vmin, vmax=vmax)
    levels    = np.linspace(vmin, vmax, 21)
    threshold = 1e-3

    fig, axes = plt.subplots(layers, NUM_DIST, figsize=(22, 18), facecolor='white')
    last_cf   = None

    for layer_idx in range(layers):
        for dist_idx, (b_min, b_max) in enumerate(zip(DIST_BINS[:-1], DIST_BINS[1:])):
            ax       = axes[layer_idx, dist_idx]
            ax.set_facecolor('white')
            bin_key  = f'layer_{layer_idx}_bin_{dist_idx}'

            if bin_key not in kde_data:
                ax.text(1, 1, 'No data', ha='center', va='center', fontsize=25)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            Z       = kde_data[bin_key][z_key]
            je_xt_b = kde_data[bin_key][xt_key]
            je_ty_b = kde_data[bin_key][ty_key]

            cf      = ax.contourf(Xi, Yi, np.ma.masked_less_equal(Z, threshold),
                                  levels=levels, cmap=cmap, norm=norm)
            last_cf = cf

            if len(je_xt_b) > 0:
                ax.scatter(np.median(je_xt_b), np.median(je_ty_b),
                           marker='*', s=150, c='lime',
                           edgecolors='darkgreen', linewidth=1.5, zorder=5)

            ax.set_xticklabels([]); ax.set_yticklabels([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)
            if dist_idx == 0:
                ax.set_ylabel(f"Block {layer_idx+1}", fontsize=FS_LABEL)
            if layer_idx == 0:
                ax.set_title(f"Dist [{b_min:.0f}-{b_max:.0f})", fontsize=FS_TITLE)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    if last_cf is not None:
        top, bottom = axes[0, 0].get_position().y1, axes[-1, 0].get_position().y0
        cbar = fig.colorbar(last_cf, cax=fig.add_axes([0.90, bottom, 0.03, top - bottom]))
        cbar.ax.tick_params(labelsize=FS_CBAR)

    folder = os.path.join(FIGURE_DIR, model_name, dataset_name, 'all')
    fname  = f"{model_name}_{dataset_name}_{valid_str}_{calcul_type}_kde_{mode}.png"
    _save(fig, folder, fname)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ratio barplot — all models, pascal + cityscape, all distance bins
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_ratio_barplot_all_models(models_data_pascal, models_data_cityscape,
                                   calcul_type='joint'):
    """
    4-column figure: Pascal-SAME, Pascal-DIFF, Cityscape-SAME, Cityscape-DIFF
    Rows: distance bins (0-10, 10-20, 20-30, 30-40)
    """
    def _model_list(mdata):
        return ([m for m in DRAW_ORDER if m in mdata] +
                [m for m in mdata   if m not in DRAW_ORDER])

    ml_p = _model_list(models_data_pascal)
    ml_c = _model_list(models_data_cityscape)
    all_models   = list(dict.fromkeys(ml_p + ml_c))
    model_colors = {m: COLOR_MAP.get(m, '#000000') for m in all_models}

    n_layers     = models_data_pascal[ml_p[0]]['je_xt_same'].shape[0]
    x_base       = np.arange(1, n_layers + 1)
    layer_labels = [f"L{i+1}" for i in range(n_layers)]

    col_configs = [
        (models_data_pascal,    ml_p, 'SAME', (0, 15)),
        (models_data_pascal,    ml_p, 'DIFF', (0, 15)),
        (models_data_cityscape, ml_c, 'SAME', (0, 10)),
        (models_data_cityscape, ml_c, 'DIFF', (0, 10)),
    ]

    fig, axes = plt.subplots(NUM_DIST, 4, figsize=(32, 22), facecolor='white')

    for col_idx, (mdata, mlist, mode, ylim) in enumerate(col_configs):
        xt_key = 'je_xt_same' if mode == 'SAME' else 'je_xt_diff'
        ty_key = 'je_ty_same' if mode == 'SAME' else 'je_ty_diff'
        marker = 'o'          if mode == 'SAME' else '^'

        for dist_idx, (b_min, b_max) in enumerate(zip(DIST_BINS[:-1], DIST_BINS[1:])):
            ax = axes[dist_idx, col_idx]
            ax.set_facecolor('white')

            for model_name in mlist:
                d        = mdata[model_name]
                color    = model_colors[model_name]
                alpha    = ALPHA_MAP.get(model_name, 0.6)
                dist     = d['distance']
                je_xt    = d[xt_key]
                je_ty    = d[ty_key]
                means, errs = [], []

                for layer_idx in range(n_layers):
                    mask  = (dist[layer_idx] >= b_min) & (dist[layer_idx] < b_max)
                    xt    = np.asarray(je_xt[layer_idx][mask], dtype=float)
                    ty    = np.asarray(je_ty[layer_idx][mask], dtype=float)
                    valid = np.isfinite(xt) & np.isfinite(ty) & (ty != 0)
                    ratio = xt[valid] / ty[valid]
                    if len(ratio) > 0:
                        means.append(np.mean(ratio))
                        errs.append(np.std(ratio, ddof=1))
                    else:
                        means.append(np.nan); errs.append(0)

                mv = np.array(means); ev = np.array(errs)
                ok = ~np.isnan(mv)
                vx, vy, ve = x_base[ok], mv[ok], ev[ok]
                if len(vx) == 0:
                    continue

                ax.bar(vx, vy, width=0.6, color=color, alpha=alpha, zorder=3, edgecolor='none')
                ax.plot(vx, vy, '-', color=color, linewidth=2,
                        alpha=min(alpha+0.15, 1.0), zorder=5)
                ax.plot(vx, vy, marker, color=color, markersize=7,
                        markeredgecolor='white', markeredgewidth=0.8,
                        alpha=min(alpha+0.15, 1.0), zorder=6)
                ax.errorbar(vx, vy, yerr=ve, fmt='none',
                            ecolor=color, elinewidth=1.2, capsize=4, capthick=1.2,
                            alpha=min(alpha+0.15, 1.0), zorder=4)

            ax.set_xticks(x_base); ax.set_xticklabels(layer_labels, fontsize=FS_BAR)
            ax.tick_params(axis='y', labelsize=FS_BAR)
            ax.set_xlim(x_base[0]-0.5, x_base[-1]+0.5)
            ax.set_ylim(*ylim)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.grid(False)
            ax.set_title(f"{mode} - Dist [{b_min:.0f}–{b_max:.0f})", fontsize=FS_BAR)
            if dist_idx == NUM_DIST - 1:
                ax.set_xlabel("Layer", fontsize=FS_BAR)
            if col_idx == 0:
                ax.set_ylabel("H(X,T) / H(T,Y)", fontsize=FS_BAR)

            if col_idx == 0 and dist_idx == 0:
                handles = [mpatches.Patch(color=model_colors[m],
                                          alpha=ALPHA_MAP.get(m, 0.6), label=m)
                           for m in ['PIGNet_GSPonly', 'ASPP', 'Mask2Former']
                           if m in model_colors]
                ax.legend(handles=handles, fontsize=FS_BAR, loc='upper left', frameon=False)

    plt.tight_layout()
    folder = os.path.join(FIGURE_DIR, 'ALL_MODELS', 'combined', 'ratio')
    fname  = f"ALL_combined_{calcul_type}_ratio_barplot.pdf"
    _save(fig, folder, fname)
