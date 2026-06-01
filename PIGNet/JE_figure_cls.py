import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'regular'

# ─── Font size constants — edit here to resize all text globally ──────────────
FS_TITLE    = 45   # subplot titles  ("Block 1", "Block 2", …)
FS_LABEL    = 45   # axis labels     (H(X,T), H(T,Y))
FS_CBAR     = 32   # colorbar tick labels
FS_LEGEND   = 60   # legend text
FS_SUBTITLE = 60   # dataset name above each panel
FS_BAR_XTICK = 20  # barplot x-axis tick label size
FS_BAR_YTICK = 20  # barplot y-axis tick label size
# ─────────────────────────────────────────────────────────────────────────────

# ─── Model colors ─────────────────────────────────────────────────────────────
COLOR_MAP = {
    'PIGNet_Backbone': '#7B1FA2',
    'PIGNet_GSP':      '#D81B60',
    'Resnet':          '#5C6BC0',
    'vit':             '#FF7043',
}

ALPHA_MAP = {
    'Resnet':          0.45,
    'vit':             0.45,
    'PIGNet_Backbone': 0.55,
    'PIGNet_GSP':      0.55,
}

LEGEND_DISPLAY_NAMES = {
    'vit': 'ViT',
}
# ─────────────────────────────────────────────────────────────────────────────

# ─── Dataset display name mapping ─────────────────────────────────────────────
DATASET_DISPLAY_NAMES = {
    'imagenet':  'ImageNet-100K',
    'CIFAR-10':  'CIFAR-10',
    'CIFAR-100': 'CIFAR-100',
}
# ─────────────────────────────────────────────────────────────────────────────

FIGURE_DIR = 'cls_final_figures'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Scatter plot — multi-dataset horizontal stack
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_scatter_combined_all_models_datasets(datasets_list, process_type, num,
                                              calcul_type=None, fname=None):
    """
    Stack each dataset's combined-model scatter figure horizontally.

    Args:
        datasets_list: list of dicts, each with:
            'dataset_name': str
            'models_list': list of {'display_name': str, 'all_layers_data': list}
        process_type: str
        num: int  (ncols = max(4, num-1))
        calcul_type: str  (key prefix for xt/ty in layer_data)
        fname: optional save path
    """
    ct              = calcul_type or 'joint'
    ncols_scatter   = max(4, num - 1)
    num_datasets    = len(datasets_list)
    num_models      = max(len(d['models_list']) for d in datasets_list)
    GAP_RATIO       = 0.30
    cols_per_ds     = ncols_scatter + 1          # scatter cols + 1 colorbar col

    # Build width_ratios: [scatter…, cbar, gap?, scatter…, cbar, …]
    width_ratios = []
    for ds_idx in range(num_datasets):
        width_ratios.extend([1.0] * ncols_scatter + [0.06])
        if ds_idx < num_datasets - 1:
            width_ratios.append(GAP_RATIO)

    total_cols = len(width_ratios)

    def col_offset(ds_idx):
        return ds_idx * (cols_per_ds + 1)      # +1 for the gap col

    fig = plt.figure(
        figsize=((5 * ncols_scatter + 1.5) * num_datasets, 5 * num_models + 1.5),
        facecolor='white',
    )
    gs = gridspec.GridSpec(
        num_models, total_cols,
        figure=fig,
        width_ratios=width_ratios,
        hspace=0.65,
        wspace=0.25,
    )

    # Vertical offset knobs (figure coords)
    SUBTITLE_Y_OFFSET = 0.055   # top of plots → dataset name label
    LEGEND_Y_OFFSET   = 0.075   # top of plots → legend (must > SUBTITLE)

    first_plot_ax = None
    row0_first_ax = {}          # ds_idx → first scatter ax of row 0
    row0_cbar_ax  = {}          # ds_idx → colorbar ax of row 0

    for ds_idx, dataset_info in enumerate(datasets_list):
        c_off       = col_offset(ds_idx)
        is_leftmost = (ds_idx == 0)

        for row_idx, model_info in enumerate(dataset_info['models_list']):
            display_name    = model_info['display_name']
            all_layers_data = model_info['all_layers_data']
            num_layers      = len(all_layers_data)

            color      = COLOR_MAP.get(display_name, '#888888')
            model_cmap = mcolors.LinearSegmentedColormap.from_list('model_cmap', ['#ffffff', color])
            last_sc    = None

            for idx, layer_data in enumerate(all_layers_data):
                ax = fig.add_subplot(gs[row_idx, c_off + idx])
                ax.set_facecolor('white')

                if first_plot_ax is None:
                    first_plot_ax = ax
                if row_idx == 0 and idx == 0 and ds_idx not in row0_first_ax:
                    row0_first_ax[ds_idx] = ax

                layer_idx = layer_data['layer_idx']
                je_xt     = np.asarray(layer_data[f'{ct}_xt'], dtype=float)
                je_ty     = np.asarray(layer_data[f'{ct}_ty'], dtype=float)

                jitter_std  = (je_ty.max() - je_ty.min()) * 0.015 if je_ty.max() > je_ty.min() else 1e-3
                je_ty_plot  = je_ty + np.random.default_rng(seed=42).normal(0, jitter_std, size=je_ty.shape)

                sc      = ax.scatter(je_xt, je_ty_plot, c=je_xt, cmap=model_cmap,
                                     s=20, alpha=0.6, edgecolors='none', vmin=0)
                last_sc = sc

                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(False)

                if is_leftmost and idx == 0:
                    ax.set_ylabel("H(T,Y)", fontsize=FS_LABEL)
                ax.set_xlabel("H(X,T)", fontsize=FS_LABEL)
                ax.set_title(f"Block {layer_idx}", fontsize=FS_TITLE)

            # Empty placeholder subplots for unused columns
            for idx in range(num_layers, ncols_scatter):
                fig.delaxes(fig.add_subplot(gs[row_idx, c_off + idx]))

            if last_sc is not None:
                cbar_ax = fig.add_subplot(gs[row_idx, c_off + ncols_scatter])
                cbar    = fig.colorbar(last_sc, cax=cbar_ax)
                cbar.ax.tick_params(labelsize=FS_CBAR)
                if row_idx == 0:
                    row0_cbar_ax[ds_idx] = cbar_ax

    plt.tight_layout(rect=[0, 0, 1, 0.85])

    # ── Dataset name subtitle (centered above each panel) ────────────────────
    for ds_idx, dataset_info in enumerate(datasets_list):
        fa = row0_first_ax.get(ds_idx)
        ca = row0_cbar_ax.get(ds_idx)
        if fa is None:
            continue
        pos_l    = fa.get_position()
        pos_r    = ca.get_position() if ca is not None else pos_l
        center_x = (pos_l.x0 + pos_r.x1) / 2
        raw_name = dataset_info['dataset_name']
        fig.text(center_x, pos_l.y1 + SUBTITLE_Y_OFFSET,
                 DATASET_DISPLAY_NAMES.get(raw_name, raw_name),
                 ha='center', va='bottom',
                 fontsize=FS_SUBTITLE, fontweight='bold',
                 transform=fig.transFigure)

    # ── Legend (aligned to leftmost panel, above subtitle) ───────────────────
    first_models   = datasets_list[0]['models_list']
    legend_handles = [
        mpatches.Patch(color=COLOR_MAP.get(m['display_name'], '#888888'),
                       label=LEGEND_DISPLAY_NAMES.get(m['display_name'], m['display_name']))
        for m in first_models
    ]
    if first_plot_ax is not None:
        pos = first_plot_ax.get_position()
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, pos.y1 + LEGEND_Y_OFFSET),
            bbox_transform=fig.transFigure,
            ncol=len(first_models),
            fontsize=FS_LEGEND,
            frameon=False,
        )

    if fname is None:
        ct_str = f"_{ct}" if ct else ""
        ds_str = "_".join(d['dataset_name'] for d in datasets_list)
        os.makedirs(FIGURE_DIR, exist_ok=True)
        fname  = os.path.join(FIGURE_DIR, f"ALL_MODELS_{ds_str}_{process_type}{ct_str}_scatter.png")

    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved: {fname}")
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Barplot — H(X,T)/H(T,Y) ratio, all models × all datasets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_ratio_barplot_all_models_classification(datasets_list, process_type,
                                                 calcul_type='joint'):
    """
    Mean ± SD ratio errorbar plot — one subplot per dataset, all in one figure.

    Args:
        datasets_list: list of {'dataset_name': str,
                                'models_data': dict  model_name → list of layer_data}
        process_type: str
        calcul_type: str
    """
    ct           = calcul_type
    num_datasets = len(datasets_list)

    # Max layers across all datasets (for shared x-axis scale)
    max_layers = max(
        max(len(v) for k, v in ds['models_data'].items() if k != 'PIGNet_GSP')
        for ds in datasets_list
    )
    x_base       = np.arange(1, max_layers + 1)
    layer_labels = [f"B{i+1}" for i in range(max_layers)]

    DRAW_ORDER = ['Resnet', 'vit', 'PIGNet_Backbone']

    fig, axes = plt.subplots(
        1, num_datasets,
        figsize=(max(6, max_layers * 1.8) * num_datasets, 5),
        facecolor='white',
    )
    if num_datasets == 1:
        axes = [axes]

    for ax_idx, (ax, ds_info) in enumerate(zip(axes, datasets_list)):
        dataset_name = ds_info['dataset_name']
        models_data  = ds_info['models_data']

        model_list = [m for m in models_data if m != 'PIGNet_GSP']
        draw_order = [m for m in DRAW_ORDER if m in model_list] + \
                     [m for m in model_list if m not in DRAW_ORDER]

        for model_name in draw_order:
            layers_data = models_data[model_name]
            color       = COLOR_MAP.get(model_name, '#000000')
            alpha       = ALPHA_MAP.get(model_name, 0.5)
            x_pos       = x_base[:len(layers_data)]

            mean_vals, err_vals = [], []
            for layer_data in layers_data:
                mi_xt = np.asarray(layer_data[f'{ct}_xt'], dtype=float).flatten()
                mi_ty = np.asarray(layer_data[f'{ct}_ty'], dtype=float).flatten()
                ratio = mi_xt / mi_ty
                mean_vals.append(np.mean(ratio))
                err_vals.append(np.std(ratio, ddof=1))

            mean_vals = np.array(mean_vals)
            err_vals  = np.array(err_vals)

            ax.bar(x_pos, mean_vals, width=0.6, color=color, alpha=alpha,
                   zorder=3, edgecolor='none')
            ax.plot(x_pos, mean_vals, '-', color=color, linewidth=2,
                    alpha=min(alpha + 0.15, 1.0), zorder=5)
            ax.plot(x_pos, mean_vals, 'o', color=color, markersize=7,
                    markeredgecolor='white', markeredgewidth=0.8,
                    alpha=min(alpha + 0.15, 1.0), zorder=6)
            ax.errorbar(x_pos, mean_vals, yerr=err_vals, fmt='none',
                        ecolor=color, elinewidth=1.2, capsize=4, capthick=1.2,
                        alpha=min(alpha + 0.15, 1.0), zorder=4)

        ax.set_xticks(x_base)
        ax.set_xticklabels(layer_labels, fontsize=25)
        ax.tick_params(axis='y', labelsize=FS_BAR_YTICK)
        ax.set_xlim(x_base[0] - 0.5, x_base[-1] + 0.5)
        ax.set_ylim(0, 2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

        if ax_idx == 1:
            ax.set_xlabel("Block", fontsize=30)

        ax.set_title(DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name),
                     fontsize=35, fontweight='bold')
        if ax_idx == 0:
            ax.set_ylabel("H(X,T) / H(T,Y)", fontsize=30)

    # Shared legend above all subplots
    all_model_names = {m for ds in datasets_list for m in ds['models_data'] if m != 'PIGNet_GSP'}
    legend_handles = [
        mpatches.Patch(color=COLOR_MAP.get(m, '#000000'),
                       alpha=ALPHA_MAP.get(m, 0.5),
                       label=LEGEND_DISPLAY_NAMES.get(m, m))
        for m in DRAW_ORDER if m in all_model_names
    ]
    if legend_handles:
        fig.legend(handles=legend_handles, fontsize=30,
                   loc='upper center', bbox_to_anchor=(0.5, 1.3),
                   ncol=len(legend_handles), frameon=False)

    plt.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    ds_str = "_".join(ds['dataset_name'] for ds in datasets_list)
    fname  = os.path.join(FIGURE_DIR, f"{ds_str}_{process_type}_{calcul_type}_ratio_barplot.pdf")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Ratio barplot saved: {fname}")
