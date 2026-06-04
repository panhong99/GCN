import numpy as np
import os
import pickle
import argparse
from JE_calcul_cls import calcul_JE, calcul_JE_kde
from JE_figure_cls import (
    plot_scatter_combined_all_models_datasets,
    plot_ratio_barplot_all_models_classification,
)

PIGNET_MODEL  = 'PIGNet_GSPonly_classification'
LAYER_NUM     = 5
BACKBONE_NUM  = 4
GSP_LAYER_NUM = 5

BASE_DATA_ROOT  = '/home/hail/pan/HDD/IB_dataset'
SCATTER_MODEL_ORDER = ['PIGNet_Backbone','Resnet', 'vit']

# ── private I/O helpers ───────────────────────────────────────────────────────
def _cache_path(base, tag, calcul_type, cluster_num):
    return os.path.join(base, f'{calcul_type}_mi_analysis_cache_{tag}_cluster{cluster_num}.pkl')

def _load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def _save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

# ── path builder ──────────────────────────────────────────────────────────────
def build_data_path(args, dataset, model_name):
    backbone_dir = f'resnet{args.backbone}' if args.backbone.isdigit() else args.backbone
    return os.path.join(
        BASE_DATA_ROOT,
        dataset,
        backbone_dir,
        args.model_type,
        model_name,
        'zoom', '1',
    )

# ── JE ────────────────────────────────────────────────────────────────────────
def load_or_calcul_je(args, dataset, model_name, ct, c):
    """Cache 있으면 로드, 없으면 JE 계산 후 저장.

    Returns:
        layers_data : list of layer dicts
        data_path   : str
    """
    is_pignet = (model_name == PIGNET_MODEL)
    data_path = build_data_path(args, dataset, model_name)

    je_cache_file       = _cache_path(data_path, 'classification',         ct, c)
    backbone_cache_file = _cache_path(data_path, 'backbone_classification', ct, c)
    gsp_cache_file      = _cache_path(data_path, 'gsp_classification',      ct, c)

    if not is_pignet and os.path.exists(je_cache_file):
        print(f"    [{dataset}/{model_name}] Loading JE cache")
        return _load_pkl(je_cache_file), data_path

    if is_pignet and os.path.exists(backbone_cache_file) and os.path.exists(gsp_cache_file):
        print(f"    [{dataset}/{model_name}] Loading JE cache (backbone+gsp)")
        return _load_pkl(backbone_cache_file) + _load_pkl(gsp_cache_file), data_path

    print(f"    [{dataset}/{model_name}] No JE cache — computing from VQ labels")
    y_in = _load_pkl(os.path.join(data_path, f'y_labels_{c}.pkl'))

    orig_model = args.model
    args.model = model_name

    if not is_pignet:
        x_in  = _load_pkl(os.path.join(data_path, f'layer_0_{c}.pkl'))
        t_in  = [_load_pkl(os.path.join(data_path, f'layer_{i}_{c}.pkl'))
                 for i in range(1, LAYER_NUM)]
        H_dim, W_dim = x_in.shape[1], x_in.shape[2]
        calcul_JE(args, x_in, t_in, y_in, H_dim, W_dim,
                  je_cache_file=je_cache_file)
        args.model = orig_model
        return _load_pkl(je_cache_file), data_path

    else:
        bx = _load_pkl(os.path.join(data_path, f'backbone_layer_0_{c}.pkl'))
        gx = _load_pkl(os.path.join(data_path, f'gsp_layer_0_{c}.pkl'))
        bt = [_load_pkl(os.path.join(data_path, f'backbone_layer_{i}_{c}.pkl'))
              for i in range(1, BACKBONE_NUM)]
        gt = [_load_pkl(os.path.join(data_path, f'gsp_layer_{i}_{c}.pkl'))
              for i in range(1, GSP_LAYER_NUM)]
        H_dim, W_dim = bx.shape[1], bx.shape[2]
        calcul_JE(args, None, None, y_in, H_dim, W_dim,
                  backbone_x_in=bx, backbone_t_in=bt,
                  gsp_x_in=gx, gsp_t_in=gt,
                  backbone_cache_file=backbone_cache_file, gsp_cache_file=gsp_cache_file,
                  backbonenum=BACKBONE_NUM, gsp_layer_num=GSP_LAYER_NUM)
        args.model = orig_model
        return _load_pkl(backbone_cache_file) + _load_pkl(gsp_cache_file), data_path

# ── KDE ───────────────────────────────────────────────────────────────────────
def load_or_calcul_kde(data_path, all_layers_data, is_pignet, ct, c):
    """Cache 있으면 스킵, 없으면 KDE 계산 후 저장."""
    kde_f    = _cache_path(data_path, 'kde_classification',          ct, c)
    kde_bb_f = _cache_path(data_path, 'kde_backbone_classification', ct, c)
    kde_gp_f = _cache_path(data_path, 'kde_gsp_classification',      ct, c)

    def _run(cache_file, layers_data):
        if os.path.exists(cache_file):
            print(f"    KDE loaded: {os.path.basename(cache_file)}")
            return
        print(f"    Computing KDE → {os.path.basename(cache_file)}")
        _save_pkl(calcul_JE_kde(layers_data, calcul_type=ct), cache_file)

    if not is_pignet:
        _run(kde_f, all_layers_data)
    else:
        bb = [d for d in all_layers_data if d.get('group') == 'Backbone']
        gp = [d for d in all_layers_data if d.get('group') == 'GSP']
        _run(kde_bb_f, bb)
        _run(kde_gp_f, gp)

# ── PIGNet helper ─────────────────────────────────────────────────────────────
def split_pignet_layers(layers_data):
    """PIGNet layers_data → {'PIGNet_Backbone': [...], 'PIGNet_GSP': [...]}"""
    bb = [d for d in layers_data if d.get('group') == 'Backbone']
    gp = [d for d in layers_data if d.get('group') == 'GSP']
    result = {}
    if bb: result['PIGNet_Backbone'] = bb
    if gp: result['PIGNet_GSP']      = gp
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',      type=str,  default='CIFAR-10',
                        help='Single dataset (ignored if --datasets is set)')
    parser.add_argument('--datasets',     type=str,  nargs='+', default=None,
                        help='e.g. --datasets CIFAR-10 CIFAR-100 imagenet')
    parser.add_argument('--model',        type=str,  default=PIGNET_MODEL,
                        help='Resnet | vit | PIGNet_GSPonly_classification')
    parser.add_argument('--backbone',     type=str,  default='101',  help='50 or 101')
    parser.add_argument('--cluster_num',  type=int,  default=50)
    parser.add_argument('--calcul_type',  type=str,  default='joint')
    parser.add_argument('--model_type',   type=str,  default='scratch')
    parser.add_argument('--all_models',   action='store_true', default=False,
                        help='Show all models in scatter and draw barplot')
    args = parser.parse_args()

    ct       = args.calcul_type
    c        = args.cluster_num
    datasets = args.datasets if args.datasets else [args.dataset]

    ALL_MODEL_NAMES  = ['Resnet', 'vit', PIGNET_MODEL]
    models_to_load   = ALL_MODEL_NAMES if args.all_models else [args.model]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. JE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n=== JE ===")
    dataset_model_layers = {ds: {} for ds in datasets}
    dataset_paths        = {ds: {} for ds in datasets}

    for ds in datasets:
        for model_name in models_to_load:
            try:
                layers, path = load_or_calcul_je(args, ds, model_name, ct, c)
                dataset_model_layers[ds][model_name] = layers
                dataset_paths[ds][model_name]        = path
            except FileNotFoundError as e:
                print(f"    [{ds}/{model_name}] Skipped — {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. Scatter Plot
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n=== Scatter Plot ===")
    datasets_list = []
    max_num = 0

    for ds in datasets:
        named = {}
        for model_name, layers in dataset_model_layers[ds].items():
            if model_name == PIGNET_MODEL:
                named.update(split_pignet_layers(layers))
            else:
                named[model_name] = layers

        models_list = [
            {'display_name': m, 'all_layers_data': named[m]}
            for m in SCATTER_MODEL_ORDER if m in named
        ]
        max_num = max(max_num, max((len(m['all_layers_data']) for m in models_list), default=0))
        datasets_list.append({'dataset_name': ds, 'models_list': models_list})

    if datasets_list:
        plot_scatter_combined_all_models_datasets(
            args,
            datasets_list,
            num=max_num + 1,
            calcul_type=ct,
        )

    # # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # # 3. KDE
    # # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # print("\n=== KDE ===")
    # for ds in datasets:
    #     if args.model not in dataset_model_layers[ds]:
    #         continue
    #     print(f"  [{ds}/{args.model}]")
    #     load_or_calcul_kde(
    #         dataset_paths[ds][args.model],
    #         dataset_model_layers[ds][args.model],
    #         is_pignet=(args.model == PIGNET_MODEL),
    #         ct=ct, c=c,
    #     )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. All-Models Barplot  (--all_models)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if not args.all_models:
        print("\nDone.")
        exit(0)

    print("\n=== All-Models Barplot ===")
    barplot_datasets = []

    for ds in datasets:
        named = {}
        for model_name, layers in dataset_model_layers[ds].items():
            if model_name == PIGNET_MODEL:
                named.update(split_pignet_layers(layers))
            else:
                named[model_name] = layers
        if named:
            barplot_datasets.append({'dataset_name': ds, 'models_data': named})

    if barplot_datasets:
        plot_ratio_barplot_all_models_classification(
            args,
            barplot_datasets,
            calcul_type=ct,
        )
    else:
        print("  No model cache found. Run each model individually first.")

    print("\nDone.")
