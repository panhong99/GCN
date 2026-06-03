import numpy as np
import os
import pickle
import argparse
from GCN.PIGNet.IB_family.seg.JE_calcul_seg import compute_and_cache_je, compute_kde_values
from GCN.PIGNet.IB_family.seg.JE_figure_seg import (
    plot_scatter_matrix,
    plot_kde_matrix,
    plot_ratio_barplot_all_models,
)

BASE_DATA_ROOT = '/home/hail/pan/HDD/IB_dataset'

# ── path builder ──────────────────────────────────────────────────────────────
def build_data_path(args, dataset_name=None, model_name=None):
    ds    = dataset_name or args.dataset
    model = model_name   or args.model
    return os.path.join(
        BASE_DATA_ROOT,
        ds,
        "resnet"+args.backbone,
        args.model_type,
        model,
        'zoom', '1',
    )

# ── raw data loader ───────────────────────────────────────────────────────────
def load_raw_data(seg_path, layer_num=4):
    """VQ pkl 파일 로드. Returns x_in, t_in, y_in."""
    def _pkl(p):
        with open(p, 'rb') as f:
            return pickle.load(f)

    y_in = _pkl(os.path.join(seg_path, 'gt_labels.pkl'))
    x_in = _pkl(os.path.join(seg_path, 'layer_0.pkl'))
    t_in = [_pkl(os.path.join(seg_path, f'layer_{i}.pkl'))
            for i in range(1, layer_num + 1)]
    return x_in, t_in, y_in

# ── JE ────────────────────────────────────────────────────────────────────────
def load_or_calcul_je(seg_path, args):
    """Cache 있으면 로드, 없으면 JE 계산 후 저장. Returns je_cache dict."""
    cache_file = os.path.join(seg_path, f'analysis_cache_same_diff_{args.calcul_type}.pkl')

    if os.path.exists(cache_file):
        print(f"  Loading JE cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("  JE cache not found — loading VQ data and computing...")
    x_in, t_in, y_in = load_raw_data(seg_path)
    print(f"  x={x_in.shape}, y={y_in.shape}, t={len(t_in)} layers")
    compute_and_cache_je(seg_path, x_in, t_in, y_in,
                         ignore_label=-1, calcul_type=args.calcul_type)
    with open(cache_file, 'rb') as f:
        return pickle.load(f)

# ── KDE ───────────────────────────────────────────────────────────────────────
def load_or_calcul_kde(seg_path, je_cache):
    """Cache 있으면 로드, 없으면 KDE 계산 후 저장. Returns kde_data dict."""
    cache_file = os.path.join(seg_path, 'kde_cache_contour.pkl')

    if os.path.exists(cache_file):
        print(f"  Loading KDE cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("  Computing KDE values...")
    kde_data = compute_kde_values(
        je_cache['je_xt_same'], je_cache['je_ty_same'],
        je_cache['je_xt_diff'], je_cache['je_ty_diff'],
        je_cache['distance'],
    )
    with open(cache_file, 'wb') as f:
        pickle.dump(kde_data, f)
    print(f"  KDE cache saved: {cache_file}\n")
    return kde_data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',         type=str, default='pascal',
                        help='pascal or cityscape')
    parser.add_argument('--model',           type=str, default='PIGNet_GSPonly',
                        help='PIGNet_GSPonly | ASPP | Mask2Former')
    parser.add_argument('--models',          type=str,
                        default='ASPP,PIGNet_GSPonly,Mask2Former',
                        help='Comma-separated model list for all-model barplot')
    parser.add_argument('--backbone',        type=str, default='101', help='50 or 101')
    parser.add_argument('--model_type',      type=str, default='scratch',
                        help='scratch or pretrained')
    parser.add_argument('--preprocess_type', type=str, default='pixel',
                        help='pixel or layer')
    parser.add_argument('--calcul_type',     type=str, default='joint',
                        help='joint or MI')
    parser.add_argument('--valid_pascal',    action='store_true', default=True)
    parser.add_argument('--vmin',            type=int, default=0)
    parser.add_argument('--vmax',            type=int, default=25)
    parser.add_argument('--all_models',      action='store_true', default=False,
                        help='Also draw all-model ratio barplot')
    args = parser.parse_args()

    seg_path = build_data_path(args)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. JE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n=== JE ===")
    je_cache = load_or_calcul_je(seg_path, args)

    distance   = je_cache['distance']
    je_xt_same = je_cache['je_xt_same']
    je_ty_same = je_cache['je_ty_same']
    je_xt_diff = je_cache['je_xt_diff']
    je_ty_diff = je_cache['je_ty_diff']

    # # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # # 2. Scatter Plot
    # # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # print("\n=== Scatter Plot ===")
    # for mode, je_xt, je_ty in [('SAME', je_xt_same, je_ty_same),
    #                             ('DIFF', je_xt_diff, je_ty_diff)]:
    #     plot_scatter_matrix(mode, je_xt, je_ty, distance,
    #                         args.model, args.dataset,
    #                         args.valid_pascal, args.calcul_type)

    # # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # # 3. KDE
    # # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # print("\n=== KDE ===")
    # kde_data = load_or_calcul_kde(seg_path, je_cache)

    # # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # # 4. KDE Matrix Plot
    # # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # print("\n=== KDE Matrix Plot ===")
    # for mode in ('SAME', 'DIFF'):
    #     plot_kde_matrix(mode, args.model, args.dataset,
    #                     args.vmin, args.vmax, kde_data,
    #                     args.valid_pascal, args.calcul_type)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. All-Models Barplot  (--all_models)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if not args.all_models:
        print("\nDone.")
        exit(0)

    print("\n=== All-Models Barplot ===")
    model_list = [m.strip() for m in args.models.split(',')]

    def load_models_data(dataset_name):
        data = {}
        for model_name in model_list:
            path  = build_data_path(args, dataset_name=dataset_name, model_name=model_name)
            cache = os.path.join(path, f'analysis_cache_same_diff_{args.calcul_type}.pkl')
            if not os.path.exists(cache):
                print(f"  [{dataset_name}/{model_name}] cache not found — skipped")
                continue
            print(f"  Loading [{dataset_name}/{model_name}]...")
            with open(cache, 'rb') as f:
                d = pickle.load(f)
            data[model_name] = {k: d[k] for k in
                                ('je_xt_same', 'je_ty_same', 'je_xt_diff', 'je_ty_diff', 'distance')}
        return data

    data_pascal    = load_models_data('pascal')
    data_cityscape = load_models_data('cityscape')

    if data_pascal and data_cityscape:
        plot_ratio_barplot_all_models(args, data_pascal, data_cityscape, args.calcul_type)
    else:
        print("  Not enough model data for barplot. Run each model first.")

    print("\nDone.")
