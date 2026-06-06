#!/usr/bin/env python3
"""
collect_img.py - Run inference for all 3 architectures and generate segmentation
result matrices (zoom / overlap / repeat) for a given backbone × dataset × model_type.

Usage:
    python collect_img.py --backbone resnet50 --dataset pascal --model_type scratch

Output layout:
    seg_result_imgs/{arch}/{R50|R101}/{dataset}/{model_type}/{process}/
    seg_result_imgs/matrices/{R50|R101}/{dataset}/{model_type}/{process}_matrix.png
"""

import os, sys, copy, warnings, argparse, yaml
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat
from torch.autograd import Variable

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_dataset import get_dataset
from seg_models   import get_model

# ── Constants ──────────────────────────────────────────────────────────────────

ROOT           = '/home/hail/pan/GCN/PIGNet'
SAVE_BASE      = f'{ROOT}/seg_result_imgs'
APPENDIX_BASE  = f'{ROOT}/appendix'
CFG_PATH       = f'{ROOT}/config_segmentation.yaml'

ARCH_NAMES = ['PIGNet_GSPonly', 'ASPP', 'Mask2Former']

# Augmentation factors (raw values used by dataset __getitem__)
# zoom split at 1.0: zoom_small = [0.1~0.7], zoom_large = [1.0~2.0]
ZOOM_SMALL_FACTORS = [0.1, np.sqrt(0.1), 0.5, np.sqrt(0.5)]
ZOOM_SMALL_LABELS  = ['0.1', '0.3', '0.5', '0.7']
ZOOM_LARGE_FACTORS = [1, 1.5, np.sqrt(2.75), 2]
ZOOM_LARGE_LABELS  = ['1.0', '1.5', '1.75', '2.0']
OVERLAP_FACTORS    = [0, 0.1, 0.2, 0.3, 0.5]
REPEAT_FACTORS     = [1, 3, 6, 9, 12]

PROCESS_CFG = {
    'zoom_small': {'factors': ZOOM_SMALL_FACTORS, 'labels': ZOOM_SMALL_LABELS},
    'zoom_large': {'factors': ZOOM_LARGE_FACTORS, 'labels': ZOOM_LARGE_LABELS},
    'overlap':    {'factors': OVERLAP_FACTORS,    'labels': [str(x) for x in OVERLAP_FACTORS]},
    'repeat':     {'factors': REPEAT_FACTORS,     'labels': [str(x) for x in REPEAT_FACTORS]},
}

# Representative images per process type (one image per factor value)
PASCAL_COLLECT = {
    'zoom_small': ['2011_000900.png', '2008_005105.png', '2007_000042.png',
                   '2007_000332.png'],
    'zoom_large': ['2007_000033.png', '2007_005608.png',
                   '2007_000783.png', '2007_002643.png'],
    'overlap':    ['2007_003143.png', '2011_003003.png', '2008_002152.png',
                   '2011_001407.png', '2008_000602.png'],
    'repeat':     ['2007_000332.png', '2007_000676.png', '2007_003051.png',
                   '2007_003714.png', '2007_000061.png'],
}

_F, _M = 'frankfurt', 'munster'
CITY_COLLECT = {
    'zoom_small': [
        f'{_F}_000001_003588_gtFine_labelTrainIds.png',
        f'{_M}_000173_000019_gtFine_labelTrainIds.png',
        f'{_F}_000001_010600_gtFine_labelTrainIds.png',
        f'{_M}_000132_000019_gtFine_labelTrainIds.png',
    ],
    'zoom_large': [
        f'{_F}_000000_000294_gtFine_labelTrainIds.png',
        f'{_F}_000000_005543_gtFine_labelTrainIds.png',
        f'{_M}_000127_000019_gtFine_labelTrainIds.png',
        f'{_F}_000001_002759_gtFine_labelTrainIds.png',
    ],
    'overlap': [
        f'{_F}_000000_001751_gtFine_labelTrainIds.png',
        f'{_F}_000000_020215_gtFine_labelTrainIds.png',
        f'{_M}_000171_000019_gtFine_labelTrainIds.png',
        f'{_M}_000158_000019_gtFine_labelTrainIds.png',
        f'{_F}_000001_005898_gtFine_labelTrainIds.png',
    ],
    'repeat': [
        f'{_F}_000000_002196_gtFine_labelTrainIds.png',
        f'{_F}_000000_000294_gtFine_labelTrainIds.png',
        f'{_F}_000000_013382_gtFine_labelTrainIds.png',
        f'{_F}_000001_005410_gtFine_labelTrainIds.png',
        f'{_M}_000119_000019_gtFine_labelTrainIds.png',
    ],
}

# Matrix display dimensions
IMG_W, IMG_H = 513, 256
GAP      = 16
LABEL_W  = 160
LABEL_H  = 70
FONT_SZ  = 60

# ── Helpers ────────────────────────────────────────────────────────────────────

def dict_to_ns(d):
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def get_colormap(dataset_name):
    if dataset_name == 'pascal':
        mat = loadmat(f'{ROOT}/data/pascal_seg_colormap.mat')['colormap']
        return (mat * 255).astype(np.uint8).flatten().tolist()
    pal = np.zeros((256, 3), dtype=np.uint8)
    city_cm = {
        0: (128,64,128), 1: (244,35,232), 2: (70,70,70), 3: (102,102,156),
        4: (190,153,153), 5: (153,153,153), 6: (250,170,30), 7: (220,220,0),
        8: (107,142,35), 9: (152,251,152), 10: (70,130,180), 11: (220,20,60),
        12: (255,0,0), 13: (0,0,142), 14: (0,0,70), 15: (0,60,100),
        16: (0,80,100), 17: (0,0,230), 18: (119,11,32), 255: (0,0,0),
    }
    for tid, color in city_cm.items():
        if tid < 256:
            pal[tid] = color
    return pal.flatten().tolist()


def colorize(pred, cmap_flat):
    img_p = Image.fromarray(pred.astype(np.uint8), mode='P')
    img_p.putpalette(cmap_flat)
    return img_p.convert('RGB')


def find_ckpt(backbone, dataset_name, model_type, arch_name):
    """
    Locate the .pth file by searching all model_num subfolders (1, 2, 3).

    Each subfolder typically holds one architecture:
      model_{50|101}/1/ → PIGNet_GSPonly
      model_{50|101}/2/ → ASPP
      model_{50|101}/3/ → Mask2Former

    Searches within each:
      model_{num}/{model_num}/segmentation/{dataset}/
      model_{num}/{model_num}/segmentation/{dataset}/{model_type}/
    """
    num = 50 if backbone == 'resnet50' else 101
    ok_types = {'pretrained': ('pretrained', 'pretrain'), 'scratch': ('scratch',)}[model_type]

    for model_num in [1, 2, 3]:
        base = f'{ROOT}/model_{num}/{model_num}/segmentation/{dataset_name}'
        if not os.path.isdir(base):
            continue
        for search_dir in [base, os.path.join(base, model_type)]:
            if not os.path.isdir(search_dir):
                continue
            for fname in sorted(os.listdir(search_dir)):
                if not fname.endswith('.pth'):
                    continue
                if not fname.startswith(arch_name):
                    continue
                if backbone not in fname:
                    continue
                if dataset_name not in fname:
                    continue
                if any(t in fname for t in ok_types):
                    return os.path.join(search_dir, fname)

    raise FileNotFoundError(
        f'No .pth for arch={arch_name} backbone={backbone} '
        f'dataset={dataset_name} model_type={model_type} '
        f'(searched model_{num}/1..3/segmentation/{dataset_name}/)'
    )


def load_ckpt(model, path, device):
    """Load checkpoint, transparently stripping 'module.' prefix if needed."""
    ckpt = torch.load(path, map_location=device)
    raw  = {k: v for k, v in ckpt['state_dict'].items() if 'tracked' not in k}
    for strip in (True, False):
        try:
            sd = {k[7:] if strip else k: v for k, v in raw.items()}
            model.load_state_dict(sd)
            return
        except RuntimeError:
            continue
    raise RuntimeError(f'Failed to load state_dict from {path}')


def make_dataset(base_cfg, process_type, factor, crop_size):
    """Create an augmented inference dataset."""
    cfg = copy.deepcopy(base_cfg)
    cfg.crop_size = crop_size
    cfg.factor    = factor
    # zoom_small / zoom_large are collect_img split keys; the dataset only knows 'zoom'
    dataset_process = 'zoom' if process_type in ('zoom_small', 'zoom_large') else process_type
    cfg.infer_params.process_type = dataset_process
    return get_dataset(cfg)


def find_index(dataset_obj, img_name, dataset_name):
    """Return the index of img_name in dataset_obj, or None if not found."""
    if dataset_name == 'cityscape':
        stem = img_name.replace('_gtFine_labelTrainIds.png', '')
        for i, p in enumerate(dataset_obj.masks):
            if stem in p:
                return i
    else:
        stem = os.path.splitext(img_name)[0]
        for i, p in enumerate(dataset_obj.images):
            if stem in p:
                return i
    return None


def load_orig(img_name, dataset_name):
    if dataset_name == 'cityscape':
        stem = img_name.replace('_gtFine_labelTrainIds.png', '')
        city = stem.split('_')[0]
        path = f'{ROOT}/data/cityscape/leftImg8bit/val/{city}/{stem}_leftImg8bit.png'
    else:
        stem = os.path.splitext(img_name)[0]
        path = f'{ROOT}/data/VOCdevkit/VOC2012/JPEGImages/{stem}.jpg'
    return Image.open(path).convert('RGB') if os.path.exists(path) else None


def load_gt(img_name, dataset_name):
    if dataset_name == 'cityscape':
        stem = img_name.replace('_gtFine_labelTrainIds.png', '')
        city = stem.split('_')[0]
        path = f'{ROOT}/data/cityscape/gtFine/val/{city}/{stem}_gtFine_color.png'
    else:
        stem = os.path.splitext(img_name)[0]
        path = f'{ROOT}/data/VOCdevkit/VOC2012/SegmentationClass/{stem}.png'
    return Image.open(path).convert('RGB') if os.path.exists(path) else None


# ── Appendix helpers ────────────────────────────────────────────────────────────

def appendix_stem(process_type, label, img_name):
    """Return filename stem used for appendix images: {process}_{label}_{img_stem}."""
    return f'{process_type}_{label}_{os.path.splitext(img_name)[0]}'


def save_appendix(img, path):
    if img is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.resize((IMG_W, IMG_H), Image.Resampling.LANCZOS).save(path)


def generate_latex_table(process_type, labels, target_imgs, arch_names,
                         dataset, backbone_label, model_type):
    """Return a LaTeX tabular string for one process type."""
    rel = f'appendix/{dataset}/{backbone_label}/{model_type}'
    disp = {'PIGNet_GSPonly': 'PIGNet', 'ASPP': 'ASPP', 'Mask2Former': 'Mask2Former'}
    arch_hdrs = ' & '.join(disp.get(a, a) for a in arch_names)
    col_spec = 'c ' * (3 + len(arch_names))

    lines = [
        f'\\begin{{tabular}}{{{col_spec.rstrip()}}}',
        f'    & Image & GT & {arch_hdrs} \\\\',
        r'    \noalign{\smallskip}',
        '',
    ]
    for i, (label, img_name) in enumerate(zip(labels, target_imgs)):
        stem = appendix_stem(process_type, label, img_name)
        row_end = r'    \\' if i == len(labels) - 1 else r'    \\[0.9cm]'
        arch_cells = ' &\n    '.join(
            f'\\myimg{{{rel}/pred_seg/{a}/{stem}}}' for a in arch_names
        )
        lines += [
            f'    {label} &',
            f'    \\myimg{{{rel}/GT/{stem}}} &',
            f'    \\myimg{{{rel}/GT_seg/{stem}}} &',
            f'    {arch_cells}',
            row_end,
            '',
        ]
    lines.append(r'\end{tabular}')
    return '\n'.join(lines)


# ── Matrix builder ─────────────────────────────────────────────────────────────

def build_matrix(img_names, aug_orig_list, aug_gt_list, model_preds, labels, arch_names):
    """
    img_names    : list[str]        — one image per row
    aug_orig_list: list[PIL | None] — augmented input image per row
    aug_gt_list  : list[PIL | None] — augmented GT color mask per row
    model_preds  : {arch: [PIL | None]}  — one entry per row
    labels       : list[str]        — row label (factor value)
    """
    col_headers = ['Image', 'GT'] + arch_names
    n_cols = len(col_headers)
    n_rows = len(img_names)
    header_h = LABEL_H + GAP
    total_w  = LABEL_W + n_cols * IMG_W + (n_cols + 1) * GAP
    total_h  = header_h + n_rows * IMG_H + (n_rows + 1) * GAP

    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    try:
        font  = ImageFont.truetype(
            '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf', FONT_SZ)
        sfont = ImageFont.truetype(
            '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf', FONT_SZ - 2)
    except Exception:
        font = sfont = ImageFont.load_default()

    # Column headers
    for c, hdr in enumerate(col_headers):
        x = LABEL_W + GAP + c * (IMG_W + GAP) + IMG_W // 2
        draw.text((x, GAP // 2), hdr, fill=(0, 0, 0), font=font, anchor='mt')

    # Rows
    for r, (label, img_name) in enumerate(zip(labels, img_names)):
        y = header_h + r * (IMG_H + GAP) + GAP
        draw.text((LABEL_W // 2, y + IMG_H // 2), label,
                  fill=(0, 0, 0), font=sfont, anchor='mm')

        for c, hdr in enumerate(col_headers):
            x = LABEL_W + GAP + c * (IMG_W + GAP)
            if hdr == 'Image':
                cell = aug_orig_list[r] if r < len(aug_orig_list) else None
            elif hdr == 'GT':
                cell = aug_gt_list[r] if r < len(aug_gt_list) else None
            else:
                preds = model_preds.get(hdr, [])
                cell  = preds[r] if r < len(preds) else None

            if cell is not None:
                canvas.paste(cell.resize((IMG_W, IMG_H), Image.Resampling.LANCZOS), (x, y))
            else:
                draw.rectangle([x, y, x + IMG_W, y + IMG_H],
                                outline=(200, 200, 200), fill=(240, 240, 240))
                draw.text((x + IMG_W // 2, y + IMG_H // 2), 'N/A',
                           fill=(150, 150, 150), font=sfont, anchor='mm')
    return canvas


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate segmentation result matrices')
    parser.add_argument('--backbone',   required=True, choices=['resnet50', 'resnet101'])
    parser.add_argument('--dataset',    required=True, choices=['pascal', 'cityscape'])
    parser.add_argument('--model_type', required=True, choices=['scratch', 'pretrained'])
    args = parser.parse_args()

    device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone_label = 'R50' if args.backbone == 'resnet50' else 'R101'
    collect_dict   = PASCAL_COLLECT if args.dataset == 'pascal' else CITY_COLLECT
    cmap           = get_colormap(args.dataset)

    # Base config from yaml
    with open(CFG_PATH) as f:
        base_cfg = dict_to_ns(yaml.safe_load(f))
    base_cfg.backbone   = args.backbone
    base_cfg.dataset    = args.dataset
    base_cfg.model_type = args.model_type
    base_cfg.mode       = 'infer'
    base_cfg.train      = False
    base_cfg.MI         = False
    base_cfg.scratch    = (args.model_type == 'scratch')

    # ── Step 1: Load all 3 models ──────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'backbone={args.backbone} | dataset={args.dataset} | model_type={args.model_type}')
    print(f'{"="*60}')
    print('\n[Step 1] Loading models...')

    # Dummy dataset (no process) to get num_classes for model init
    init_cfg = copy.deepcopy(base_cfg)
    init_cfg.crop_size = 513
    init_cfg.factor    = 1
    init_cfg.infer_params.process_type = None
    base_dataset = get_dataset(init_cfg)

    loaded_models = {}  # arch_name → nn.Module
    for arch_name in ARCH_NAMES:
        try:
            ckpt_path = find_ckpt(args.backbone, args.dataset, args.model_type, arch_name)
        except FileNotFoundError as e:
            print(f'  [{arch_name}] SKIP — {e}')
            continue

        m_cfg = copy.deepcopy(base_cfg)
        m_cfg.model     = arch_name
        m_cfg.crop_size = 512 if arch_name == 'Mask2Former' else 513

        model = get_model(m_cfg, base_dataset)
        model.to(device).eval()
        load_ckpt(model, ckpt_path, device)
        loaded_models[arch_name] = model
        print(f'  [{arch_name}] {ckpt_path}')

    # ── Step 2 & 3: Inference per augmentation ─────────────────────────────────
    print('\n[Step 2] Running inference...')
    arch_names = list(loaded_models.keys())  # only models that were successfully loaded
    # all_preds[process_type][arch_name] = list[PIL | None]  (one per factor)
    all_preds     = {pt: {a: [] for a in arch_names} for pt in PROCESS_CFG}
    # aug_orig/gt: augmented Image and GT PIL per (process_type, factor_idx)
    aug_orig_imgs = {pt: [] for pt in PROCESS_CFG}
    aug_gt_imgs   = {pt: [] for pt in PROCESS_CFG}
    orig_imgs = {}  # fallback: raw disk images keyed by img_name
    gt_imgs   = {}

    total_runs = sum(len(v['factors']) for v in PROCESS_CFG.values()) * len(arch_names)
    run_no = 0

    for process_type, pcfg in PROCESS_CFG.items():
        factors     = pcfg['factors']
        labels      = pcfg['labels']
        target_imgs = collect_dict[process_type]

        print(f'\n  -- {process_type} ({len(factors)} factors) --')

        for factor_idx, (factor_val, label, img_name) in enumerate(
                zip(factors, labels, target_imgs)):

            # Fallback: raw images loaded from disk once per unique image
            if img_name not in orig_imgs:
                o = load_orig(img_name, args.dataset)
                g = load_gt(img_name, args.dataset)
                if o: orig_imgs[img_name] = o
                if g: gt_imgs[img_name]   = g

            app_base = os.path.join(
                APPENDIX_BASE, args.dataset, backbone_label, args.model_type)
            stem = appendix_stem(process_type, label, img_name)

            # Two datasets: crop_size=513 (PIGNet/ASPP) and 512 (Mask2Former)
            ds_513 = make_dataset(base_cfg, process_type, factor_val, crop_size=513)
            ds_512 = make_dataset(base_cfg, process_type, factor_val, crop_size=512)

            idx_513 = find_index(ds_513, img_name, args.dataset)
            idx_512 = find_index(ds_512, img_name, args.dataset)

            # Augmented Image / GT captured from first non-Mask2Former sample
            row_aug_img = None
            row_aug_gt  = None

            for arch_name, model in loaded_models.items():
                run_no += 1
                ds  = ds_512 if arch_name == 'Mask2Former' else ds_513
                idx = idx_512 if arch_name == 'Mask2Former' else idx_513

                print(f'  [{run_no}/{total_runs}] {arch_name} | {process_type}={label} | {img_name}')

                if idx is None:
                    print(f'    WARNING: image not found in dataset, skipping')
                    all_preds[process_type][arch_name].append(None)
                    continue

                sample = ds[idx]
                if sample[0] is None:
                    print(f'    WARNING: augmentation returned None, skipping')
                    all_preds[process_type][arch_name].append(None)
                    continue

                # Capture augmented Image & GT from first non-Mask2Former success
                if row_aug_img is None and arch_name != 'Mask2Former':
                    try:
                        pil_img = sample[2]  # unnorm PIL image
                        pil_gt  = sample[3]  # color_target PIL image
                        if pil_img is not None:
                            row_aug_img = pil_img.convert('RGB').resize(
                                (IMG_W, IMG_H), Image.Resampling.LANCZOS)
                        if pil_gt is not None:
                            row_aug_gt = pil_gt.convert('RGB').resize(
                                (IMG_W, IMG_H), Image.Resampling.LANCZOS)
                    except Exception:
                        pass

                inp = Variable(sample[0].to(device))
                with torch.no_grad():
                    if arch_name == 'Mask2Former':
                        out = model(inp.unsqueeze(0))
                        if isinstance(out, tuple):
                            out = out[0]
                    else:
                        out, _ = model(inp.unsqueeze(0))

                _, pred = torch.max(out, 1)
                pred = pred.cpu().numpy().squeeze().astype(np.uint8)

                if args.dataset == 'cityscape':
                    sample_mask = sample[1].numpy().astype(np.uint8)
                    pred[sample_mask == 255] = 255

                pred_img = colorize(pred, cmap).convert('RGB').crop((0, 0, IMG_W, IMG_H))

                # Save individual mask (seg_result_imgs)
                save_dir = os.path.join(
                    SAVE_BASE, arch_name, backbone_label,
                    args.dataset, args.model_type, process_type)
                os.makedirs(save_dir, exist_ok=True)
                pred_img.save(os.path.join(save_dir, f'{label}_{img_name}'))

                # Save to appendix/pred_seg/{arch}/
                app_pred_dir = os.path.join(
                    APPENDIX_BASE, args.dataset, backbone_label,
                    args.model_type, 'pred_seg', arch_name)
                os.makedirs(app_pred_dir, exist_ok=True)
                pred_img.save(os.path.join(app_pred_dir, f'{stem}.png'))

                all_preds[process_type][arch_name].append(pred_img)

            # Fallback to disk image if augmented capture failed
            aug_img = row_aug_img or orig_imgs.get(img_name)
            aug_gt  = row_aug_gt  or gt_imgs.get(img_name)
            aug_orig_imgs[process_type].append(aug_img)
            aug_gt_imgs[process_type].append(aug_gt)

            # Save augmented GT to appendix
            save_appendix(aug_img, os.path.join(app_base, 'GT',     f'{stem}.png'))
            save_appendix(aug_gt,  os.path.join(app_base, 'GT_seg', f'{stem}.png'))

    # ── Step 4: Build and save matrices ────────────────────────────────────────
    print('\n[Step 3] Building matrices...')
    matrix_dir = os.path.join(SAVE_BASE, 'matrices', backbone_label,
                              args.model_type, args.dataset)
    os.makedirs(matrix_dir, exist_ok=True)

    for process_type, pcfg in PROCESS_CFG.items():
        labels      = pcfg['labels']
        target_imgs = collect_dict[process_type]
        mat_preds   = {a: all_preds[process_type][a] for a in arch_names}

        matrix   = build_matrix(target_imgs,
                                aug_orig_imgs[process_type],
                                aug_gt_imgs[process_type],
                                mat_preds, labels, arch_names)
        out_path = os.path.join(matrix_dir, f'{process_type}_matrix.png')
        matrix.save(out_path, dpi=(150, 150))
        print(f'  Saved: {out_path}')

    # ── Step 5: Generate LaTeX tables ──────────────────────────────────────────
    print('\n[Step 4] Generating LaTeX tables...')
    latex_dir = os.path.join(
        APPENDIX_BASE, args.dataset, backbone_label, args.model_type)
    os.makedirs(latex_dir, exist_ok=True)

    for process_type, pcfg in PROCESS_CFG.items():
        tex = generate_latex_table(
            process_type,
            pcfg['labels'],
            collect_dict[process_type],
            arch_names,
            args.dataset,
            backbone_label,
            args.model_type,
        )
        tex_path = os.path.join(latex_dir, f'{process_type}_table.tex')
        with open(tex_path, 'w') as f:
            f.write(tex)
        print(f'  Saved: {tex_path}')

    print(f'\nDone! Results in {SAVE_BASE} and {APPENDIX_BASE}')


if __name__ == '__main__':
    main()
