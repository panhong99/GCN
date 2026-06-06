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

ROOT      = '/home/hail/pan/GCN/PIGNet'
SAVE_BASE = f'{ROOT}/seg_result_imgs'
CFG_PATH  = f'{ROOT}/config_segmentation.yaml'

ARCH_NAMES = ['PIGNet_GSPonly', 'ASPP', 'Mask2Former']

# Augmentation factors (raw values used by dataset __getitem__)
ZOOM_FACTORS   = [0.1, np.sqrt(0.1), 0.5, np.sqrt(0.5), 1, 1.5, np.sqrt(2.75), 2]
ZOOM_LABELS    = ['0.1', '0.3', '0.5', '0.7', '1.0', '1.5', '1.75', '2.0']
OVERLAP_FACTORS = [0, 0.1, 0.2, 0.3, 0.5]
REPEAT_FACTORS  = [1, 3, 6, 9, 12]

PROCESS_CFG = {
    'zoom':    {'factors': ZOOM_FACTORS,    'labels': ZOOM_LABELS},
    'overlap': {'factors': OVERLAP_FACTORS, 'labels': [str(x) for x in OVERLAP_FACTORS]},
    'repeat':  {'factors': REPEAT_FACTORS,  'labels': [str(x) for x in REPEAT_FACTORS]},
}

# Representative images per process type (one image per factor value)
PASCAL_COLLECT = {
    'zoom':    ['2011_000900.png', '2008_005105.png', '2007_000042.png',
                '2007_000332.png', '2007_000033.png', '2007_005608.png',
                '2007_000783.png', '2007_002643.png'],
    'overlap': ['2007_003143.png', '2011_003003.png', '2008_002152.png',
                '2011_001407.png', '2008_000602.png'],
    'repeat':  ['2007_000332.png', '2007_000676.png', '2007_003051.png',
                '2007_003714.png', '2007_000061.png'],
}

_F, _M = 'frankfurt', 'munster'
CITY_COLLECT = {
    'zoom': [
        f'{_F}_000001_003588_gtFine_labelTrainIds.png',
        f'{_M}_000173_000019_gtFine_labelTrainIds.png',
        f'{_F}_000001_010600_gtFine_labelTrainIds.png',
        f'{_M}_000132_000019_gtFine_labelTrainIds.png',
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
GAP      = 8
LABEL_W  = 50
LABEL_H  = 20
FONT_SZ  = 14

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


def find_ckpt(backbone, dataset_name, model_type, arch_name, model_num):
    """
    Locate the .pth file for the given combination.

    Searches two locations (some architectures are in the dataset-level dir,
    others are stored in a pretrained/ or scratch/ subdir):
      1. model_{num}/{model_num}/segmentation/{dataset}/
      2. model_{num}/{model_num}/segmentation/{dataset}/{model_type}/
    """
    num  = 50 if backbone == 'resnet50' else 101
    base = f'{ROOT}/model_{num}/{model_num}/segmentation/{dataset_name}'
    if not os.path.isdir(base):
        raise FileNotFoundError(f'Checkpoint dir not found: {base}')

    # File names use both 'pretrained'/'pretrain' for the pretrained type
    ok_types = {'pretrained': ('pretrained', 'pretrain'), 'scratch': ('scratch',)}[model_type]

    # Search base dir first, then the model_type subdir
    search_dirs = [base, os.path.join(base, model_type)]

    candidates = []
    for search_dir in search_dirs:
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
                candidates.append(os.path.join(search_dir, fname))

    if not candidates:
        raise FileNotFoundError(
            f'No .pth for arch={arch_name} backbone={backbone} '
            f'dataset={dataset_name} model_type={model_type} '
            f'in {base} or {base}/{model_type}')
    # Prefer the shortest (simplest) path when multiple matches exist
    return sorted(candidates, key=len)[0]


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
    cfg.infer_params.process_type = process_type
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


# ── Matrix builder ─────────────────────────────────────────────────────────────

def build_matrix(img_names, orig_imgs, gt_imgs, model_preds, labels, arch_names):
    """
    img_names   : list[str]   — one image per row
    model_preds : {arch: [PIL | None]}  — one entry per row
    labels      : list[str]   — row label (factor value)
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
                cell = orig_imgs.get(img_name)
            elif hdr == 'GT':
                cell = gt_imgs.get(img_name)
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
    parser.add_argument('--model_num',  required=True, type=int, choices=[1, 2, 3],
                        help='Statistical run index (1/2/3). Each folder contains all 3 arch checkpoints.')
    args = parser.parse_args()

    device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone_label = 'R50' if args.backbone == 'resnet50' else 'R101'
    collect_dict   = PASCAL_COLLECT if args.dataset == 'pascal' else CITY_COLLECT
    cmap           = get_colormap(args.dataset)
    model_num      = args.model_num

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
    print(f'backbone={args.backbone} | model_num={model_num} | '
          f'dataset={args.dataset} | model_type={args.model_type}')
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
            ckpt_path = find_ckpt(args.backbone, args.dataset, args.model_type,
                                  arch_name, model_num)
        except FileNotFoundError as e:
            print(f'  [{arch_name}] SKIP — {e}')
            continue

        m_cfg = copy.deepcopy(base_cfg)
        m_cfg.model        = arch_name
        m_cfg.model_number = model_num
        m_cfg.crop_size    = 512 if arch_name == 'Mask2Former' else 513

        model = get_model(m_cfg, base_dataset)
        model.to(device).eval()
        load_ckpt(model, ckpt_path, device)
        loaded_models[arch_name] = model
        print(f'  [{arch_name}] {ckpt_path}')

    # ── Step 2 & 3: Inference per augmentation ─────────────────────────────────
    print('\n[Step 2] Running inference...')
    arch_names = list(loaded_models.keys())  # only models that were successfully loaded
    # all_preds[process_type][arch_name] = list[PIL | None]  (one per factor)
    all_preds = {pt: {a: [] for a in arch_names} for pt in PROCESS_CFG}
    orig_imgs = {}
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

            # Original and GT (loaded once per unique image)
            if img_name not in orig_imgs:
                o = load_orig(img_name, args.dataset)
                g = load_gt(img_name, args.dataset)
                if o: orig_imgs[img_name] = o
                if g: gt_imgs[img_name]   = g

            # Two datasets: crop_size=513 (PIGNet/ASPP) and 512 (Mask2Former)
            ds_513 = make_dataset(base_cfg, process_type, factor_val, crop_size=513)
            ds_512 = make_dataset(base_cfg, process_type, factor_val, crop_size=512)

            idx_513 = find_index(ds_513, img_name, args.dataset)
            idx_512 = find_index(ds_512, img_name, args.dataset)

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
                pred_img = colorize(pred, cmap).resize((IMG_W, IMG_H), Image.Resampling.LANCZOS)

                # Save individual mask
                save_dir = os.path.join(
                    SAVE_BASE, arch_name, backbone_label,
                    args.dataset, args.model_type, process_type)
                os.makedirs(save_dir, exist_ok=True)
                pred_img.save(os.path.join(save_dir, f'{label}_{img_name}'))

                all_preds[process_type][arch_name].append(pred_img)

    # ── Step 4: Build and save matrices ────────────────────────────────────────
    print('\n[Step 3] Building matrices...')
    matrix_dir = os.path.join(SAVE_BASE, 'matrices', backbone_label,
                              str(model_num), args.model_type, args.dataset)
    os.makedirs(matrix_dir, exist_ok=True)

    for process_type, pcfg in PROCESS_CFG.items():
        labels      = pcfg['labels']
        target_imgs = collect_dict[process_type]
        mat_preds   = {a: all_preds[process_type][a] for a in arch_names}

        matrix   = build_matrix(target_imgs, orig_imgs, gt_imgs, mat_preds, labels, arch_names)
        out_path = os.path.join(matrix_dir, f'{process_type}_matrix.png')
        matrix.save(out_path, dpi=(150, 150))
        print(f'  Saved: {out_path}')

    print(f'\nDone! Results in {SAVE_BASE}')


if __name__ == '__main__':
    main()
