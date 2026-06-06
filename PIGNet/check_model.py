#!/usr/bin/env python3
"""
check_model.py - Run inference with every checkpoint version and save individual images.

Scans model_{num}/ directories for .pth files. For each checkpoint a
separate output folder is created:

    eval_imgs/check/{arch}/{R50|R101}/{dataset}/{model_type}/{sub_dir}__{version}/
        {img_stem}.png          ← prediction
        {img_stem}_orig.png     ← original image
        {img_stem}_gt.png       ← ground-truth

A legend.txt at the arch level maps every folder name to its full .pth path.

Usage:
    python check_model.py --backbone resnet101 --dataset pascal --model_type pretrained
    python check_model.py --backbone resnet50  --dataset cityscape --model_type scratch
    # limit to one experiment folder:
    python check_model.py --backbone resnet101 --dataset pascal --model_type pretrained --exp_num 1
"""

import os, sys, copy, warnings, argparse, yaml, glob, re
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_dataset import get_dataset
from seg_models   import get_model

ROOT      = '/home/hail/pan/GCN/PIGNet'
SAVE_BASE = f'{ROOT}/eval_imgs/check'
CFG_PATH  = f'{ROOT}/config_segmentation.yaml'

# ── Representative images (zoom=1, no augmentation) ───────────────────────────
PASCAL_EVAL = [
    '2007_000033.png',
    '2007_000332.png',
    '2007_003143.png',
    '2007_000676.png',
    '2008_005105.png',
]

_F, _M = 'frankfurt', 'munster'
CITY_EVAL = [
    f'{_F}_000000_000294_gtFine_labelTrainIds.png',
    f'{_F}_000001_003588_gtFine_labelTrainIds.png',
    f'{_F}_000000_001751_gtFine_labelTrainIds.png',
    f'{_F}_000000_002196_gtFine_labelTrainIds.png',
    f'{_M}_000173_000019_gtFine_labelTrainIds.png',
]

IMG_W, IMG_H = 513, 256

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


def load_ckpt(model, path, device):
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


def find_index(dataset_obj, img_name, dataset_name):
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


# ── Checkpoint scanning ────────────────────────────────────────────────────────

# Arch name → prefix patterns for filename matching
ARCH_PATTERNS = {
    'PIGNet_GSPonly': ('PIGNet_GSPonly',),
    'ASPP':          ('ASPP_',),
    'Mask2Former':   ('Mask2Former_',),
}

# Strip only arch+backbone+dataset, keep model_type → "pretrained_v3" / "scratch_v3"
_STRIP_RE = re.compile(
    r'^(?:PIGNet_GSPonly|ASPP|Mask2Former)_resnet(?:50|101)_(?:pascal|cityscape)_?|'
    r'^(?:PIGNet_GSPonly|ASPP|Mask2Former)_resnet(?:50|101)_'
)

# Full prefix including dataset (for stripping arch+backbone+type+dataset)
_STRIP_FULL_RE = re.compile(
    r'^(?:PIGNet_GSPonly|ASPP|Mask2Former)_resnet(?:50|101)_(?:pretrained|pretrain|scratch)_(?:pascal|cityscape)_?'
)


def _version_tag(fname):
    """
    Return a compact version string from the filename (without .pth).
    Keeps model_type prefix (pretrained/scratch) so same-subdir checkpoints
    don't collide:
      ASPP_resnet101_pretrained_cityscape_v3.pth  →  pretrained_v3
      ASPP_resnet101_scratch_cityscape_v3.pth     →  scratch_v3
    """
    stem = fname.replace('.pth', '')
    # Strip arch + backbone, keep type + dataset-suffix
    stripped = re.sub(
        r'^(?:PIGNet_GSPonly|ASPP|Mask2Former)_resnet(?:50|101)_', '', stem)
    # Strip dataset name (pascal/cityscape) from what remains
    stripped = re.sub(r'^(?:pretrained|pretrain|scratch)_(?:pascal|cityscape)_?',
                      lambda m: m.group(0).split('_')[0] + '_', stripped)
    return stripped or stem


def _short_label(path):
    """
    Build a compact label:  {sub_dir}/{version}
    - model_1/pretrained_v3  (sub_dir=model_1, type visible in version)
    - pretrained/v3          (sub_dir already encodes type, so strip type from version)
    """
    rel   = os.path.relpath(path, ROOT)
    parts = rel.split(os.sep)
    try:
        seg_idx = parts.index('segmentation')
        sub_dir = parts[seg_idx + 2] if len(parts) > seg_idx + 3 else 'root'
    except ValueError:
        sub_dir = 'root'
    version = _version_tag(os.path.basename(path))
    # Avoid "pretrained/pretrained_v3" → strip leading type when sub_dir already names the type
    if sub_dir in ('pretrained', 'scratch', 'pretrain'):
        version = re.sub(r'^(?:pretrained|pretrain|scratch)_', '', version)
    return f'{sub_dir}/{version}' if version else sub_dir


def scan_ckpts(backbone, dataset_name, model_type, exp_num=None):
    """
    Scan for all relevant .pth files.
    Returns {arch_name: [(short_label, full_path), ...]}

    Rules:
      - PIGNet_GSPonly: only filenames starting with 'PIGNet_GSPonly' (skip plain 'PIGNet_')
      - ASPP, Mask2Former: all matching files
    """
    num = 50 if backbone == 'resnet50' else 101

    if exp_num is not None:
        search_root = f'{ROOT}/model_{num}/{exp_num}/segmentation/{dataset_name}'
    else:
        search_root = f'{ROOT}/model_{num}'

    ok_types    = None if model_type == 'all' else \
                  {'pretrained': ('pretrained', 'pretrain'), 'scratch': ('scratch',)}[model_type]
    backbone_str = backbone  # 'resnet50' or 'resnet101'

    result = {}
    for path in sorted(glob.glob(f'{search_root}/**/*.pth', recursive=True)):
        # Only segmentation checkpoints
        if '/segmentation/' not in path.replace(os.sep, '/'):
            continue

        fname = os.path.basename(path)

        # Detect arch; skip plain PIGNet
        arch = None
        for aname, prefixes in ARCH_PATTERNS.items():
            if any(fname.startswith(p) for p in prefixes):
                arch = aname
                break
        if arch is None:
            continue

        # Filters: backbone, dataset, model_type
        if backbone_str not in fname:
            continue
        if dataset_name not in fname:
            continue
        if ok_types is not None and not any(t in fname for t in ok_types):
            continue

        label = _short_label(path)
        result.setdefault(arch, []).append((label, path))

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Compare all checkpoint versions per architecture')
    parser.add_argument('--backbone',   required=True, choices=['resnet50', 'resnet101'])
    parser.add_argument('--dataset',    required=True, choices=['pascal', 'cityscape'])
    parser.add_argument('--model_type', required=True, choices=['scratch', 'pretrained', 'all'],
                        help='"all" scans both pretrained and scratch checkpoints')
    parser.add_argument('--exp_num',    type=str, default=None,
                        help='Limit scan to one experiment dir (e.g. --exp_num 1). '
                             'Default: scan all experiment dirs.')
    args = parser.parse_args()

    device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone_label = 'R50' if args.backbone == 'resnet50' else 'R101'
    eval_list      = PASCAL_EVAL if args.dataset == 'pascal' else CITY_EVAL
    cmap           = get_colormap(args.dataset)

    # Base config
    with open(CFG_PATH) as f:
        base_cfg = dict_to_ns(yaml.safe_load(f))
    base_cfg.backbone   = args.backbone
    base_cfg.dataset    = args.dataset
    base_cfg.model_type = args.model_type
    base_cfg.mode       = 'infer'
    base_cfg.train      = False
    base_cfg.MI         = False
    base_cfg.scratch    = (args.model_type == 'scratch')  # 'all' → defaults to False (pretrained mode)

    # Datasets: zoom=1 (no distortion) for clean baseline inference
    def make_ds(crop_size):
        cfg = copy.deepcopy(base_cfg)
        cfg.crop_size = crop_size
        cfg.factor    = 1
        cfg.infer_params.process_type = 'zoom'
        return get_dataset(cfg)

    ds_513 = make_ds(513)
    ds_512 = make_ds(512)

    # Preload original / GT images
    orig_imgs, gt_imgs = {}, {}
    for img_name in eval_list:
        o = load_orig(img_name, args.dataset)
        g = load_gt(img_name, args.dataset)
        if o: orig_imgs[img_name] = o
        if g: gt_imgs[img_name]   = g

    # Scan checkpoints
    ckpt_map = scan_ckpts(args.backbone, args.dataset, args.model_type, args.exp_num)

    if not ckpt_map:
        print('No checkpoints found. Check --backbone / --dataset / --model_type.')
        return

    print(f'\n{"="*60}')
    print(f'backbone={args.backbone}  dataset={args.dataset}  model_type={args.model_type}')
    for arch, ckpts in ckpt_map.items():
        print(f'\n  [{arch}]  {len(ckpts)} checkpoint(s)')
        for lbl, path in ckpts:
            print(f'    {lbl:30s}  {path}')
    print(f'{"="*60}\n')

    # ── Per-architecture: one folder per checkpoint, individual images ────────
    for arch_name, ckpt_list in ckpt_map.items():
        print(f'\n[{arch_name}]  running {len(ckpt_list)} checkpoint(s)...')

        ds      = ds_512 if arch_name == 'Mask2Former' else ds_513
        idx_map = {n: find_index(ds, n, args.dataset) for n in eval_list}

        arch_root   = os.path.join(SAVE_BASE, arch_name, backbone_label,
                                   args.dataset, args.model_type)
        legend_rows = []

        for label, ckpt_path in ckpt_list:
            # folder name: replace '/' with '__'  e.g. model_1__v3
            folder_name = label.replace('/', '__')
            ckpt_dir    = os.path.join(arch_root, folder_name)
            os.makedirs(ckpt_dir, exist_ok=True)

            legend_rows.append(f'{folder_name:40s}  {ckpt_path}')
            print(f'  Loading [{label}]  {ckpt_path}')

            m_cfg = copy.deepcopy(base_cfg)
            m_cfg.model        = arch_name
            m_cfg.model_number = 1
            m_cfg.crop_size    = 512 if arch_name == 'Mask2Former' else 513

            try:
                model = get_model(m_cfg, ds_513)
                model.to(device).eval()
                load_ckpt(model, ckpt_path, device)
            except Exception as e:
                print(f'    ERROR: {e}')
                continue

            for img_name in eval_list:
                stem = os.path.splitext(img_name)[0].replace('_gtFine_labelTrainIds', '')
                idx  = idx_map.get(img_name)

                # Save orig + GT alongside predictions (only once per image)
                if orig_imgs.get(img_name):
                    orig_imgs[img_name].resize((IMG_W, IMG_H), Image.Resampling.LANCZOS)\
                        .save(os.path.join(ckpt_dir, f'{stem}_orig.png'))
                if gt_imgs.get(img_name):
                    gt_imgs[img_name].resize((IMG_W, IMG_H), Image.Resampling.LANCZOS)\
                        .save(os.path.join(ckpt_dir, f'{stem}_gt.png'))

                if idx is None:
                    print(f'    WARN: {img_name} not in dataset')
                    continue

                sample = ds[idx]
                if sample[0] is None:
                    continue

                inp = Variable(sample[0].to(device))
                with torch.no_grad():
                    if arch_name == 'Mask2Former':
                        out = model(inp.unsqueeze(0))
                        if isinstance(out, tuple):
                            out = out[0]
                    else:
                        out, _ = model(inp.unsqueeze(0))

                _, pred_t = torch.max(out, 1)
                pred     = pred_t.cpu().numpy().squeeze().astype(np.uint8)
                pred_img = colorize(pred, cmap).resize((IMG_W, IMG_H), Image.Resampling.LANCZOS)
                pred_img.save(os.path.join(ckpt_dir, f'{stem}_pred.png'))
                print(f'    OK  {stem}')

            del model
            torch.cuda.empty_cache()

        # Legend file (one per arch)
        os.makedirs(arch_root, exist_ok=True)
        legend_path = os.path.join(arch_root, 'legend.txt')
        with open(legend_path, 'w') as f:
            f.write(f'arch={arch_name}  backbone={backbone_label}  '
                    f'dataset={args.dataset}  model_type={args.model_type}\n')
            f.write('='*70 + '\n')
            for row in legend_rows:
                f.write(row + '\n')
        print(f'  Legend  →  {legend_path}')

    print(f'\nDone.  Results in  {SAVE_BASE}')


if __name__ == '__main__':
    main()
