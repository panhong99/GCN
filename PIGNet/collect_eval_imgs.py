#!/usr/bin/env python3
"""
collect_eval_imgs.py - Copy representative eval images from collect_img.py into
    eval_imgs/pascal/
    eval_imgs/cityscape/
"""

import os, shutil

ROOT     = '/home/hail/pan/GCN/PIGNet'
OUT_BASE = f'{ROOT}/eval_imgs'

_F, _M = 'frankfurt', 'munster'

PASCAL_IMGS = list(dict.fromkeys([
    '2011_000900.png', '2008_005105.png', '2007_000042.png',
    '2007_000332.png', '2007_000033.png', '2007_005608.png',
    '2007_000783.png', '2007_002643.png',
    '2007_003143.png', '2011_003003.png', '2008_002152.png',
    '2011_001407.png', '2008_000602.png',
    '2007_000676.png', '2007_003051.png', '2007_003714.png', '2007_000061.png',
]))

CITY_IMGS = list(dict.fromkeys([
    f'{_F}_000001_003588_gtFine_labelTrainIds.png',
    f'{_M}_000173_000019_gtFine_labelTrainIds.png',
    f'{_F}_000001_010600_gtFine_labelTrainIds.png',
    f'{_M}_000132_000019_gtFine_labelTrainIds.png',
    f'{_F}_000000_000294_gtFine_labelTrainIds.png',
    f'{_F}_000000_005543_gtFine_labelTrainIds.png',
    f'{_M}_000127_000019_gtFine_labelTrainIds.png',
    f'{_F}_000001_002759_gtFine_labelTrainIds.png',
    f'{_F}_000000_001751_gtFine_labelTrainIds.png',
    f'{_F}_000000_020215_gtFine_labelTrainIds.png',
    f'{_M}_000171_000019_gtFine_labelTrainIds.png',
    f'{_M}_000158_000019_gtFine_labelTrainIds.png',
    f'{_F}_000001_005898_gtFine_labelTrainIds.png',
    f'{_F}_000000_002196_gtFine_labelTrainIds.png',
    f'{_F}_000000_013382_gtFine_labelTrainIds.png',
    f'{_F}_000001_005410_gtFine_labelTrainIds.png',
    f'{_M}_000119_000019_gtFine_labelTrainIds.png',
]))


def copy_pascal():
    out_dir = os.path.join(OUT_BASE, 'pascal')
    os.makedirs(out_dir, exist_ok=True)
    ok, miss = 0, []
    for fname in PASCAL_IMGS:
        stem = os.path.splitext(fname)[0]
        src  = f'{ROOT}/data/VOCdevkit/VOC2012/JPEGImages/{stem}.jpg'
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, f'{stem}.jpg'))
            ok += 1
        else:
            miss.append(src)
    print(f'[Pascal] copied {ok}/{len(PASCAL_IMGS)}  →  {out_dir}')
    for m in miss:
        print(f'  MISSING: {m}')


def copy_cityscape():
    out_dir = os.path.join(OUT_BASE, 'cityscape')
    os.makedirs(out_dir, exist_ok=True)
    ok, miss = 0, []
    for fname in CITY_IMGS:
        stem = fname.replace('_gtFine_labelTrainIds.png', '')
        city = stem.split('_')[0]
        src  = f'{ROOT}/data/cityscape/leftImg8bit/val/{city}/{stem}_leftImg8bit.png'
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, f'{stem}_leftImg8bit.png'))
            ok += 1
        else:
            miss.append(src)
    print(f'[Cityscape] copied {ok}/{len(CITY_IMGS)}  →  {out_dir}')
    for m in miss:
        print(f'  MISSING: {m}')


if __name__ == '__main__':
    copy_pascal()
    copy_cityscape()
