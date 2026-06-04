import os
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat

# ── 설정 ──────────────────────────────────────────────────────────────────────
dataset = "cityscape"   # "pascal" or "cityscape"

task = ["zoom", "overlap", "repeat"]
zoom_factor       = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 1.7, 2]
overlap_percentage   = [0.1, 0.2, 0.3, 0.5, 1]
pattern_repeat_count = [1, 3, 6, 9, 12]
ratio_dict = {"zoom": zoom_factor, "overlap": overlap_percentage, "repeat": pattern_repeat_count}

# Pascal collect 리스트
pascal_zoom_collect    = ["2011_000900.png", "2008_005105.png", "2007_000042.png",
                          "2007_000332.png", "2007_000033.png", "2007_005608.png",
                          "2007_000783.png", "2007_002643.png"]
pascal_overlap_collect = ["2007_003143.png", "2011_003003.png", "2008_002152.png",
                          "2011_001407.png", "2008_000602.png"]
pascal_repeat_collect  = ["2007_000332.png", "2007_000676.png", "2007_003051.png",
                          "2007_003714.png", "2007_000061.png"]

# Cityscape collect 리스트
f_name = "frankfurt"
m_name = "munster"
city_zoom_collect    = [f"{f_name}_000001_003588_gtFine_labelTrainIds.png",
                        f"{m_name}_000173_000019_gtFine_labelTrainIds.png",
                        f"{f_name}_000001_010600_gtFine_labelTrainIds.png",
                        f"{m_name}_000132_000019_gtFine_labelTrainIds.png",
                        f"{f_name}_000000_000294_gtFine_labelTrainIds.png",
                        f"{f_name}_000000_005543_gtFine_labelTrainIds.png",
                        f"{m_name}_000127_000019_gtFine_labelTrainIds.png",
                        f"{f_name}_000001_002759_gtFine_labelTrainIds.png"]
city_overlap_collect = [f"{f_name}_000000_001751_gtFine_labelTrainIds.png",
                        f"{f_name}_000000_020215_gtFine_labelTrainIds.png",
                        f"{m_name}_000171_000019_gtFine_labelTrainIds.png",
                        f"{m_name}_000158_000019_gtFine_labelTrainIds.png",
                        f"{f_name}_000001_005898_gtFine_labelTrainIds.png"]
city_repeat_collect  = [f"{f_name}_000000_002196_gtFine_labelTrainIds.png",
                        f"{f_name}_000000_000294_gtFine_labelTrainIds.png",
                        f"{f_name}_000000_013382_gtFine_labelTrainIds.png",
                        f"{f_name}_000001_005410_gtFine_labelTrainIds.png",
                        f"{m_name}_000119_000019_gtFine_labelTrainIds.png"]

if dataset == "pascal":
    zoom_collect    = pascal_zoom_collect
    overlap_collect = pascal_overlap_collect
    repeat_collect  = pascal_repeat_collect
    orig_base = "/home/hail/pan/GCN/PIGNet/data/VOCdevkit/VOC2012/JPEGImages"
    gt_base   = "/home/hail/pan/GCN/PIGNet/data/VOCdevkit/VOC2012/SegmentationClass"
else:
    zoom_collect    = city_zoom_collect
    overlap_collect = city_overlap_collect
    repeat_collect  = city_repeat_collect
    orig_base = "/home/hail/pan/GCN/PIGNet/data/cityscape/leftImg8bit/val"
    gt_base   = "/home/hail/pan/GCN/PIGNet/data/cityscape/gtFine/val"

collect_dict = {"zoom": zoom_collect, "overlap": overlap_collect, "repeat": repeat_collect}

PKL_DIR       = "/home/hail/pan/GCN/PIGNet/infer_output"
SAVE_BASE     = "/home/hail/pan/GCN/PIGNet/SEG_family/result_imgs"
MAKE_DIR_PATH = "/home/hail/pan/GCN/PIGNet/final_output_image/collect_img"

IMG_W, IMG_H = 513, 256
GAP      = 8
LABEL_W  = 50
LABEL_H  = 20
FONT_SIZE = 14

# ── 컬러맵 준비 ───────────────────────────────────────────────────────────────
if dataset == "pascal":
    cmap_mat = loadmat("/home/hail/pan/GCN/PIGNet/data/pascal_seg_colormap.mat")["colormap"]
    cmap_flat = (cmap_mat * 255).astype(np.uint8).flatten().tolist()
else:
    cityscapes_colormap = {
        0: (128, 64, 128), 1: (244, 35, 232), 2: (70, 70, 70),
        3: (102, 102, 156), 4: (190, 153, 153), 5: (153, 153, 153),
        6: (250, 170, 30), 7: (220, 220, 0), 8: (107, 142, 35),
        9: (152, 251, 152), 10: (70, 130, 180), 11: (220, 20, 60),
        12: (255, 0, 0), 13: (0, 0, 142), 14: (0, 0, 70),
        15: (0, 60, 100), 16: (0, 80, 100), 17: (0, 0, 230),
        18: (119, 11, 32), 255: (0, 0, 0)
    }
    palette = np.zeros((256, 3), dtype=np.uint8)
    for tid, color in cityscapes_colormap.items():
        if tid < 256:
            palette[tid] = color
    cmap_flat = palette.flatten().tolist()

def colorize(pred_array):
    """numpy uint8 pred → colorized RGB PIL.Image"""
    img_p = Image.fromarray(pred_array.astype(np.uint8), mode="P")
    img_p.putpalette(cmap_flat)
    return img_p.convert("RGB")

# ── pkl 로드: (model, process_type, factor) → {stem: pred_array} ──────────────
# pkl 파일명: {dataset}_{model}_{process_type}_{factor}_number_{model_number}.pkl
pkl_index = {}  # key: (model, process_type, str(factor)) → {name_stem: pred_array}

for fname in os.listdir(PKL_DIR):
    if not fname.endswith(".pkl"):
        continue
    parts = fname.replace(".pkl", "").split("_")
    # dataset은 맨 앞 단어 — 현재 dataset과 다르면 skip
    if not fname.startswith(dataset + "_"):
        continue

    # dataset prefix 제거 후 파싱: {model}_{process_type}_{factor}_number_{model_number}
    remainder = fname[len(dataset) + 1:].replace(".pkl", "")
    # "number_{n}" 뒤쪽 제거
    remainder = remainder[:remainder.rfind("_number_")]
    # 마지막 _ 구분: process_type과 factor는 각각 한 토큰
    tokens = remainder.split("_")
    # tokens: [model_parts..., process_type, factor]
    factor_str  = tokens[-1]
    process_type = tokens[-2]
    model_name  = "_".join(tokens[:-2])

    with open(os.path.join(PKL_DIR, fname), "rb") as f:
        data = pickle.load(f)

    stem_to_pred = {}
    for pred, iname in zip(data["pred_img"], data["img_name"]):
        stem = os.path.splitext(iname)[0]  # 확장자 제거
        stem_to_pred[stem] = pred

    key = (model_name, process_type, factor_str)
    pkl_index[key] = stem_to_pred
    print(f"Loaded pkl: model={model_name}, type={process_type}, factor={factor_str}, n={len(stem_to_pred)}")

# ── 이미지 수집 ───────────────────────────────────────────────────────────────
# collected_data[task][ratio_str][img_name] = {model: PIL.Image (RGB)}
collected_data = {t: {} for t in task}
missing_images = []

model_list = sorted({k[0] for k in pkl_index})

for key_task, ratio_list in ratio_dict.items():
    target_img_list = collect_dict[key_task]
    for ratio_key, img_name in zip(ratio_list, target_img_list):
        ratio_str = str(ratio_key)
        img_stem  = os.path.splitext(img_name)[0]

        for model_name in model_list:
            pkl_key = (model_name, key_task, ratio_str)
            if pkl_key not in pkl_index:
                missing_images.append((model_name, key_task, ratio_key, img_name, "no pkl"))
                continue

            stem_map = pkl_index[pkl_key]
            if img_stem not in stem_map:
                missing_images.append((model_name, key_task, ratio_key, img_name, "not in pkl"))
                continue

            pred = stem_map[img_stem]
            pred_img = colorize(pred).resize((IMG_W, IMG_H), Image.Resampling.LANCZOS)

            # 개별 이미지 저장
            save_dir = os.path.join(SAVE_BASE, dataset, model_name, key_task, ratio_str)
            os.makedirs(save_dir, exist_ok=True)
            pred_img.save(os.path.join(save_dir, img_name))

            indiv_dir = os.path.join(MAKE_DIR_PATH, dataset, "pred", model_name)
            os.makedirs(indiv_dir, exist_ok=True)
            pred_img.save(os.path.join(indiv_dir, f"{key_task}_{ratio_key}_{img_name}"))
            print(f"Saved: {model_name} / {key_task} / {ratio_key} / {img_name}")

            # matrix 조립용
            collected_data[key_task].setdefault(ratio_str, {}).setdefault(img_name, {})
            collected_data[key_task][ratio_str][img_name][model_name] = pred_img

# ── 원본 / GT 로드 헬퍼 ───────────────────────────────────────────────────────
def load_orig(img_name):
    if dataset == "cityscape":
        stem = img_name.replace("_gtFine_labelTrainIds.png", "")
        city = stem.split("_")[0]
        path = os.path.join(orig_base, city, f"{stem}_leftImg8bit.png")
    else:
        path = os.path.join(orig_base, img_name)
    if os.path.exists(path):
        return Image.open(path).convert("RGB").resize((IMG_W, IMG_H), Image.Resampling.LANCZOS)
    return None

def load_gt(img_name):
    if dataset == "cityscape":
        stem = img_name.replace("_gtFine_labelTrainIds.png", "")
        city = stem.split("_")[0]
        path = os.path.join(gt_base, city, f"{stem}_gtFine_color.png")
    else:
        path = os.path.join(gt_base, img_name)
    if os.path.exists(path):
        return Image.open(path).convert("RGB").resize((IMG_W, IMG_H), Image.Resampling.LANCZOS)
    return None

# ── Matrix 이미지 생성 ─────────────────────────────────────────────────────────
def make_matrix(task_key, ratio_img_model_dict):
    ratio_list = sorted(ratio_img_model_dict.keys(), key=lambda x: float(x))
    all_models = sorted({m for r in ratio_img_model_dict.values()
                           for img in r.values() for m in img.keys()})
    col_headers = ["Image", "GT"] + all_models
    n_cols = len(col_headers)
    n_rows = len(ratio_list)

    header_h  = LABEL_H + GAP
    total_w = LABEL_W + n_cols * IMG_W + (n_cols + 1) * GAP
    total_h = header_h + n_rows * IMG_H + (n_rows + 1) * GAP

    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    try:
        font       = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", FONT_SIZE)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", FONT_SIZE - 2)
    except Exception:
        font = small_font = ImageFont.load_default()

    # 열 헤더
    for c_idx, col_name in enumerate(col_headers):
        x = LABEL_W + GAP + c_idx * (IMG_W + GAP) + IMG_W // 2
        draw.text((x, GAP // 2), col_name, fill=(0, 0, 0), font=font, anchor="mt")

    # 행
    for r_idx, ratio_str in enumerate(ratio_list):
        y_top = header_h + r_idx * (IMG_H + GAP) + GAP

        draw.text((LABEL_W // 2, y_top + IMG_H // 2),
                  ratio_str, fill=(0, 0, 0), font=small_font, anchor="mm")

        img_name   = list(ratio_img_model_dict[ratio_str].keys())[0]
        model_dict = ratio_img_model_dict[ratio_str][img_name]

        for c_idx, col_name in enumerate(col_headers):
            x_left = LABEL_W + GAP + c_idx * (IMG_W + GAP)
            if col_name == "Image":
                cell = load_orig(img_name)
            elif col_name == "GT":
                cell = load_gt(img_name)
            else:
                cell = model_dict.get(col_name)

            if cell is not None:
                canvas.paste(cell.resize((IMG_W, IMG_H), Image.Resampling.LANCZOS), (x_left, y_top))
            else:
                draw.rectangle([x_left, y_top, x_left + IMG_W, y_top + IMG_H],
                                outline=(200, 200, 200), fill=(240, 240, 240))
                draw.text((x_left + IMG_W // 2, y_top + IMG_H // 2),
                          "N/A", fill=(150, 150, 150), font=small_font, anchor="mm")
    return canvas


matrix_out_dir = os.path.join(MAKE_DIR_PATH, dataset, "matrix")
os.makedirs(matrix_out_dir, exist_ok=True)

for task_key, ratio_img_model_dict in collected_data.items():
    if not ratio_img_model_dict:
        continue
    print(f"\nCreating matrix: {task_key}")
    matrix_img = make_matrix(task_key, ratio_img_model_dict)
    out_path = os.path.join(matrix_out_dir, f"{task_key}_matrix.png")
    matrix_img.save(out_path, dpi=(150, 150))
    print(f"Matrix saved: {out_path}")

# 누락 보고
if missing_images:
    print("\nMissing images:")
    for item in missing_images:
        print(f"  model={item[0]}, task={item[1]}, ratio={item[2]}, img={item[3]}, reason={item[4]}")
else:
    print("\nAll images found and processed.")

print("Finished!!")
