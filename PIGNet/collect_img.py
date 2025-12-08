import os
from PIL import Image
import numpy as np

task = ["zoom", "overlap", "repeat"]

zoom_factor = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 1.7, 2]
overlap_percentage = [0, 0.1, 0.2, 0.3, 0.5]
pattern_repeat_count = [1, 3, 6, 9, 12]

f_name = "frankfurt"
m_name = "munster"

# zoom_collect = ["2011_000900.png","2008_005105.png","2007_000042.png","2007_000332.png","2007_000033.png","2007_005608.png","2007_000783.png","2007_002643.png"]

# overlap_collect = ["2007_003143.png","2011_003003.png","2008_002152.png","2011_001407.png","2008_000602.png"]

# repeat_collect = ["2007_000332.png","2007_000676.png","2007_003051.png","2007_003714.png","2007_000061.png"]

zoom_collect =  [f"{f_name}_000001_003588_gtFine_labelTrainIds.png",
                   f"{m_name}_000173_000019_gtFine_labelTrainIds.png",
                   f"{f_name}_000001_010600_gtFine_labelTrainIds.png",
                   f"{m_name}_000132_000019_gtFine_labelTrainIds.png",
                   f"{f_name}_000000_000294_gtFine_labelTrainIds.png",
                   f"{f_name}_000000_005543_gtFine_labelTrainIds.png",
                   f"{m_name}_000127_000019_gtFine_labelTrainIds.png",
                   f"{f_name}_000001_002759_gtFine_labelTrainIds.png",]

overlap_collect = [f"{f_name}_000000_001751_gtFine_labelTrainIds.png",
                   f"{f_name}_000000_020215_gtFine_labelTrainIds.png",
                   f"{m_name}_000171_000019_gtFine_labelTrainIds.png",
                   f"{m_name}_000158_000019_gtFine_labelTrainIds.png",
                 f"{f_name}_000001_005898_gtFine_labelTrainIds.png"]

repeat_collect = [f"{f_name}_000000_002196_gtFine_labelTrainIds.png",
                 f"{f_name}_000000_000294_gtFine_labelTrainIds.png",
                   f"{f_name}_000000_013382_gtFine_labelTrainIds.png",
                   f"{f_name}_000001_005410_gtFine_labelTrainIds.png",
                   f"{m_name}_000119_000019_gtFine_labelTrainIds.png"]

ratio_dict = {task[0]: zoom_factor, task[1]: overlap_percentage, task[2]: pattern_repeat_count}

dataset = "cityscape"
image_type = "pred"
make_dir_path = "/home/hail/pan/GCN/PIGNet/final_output_image/collect_img"
model_path = f"/home/hail/pan/GCN/PIGNet/final_output_image/pred_segmentation_masks/{dataset}"
mname_list = os.listdir(model_path)

# 0.5cm를 픽셀로 변환 (96 DPI 기준)
gap_cm = 0.5
dpi = 96
gap_px = int((dpi / 2.54) * gap_cm)

# 없는 이미지를 기록할 리스트
missing_images = []

for mname in sorted(mname_list):
    for key, value in ratio_dict.items():
        
        if key == "zoom":
            target_img_list = zoom_collect
        elif key == "overlap":
            target_img_list = overlap_collect
        elif key == "repeat":
            target_img_list = repeat_collect
        
        for ratio_key, img_name in zip(value, target_img_list):
            path = f"{model_path}/{mname}/{key}/{ratio_key}"
            
            output_dir = f"{make_dir_path}/{dataset}/{image_type}/{mname}"
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                img_name_list = os.listdir(path)
                if img_name in img_name_list:
                    img = Image.open(os.path.join(path, img_name))
                    img = img.resize((513, 256), Image.Resampling.LANCZOS)  # Resize to 513x256
                    
                    # 개별 이미지 저장
                    output_filename = f"{key}_{ratio_key}_{img_name}"
                    img.save(os.path.join(output_dir, output_filename))
                    
                    print(f"Image saved at: {os.path.join(output_dir, output_filename)}")
                else:
                    missing_images.append((mname, key, ratio_key, img_name))
                    print(f"Image not found: {os.path.join(path, img_name)}")

            except FileNotFoundError:
                missing_images.append((mname, key, ratio_key, img_name))
                print(f"Directory not found: {path}")

# 코드 실행 후 없는 이미지 보고
if missing_images:
    print("\nMissing images summary:")
    for mname, key, ratio_key, img_name in missing_images:
        print(f"Model: {mname}, Task: {key}, Ratio: {ratio_key}, Image: {img_name}")
else:
    print("\nAll images were found and processed.")

print("Finished!!")


