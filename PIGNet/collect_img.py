import os
from PIL import Image

task = ["zoom", "overlap", "repeat"]

zoom_factor = [1 , 0.1 , 0.5 ,  1.5 , 2] # zoom in, out value 양수면 줌 음수면 줌아웃

overlap_percentage = [0, 0.1 , 0.2 , 0.3 , 0.5] #겹치는 비율 0~1 사이 값으로 0.8 이상이면 shape 이 안맞음

pattern_repeat_count = [1, 3, 6, 9, 12]

# zoom_collect = ["2007_000061.png", "2007_000123.png", "2007_000033.png", "2007_000783.png", "2007_007084.png"]

# overlap_collect = ["2010_004519.png", "2007_002618.png", "2009_003217.png", "2009_003810.png", "2009_001332.png"]

# repeat_collect = ["2007_002619.png", "2207_000676.png", "2007_003349.png", "2009_000924.png", "2007_000061.png"]

zoom_collect =  ["frankfurt_000000_000576_gtFine_labelTrainIds.png",
                   "frankfurt_000000_000294_gtFine_labelTrainIds.png",
                   "frankfurt_000001_032018_gtFine_labelTrainIds.png",
                   "frankfurt_000000_013382_gtFine_labelTrainIds.png",
                   "frankfurt_000000_003025_gtFine_labelTrainIds.png"]

overlap_collect = ["frankfurt_000000_003357_gtFine_labelTrainIds.png",
                   "frankfurt_000000_010763_gtFine_labelTrainIds.png",
                   "frankfurt_000000_000294_gtFine_labelTrainIds.png",
                   "frankfurt_000000_000294_gtFine_labelTrainIds.png",
                   "munster_000016_000019_gtFine_labelTrainIds.png"]

repeat_collect = ["frankfurt_000001_025512_gtFine_labelTrainIds.png",
                   "munster_000016_000019_gtFine_labelTrainIds.png",
                   "frankfurt_000001_009854_gtFine_labelTrainIds.png",
                   "frankfurt_000000_000576_gtFine_labelTrainIds.png",
                   "frankfurt_000001_052594_gtFine_labelTrainIds.png"]


ratio_dict = {task[0]: zoom_factor, task[1]: overlap_percentage, task[2]: pattern_repeat_count}

dataset = "cityscape"
image_type = "GT_masks"
make_dir_path = "/home/hail/pan/GCN/PIGNet/collect_img"
model_path = f"/home/hail/pan/GCN/PIGNet/GT_segmentation_masks/{dataset}"
mname_list = os.listdir(model_path)

for key, value in ratio_dict.items():

    # assign target_img_list
    if key == "zoom":
        target_img_list = zoom_collect
    elif key == "overlap":
        target_img_list = overlap_collect
    elif key == "repeat":
        target_img_list = repeat_collect
        
    # set ratio_key img_name pair
    for ratio_key, img_name in zip(value, target_img_list):
        for mname in sorted(mname_list):
            path = f"{model_path}/{mname}/{key}/{ratio_key}"
            
            img_name_list = os.listdir(path)
            
            if img_name in img_name_list:
                img = Image.open(os.path.join(path, img_name))
                os.makedirs(f"{make_dir_path}/{dataset}/{image_type}/{key}/{ratio_key}", exist_ok=True)
                img.save(os.path.join(f"{make_dir_path}/{dataset}/{image_type}/{key}/{ratio_key}", mname+"_"+img_name))
                print(f"img save complete at : {make_dir_path}/{key}/{ratio_key}")
                print(f"img name is {mname+ '_' +img_name}")

print("Finished!!")
                    
                    
