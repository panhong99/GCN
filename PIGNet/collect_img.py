import os
from PIL import Image

task = ["zoom", "overlap", "repeat"]

zoom_factor = [1 , 0.1 , 0.5 ,  1.5 , 2] # zoom in, out value 양수면 줌 음수면 줌아웃

overlap_percentage = [0, 0.1 , 0.2 , 0.3 , 0.5] #겹치는 비율 0~1 사이 값으로 0.8 이상이면 shape 이 안맞음

pattern_repeat_count = [1, 3, 6, 9, 12]

zoom_collect = ["2007_002094.png", "2007_000799.png", "2007_000033.png", "2007_004189.png", "2009_000705.png"]

overlap_collect = ["2007_005813.png", "2008_005691.png", "2007_003143.png", "2009_003217.png", "2008_000661.png"]

repeat_collect = ["2007_005828.png", "2007_007084.png", "2008_000911.png", "2008_007497.png", "2007_000061.png"]

ratio_dict = {task[0]: zoom_factor, task[1]: overlap_percentage, task[2]: pattern_repeat_count}

make_dir_path = "/home/hail/Desktop/HDD/pan/GCN/PIGNet/collect_img"
model_path = "/home/hail/Desktop/HDD/pan/GCN/PIGNet/pred_segmentation_masks/pascal"
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
                os.makedirs(f"{make_dir_path}/{key}/{ratio_key}", exist_ok=True)
                img.save(os.path.join(f"{make_dir_path}/{key}/{ratio_key}", mname+"_"+img_name))
                print(f"img save complete at : {make_dir_path}/{key}/{ratio_key}")
                print(f"img name is {mname+ '_' +img_name}")

print("Finished!!")
                    
                    
