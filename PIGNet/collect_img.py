import os
from PIL import Image

task = ["zoom", "overlap", "repeat"]

zoom_factor = [0.1,0.3,0.5,0.7,1,1.5,1.7,2] # zoom in, out value 양수면 줌 음수면 줌아웃

overlap_percentage = [0, 0.1 , 0.2 , 0.3 , 0.5] #겹치는 비율 0~1 사이 값으로 0.8 이상이면 shape 이 안맞음

pattern_repeat_count = [1, 3, 6, 9, 12]

f_name = "frankfurt"
m_name = "munster"

zoom_collect = ["2011_000900.png","2007_006650.png","2007_000042.png","2007_000332.png","2007_000033.png","2007_005608.png","2007_000783.png","2007_002643.png"]

overlap_collect = ["2007_003143.png","2011_003003.png","2008_002152.png","2011_001407.png","2008_000602.png"]

repeat_collect = ["2007_000332.png","2007_000676.png","2007_003051.png","2007_003714.png","2007_000061.png"]

# zoom_collect =  [f"{f_name}_000001_003588_gtFine_labelTrainIds.png",
#                    f"{m_name}_000173_000019_gtFine_labelTrainIds.png",
#                    f"{f_name}_000001_010600_gtFine_labelTrainIds.png",
#                    f"{m_name}_000132_000019_gtFine_labelTrainIds.png",
#                    f"{f_name}_000000_000294_gtFine_labelTrainIds.png",
#                    f"{f_name}_000000_005543_gtFine_labelTrainIds.png",
#                    f"{m_name}_000127_000019_gtFine_labelTrainIds.png",
#                    f"{f_name}_000001_002759_gtFine_labelTrainIds.png",]

# overlap_collect = [f"{f_name}_000000_001751_gtFine_labelTrainIds.png",
#                    f"{f_name}_000000_020215_gtFine_labelTrainIds.png",
#                    f"{m_name}_000171_000019_gtFine_labelTrainIds.png",
#                    f"{m_name}_000158_000019_gtFine_labelTrainIds.png",
#                  f"{f_name}_000001_005898_gtFine_labelTrainIds.png"]

# repeat_collect = [f"{f_name}_000000_002196_gtFine_labelTrainIds.png",
#                  f"{f_name}_000000_000294_gtFine_labelTrainIds.png",
#                    f"{f_name}_000000_013382_gtFine_labelTrainIds.png",
#                    f"{f_name}_000001_005410_gtFine_labelTrainIds.png",
#                    f"{m_name}_000119_000019_gtFine_labelTrainIds.png"]


ratio_dict = {task[0]: zoom_factor, task[1]: overlap_percentage, task[2]: pattern_repeat_count}

dataset = "pascal"
image_type = "GT"
make_dir_path = "/home/hail/pan/GCN/PIGNet/collect_img"
model_path = f"/home/hail/pan/GCN/PIGNet/final_output_image/GT_input_images/{dataset}"
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
                    
                    
