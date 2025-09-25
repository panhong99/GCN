import os
from PIL import Image

task = ["overlap", "repeat", "zoom"]

zoom_factor = [1 , 0.1 , 0.5 ,  1.5 , 2] # zoom in, out value 양수면 줌 음수면 줌아웃

overlap_percentage = [0, 0.1 , 0.2 , 0.3 , 0.5] #겹치는 비율 0~1 사이 값으로 0.8 이상이면 shape 이 안맞음

pattern_repeat_count = [1, 3, 6, 9, 12]

ratio_dict = {task[2]: zoom_factor, task[0]: overlap_percentage, task[1]: pattern_repeat_count}
model_path = "/home/hail/Desktop/HDD/pan/GCN/PIGNet/pred_segmentation_masks/pascal"

mname_list = os.listdir(model_path)

zoom_dict = {}
overlap_dict = {}
repeat_dict = {}

for key, value in ratio_dict.items():
    print(key)
    
    if key == "zoom":
        file_name_dict = zoom_dict
    elif key == "overlap":
        file_name_dict = overlap_dict
    elif key == "repeat":
        file_name_dict = repeat_dict
    
    junk_file_names = []
    
    for ratio in value:
        print(ratio)
        file_name_dict[ratio] = []

        for mname in sorted(mname_list):
            print(mname)
            
            path = f"{model_path}/{mname}/{key}/{ratio}"
            file_name = os.listdir(path)
            junk_file_names = file_name
            
            if file_name_dict[ratio] == []:
                file_name_dict[ratio].extend(file_name)
                junk_file_names = []
            
            elif file_name_dict[ratio] != []: # file_names != []
                file_name_dict[ratio] = set(file_name_dict[ratio])
                junk_file_names = set(junk_file_names)
                file_name_dict[ratio] = file_name_dict[ratio].intersection(junk_file_names)
                file_name_dict[ratio] = list(file_name_dict[ratio])
        try:
            if len(file_name_dict) == 0:
                raise ValueError("length of list is zero!!")
        
        except ValueError as e:
            print(f"empty error: {e}")
            
        for mname in sorted(mname_list):
            path_ = f"{model_path}/{mname}/{key}/{ratio}"
            new_img_path = f"/home/hail/Desktop/HDD/pan/GCN/PIGNet/commom_pred_mask/{mname}/{key}/{ratio}"
            
            for img_name in file_name_dict[ratio]:
                
                if os.path.exists(path_):
                    img_path = os.path.join(path_, img_name)
                else:
                    print("can't find folder")
                
                img = Image.open(img_path)
                
                if os.path.exists(new_img_path):
                       img.save(os.path.join(new_img_path, img_name))
                
                else: # os.path.exists(new_img_path) == False
                    os.makedirs(new_img_path)
                    img.save(os.path.join(new_img_path, img_name))

        print(f"commom {mname}/{key} imgs save complete!!")      