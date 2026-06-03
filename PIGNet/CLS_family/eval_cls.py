import argparse
import numpy as np
import torch
import pandas as pd
import os
import warnings
import torchvision.transforms.functional as TF
import re
import yaml
from tqdm.auto import tqdm
from GCN.PIGNet.CLS_family.cls_dataset import get_dataset
from GCN.PIGNet.CLS_family.cls_models import get_model
warnings.filterwarnings("ignore")

def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    print(f"cuda available : {device}")

    is_training = (config.mode == "train")

    config.model = config.model
    print(f"Mode: {config.mode} | Model: {config.model} | Dataset: {config.dataset} | Device: {device}")
    
    if config.backbone == "resnet50":
        num=50
    elif config.backbone == "resnet101":
        num=101

    model_filename = config.infer_params.model_filename
    m_name = re.search(fr"classification_(.*?)_{config.backbone}", model_filename)
    if m_name:
        config.model = m_name.group(1)
    elif "vit" in model_filename:
        config.model = "vit"

    # define model, dataset
    dataset, dataset_loader, valid_dataset = get_dataset(config)
    model = get_model(config, dataset)
    
    print("-" * 60)
    print("Evaluating !!! ")
    print("-" * 60)

    cls_infer_image_dir = "cls_infer_image"
    os.makedirs(cls_infer_image_dir, exist_ok=True)

    checkpoint = torch.load(f'/home/hail/pan/GCN/PIGNet/model_{num}/{config.model_number}/classification/{config.dataset}/{config.model_type}/{model_filename}', map_location=device)
    
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    model.load_state_dict(state_dict)
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(tqdm(valid_dataset)):

            # PIGNet 모델인 경우에만 이미지를 저장
            if config.model == "PIGNet_classification" and config.model == 1:
                # 각 ratio마다 첫 번째 배치의 첫 번째 이미지 한 장만 저장
                if i == 0:  # 첫 번째 배치만
                    batch_size = inputs.size(0)
                    if batch_size > 0:  # 배치에 이미지가 있는 경우
                        img_tensor = inputs[0]  # 첫 번째 이미지 (shape: (C, H, W))
                        # CIFAR 데이터셋의 경우 정규화된 텐서를 역정규화하여 [0, 1] 범위로 복원
                        if config.dataset in ['CIFAR-10', 'CIFAR-100']:
                            img_tensor = (img_tensor + 1) / 2  # denormalize
                        # 텐서를 PIL 이미지로 변환
                        img_pil = TF.to_pil_image(img_tensor)
                        img_filename = f"{config.infer_params.process_type}_{config.factor}.png"
                        img_path = os.path.join(cls_infer_image_dir, img_filename)
                        img_pil.save(img_path)

            inputs = inputs.to(device)
            labels = torch.tensor(labels).to(device)

            if config.model == "vit":

                outputs = model(inputs)

            elif config.model == "Resnet":
                outputs, layers_output = model(inputs)

            else:
                outputs, layers_output, backbone_layers_output = model(inputs)

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        backbone_path = f"/home/hail/pan/GCN/PIGNet/layers_activity/{config.dataset}/{config.model}/{config.infer_params.process_type}/{config.backbone}/backbone_activity"
        layers_path = f"/home/hail/pan/GCN/PIGNet/layers_activity/{config.dataset}/{config.model}/{config.infer_params.process_type}/{config.factor}/layers_activity"
        
        if (config.model == "PIGNet_GSPonly_classification" or config.model == "PIGNet_classification") and (config.factor == 1) and (config.model_number == 1) and (config.dataset == "imagenet"):

            if os.path.exists(backbone_path):
                pass
            else: # not exists(path)
                os.makedirs(backbone_path, exist_ok=True)
                torch.save(backbone_layers_output, os.path.join(backbone_path, "backbone.pth"))                
            
            if os.path.exists(layers_path):
                pass
            else: # not exists(path)
                os.makedirs(layers_path, exist_ok=True)
                torch.save(layers_output, os.path.join(layers_path, "model_layers.pth"))                

        elif config.model == "Resnet":

            if os.path.exists(layers_path):
                pass
            else: # not exists(path)
                os.makedirs(layers_path, exist_ok=True)
                torch.save(layers_output, os.path.join(layers_path, "model_layers.pth"))                
        
        else: # vit
            pass

        accuracy = 100 * correct / total
        print('Accuracy: {:.2f}%'.format(accuracy))
        return accuracy
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Load configuration from config.yaml")
    parser.add_argument("--config", type = str, default = "/home/hail/pan/GCN/PIGNet/config_classification.yaml", help = "path to config.yaml")
    parser.add_argument("--mode", type = str, default = "eval", help = "evaluation mode")
    cli_args = parser.parse_args()
    
    try:
        with open(cli_args.config, "r") as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error config.yaml not found at {cli_args.config}")
        exit()
    except Exception as e:
        print(f"Error parsing YAML file: {e}")
        exit()
        
    def dict_to_namespace(d):
        namespace = argparse.Namespace()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(namespace, k, dict_to_namespace(v))
            else:
                setattr(namespace, k, v)
        return namespace
    
    config = dict_to_namespace(config_dict)
    config.mode = cli_args.mode
        
    print("-- Starting Infer Mode --")
    
    if config.backbone == "resnet101":
        num = 101
    else:
        num = 50
    
    path = f"/home/hail/pan/GCN/PIGNet/model_{num}/{config.model_number}/classification/{config.dataset}/{config.model_type}"
            
    try:
        model_list = sorted(os.listdir(path))
        print(f"[INFO] Found {len(model_list)} models in '{path}'")
    except FileNotFoundError:
        print(f"[ERROR] Model directory not found at '{path}'")
        exit()

    zoom_ratio = [0.1, np.sqrt(0.1), 0.5, np.sqrt(0.5), 1, 1.5, np.sqrt(2.75), 2] # zoom in, out value 양수면 줌 음수면 줌아웃
    rotate_degree = [0, 180, 150, 120, 90, 60, 45, 30, 15, -15, -30, -45, -60, -90, -120, -150, -180]
    process_dict = {"zoom": zoom_ratio, "rotate": rotate_degree}
    output_dict = {model_name: {"zoom": [], "rotate": []} for model_name in model_list}

    for name in model_list:
        for process_key, factor_list in process_dict.items():
            for factor_value in factor_list:

                # 루프마다 config 객체를 복사하여 수정 (원본 config는 유지)
                iter_config = argparse.Namespace()
                for k, v in vars(config).items():
                    if isinstance(v, argparse.Namespace):
                        setattr(iter_config, k, argparse.Namespace(**vars(v)))
                    else:
                        setattr(iter_config, k, v)
                
                iter_config.infer_params.model_filename = name
                iter_config.infer_params.process_type = process_key
                iter_config.factor = float(factor_value)
                iter_config.crop_size = 512 if config.model == "vit" else 513

                print("-" * 60)
                print(f"Testing model: {name} | Process: {process_key} | Factor: {factor_value}")
                print("\n")

                accuracy = main(iter_config)
                if accuracy is not None:
                    output_dict[name][process_key].append(accuracy)

    print("\n--- Inference Results Summary ---")

    records = []
    for model_name, result_dict in output_dict.items():
        for task, values in result_dict.items():
            factors = process_dict.get(task, [])
            for i, val in enumerate(values):
                records.append({
                    "model": model_name,
                    "task": task,
                    "factor": factors[i] if i < len(factors) else "N/A",
                    "accuracy": val
                })
    df_long = pd.DataFrame(records)
    
    df_wide = df_long.pivot_table(index=['model', 'task'], 
                                    columns='factor', 
                                    values='accuracy').reset_index()
    
    df_wide.rename_axis(columns=None, inplace=True)
    output_filename = f"output_wide_{config.model_number}_{config.model_type}_{config.dataset}.csv"
    df_wide.to_csv(output_filename, index=False)
    
    print(f"\n[SUCCESS] Reshaped results saved to '{output_filename}'")
    print(df_wide)
