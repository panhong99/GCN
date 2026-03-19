import argparse
import os
import torch
import numpy as np
import re
import yaml
import warnings
import pickle
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from make_classification_dataset import get_dataset
from make_classification_model import get_model
import random

# TODO
from gc import collect

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.experimental = False

set_seed(42)
    
warnings.filterwarnings("ignore")

def main(config, model_file, model_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # mode == 'train' -> shuffle = True
    dataset, _, MI_dataset_loader = get_dataset(config)
    model = get_model(config, dataset)
    checkpoint = torch.load(os.path.join(model_path, model_file), map_location=device)
    
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    gt_labels, pred_labels = [], []    

    # make kmeans models per layers
    layer_0_kmeans = MiniBatchKMeans(n_clusters=50, random_state=42, n_init=5)
    layer_1_kmeans = MiniBatchKMeans(n_clusters=50, random_state=42, n_init=5)
    layer_2_kmeans = MiniBatchKMeans(n_clusters=50, random_state=42, n_init=5)
    layer_3_kmeans = MiniBatchKMeans(n_clusters=50, random_state=42, n_init=5)
    layer_4_kmeans = MiniBatchKMeans(n_clusters=50, random_state=42, n_init=5)

    kmeans_models = [layer_0_kmeans, layer_1_kmeans, layer_2_kmeans, layer_3_kmeans, layer_4_kmeans]
        
    # fit Kmeans models per layers
    for inputs, labels in tqdm(MI_dataset_loader, desc=f" {config.factor_name}", leave=False):
        inputs = inputs.to(device)
        with torch.no_grad():
            if config.model == "Resnet":
                outputs, layers_output = model(inputs)
            else:
                outputs, layers_output, _ = model(inputs)
        
        # do batch training per layers                
        for activities_d, kmeans_model in zip(layers_output, kmeans_models):
            # activities_d = np.transpose(activities_d.cpu().numpy(), (0, 2, 3, 1))
            activities_d = np.transpose(activities_d, (0, 2, 3, 1))
            b, h, w, c = activities_d.shape
            activities_d = activities_d.reshape(-1, c)
            kmeans_model.batch_size = b * h * w
            kmeans_model.partial_fit(activities_d)

    del activities_d, layers_output, outputs, MI_dataset_loader, inputs
    collect()
    
    config.MI = False
    # no shuffle
    _, _, MI_dataset_loader = get_dataset(config)
    
    # predict VQ labels
    for m_idx, kmean_model in enumerate(kmeans_models):
        vq_labels = []
        for inputs, labels in tqdm(MI_dataset_loader, desc=f" {config.factor_name}", leave=False):
            inputs = inputs.to(device)           
            with torch.no_grad():
                if config.model == "Resnet":
                    outputs, layers_output = model(inputs)
                else:
                    outputs, layers_output, _ = model(inputs)
                        
            b, h, w, c = layers_output[0].shape

            if m_idx == 0:
                gt_labels.append(labels.cpu().numpy())
                pred_labels.append(torch.argmax(outputs, dim=1).cpu().numpy())    
                                            
            if (config.model == "Resnet") and (m_idx == 4):
                continue
            else:
                activities_p = np.transpose(layers_output[m_idx].cpu().numpy(), (0, 2, 3, 1))
                b, h, w, c = activities_p.shape
                layer_data_flat = activities_p.reshape(-1, c)
                vq_label = kmean_model.predict(layer_data_flat)       
                vq_label = vq_label.reshape(b, h, w).astype(np.int8)        
                vq_labels.append(vq_label)

        if (config.model == "Resnet") and (m_idx == 4):
            continue
        else:
            with open(config.output_folder + f'/layer_{m_idx}.pkl', 'wb') as f:
                pickle.dump(np.concatenate(vq_labels, axis=0), f)
                print(f"save layer_{m_idx} complete")
    
        del vq_labels, activities_p, layers_output, outputs, inputs, layer_data_flat
        collect()

    if config.dataset == "CIFAR-10" or config.dataset == "CIFAR-100":
        # int64로 되어있길래 dtype을 변경해주면 저장할 때 메모리를 절약할 수 있지 않을까 해서 변경
        gt_labels = np.concatenate(gt_labels).astype(np.int8)
        pred_labels = np.concatenate(pred_labels).astype(np.int8)
    else: # imagenet
        gt_labels = np.concatenate(gt_labels).astype(np.int16)
        pred_labels = np.concatenate(pred_labels).astype(np.int16)

    with open(config.output_folder + f'/gt_labels.pkl', 'wb') as f:
        pickle.dump(gt_labels, f)

    with open(config.output_folder + f'/pred_labels.pkl', 'wb') as f:
        pickle.dump(pred_labels, f)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/hail/pan/GCN/PIGNet/config_cls_MI.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        
    def dict_to_namespace(d):
        namespace = argparse.Namespace()
        for k, v in d.items():
            setattr(namespace, k, dict_to_namespace(v) if isinstance(v, dict) else v)
        return namespace
    
    config = dict_to_namespace(config_dict)

    # infer task 및 ratio
    # zoom_ratio = [0.1, np.sqrt(0.1), 0.5, np.sqrt(0.5), 1, 1.5, np.sqrt(2.75), 2]
    zoom_ratio = [1]
    # rotate_degree = [0, 180, 150, 120, 90, 60, 45, 30, 15, -15, -30, -45, -60, -90, -120, -150, -180]
    rotate_degree = [0]
    zoom_name_map = {0.1: "0.1", np.sqrt(0.1): "0.3", 0.5: "0.5", np.sqrt(0.5): "0.7", 
                     1: "1", 1.5: "1.5", np.sqrt(2.75): "1.75", 2: "2.0"}

    if config.backbone == "resnet50":
        num = 50
    elif config.backbone == "resnet101":
        num = 101

    model_path = f"/home/hail/pan/GCN/PIGNet/model_{num}/{config.model_number}/classification/{config.dataset}/{config.model_type}"
    model_files = sorted(os.listdir(model_path))

    for model_file in model_files:
        m_name = re.search(fr"classification_(.*?)_{config.backbone}", model_file)
        model_key = m_name.group(1) if m_name else ("vit" if "vit" in model_file else "unknown")

        if model_key in ["vit", "PIGNet_classification"]:
            continue

        print(f"Processing Model: {model_key}")
        
        for p_key, f_list in {"zoom": zoom_ratio, "rotate": rotate_degree}.items():
            for f_val in f_list:
                f_name = zoom_name_map.get(f_val, str(int(f_val))) if p_key == "zoom" else str(int(f_val))
                
                output_folder = os.path.join(
                    "/home/hail/pan/HDD/MI_dataset", 
                    config.dataset, 
                    config.backbone, 
                    config.model_type, 
                    model_key,
                    p_key,
                    f_name
                )

                os.makedirs(output_folder, exist_ok=True)

                iter_config = argparse.Namespace(**vars(config))
                iter_config.model = model_key
                iter_config.factor = float(f_val)
                iter_config.factor_name = f_name
                iter_config.infer_params.process_type = p_key
                iter_config.output_folder = output_folder
                
                main(iter_config, model_file, model_path)
