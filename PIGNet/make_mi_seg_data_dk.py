import argparse
import os
import torch
import numpy as np
import re
import yaml
import warnings
import pickle
import copy
from tqdm.auto import tqdm
from sklearn.cluster import MiniBatchKMeans
from make_segmentation_dataset import get_dataset
from make_segmentation_model import get_model
from torch.autograd import Variable
from sklearn.cluster import KMeans
import random
import pickle
import utils_segmentation as utils_segmentation
from gc import collect
from functools import partial

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.experimental = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

set_seed(42)

warnings.filterwarnings("ignore")

def mi_seg_data_loader(config, dataset, shuffle=True):
    """DataLoader 생성 함수 - 동일한 셔플 순서 보장"""
    feature_shape = (2048, 33, 33)
    collate_fn = partial(utils_segmentation.make_batch_fn, batch_size=config.batch_size, feature_shape=feature_shape)
    
    # 고정된 seed로 동일한 셔플 순서 보장
    generator = torch.Generator()
    generator.manual_seed(42)
    
    MI_dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=config.workers,
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=seed_worker
    )
    
    return MI_dataset_loader

def main(config, model_file, model_path):

    device = f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu"

    dataset = get_dataset(config)
    model = get_model(config, dataset)

    checkpoint = torch.load(os.path.join(model_path, model_file), map_location=device)
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    # state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}

    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    for layer_idx in range(5):

        set_seed(42)
        dataset = get_dataset(config)

        # 학습용 DataLoader 생성 (shuffle=True, 동일한 순서 보장)
        MI_dataset_loader = mi_seg_data_loader(config, dataset, shuffle=True)
        pixel_by_kmeans = [MiniBatchKMeans(n_clusters=50, random_state=42, batch_size = config.batch_size ,n_init=1) for _ in range(33**2)]

        for inputs, _ in tqdm(MI_dataset_loader , desc=f" {config.factor_name}", leave=False):
                
                if inputs is None:
                        continue

                with torch.no_grad():
                    inputs = Variable(inputs.to(device))
                    outputs, layers_output, _ = model(inputs)

                _, _, H, W = layers_output[0].shape

                for h in range(H):
                    for w in range(W):
                        pixel_activities = layers_output[layer_idx][:,:,h,w].cpu().numpy()  # (batch, channels)
                        pixel_model = pixel_by_kmeans[h*W+w]
                        pixel_model.partial_fit(pixel_activities)

        del layers_output, outputs, inputs, MI_dataset_loader
        collect()

        gt_masks = []
        pred_masks = []
        
        config.MI = False
        dataset = get_dataset(config)
        # 예측용 DataLoader 생성 (shuffle=False, 동일한 순서 보장)
        MI_dataset_loader = mi_seg_data_loader(config, dataset, shuffle=False) 

        vq_labels = [[] for _ in range(H*W)]

        for inputs, targets in tqdm(MI_dataset_loader, desc=f" {config.factor_name}", leave=False):

            if inputs is None:
                continue

            with torch.no_grad():        
                inputs = Variable(inputs.to(device))
                outputs, layers_output, _ = model(inputs)
                        
            _, _, H, W = layers_output[0].shape
            
            if layer_idx == 0:
                _, pred = torch.max(outputs, 1)
                preds = pred.cpu().numpy().astype(np.uint8)
                mask = targets.numpy().astype(np.uint8)
                pred_masks.append(preds)
                gt_masks.append(mask)

            for h in range(H):
                for w in range(W):
                    pixel_activities = layers_output[layer_idx][:,:,h,w].cpu().numpy()  # (batch, channels)
                    pixel_model = pixel_by_kmeans[h*W+w]
                    vq_label = pixel_model.predict(pixel_activities)
                    vq_labels[h*W+w].append(vq_label)

        vq_labels = np.array([np.concatenate(vq_labels[i]) for i in range(H*W)])  # 각 픽셀별로 합치기
        vq_labels = vq_labels.T  # (total_samples, 33**2)
        vq_labels = vq_labels.reshape(-1, H, W)  # (total_samples, 33, 33)

        with open(config.output_folder + f'/layer_{layer_idx}.pkl', 'wb') as f:
            pickle.dump(vq_labels, f)
            print(f"save layer_{layer_idx} complete")
    
        if layer_idx == 0:
            with open(config.output_folder + f'/gt_labels.pkl', 'wb') as f:
                pickle.dump(np.concatenate(gt_masks, axis=0), f)

            with open(config.output_folder + f'/pred_labels.pkl', 'wb') as f:
                pickle.dump(np.concatenate(pred_masks, axis=0), f)

        del vq_labels, pixel_activities, layers_output, outputs, inputs
        collect()
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/hail/pan/GCN/PIGNet/config_seg_MI.yaml")
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

    # 가공 조건 및 이름 매핑
    process_dict = {
        "zoom": [0.1, np.sqrt(0.1), 0.5, np.sqrt(0.5), 1, 1.5, np.sqrt(2.75), 2],
        "overlap": [0, 0.1, 0.2, 0.3, 0.5],
        "repeat": [1, 3, 6, 9, 12]
    }

    zoom_name_map = {0.1: "0.1", np.sqrt(0.1): "0.3", 0.5: "0.5", np.sqrt(0.5): "0.7", 
                     1: "1", 1.5: "1.5", np.sqrt(2.75): "1.75", 2: "2.0"}
    
    if config.backbone == "resnet50":
        num=50
    elif config.backbone == "resnet101":
        num=101
    
    model_path = f"/home/hail/pan/GCN/PIGNet/model_{num}/{config.model_number}/segmentation/{config.dataset}/{config.model_type}"
    model_files = sorted(os.listdir(model_path))

    for model_file in model_files:
        m_name = re.search(fr"(.*?)_{config.backbone}", model_file)
        model_key = m_name.group(1) if m_name else ("Mask2Former" if "Mask2Former" in model_file else "unknown")

        # PIGNet, Mask2Former 제외 처리
        if model_key in ["PIGNet", "Mask2Former"]:
            continue

        print(f"\n>>> Processing Model: {model_key}")
        
        for p_key, f_list in process_dict.items():
            for f_val in f_list:
                f_name = zoom_name_map.get(f_val, str(f_val)) if p_key == "zoom" else str(f_val)
                
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

                # 인퍼런스용 독립 설정
                iter_config = argparse.Namespace(**vars(config))
                iter_config.model = model_key
                iter_config.factor = f_val
                iter_config.factor_name = f_name
                iter_config.infer_params.process_type = p_key
                iter_config.output_folder = output_folder

                main(iter_config, model_file, model_path)
                