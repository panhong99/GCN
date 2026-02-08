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
import cv2

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

def resize_gt_fast(gt_masks, target_size=33):
    """
    빠른 버전 - cv2.resize 사용
    gt_masks: (B, H, W)
    return: (B, target_size, target_size)
    """
    B, H, W = gt_masks.shape
    gt_resized = np.zeros((B, target_size, target_size), dtype=np.uint8)
    
    for b in range(B):
        # cv2.resize는 (W, H) 순서로 받음
        resized = cv2.resize(gt_masks[b].astype(np.uint8), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        # 255는 invalid로 처리 (0으로 변경)
        resized = np.where(resized == 255, 0, resized)
        gt_resized[b] = resized
    
    return gt_resized

def main(config, model_file, model_path):

    device = f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu"

    config.batch_size = 128
    dataset = get_dataset(config)
    model = get_model(config, dataset)

    checkpoint = torch.load(os.path.join(model_path, model_file), map_location=device)

    if config.model == "PIGNet_GSPonly":
        layer_num = 5
        state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}

    else: # ASPP
        layer_num = 5
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}

    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    n_clusters = 50
    print(f"KMeans clusters: {n_clusters} (fixed)")

    for layer_idx in range(5):
        print(f"\n{'='*60}")
        print(f"Processing Layer {layer_idx}")
        print(f"{'='*60}")

        set_seed(42)
        dataset = get_dataset(config)

        # ===== PASS 1: 모든 배치에서 valid 데이터 수집 =====
        print(f"[PASS 1] Collecting valid data for layer {layer_idx}...")
        all_pixel_data = [[] for _ in range(33*33)]
        gt_masks = []
        
        MI_dataset_loader = mi_seg_data_loader(config, dataset, shuffle=True)

        for inputs, targets in tqdm(MI_dataset_loader, desc=f"Pass 1: {config.factor_name}", leave=False):
            if inputs is None:
                continue

            with torch.no_grad():
                inputs = Variable(inputs.to(device))
                outputs, layers_output = model(inputs)

            _, _, H, W = layers_output[0].shape
            targets_np = targets.numpy().astype(np.uint8)

            # GT mask 수집 (layer 0에서만)
            if layer_idx == 0:
                gt_masks.append(targets_np)

            # Binary 처리
            layer_data = layers_output[layer_idx].cpu().numpy()
            if config.binary == True:
                layer_data = (layer_data > 0).astype(np.float32)

            # GT 유효성 마스크
            valid_mask_resized = resize_gt_fast(targets_np, target_size=H)
            gt_mask = (valid_mask_resized != 0)

            # 각 픽셀별로 valid 데이터 수집
            for h in range(H):
                for w in range(W):
                    pixel_mask = gt_mask[:, h, w]
                    pixel_activities = layer_data[pixel_mask, :, h, w]
                    
                    if pixel_activities.shape[0] > 50:
                        all_pixel_data[h*W+w].append(pixel_activities)

        del layers_output, outputs, inputs, MI_dataset_loader
        collect()

        # numpy 배열로 변환
        print(f"[PASS 1] Converting to numpy arrays...")
        for idx in range(33*33):
            if len(all_pixel_data[idx]) > 0:
                all_pixel_data[idx] = np.vstack(all_pixel_data[idx])
            else:
                all_pixel_data[idx] = None

        # ===== PASS 2: 각 픽셀별 KMeans fit_predict =====
        print(f"[PASS 2] Fitting KMeans models for layer {layer_idx}...")
        pixel_by_kmeans = [None] * (33*33)
        vq_labels = [[] for _ in range(33*33)]
        
        for idx in range(33*33):
            if all_pixel_data[idx] is not None and len(all_pixel_data[idx]) >= n_clusters:
                model_km = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
                labels = model_km.fit_predict(all_pixel_data[idx])
                vq_labels[idx] = labels
                pixel_by_kmeans[idx] = model_km

        del all_pixel_data
        collect()

        # 결과 저장
        vq_labels = np.array([np.array(vq_labels[i]) if len(vq_labels[i]) > 0 else np.array([]) for i in range(33*33)])
        vq_labels = vq_labels.T
        vq_labels = vq_labels.reshape(-1, H, W)

        with open(config.output_folder + f'/layer_{layer_idx}.pkl', 'wb') as f:
            pickle.dump(vq_labels, f)
            print(f"save layer_{layer_idx} complete")

        if layer_idx == 0:
            with open(config.output_folder + f'/gt_labels.pkl', 'wb') as f:
                pickle.dump(np.concatenate(gt_masks, axis=0), f)

        del vq_labels, pixel_by_kmeans, MI_dataset_loader
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
    # process_dict = {
    #     "zoom": [0.1, np.sqrt(0.1), 0.5, np.sqrt(0.5), 1, 1.5, np.sqrt(2.75), 2],
    #     "overlap": [0, 0.1, 0.2, 0.3, 0.5],
    #     "repeat": [1, 3, 6, 9, 12]
    # }

    process_dict = {
        "zoom": [1],
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
                    "pixel_dataset",
                    config.backbone, 
                    config.model_type, 
                    model_key, 
                    p_key,
                    f_name,
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
                