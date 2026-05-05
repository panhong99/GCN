import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import re
import yaml
import warnings
import pickle
import copy
from tqdm.auto import trange
from sklearn.cluster import MiniBatchKMeans
from make_segmentation_dataset import get_dataset
from make_segmentation_model import get_model
from torch.autograd import Variable
from sklearn.cluster import KMeans
import random
import pickle
import cv2
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

def resize_gt(gt_masks, target_size=33, config_dataset=None):
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
        if config_dataset == "pascal":
            resized = np.where(resized == 255, 255, resized)
        else:
            resized = np.where(resized == 255, 255, resized)
        gt_resized[b] = resized

    return gt_resized

def main(config, model_file, model_path):

    device = f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu"

    # if config.dataset == "cityscapes":
    #     invalid_cls = 255
    # else: # pascal
    #     invalid_cls = 255

    dataset = get_dataset(config)
    model = get_model(config, dataset)

    checkpoint = torch.load(os.path.join(model_path, model_file), map_location=device)

    # if config.model == "ASPP" or config.model == "PIGNet_GSPonly":
    # if config.model == "ASPP":
    #     state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    # else:
    #     state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}

    state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}

    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    set_seed(42)
    dataset = get_dataset(config)

    # ★ Feature map size 결정
    H_size = W_size = 33

    print(f"\n{'='*60}")
    print(f"Training KMeans models for all layers & pixels (batch-wise)")
    print(f"{'='*60}")
    print(f"Grid size: {H_size} x {W_size}")
    print(f"Total KMeans models: {5 * H_size * W_size} (5 layers × {H_size} × {W_size})")

    # ★ PIXEL별 KMeans: 각 (layer_idx, h, w)마다 독립적인 모델
    vq_models = {}
    for layer_idx in range(5):
        for h in range(H_size):
            for w in range(W_size):
                key = (layer_idx, h, w)
                vq_models[key] = MiniBatchKMeans(
                    n_clusters=50,
                    random_state=42,
                    n_init=1,
                    batch_size=50,
                    verbose=0
                )

    # 두 번째 pass: 배치별로 바로 fit
    print(f"\nFitting KMeans models with batch data...")
    loader = mi_seg_data_loader(config, dataset, shuffle=False)
    loader_list = list(loader)

    for idx in trange(len(loader_list), desc="Training KMeans", leave=False):
        inputs, targets = loader_list[idx]
        if inputs is None:
            continue

        with torch.no_grad():
            inputs = Variable(inputs.to(device))
            outputs, layers_output = model(inputs)

        targets_np = targets.numpy().astype(np.uint8)

        if config.model == "Mask2Former":
            # 특정 인덱스만 선택 [0, 2, 5, 8, 9]
            layers_output = [layers_output[i] for i in [0, 2, 5, 8, 9]]
            # 각 layer를 (bs, Q, H, W) → (bs, Q, 33, 33)으로 리사이즈
            layers_output_resized = []
            for layer in layers_output:
                resized = torch.nn.functional.interpolate(
                    layer, size=(33, 33), mode='bilinear', align_corners=False
                )
                layers_output_resized.append(resized)
            layers_output = layers_output_resized

        # ★ PIXEL별 처리: 각 픽셀마다 해당 모델에만 fit (전체 데이터 사용)
        for layer_idx in range(len(layers_output)):
            layer_data = layers_output[layer_idx].cpu().numpy()
            B, C, H, W = layer_data.shape

            # Binary 처리: 각 채널값이 0보다 크면 1, 아니면 0
            if config.binary == True:
                layer_data = (layer_data > 0).astype(np.float32)  # (B, C, H, W)

            # ★ 각 픽셀 (h, w)에 대해
            for h in range(H):
                for w in range(W):
                    # 전체 배치 데이터 사용: (B, C)
                    pixel_features = layer_data[:, :, h, w]  # (B, C)

                    # 해당 픽셀 전용 KMeans 모델에만 fit
                    key = (layer_idx, h, w)
                    vq_models[key].partial_fit(pixel_features)

        del outputs, layers_output, inputs
        collect()

    print(f"\n{'='*60}")
    print(f"Predicting VQ labels for all layers & pixels")
    print(f"{'='*60}")

    gt_masks = []
    pred_masks = []
    all_vq_preds = {layer_idx: [] for layer_idx in range(5)}
    all_vq_valid_masks = {layer_idx: [] for layer_idx in range(5)}

    config.MI = False
    dataset = get_dataset(config)
    infer_loader = mi_seg_data_loader(config, dataset, shuffle=False)
    infer_loader_list = list(infer_loader)

    for idx in trange(len(infer_loader_list), desc="Predicting", leave=False):
        inputs, targets = infer_loader_list[idx]
        if inputs is None:
            continue

        with torch.no_grad():
            inputs = Variable(inputs.to(device))
            outputs, layers_output = model(inputs)

        targets_np = targets.numpy().astype(np.uint8)

        # Mask2Former 레이어 선택 및 리사이즈
        if config.model == "Mask2Former":
            # 특정 인덱스만 선택 [0, 2, 5, 8, 9]
            layers_output = [layers_output[i] for i in [0, 2, 5, 8, 9]]
            # 각 layer를 (bs, Q, H, W) → (bs, Q, 33, 33)으로 리사이즈
            layers_output_resized = []
            for layer in layers_output:
                resized = torch.nn.functional.interpolate(
                    layer, size=(33, 33), mode='bilinear', align_corners=False
                )
                layers_output_resized.append(resized)
            layers_output = layers_output_resized

        # GT 라벨 저장
        valid_mask_resized = resize_gt(targets_np, target_size=33, config_dataset=config.dataset)
        gt_masks.append(valid_mask_resized)

        # ★ PIXEL별 Prediction: 각 픽셀마다 해당 모델로 predict (전체 데이터 사용)
        for layer_idx in range(len(layers_output)):
            layer_data = layers_output[layer_idx].cpu().numpy()  # (batch, C, H, W)
            B, C, H, W = layer_data.shape

            # Binary 처리: 각 채널값이 0보다 크면 1, 아니면 0
            if config.binary == True:
                layer_data = (layer_data > 0).astype(np.float32)  # (B, C, H, W)

            # 결과를 담을 배열
            vq_pred_mask = np.zeros((B, H, W), dtype=np.int32)

            # ★ 각 픽셀 (h, w)에 대해
            for h in range(H):
                for w in range(W):
                    # 전체 배치 데이터 사용
                    pixel_features = layer_data[:, :, h, w]  # (B, C)

                    # 해당 픽셀 전용 모델로 predict
                    key = (layer_idx, h, w)
                    pixel_pred = vq_models[key].predict(pixel_features)  # (B,)

                    # 예측값 저장
                    vq_pred_mask[:, h, w] = pixel_pred

            all_vq_preds[layer_idx].append(vq_pred_mask)

        del outputs, layers_output, inputs
        collect()

    # ===== Step 5: 결과 저장 =====
    print(f"\nSaving results...")

    for layer_idx in range(5):
        vq_labels = np.concatenate(all_vq_preds[layer_idx], axis=0)

        with open(config.output_folder + f'/layer_{layer_idx}.pkl', 'wb') as f:
            pickle.dump(vq_labels, f)

    # GT 라벨 저장 (invalid 값을 -1로 치환)
    gt_final = np.concatenate(gt_masks, axis=0).astype(np.int32)

    gt_final = np.where(gt_final == 255, -1, gt_final)

    # if config.dataset == "cityscapes":
    #     gt_final = np.where(gt_final == 255, -1, gt_final)
    # else:  # pascal
    #     gt_final = np.where((gt_final == 0) | (gt_final == 255), -1, gt_final)

    with open(config.output_folder + f'/gt_labels.pkl', 'wb') as f:
        pickle.dump(gt_final, f)

    print(f"Results saved to {config.output_folder}")

    del vq_models, all_vq_preds, gt_masks, pred_masks
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
        if model_key in ["PIGNet"]:
            continue

        print(f"\n>>> Processing Model: {model_key}")

        for p_key, f_list in process_dict.items():
            for f_val in f_list:
                f_name = zoom_name_map.get(f_val, str(f_val)) if p_key == "zoom" else str(f_val)

                output_folder = os.path.join(
                    "/home/hail/pan/HDD/MI_dataset",
                    "pixel_dataset",  # ★ layer_dataset에서 pixel_dataset으로 변경
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
                iter_config.crop_size = 512 if iter_config.model == "Mask2Former" else 513
                iter_config.batch_size = 50
                main(iter_config, model_file, model_path)
