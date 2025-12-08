import torch
import os
import yaml
import argparse

def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    def dict_to_namespace(d):
        namespace = argparse.Namespace()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(namespace, k, dict_to_namespace(v))
            else:
                setattr(namespace, k, v)
        return namespace
    return dict_to_namespace(config_dict)

def get_checkpoint_path(config, model_number):
    # config의 model_number를 임시로 변경
    original_model_number = config.model_number
    config.model_number = model_number
    path = f'/home/hail/pan/GCN/PIGNet/model/{config.model_number}/classification/{config.dataset}/{config.model_type}'
    config.model_number = original_model_number  # 복원
    return path

def load_state_dict(checkpoint_path, model_filename, device):
    checkpoint = torch.load(os.path.join(checkpoint_path, model_filename), map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    return state_dict

def compare_state_dicts(state_dict_1, state_dict_2):
    keys_1 = set(state_dict_1.keys())
    keys_2 = set(state_dict_2.keys())
    
    only_in_1 = keys_1 - keys_2
    only_in_2 = keys_2 - keys_1
    common = keys_1 & keys_2
    
    print(f"Keys only in model 1: {len(only_in_1)}")
    for k in sorted(only_in_1):
        print(f"  {k}: {state_dict_1[k].shape}")
    
    print(f"\nKeys only in model 2: {len(only_in_2)}")
    for k in sorted(only_in_2):
        print(f"  {k}: {state_dict_2[k].shape}")
    
    print(f"\nCommon keys: {len(common)}")
    shape_diffs = []
    for k in sorted(common):
        shape_1 = state_dict_1[k].shape
        shape_2 = state_dict_2[k].shape
        if shape_1 != shape_2:
            shape_diffs.append((k, shape_1, shape_2))
            print(f"  {k}: model1 {shape_1} vs model2 {shape_2}")
    
    if not shape_diffs:
        print("  All common keys have matching shapes.")

if __name__ == "__main__":
    config_path = "/home/hail/pan/GCN/PIGNet/config_classification.yaml"
    config = load_config(config_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model 1 (e.g., model_number 1)
    model_number_1 = 1
    path_1 = get_checkpoint_path(config, model_number_1)
    print(f"Path for model {model_number_1}: {path_1}")
    
    # Model 2 (e.g., model_number 3)
    model_number_2 = 3
    path_2 = get_checkpoint_path(config, model_number_2)
    print(f"Path for model {model_number_2}: {path_2}")
    
    # Assume model_filename is the same or find one
    try:
        model_files_1 = os.listdir(path_1)
        model_files_2 = os.listdir(path_2)
        # For simplicity, take the first file or a specific one
        model_filename = model_files_1[0] if model_files_1 else None
        if model_filename not in model_files_2:
            print(f"Model file {model_filename} not found in path_2")
            exit()
    except FileNotFoundError:
        print("Paths not found")
        exit()
    
    print(f"Using model file: {model_filename}")
    
    state_dict_1 = load_state_dict(path_1, model_filename, device)
    state_dict_2 = load_state_dict(path_2, model_filename, device)
    
    print("\n--- State Dict Comparison ---")
    compare_state_dicts(state_dict_1, state_dict_2)