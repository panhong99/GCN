import pickle
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score

if __name__ == "__main__":
    
    path = "/home/hail/pan/HDD/MI_dataset/CIFAR-10/resnet101/pretrained/zoom/PIGNet_GSPonly_classification/1/1.0.pkl"
    
    data = pickle.load(open(path, "rb"))
    
    activity_layer_0 = data['activity_layer_0']
    activity_layer_1 = data['activity_layer_1']
    activity_layer_2 = data['activity_layer_2']
    activity_layer_3 = data['activity_layer_3']
    activity_layer_4 = data['activity_layer_4']
    
    # 모든 layer를 list로 정리
    layers_activities = [activity_layer_0, activity_layer_1, activity_layer_2, 
                         activity_layer_3, activity_layer_4]
    
    # 설정
    n_vq = 100
    
    # 각 layer를 VQ (Vector Quantization)
    vq_layers = []
    
    for idx, layer_activity in enumerate(layers_activities):
        # torch tensor를 numpy로 변환 (필요시)
        if isinstance(layer_activity, torch.Tensor):
            layer_activity = layer_activity.cpu().numpy()
        
        print(f"\nLayer {idx} 처리 중...")
        print(f"Original shape: {layer_activity.shape}")
        
        # (batch, channels, height, width) → (batch, height, width, channels)
        layer_activity = np.transpose(layer_activity, (0, 2, 3, 1))
        print(f"After permute shape: {layer_activity.shape}")
        
        # (batch*height*width, channels)로 reshape
        batch_size, height, width, channels = layer_activity.shape
        layer_activity_flat = layer_activity.reshape(-1, channels)
        print(f"After reshape shape: {layer_activity_flat.shape}")
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_vq, random_state=42, n_init=10)
        vq_labels = kmeans.fit_predict(layer_activity_flat)
        
        # 원래 spatial shape로 복원
        vq_labels = vq_labels.reshape(batch_size, height, width)
        print(f"VQ labels shape: {vq_labels.shape}")
        print(f"Unique clusters: {np.unique(vq_labels)}")
        
        vq_layers.append(vq_labels)
    
    print("\n모든 layer 처리 완료!")
    print(f"VQ layers 개수: {len(vq_layers)}")
    for i, vq_layer in enumerate(vq_layers):
        print(f"VQ Layer {i} shape: {vq_layer.shape}") 