import pickle
import numpy as np

# Pickle 파일 저장
def save_pkl(filepath, data):
    """Pickle 파일로 데이터 저장"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved: {filepath}")

# Pickle 파일 로드
def load_pkl(filepath):
    """Pickle 파일에서 데이터 로드"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded: {filepath}")
    return data

# 사용 예시
if __name__ == "__main__":
    # 저장 예시
    # vq_labels = np.array(...)
    # save_pkl(config.output_folder + f'/layer_{layer_idx}.pkl', vq_labels)
    
    # 로드 예시
    # layer_data = load_pkl(config.output_folder + f'/layer_{layer_idx}.pkl')
    
    # 실제 사용 (경로 수정 필요)
    filepath = "/home/hail/pan/HDD/MI_dataset/pascal/total_mask_dataset/resnet101/pretrained/PIGNet_GSPonly/zoom/1/mi_analysis_cache.pkl"
    data = load_pkl(filepath)
    print(f"Data shape: {data.shape}")
    print(f"Data type: {type(data)}")   