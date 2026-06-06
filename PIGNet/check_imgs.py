import pickle
import os
from pathlib import Path

def load_pkl_file(pkl_path):
    """
    pkl 파일을 읽어서 내용 반환
    
    Args:
        pkl_path: pkl 파일 경로
        
    Returns:
        pkl 파일의 내용
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None


def load_all_pkl_files(directory):
    """
    디렉토리 내의 모든 pkl 파일 읽기
    
    Args:
        directory: pkl 파일들이 있는 디렉토리 경로
        
    Returns:
        {파일이름: 내용} 형태의 딕셔너리
    """
    results = {}
    pkl_files = Path(directory).glob('*.pkl')
    
    for pkl_file in pkl_files:
        print(f"Loading {pkl_file.name}...")
        data = load_pkl_file(str(pkl_file))
        if data is not None:
            results[pkl_file.name] = data
            print(f"  ✓ Loaded successfully")
    
    return results


def inspect_pkl_file(pkl_path):
    """
    pkl 파일의 구조와 정보 출력
    
    Args:
        pkl_path: pkl 파일 경로
    """
    data = load_pkl_file(pkl_path)
    if data is None:
        return
    
    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(pkl_path)}")
    print(f"{'='*60}")
    print(f"Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"  {key}: {type(value)} - shape: {getattr(value, 'shape', 'N/A')}")
    elif isinstance(data, (list, tuple)):
        print(f"Length: {len(data)}")
        print(f"First element type: {type(data[0]) if len(data) > 0 else 'N/A'}")
    else:
        print(f"Data shape: {getattr(data, 'shape', 'N/A')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    infer_output_path = "/home/hail/pan/GCN/PIGNet/infer_output/resenet101/pretrained"
    
    # 방법 1: 단일 pkl 파일 읽기
    pkl_file = os.path.join(infer_output_path, "pascal_ASPP_zoom_0.7_number_2.pkl")
    if os.path.exists(pkl_file):
        data = load_pkl_file(pkl_file)
        print(f"Loaded data type: {type(data)}")
    
    # 방법 2: pkl 파일 구조 확인
    if os.path.exists(pkl_file):
        inspect_pkl_file(pkl_file)
    
    # 방법 3: 디렉토리 내 모든 pkl 파일 읽기
    all_data = load_all_pkl_files(infer_output_path)
    print(f"\nTotal pkl files loaded: {len(all_data)}")
