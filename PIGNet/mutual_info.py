import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main(pth_path):
    print(f"Loading: {pth_path}\n")
    data = torch.load(pth_path, map_location='cpu')
    
    print(f"Top-level type: {type(data)}")
    print("="*60)
    
    # 전체 데이터를 모으기
    all_values = []
    layer_data = {}
    
    if isinstance(data, (list, tuple)):
        print(f"{type(data).__name__} with {len(data)} elements:")
        for i, item in enumerate(data):
            if isinstance(item, torch.Tensor):
                arr = item.detach().cpu().numpy()
                all_values.append(arr.ravel())  # ravel == flatten
                layer_data[f"element_{i}"] = arr
                print(f"\n[{i}] Tensor shape={arr.shape}")
            elif isinstance(item, (list, tuple)):
                print(f"\n[{i}] {type(item).__name__} len={len(item)}")
            else:
                print(f"[{i}] {type(item)}")
    
    if not all_values:
        print("No tensor data found.")
        return
    
    # 전체 데이터 concat
    all_concat = np.concatenate(all_values)
    print(f"\nTotal elements across all layers: {len(all_concat)}")

    # 1) 전체 데이터 KDE plot (기본 bin=50)
    print("\nSaving global KDE plot...")
    plt.figure(figsize=(8, 6))
    sns.kdeplot(all_concat, fill=True, color='blue', alpha=0.7)
    plt.title('Global KDE Plot')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True)
    out_path = "global_kde.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    # 1.5) 전체 데이터 히스토그램 plot
    print("\nSaving global histogram...")
    plt.figure(figsize=(8, 6))
    plt.hist(all_concat, bins=1000, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Global Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    out_path = "global_hist.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    # 2) 개별 layer KDE plot (각 layer의 실제 min/max 범위 사용)
    print("\nSaving individual layer KDE plots...")
    for name, arr in layer_data.items():
        plt.figure(figsize=(8, 6))
        sns.kdeplot(arr.ravel(), fill=True, color='blue', alpha=0.7)
        plt.title(f'KDE Plot of {name}')
        plt.xlabel('Value')
        plt.ylabel('Density')

        # 각 layer의 실제 min/max 값으로 x축 범위 설정
        layer_min = arr.min()
        layer_max = arr.max()
        plt.xlim(left=layer_min, right=layer_max)

        plt.grid(True)
        out_path = f"{name}_kde.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path} (range: {layer_min:.6f} ~ {layer_max:.6f})")

    # 3) 개별 layer 히스토그램 plot
    print("\nSaving individual layer histogram plots...")
    for name, arr in layer_data.items():
        plt.figure(figsize=(8, 6))
        plt.hist(arr.ravel(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Histogram of {name}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        # 각 layer의 실제 min/max 값으로 x축 범위 설정
        layer_min = arr.min()
        layer_max = arr.max()
        plt.xlim(left=layer_min, right=layer_max)

        plt.grid(True)
        out_path = f"{name}_hist.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path} (range: {layer_min:.6f} ~ {layer_max:.6f})")

    print("Done.")

if __name__ == '__main__':
    # 디버깅 모드: 디버거가 연결되었거나 DEBUG=1이면 기본 경로 사용
    if (hasattr(sys, 'gettrace') and sys.gettrace()) or os.getenv('DEBUG', '') == '1':
        print('Debug mode detected -> using default path')
        pth_path = '/home/hail/pan/GCN/PIGNet/layers_activity/pascal/PIGNet/zoom/1/layers_activity/model_layers.pth'  # 기본 경로 (실제 파일 경로로 변경)
    else:
        if len(sys.argv) < 2:
            print("사용법: python mutual_info.py <.pth 파일 경로>")
            print("예: python mutual_info.py /path/to/activations.pth")
            sys.exit(1)
        pth_path = sys.argv[1]
    
    if not os.path.exists(pth_path):
        print(f"파일을 찾을 수 없음: {pth_path}")
        sys.exit(1)
    
    main(pth_path)