import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd  # 추가
# from sklearn.preprocessing import KBinsDiscretizer  # 제거: 직접 quantile binning 구현

def discretize_layer_data(layer_data, num_bins=10, strategy='quantile'):
    """
    각 layer의 activation 데이터를 quantile 기반으로 discretize (CDF 기반).
    strategy는 무시하고 quantile binning만 사용 (CDF 균등 분배).
    
    :param layer_data: {'layer_name': np.array, ...}
    :param num_bins: bin 개수 (default: 10)
    :param strategy: 무시 (호환성 유지)
    :return: discretized_layer_data: {'layer_name': {'discretized': np.array, 'total_samples': int, 'bin_edges': list, 'bin_counts': list, 'cdf': list}, ...}
    """
    discretized_layer_data = {}
    for name, arr in layer_data.items():
        flat_arr = arr.ravel()
        total_samples = len(flat_arr)
        
        # Quantile 기반 bin edges 계산 (CDF 균등)
        quantiles = np.linspace(0, 1, num_bins + 1)
        bin_edges = np.quantile(flat_arr, quantiles).tolist()
        
        # Discretize: 각 값에 bin index 할당 (0 ~ num_bins-1)
        discretized_flat = np.digitize(flat_arr, bin_edges[:-1]) - 1
        discretized_flat = np.clip(discretized_flat, 0, num_bins - 1)  # 범위 제한
        
        # Bin counts 계산
        bin_counts, _ = np.histogram(flat_arr, bins=bin_edges)
        bin_counts = bin_counts.tolist()
        
        # CDF 계산 (누적 비율)
        cumulative_counts = np.cumsum(bin_counts).astype(np.float64)
        cdf = (cumulative_counts / total_samples).tolist()
        
        # 원래 shape로 reshape
        discretized_reshaped = discretized_flat.reshape(arr.shape)
        
        discretized_layer_data[name] = {
            'discretized': discretized_reshaped,
            'total_samples': total_samples,
            'bin_edges': bin_edges,
            'bin_counts': bin_counts,
            'cdf': cdf
        }
        
        print(f"Discretized {name}: {num_bins} bins, total_samples={total_samples}, bin_counts={bin_counts}")
    
    return discretized_layer_data

def calculate_mutual_info(discretized_layer_data):
    """
    Discretized layer data를 사용해서 layer 간 mutual information을 계산합니다.
    
    :param discretized_layer_data: {'layer_name': discretized_np.array, ...}
    :return: mi_results: {'layer1_vs_layer2': mi_score, ...}
    """
    from sklearn.metrics import mutual_info_score
    import itertools
    
    mi_results = {}
    layer_names = list(discretized_layer_data.keys())
    
    for layer1, layer2 in itertools.combinations(layer_names, 2):
        data1 = discretized_layer_data[layer1].flatten()
        data2 = discretized_layer_data[layer2].flatten()
        
        # 데이터가 너무 크면 샘플링
        max_samples = 10000
        if len(data1) > max_samples:
            indices = np.random.choice(len(data1), max_samples, replace=False)
            data1 = data1[indices]
            data2 = data2[indices]
        
        mi_score = mutual_info_score(data1, data2)
        mi_results[f"{layer1}_vs_{layer2}"] = mi_score
        print(f"MI between {layer1} and {layer2}: {mi_score:.4f}")
    
    return mi_results

def plot_and_save_distributions(layer_data, prefix, out_dir):
    """
    주어진 layer 데이터에 대해 전체 및 개별 히스토그램/CDF를 생성하고 저장합니다.
    각 bin의 누적 개수와 %를 CSV로 정리하여 저장합니다.
    
    :param layer_data: {'layer_name': np.array, ...}
    :param prefix: 파일명/타이틀 접두사 (예: 'PIGNet_SPP', 'ASPP', 'GSP_Only')
    :param out_dir: 저장 디렉토리
    :return: 전체 히스토그램 CSV 경로
    """
    if not layer_data:
        print(f"[{prefix}] No data to plot.")
        return None

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n--- Plotting for [{prefix}] ---")

    # 1) 전체 히스토그램
    all_concat = np.concatenate([arr.ravel() for arr in layer_data.values()])
    fig_hist, ax_hist = plt.subplots(figsize=(12, 8))
    counts, bins, _ = ax_hist.hist(all_concat, bins=3000, alpha=0.7, color='blue', edgecolor='black')
    ax_hist.set_title(f'Global Histogram for {prefix} (3000 bins)')
    ax_hist.set_xlabel('Value')
    ax_hist.set_ylabel('Frequency')
    ax_hist.grid(True)
    out_path_hist = os.path.join(out_dir, f"{prefix}_global_hist.png")
    fig_hist.savefig(out_path_hist)
    plt.close(fig_hist)
    print(f"Saved: {out_path_hist}")

    # 2) 전체 CDF
    fig_cdf, ax_cdf = plt.subplots(figsize=(12, 6))
    cdf = np.cumsum(counts).astype(np.float64)
    if cdf.size > 0 and cdf[-1] > 0:
        cdf /= cdf[-1]
    ax_cdf.plot(bins[1:], cdf, color='red', linestyle='-', linewidth=2)
    ax_cdf.set_title(f'Global CDF for {prefix}')
    ax_cdf.set_xlabel('Value')
    ax_cdf.set_ylabel('Cumulative Fraction')
    ax_cdf.set_ylim(0, 1)
    ax_cdf.grid(True)
    out_path_cdf = os.path.join(out_dir, f"{prefix}_global_cdf.png")
    fig_cdf.savefig(out_path_cdf)
    plt.close(fig_cdf)
    print(f"Saved: {out_path_cdf}")

    # 3) 전체 히스토그램 CSV: 각 bin의 정보
    total_samples = len(all_concat)
    cumulative_counts = np.cumsum(counts)
    cumulative_percent = (cumulative_counts / total_samples) * 100

    bin_info = []
    for i in range(len(bins) - 1):
        bin_info.append({
            'bin_start': bins[i],
            'bin_end': bins[i+1],
            'count': int(counts[i]),
            'cumulative_count': int(cumulative_counts[i]),
            'cumulative_percent': cumulative_percent[i]
        })

    df_bins = pd.DataFrame(bin_info)
    csv_path = os.path.join(out_dir, f"{prefix}_global_bins.csv")
    df_bins.to_csv(csv_path, index=False)
    print(f"Saved bin info CSV: {csv_path}")

    # 4) 개별 레이어 히스토그램/CDF 및 CSV
    for name, arr in layer_data.items():
        fig_hist_ind, ax_hist_ind = plt.subplots(figsize=(12, 8))
        counts_ind, bins_ind, _ = ax_hist_ind.hist(
            arr.ravel(),
            bins=3000,
            alpha=0.7,
            color='blue',
            edgecolor='black'
        )
        ax_hist_ind.set_title(f'Histogram for {prefix} - {name} (3000 bins)')
        ax_hist_ind.set_xlabel('Value')
        ax_hist_ind.set_ylabel('Frequency')
        ax_hist_ind.grid(True)
        out_path_hist_ind = os.path.join(out_dir, f"{prefix}_{name}_hist.png")
        fig_hist_ind.savefig(out_path_hist_ind)
        plt.close(fig_hist_ind)
        print(f"Saved: {out_path_hist_ind}")

        # CDF
        fig_cdf_ind, ax_cdf_ind = plt.subplots(figsize=(12, 6))
        cdf_ind = np.cumsum(counts_ind).astype(np.float64)
        if cdf_ind.size > 0 and cdf_ind[-1] > 0:
            cdf_ind /= cdf_ind[-1]
        ax_cdf_ind.plot(bins_ind[1:], cdf_ind, color='purple', linestyle='-', linewidth=2)
        ax_cdf_ind.set_title(f'CDF for {prefix} - {name}')
        ax_cdf_ind.set_xlabel('Value')
        ax_cdf_ind.set_ylabel('Cumulative Fraction')
        ax_cdf_ind.set_ylim(0, 1)
        ax_cdf_ind.grid(True)
        out_path_cdf_ind = os.path.join(out_dir, f"{prefix}_{name}_cdf.png")
        fig_cdf_ind.savefig(out_path_cdf_ind)
        plt.close(fig_cdf_ind)
        print(f"Saved: {out_path_cdf_ind}")

        # 개별 레이어 CSV: 각 bin의 정보
        total_samples_ind = len(arr.ravel())
        cumulative_counts_ind = np.cumsum(counts_ind)
        cumulative_percent_ind = (cumulative_counts_ind / total_samples_ind) * 100

        bin_info_ind = []
        for i in range(len(bins_ind) - 1):
            bin_info_ind.append({
                'bin_start': bins_ind[i],
                'bin_end': bins_ind[i+1],
                'count': int(counts_ind[i]),
                'cumulative_count': int(cumulative_counts_ind[i]),
                'cumulative_percent': cumulative_percent_ind[i]
            })

        df_bins_ind = pd.DataFrame(bin_info_ind)
        csv_path_ind = os.path.join(out_dir, f"{prefix}_{name}_bins.csv")
        df_bins_ind.to_csv(csv_path_ind, index=False)
        print(f"Saved bin info CSV for {name}: {csv_path_ind}")

    return csv_path  # 전체 히스토그램 CSV 경로 return

def process_activity_file(model_name, pth_path, num_bins=10, strategy='uniform'):
    """
    모델 이름과 pth 경로를 기반으로 activity를 분석하고 플롯을 저장합니다.
    Discretization을 적용합니다.
    """
    print(f"\n{'='*20} Processing: {model_name} ({os.path.basename(pth_path)}) {'='*20}")
    
    if not os.path.exists(pth_path):
        print(f"File not found: {pth_path}", file=sys.stderr)
        return

    data = torch.load(pth_path, map_location='cpu')
    out_dir = f"activity_plots_{model_name}"

    if model_name.upper() == 'PIGNET':
        # 기대 형태: list/tuple, 최소 8개 텐서
        if not isinstance(data, (list, tuple)) or len(data) < 8:
            print(f"Error: PIGNet data is not a list of at least 8 tensors.", file=sys.stderr)
            return

        # SPP (기존 backbone이라 부르던 0~3 인덱스)
        spp_data = {f"spp_{i}": data[i].detach().cpu().numpy() for i in range(4)}
        spp_discretized = discretize_layer_data(spp_data, num_bins, strategy)
        spp_discretized_arrays = {name: info['discretized'] for name, info in spp_discretized.items()}
        plot_and_save_distributions(spp_discretized_arrays, f"{model_name}_SPP", out_dir)
        
        # SPP MI 계산
        spp_mi = calculate_mutual_info(spp_discretized_arrays)
        spp_mi_df = pd.DataFrame(list(spp_mi.items()), columns=['Layer_Pair', 'MI_Score'])
        spp_mi_csv = os.path.join(out_dir, f"{model_name}_SPP_mutual_info.csv")
        spp_mi_df.to_csv(spp_mi_csv, index=False)
        print(f"Saved SPP MI: {spp_mi_csv}")

        # GSP (4~7)
        gsp_data = {f"gsp_{i-4}": data[i].detach().cpu().numpy() for i in range(4, 8)}
        gsp_discretized = discretize_layer_data(gsp_data, num_bins, strategy)
        gsp_discretized_arrays = {name: info['discretized'] for name, info in gsp_discretized.items()}
        plot_and_save_distributions(gsp_discretized_arrays, f"{model_name}_GSP", out_dir)
        
        # GSP MI 계산
        gsp_mi = calculate_mutual_info(gsp_discretized_arrays)
        gsp_mi_df = pd.DataFrame(list(gsp_mi.items()), columns=['Layer_Pair', 'MI_Score'])
        gsp_mi_csv = os.path.join(out_dir, f"{model_name}_GSP_mutual_info.csv")
        gsp_mi_df.to_csv(gsp_mi_csv, index=False)
        print(f"Saved GSP MI: {gsp_mi_csv}")

    else:
        # ASPP, GSP_Only 등
        if isinstance(data, (list, tuple)):
            all_layer_data = {f"element_{i}": item.detach().cpu().numpy()
                              for i, item in enumerate(data) if isinstance(item, torch.Tensor)}
        elif isinstance(data, dict):
            all_layer_data = {name: item.detach().cpu().numpy()
                              for name, item in data.items() if isinstance(item, torch.Tensor)}
        else:
            print(f"Error: Unsupported data type '{type(data)}' for model {model_name}", file=sys.stderr)
            return

        all_discretized = discretize_layer_data(all_layer_data, num_bins, strategy)
        all_discretized_arrays = {name: info['discretized'] for name, info in all_discretized.items()}
        plot_and_save_distributions(all_discretized_arrays, model_name, out_dir)
        
        # MI 계산
        all_mi = calculate_mutual_info(all_discretized_arrays)
        all_mi_df = pd.DataFrame(list(all_mi.items()), columns=['Layer_Pair', 'MI_Score'])
        all_mi_csv = os.path.join(out_dir, f"{model_name}_mutual_info.csv")
        all_mi_df.to_csv(all_mi_csv, index=False)
        print(f"Saved MI: {all_mi_csv}")

    print(f"Finished processing for {model_name}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze layer activities from different models.")
    parser.add_argument('--pignet_path', type=str, default="/home/hail/pan/GCN/PIGNet/layers_activity/cityscape/PIGNet/zoom/1/layers_activity/model_layers.pth", help='Path to PIGNet activity file.')
    parser.add_argument('--aspp_path', type=str, default="/home/hail/pan/GCN/PIGNet/layers_activity/cityscape/ASPP/zoom/1/layers_activity/model_layers.pth", help='Path to ASPP activity file.')
    parser.add_argument('--gsp_only_path', type=str, default="/home/hail/pan/GCN/PIGNet/layers_activity/cityscape/PIGNet_GSPonly/zoom/1/layers_activity/model_layers.pth", help='Path to GSP_Only activity file.')
    parser.add_argument('--num_bins', type=int, default=10, help='Number of bins for discretization (default: 10).')
    parser.add_argument('--strategy', type=str, default='quantile', help='Binning strategy (ignored, always uses quantile for CDF-based discretization).')
    
    args = parser.parse_args()

    if args.pignet_path:
        process_activity_file('PIGNet', args.pignet_path, args.num_bins, args.strategy)
    if args.aspp_path:
        process_activity_file('ASPP', args.aspp_path, args.num_bins, args.strategy)
    if args.gsp_only_path:
        process_activity_file('GSP_Only', args.gsp_only_path, args.num_bins, args.strategy)

    if not any([args.pignet_path, args.aspp_path, args.gsp_only_path]):
        print("No activity file paths provided. Use --pignet_path, --aspp_path, or --gsp_only_path.")
        print("\nExample usage:")
        print("python mutual_info.py --pignet_path /path/to/pignet.pth --aspp_path /path/to/aspp.pth --num_bins 20")