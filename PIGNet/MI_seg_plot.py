"""
MI Plotting Module
모든 plot 함수 전담
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_scatter_same_diff(mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff, 
                           distance, layer_idx, model_name, dataset_name, process_type):
    """
    Plot scatter maps for SAME and DIFF separately in Information Plane.
    Color intensity is based on distance.
    """
    
    # Plot SAME
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter_same = ax.scatter(mi_xt_same, mi_ty_same, c=distance, cmap='Reds', 
                              s=50, alpha=0.7, edgecolors='darkred', linewidth=0.5)
    
    cbar_same = plt.colorbar(scatter_same, ax=ax)
    cbar_same.set_label('Euclidean Distance', fontsize=11)
    
    ax.set_xlabel("I(X; T)", fontsize=12, fontweight='bold')
    ax.set_ylabel("I(T; Y)", fontsize=12, fontweight='bold')
    ax.set_title(f"Layer {layer_idx+1} - SAME Class Conditional Information Plane", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_{dataset_name}_{process_type}_scatter_layer{layer_idx+1}_SAME.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Layer {layer_idx+1} SAME scatter plot saved.")
    
    # Plot DIFF
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter_diff = ax.scatter(mi_xt_diff, mi_ty_diff, c=distance, cmap='Blues', 
                              s=50, alpha=0.7, edgecolors='darkblue', linewidth=0.5)
    
    cbar_diff = plt.colorbar(scatter_diff, ax=ax)
    cbar_diff.set_label('Euclidean Distance', fontsize=11)
    
    ax.set_xlabel("I(X; T)", fontsize=12, fontweight='bold')
    ax.set_ylabel("I(T; Y)", fontsize=12, fontweight='bold')
    ax.set_title(f"Layer {layer_idx+1} - DIFF Class Conditional Information Plane", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_{dataset_name}_{process_type}_scatter_layer{layer_idx+1}_DIFF.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Layer {layer_idx+1} DIFF scatter plot saved.")


def plot_scatter_with_distance_bins(mi_xt_same, mi_ty_same, mi_xt_diff, mi_ty_diff, 
                                     distance, layer_idx, model_name, dataset_name, process_type):
    """
    Plot scatter maps with distance binning (10-unit intervals).
    """
    
    max_distance = np.max(distance) + 1
    distance_bins = np.arange(0, max_distance + 10, 10)
    
    for bin_idx in range(len(distance_bins) - 1):
        bin_min = distance_bins[bin_idx]
        bin_max = distance_bins[bin_idx + 1]
        
        mask = (distance >= bin_min) & (distance < bin_max)
        
        if not np.any(mask):
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot SAME
        dist_bin = distance[mask]
        scatter_same = axes[0].scatter(mi_xt_same[mask], mi_ty_same[mask], 
                                       c=dist_bin, cmap='Reds', 
                                       s=50, alpha=0.7, edgecolors='darkred', linewidth=0.5)
        
        cbar_same = plt.colorbar(scatter_same, ax=axes[0])
        cbar_same.set_label('Euclidean Distance', fontsize=10)
        
        axes[0].set_xlabel("I(X; T)", fontsize=11, fontweight='bold')
        axes[0].set_ylabel("I(T; Y)", fontsize=11, fontweight='bold')
        axes[0].set_title(f"SAME Class - Distance [{bin_min}-{bin_max})", fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Plot DIFF
        scatter_diff = axes[1].scatter(mi_xt_diff[mask], mi_ty_diff[mask], 
                                       c=dist_bin, cmap='Blues', 
                                       s=50, alpha=0.7, edgecolors='darkblue', linewidth=0.5)
        
        cbar_diff = plt.colorbar(scatter_diff, ax=axes[1])
        cbar_diff.set_label('Euclidean Distance', fontsize=10)
        
        axes[1].set_xlabel("I(X; T)", fontsize=11, fontweight='bold')
        axes[1].set_ylabel("I(T; Y)", fontsize=11, fontweight='bold')
        axes[1].set_title(f"DIFF Class - Distance [{bin_min}-{bin_max})", fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f"Layer {layer_idx+1} - Scatter Plot Comparison (Distance Bin: {bin_min}-{bin_max})", 
                     fontsize=13, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(f"{model_name}_{dataset_name}_{process_type}_scatter_layer{layer_idx+1}_dist{int(bin_min)}-{int(bin_max)}.png",
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Layer {layer_idx+1} distance [{bin_min}-{bin_max}) scatter plot saved.")


def plot_scatter_matrix_same(mi_xt_same, mi_ty_same, distance, 
                             model_name, dataset_name, process_type):
    """
    Matrix plot: Layer (y축) x Distance (x축) for SAME mode
    4x4 grid (4 layers, 4 distance bins)
    """
    layers = mi_xt_same.shape[0]
    dist_bins = np.arange(0, 50, 10)  # 0-10, 10-20, 20-30, 30-40
    num_dist = len(dist_bins) - 1
    
    fig, axes = plt.subplots(layers, num_dist, figsize=(20, 18), facecolor='white')
    
    for layer_idx in range(layers):
        for dist_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
            ax = axes[layer_idx, dist_idx]
            ax.set_facecolor('white')
            
            # Distance bin mask
            mask = (distance[layer_idx] >= b_min) & (distance[layer_idx] < b_max)
            
            if not np.any(mask):
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12,
                       transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Scatter plot
            dist_bin = distance[layer_idx][mask]
            scatter = ax.scatter(mi_xt_same[layer_idx][mask], 
                                mi_ty_same[layer_idx][mask], 
                                c=dist_bin, cmap='Reds', 
                                s=30, alpha=0.6, edgecolors='darkred', linewidth=0.3)
            
            # Ticks 제거
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True, alpha=0.2)
            
            # Y축 레이블 (좌측만)
            if dist_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx+1}", fontsize=11, fontweight='bold')
            
            # X축 레이블 (상단만)
            if layer_idx == 0:
                ax.set_title(f"Dist [{b_min:.0f}-{b_max:.0f})", fontsize=11, fontweight='bold')
    
    plt.suptitle(f"{model_name}_{dataset_name}_Scatter Matrix - SAME Mode", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    fname = f"{model_name}_{dataset_name}_{process_type}_scatter_matrix_SAME.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ SAME scatter matrix plot saved: {fname}")


def plot_scatter_matrix_diff(mi_xt_diff, mi_ty_diff, distance, 
                             model_name, dataset_name, process_type):
    """
    Matrix plot: Layer (y축) x Distance (x축) for DIFF mode
    4x4 grid (4 layers, 4 distance bins)
    """
    layers = mi_xt_diff.shape[0]
    dist_bins = np.arange(0, 50, 10)  # 0-10, 10-20, 20-30, 30-40
    num_dist = len(dist_bins) - 1
    
    fig, axes = plt.subplots(layers, num_dist, figsize=(20, 18), facecolor='white')
    
    for layer_idx in range(layers):
        for dist_idx, (b_min, b_max) in enumerate(zip(dist_bins[:-1], dist_bins[1:])):
            ax = axes[layer_idx, dist_idx]
            ax.set_facecolor('white')
            
            # Distance bin mask
            mask = (distance[layer_idx] >= b_min) & (distance[layer_idx] < b_max)
            
            if not np.any(mask):
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12,
                       transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Scatter plot
            dist_bin = distance[layer_idx][mask]
            scatter = ax.scatter(mi_xt_diff[layer_idx][mask], 
                                mi_ty_diff[layer_idx][mask], 
                                c=dist_bin, cmap='Blues', 
                                s=30, alpha=0.6, edgecolors='darkblue', linewidth=0.3)
            
            # Ticks 제거
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True, alpha=0.2)
            
            # Y축 레이블 (좌측만)
            if dist_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx+1}", fontsize=11, fontweight='bold')
            
            # X축 레이블 (상단만)
            if layer_idx == 0:
                ax.set_title(f"Dist [{b_min:.0f}-{b_max:.0f})", fontsize=11, fontweight='bold')
    
    plt.suptitle(f"{model_name}_{dataset_name}_Scatter Matrix - DIFF Mode", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    fname = f"{model_name}_{dataset_name}_{process_type}_scatter_matrix_DIFF.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ DIFF scatter matrix plot saved: {fname}")
