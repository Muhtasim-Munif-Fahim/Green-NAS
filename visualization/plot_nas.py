"""
Generate Figures for NAS Results - CORRECTED (v2)
Uses corrected baseline results and v2 transfer learning data
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matplotlib.colors import LinearSegmentedColormap

# Professional Style Configuration
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

# Gradient Color Palette (Professional blue-green-purple gradient)
GRADIENT_COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe']  # Purple to blue gradient
GRADIENT_COLORS_ALT = ['#43e97b', '#38f9d7', '#667eea', '#764ba2']  # Green to purple

# Helper function for small caps text
def small_caps(text):
    """Convert text to small caps style (ALL CAPS with reduced size for lowercase)"""
    return text.upper()

# File paths
NAS_RESULTS_FILE = 'nas_results/nas_results_20251204_123336.json'
DISCOVERED_RESULTS_FILE = 'discovered_models_results.json'
BASELINE_RESULTS_FILE = 'baseline_models_results_v2.json'
TL_RESULTS_FILE = 'transfer_learning_results_v2.json'

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def main():
    print("Generating corrected NAS figures...")
    
    nas_results = load_json(NAS_RESULTS_FILE)
    trained_results = load_json(DISCOVERED_RESULTS_FILE)
    baseline_results = load_json(BASELINE_RESULTS_FILE)
    tl_results = load_json(TL_RESULTS_FILE)
    
    if not nas_results:
        print("Error: NAS results not found")
        return

    # ==========================================
    # Figure 1: Pareto Front
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Extract Pareto front data
    pareto_front = nas_results['final_pareto_front']
    params = [p['params'] for p in pareto_front]
    rmse = [p['rmse'] for p in pareto_front]
    
    # Plot all Pareto points with gradient color
    ax.scatter(params, rmse, s=100, c=GRADIENT_COLORS[0], marker='o', alpha=0.7, 
               edgecolors='none', label=small_caps('Pareto Front (N=20)'), zorder=2)
    
    # Highlight selected architectures with gradient colors
    selected = nas_results['selected_architectures']
    gradient_arch_colors = [GRADIENT_COLORS[1], GRADIENT_COLORS[2], GRADIENT_COLORS[3]]
    
    for idx, (name, arch) in enumerate(selected.items()):
        ax.scatter(arch['params'], arch['rmse'], s=300, c=gradient_arch_colors[idx], 
                   marker='*', edgecolors='none', linewidth=0,
                   label=small_caps(name.split('(')[0].strip()), zorder=3)
    
    # Add GraphCast reference
    ax.scatter(36700000, 0.015, s=200, c='#9b59b6', marker='D', 
               edgecolors='none', linewidth=0, label=small_caps('GraphCast (SOTA)'), alpha=0.7, zorder=2)
    
    ax.set_xlabel(small_caps('Parameters (count)'), fontsize=14, fontweight='bold')
    ax.set_ylabel(small_caps('Validation RMSE'), fontsize=14, fontweight='bold')
    ax.set_title(small_caps('Pareto Front of NAS-Discovered Architectures'), fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure_2_pareto_front_nas.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure_2_pareto_front_nas.svg', bbox_inches='tight')
    print("Saved: figure_2_pareto_front_nas.png and .svg")
    
    # ==========================================
    # Figure 2: Architecture Comparison Bar Chart
    # ==========================================
    # Get best baseline (Manual GRU or Transformer)
    baseline_rmse = 0.0967 # Fallback
    baseline_params = 153000
    baseline_latency = 1.0
    
    if baseline_results:
        # Find best performing baseline
        best_base = min(baseline_results.items(), key=lambda x: x[1]['rmse'])
        baseline_name = best_base[0]
        baseline_rmse = best_base[1]['rmse']
        baseline_params = best_base[1]['params']
        baseline_latency = best_base[1]['latency_ms']
        print(f"Using best baseline: {baseline_name} (RMSE={baseline_rmse:.4f})")
    
    # Use Green-NAS-A results from Transfer Learning (v2) if available (more accurate 8-feature)
    nas_a_rmse = trained_results['Green-NAS-A (High Accuracy)']['rmse_target']
    if tl_results and 'zero_shot_rmse' in tl_results:
        # Zero-shot on target is comparable to "rmse_target"
        nas_a_rmse = tl_results['zero_shot_rmse']
        print(f"Using Green-NAS-A RMSE from TL v2: {nas_a_rmse:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = ['Green-NAS-A', 'Green-NAS-B', 'Green-NAS-C', 'Baseline']
    rmse_values = [
        nas_a_rmse,
        trained_results['Green-NAS-B (Balanced)']['rmse_target'],
        trained_results['Green-NAS-C (High Efficiency)']['rmse_target'],
        baseline_rmse
    ]
    params_values = [
        trained_results['Green-NAS-A (High Accuracy)']['params'] / 1000,
        trained_results['Green-NAS-B (Balanced)']['params'] / 1000,
        trained_results['Green-NAS-C (High Efficiency)']['params'] / 1000,
        baseline_params / 1000
    ]
    latency_values = [
        trained_results['Green-NAS-A (High Accuracy)']['latency_ms'],
        trained_results['Green-NAS-B (Balanced)']['latency_ms'],
        trained_results['Green-NAS-C (High Efficiency)']['latency_ms'],
        baseline_latency
    ]
    
    # Gradient color palette for bars
    bar_colors = GRADIENT_COLORS
    models_caps = [small_caps(m) for m in models]
    
    # RMSE
    axes[0].bar(models_caps, rmse_values, color=bar_colors, edgecolor='none', linewidth=0)
    axes[0].set_ylabel(small_caps('Target RMSE'), fontsize=12, fontweight='bold')
    axes[0].set_title(small_caps('(A) Accuracy'), fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Parameters
    axes[1].bar(models_caps, params_values, color=bar_colors, edgecolor='none', linewidth=0)
    axes[1].set_ylabel(small_caps('Parameters (k)'), fontsize=12, fontweight='bold')
    axes[1].set_title(small_caps('(B) Model Size'), fontsize=13, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_yscale('log')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Latency
    axes[2].bar(models_caps, latency_values, color=bar_colors, edgecolor='none', linewidth=0)
    axes[2].set_ylabel(small_caps('Latency (ms)'), fontsize=12, fontweight='bold')
    axes[2].set_title(small_caps('(C) Inference Speed'), fontsize=13, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure_3_architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure_3_architecture_comparison.svg', bbox_inches='tight')
    print("Saved: figure_3_architecture_comparison.png and .svg")
    
    # ==========================================
    # Figure 3: Evolution Convergence
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = [g['generation'] for g in nas_results['generation_history']]
    best_rmse = [g['best_rmse'] for g in nas_results['generation_history']]
    avg_rmse = [g['avg_rmse'] for g in nas_results['generation_history']]
    
    # Use gradient colors for lines
    ax.plot(generations, best_rmse, marker='o', linewidth=2.5, markersize=8, 
            color=GRADIENT_COLORS_ALT[2], label=small_caps('Best RMSE'), 
            markeredgecolor='none', zorder=3)
    ax.plot(generations, avg_rmse, marker='s', linewidth=2.5, markersize=8, 
            color=GRADIENT_COLORS_ALT[0], label=small_caps('Average RMSE'), 
            markeredgecolor='none', alpha=0.7, zorder=2)
    
    ax.set_xlabel(small_caps('Generation'), fontsize=14, fontweight='bold')
    ax.set_ylabel(small_caps('RMSE'), fontsize=14, fontweight='bold')
    ax.set_title(small_caps('NSGA-II Convergence Over Generations'), fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(generations)
    
    plt.tight_layout()
    plt.savefig('figure_4_evolution_convergence.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure_4_evolution_convergence.svg', bbox_inches='tight')
    print("Saved: figure_4_evolution_convergence.png and .svg")
    
    print("\n" + "="*80)
    print("All figures generated successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
