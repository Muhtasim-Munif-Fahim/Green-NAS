"""
Generate Transfer Learning Figure (N=10 Trials) - CORRECTED (v2)
Uses v2 results file
"""

import matplotlib.pyplot as plt
import json
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Professional Style Configuration
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

# Gradient Color Palette
GRADIENT_COLORS = ['#43e97b', '#38f9d7', '#667eea']  # Green to purple gradient

# Helper function for small caps
def small_caps(text):
    return text.upper()

# Config
RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "transfer_learning_results.json")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "transfer_curve.png")

def generate_figure():
    if not os.path.exists(RESULTS_FILE):
        print(f"Results file not found: {RESULTS_FILE}")
        return

    # Load results
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
    
    percentages = [1, 10, 50, 100]
    transfer_means = []
    transfer_stds = []
    scratch_means = []
    scratch_stds = []
    
    # Check if data exists for all percentages
    available_pcts = []
    
    for pct_val, pct_key in zip(percentages, ["1%", "10%", "50%", "100%"]):
        if pct_key in data['experiments']:
            stats = data['experiments'][pct_key]['statistics']
            transfer_means.append(stats['transfer_mean'])
            transfer_stds.append(stats['transfer_std'])
            scratch_means.append(stats['scratch_mean'])
            scratch_stds.append(stats['scratch_std'])
            available_pcts.append(pct_val)
        else:
            print(f"Warning: Missing data for {pct_key}")
    
    if not available_pcts:
        print("No data available to plot")
        return

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Plot lines with error bars using gradient colors
    plt.errorbar(available_pcts, scratch_means, yerr=scratch_stds, fmt='-o', 
                 label=small_caps('Training from Scratch'), color=GRADIENT_COLORS[0], 
                 capsize=5, linewidth=2.5, markeredgecolor='none')
    plt.errorbar(available_pcts, transfer_means, yerr=transfer_stds, fmt='-o', 
                 label=small_caps('Transfer Learning (Green-NAS-A)'), color=GRADIENT_COLORS[2], 
                 capsize=5, linewidth=2.5, markeredgecolor='none')
    
    # Zero-shot baseline line
    if 'zero_shot_rmse' in data:
        zero_shot = data['zero_shot_rmse']
        plt.axhline(y=zero_shot, color=GRADIENT_COLORS[1], linestyle='--', 
                    label=small_caps(f'Zero-Shot Baseline ({zero_shot:.4f})'))
    
    # Formatting with small caps
    plt.xscale('log')
    plt.xticks(percentages, [f"{p}%" for p in percentages])
    plt.xlabel(small_caps('Target Data Percentage (Log Scale)'), fontsize=12, fontweight='bold')
    plt.ylabel(small_caps('RMSE (Lower is Better)'), fontsize=12, fontweight='bold')
    plt.title(small_caps('Transfer Learning Efficiency: Fine-Tuning vs. From Scratch'), 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Annotate improvement at 1% if available
    if "1%" in data['experiments']:
        imp_1pct = data['experiments']['1%']['statistics']['improvement_pct']
        # Position annotation based on data
        y_pos = transfer_means[0]
        plt.annotate(f"{imp_1pct:.1f}% Improvement", 
                     xy=(1, y_pos), xytext=(1.2, y_pos + 0.005),
                     arrowprops=dict(facecolor=GRADIENT_COLORS[2], shrink=0.05, edgecolor='none'))
    
    plt.tight_layout()
    # Save PNG
    plt.savefig(OUTPUT_FILE, dpi=300)
    # Save SVG
    svg_file = OUTPUT_FILE.replace('.png', '.svg')
    plt.savefig(svg_file)
    print(f"Figure saved to {OUTPUT_FILE} and {svg_file}")

if __name__ == "__main__":
    generate_figure()
