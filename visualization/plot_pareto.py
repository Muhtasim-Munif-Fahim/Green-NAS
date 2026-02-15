"""
Pareto Front Visualization
Generates a plot showing the trade-off between Accuracy (RMSE) and Efficiency (Params).
"""

import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from pathlib import Path

def plot_pareto():
    print("🚀 Generating Pareto Front Plot")
    
    # Load NAS results (Generation 9)
    results_path = "models/nas_results/gen_9.json"
    if not os.path.exists(results_path):
        print("⚠️ NAS results not found.")
        return

    with open(results_path, 'r') as f:
        population = json.load(f)
        
    # Extract data
    data = []
    for p in population:
        rmse = p['fitness'][0]
        params = p['fitness'][1]
        model_type = p['genes']['model_type']
        data.append({'rmse': rmse, 'params': params, 'type': model_type})
        
    df = pd.DataFrame(data)
    
    # Identify Pareto optimal points (simple heuristic for plot: lower is better for both)
    # In a real plot, we'd calculate the hull, but scatter is fine.
    
    plt.figure(figsize=(10, 6))
    
    # Color by model type
    colors = {'lstm': 'blue', 'gru': 'green', 'hybrid': 'red', 'attention': 'purple', 'mlp': 'gray'}
    
    for mtype in df['type'].unique():
        subset = df[df['type'] == mtype]
        plt.scatter(subset['params'], subset['rmse'], 
                    label=mtype.upper(), color=colors.get(mtype, 'black'), alpha=0.7, edgecolors='w', s=80)
        
    plt.xscale('log') # Params vary wildly
    plt.xlabel('Parameters (Log Scale)')
    plt.ylabel('Validation RMSE (Lower is Better)')
    plt.title('NAS Pareto Front: Accuracy vs. Efficiency')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Annotate "Sweet Spot"
    best_hybrid = df[df['type'] == 'hybrid'].sort_values('rmse').iloc[0]
    plt.annotate('Best Hybrid\n(High Acc, Low Params)', 
                 xy=(best_hybrid['params'], best_hybrid['rmse']), 
                 xytext=(best_hybrid['params']*2, best_hybrid['rmse']*1.2),
                 arrowprops=dict(facecolor='black', shrink=0.05))
                 
    output_path = "data/metadata/pareto_front.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Saved Pareto plot to {output_path}")

if __name__ == "__main__":
    plot_pareto()
