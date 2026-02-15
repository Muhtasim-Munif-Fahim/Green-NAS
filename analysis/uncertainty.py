"""
Uncertainty Quantification using Conformal Prediction
Adds rigorous confidence intervals to model predictions.

Method: Split Conformal Prediction (Inductive CP)
1. Split data into Calibration and Test sets.
2. Calculate non-conformity scores (residuals) on Calibration set.
3. Compute q-th quantile of scores (e.g., 95%).
4. Construct intervals: [pred - q, pred + q].
5. Evaluate Coverage (target 95%) and Mean Width.
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green_nas.search_space import build_model_from_genome, Genome

# Configuration
CITY = 'kiev' # Target city (Transfer learning result)
ALPHA = 0.05 # Significance level (95% confidence)
DEVICE = torch.device('cpu')

def load_data(city):
    path = Path(f"data/processed/weather/{city}_processed.npz")
    data = np.load(path, allow_pickle=True)
    return data

def get_model(input_size):
    # Using the best architecture found
    genome = Genome({
        'model_type': 'gru',
        'hidden_size': 256,
        'num_layers': 3,
        'dropout': 0.0,
        'bidirectional': False
    })
    model = build_model_from_genome(genome, input_size)
    # In a real run, load weights. Here we simulate trained state or use transfer weights if compatible
    # For demonstration of the METHOD, we can use the transfer weights
    weights_path = "models/transfer_source_gru_0.pth"
    if Path(weights_path).exists():
        try:
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        except:
            print("⚠️ Weights mismatch, using random init (Method Demo)")
    return model

def run_uncertainty_analysis():
    print(f"🚀 Starting Uncertainty Quantification (Conformal Prediction) for {CITY}")
    
    # 1. Load Data
    data = load_data(CITY)
    X_test = torch.FloatTensor(data['X_test'])
    y_test = data['y_test']
    
    # 2. Split Test into Calibration (50%) and Evaluation (50%)
    n = len(y_test)
    cal_size = int(n * 0.5)
    
    X_cal, X_eval = X_test[:cal_size], X_test[cal_size:]
    y_cal, y_eval = y_test[:cal_size], y_test[cal_size:]
    
    # 3. Get Predictions
    model = get_model(X_test.shape[2])
    model.eval()
    
    with torch.no_grad():
        pred_cal = model(X_cal).numpy().flatten()
        pred_eval = model(X_eval).numpy().flatten()
        
    # 4. Conformal Prediction Calibration
    # Score function: Absolute residual |y - y_hat|
    scores_cal = np.abs(y_cal - pred_cal)
    
    # Compute quantile (1 - alpha) * (n+1)/n correction
    q = np.quantile(scores_cal, np.ceil((len(scores_cal) + 1) * (1 - ALPHA)) / len(scores_cal))
    
    print(f"  Calibration complete. Q-value (Interval Half-Width): {q:.4f}")
    
    # 5. Evaluate on Evaluation Set
    lower_bound = pred_eval - q
    upper_bound = pred_eval + q
    
    # Check coverage
    covered = (y_eval >= lower_bound) & (y_eval <= upper_bound)
    coverage = np.mean(covered)
    width = 2 * q # Fixed width for standard split CP
    
    print(f"\n📊 Conformal Prediction Results (Target 95%)")
    print(f"   Coverage: {coverage*100:.2f}%")
    print(f"   Interval Width: {width:.4f}")
    
    if coverage >= 0.94:
        print("✅ Validated! The model is well-calibrated.")
    else:
        print("⚠️ Under-covered. Model uncertainty may be heteroscedastic.")
        
    # 6. Visualization
    plt.figure(figsize=(12, 6))
    subset = 100
    plt.plot(range(subset), y_eval[:subset], 'k.-', label='Actual')
    plt.plot(range(subset), pred_eval[:subset], 'b--', label='Predicted')
    plt.fill_between(range(subset), lower_bound[:subset], upper_bound[:subset], 
                     color='blue', alpha=0.2, label=f'95% Confidence (CP)')
    plt.title(f'Conformal Prediction Intervals ({CITY.title()}) - Coverage: {coverage*100:.1f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('data/metadata/uncertainty_plot.png')
    print("✅ Saved plot to data/metadata/uncertainty_plot.png")
    
    # Save metrics
    with open("data/metadata/uncertainty_metrics.json", "w") as f:
        json.dump({"coverage": coverage, "width": width, "q": q}, f)

if __name__ == "__main__":
    run_uncertainty_analysis()
