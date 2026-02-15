"""
Explainability Analysis (XAI) using SHAP
Interprets the best NAS-discovered architectures to understand feature importance.

Methods:
1. DeepExplainer/GradientExplainer for feature attribution
2. Summary plots
3. Temporal importance analysis
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.baseline_models import BaselineLSTM, BaselineGRU, HybridCNNLSTM
from green_nas.search_space import build_model_from_genome, Genome

# Configuration
CITY = 'dhaka' # Analyze primary city
DEVICE = torch.device('cpu') # SHAP often works better on CPU for compatibility
SAMPLES = 100 # Number of background samples
TEST_SAMPLES = 50 # Number of test samples to explain

def load_data(city):
    path = Path(f"data/processed/weather/{city}_processed.npz")
    data = np.load(path, allow_pickle=True)
    return data

def get_best_model(input_size):
    """Load best model (hardcoded from NAS results for demo)"""
    # Best Accuracy: GRU (3 layers, 256 hidden)
    genome = Genome({
        'model_type': 'gru',
        'hidden_size': 256,
        'num_layers': 3,
        'dropout': 0.0,
        'bidirectional': False
    })
    
    model = build_model_from_genome(genome, input_size)
    
    # Load weights if available (using transfer source weights as proxy for best)
    # In real scenario, we'd load the exact checkpoint
    weights_path = "models/transfer_source_gru_0.pth"
    if Path(weights_path).exists():
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print("✅ Loaded pre-trained weights")
    else:
        print("⚠️ Weights not found, using random init (for demo structure)")
        
    model.to(DEVICE)
    model.eval()
    return model

def run_shap_analysis():
    print(f"🚀 Starting SHAP Analysis for {CITY}")
    
    # 1. Load Data
    data = load_data(CITY)
    X_train = torch.FloatTensor(data['X_train'])
    X_test = torch.FloatTensor(data['X_test'])
    
    # Feature names (based on Open-Meteo)
    feature_names = [
        'temp', 'humidity', 'precip', 'pressure', 'cloud', 'wind_speed',
        'wind_dir', 'radiation', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
    ]
    
    # 2. Load Model
    input_size = X_train.shape[2]
    model = get_best_model(input_size)
    
    # 3. Prepare SHAP
    # Use a random subset of training data as background
    background = X_train[np.random.choice(X_train.shape[0], SAMPLES, replace=False)]
    
    # Use a subset of test data to explain
    to_explain = X_test[:TEST_SAMPLES]
    
    print("  Computing SHAP values (this may take a moment)...")
    
    # DeepExplainer or GradientExplainer
    # Note: DeepExplainer can be tricky with RNNs, GradientExplainer is often more robust
    e = shap.GradientExplainer(model, background)
    shap_values = e.shap_values(to_explain)
    
    # shap_values is list of tensors (one for each output). We have 1 output.
    # Shape: (samples, seq_len, features)
    vals = shap_values[0] if isinstance(shap_values, list) else shap_values
    print(f"  SHAP values shape: {np.array(vals).shape}")
    
    # 4. Global Feature Importance
    # Sum absolute SHAP values across time and samples
    # vals shape should be (samples, seq_len, features)
    # We want (features,)
    global_importance = np.abs(vals).mean(axis=(0, 1))
    
    # Ensure 1D
    global_importance = np.array(global_importance).flatten()
    print(f"  Global importance shape: {global_importance.shape}")
    print(f"  Feature names count: {len(feature_names)}")
    
    if len(global_importance) != len(feature_names):
        print(f"⚠️ Mismatch! Adjusting feature names to match data ({len(global_importance)})")
        feature_names = [f"Feature_{i}" for i in range(len(global_importance))]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': global_importance
    }).sort_values('Importance', ascending=False)
    
    print("\n📊 Global Feature Importance:")
    print(importance_df)
    
    # Save importance
    importance_df.to_csv("data/metadata/shap_importance.csv", index=False)
    
    # 5. Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(f'SHAP Feature Importance for Weather Forecasting ({CITY.title()})')
    plt.tight_layout()
    plt.savefig('data/metadata/shap_importance.png')
    print("✅ Saved importance plot to data/metadata/shap_importance.png")
    
    # 6. Temporal Importance (Average across features)
    # Shape: (seq_len,)
    temporal_importance = np.abs(vals).mean(axis=(0, 2))
    
    plt.figure(figsize=(10, 4))
    plt.plot(range(24), temporal_importance, marker='o')
    plt.xlabel('Lag (Hours Past)')
    plt.ylabel('Mean Absolute SHAP Value')
    plt.title('Temporal Importance: Which past hours matter most?')
    plt.grid(True)
    plt.savefig('data/metadata/shap_temporal.png')
    print("✅ Saved temporal plot to data/metadata/shap_temporal.png")

if __name__ == "__main__":
    run_shap_analysis()
