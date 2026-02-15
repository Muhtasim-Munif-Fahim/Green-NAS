"""
Robustness & Validation Testing for Weather Forecasting Models
Stress-tests the model to ensure results are not due to lucky splits or overfitting.

Methods:
1. Time-Series Cross-Validation (Rolling Origin):
   - Train on [0..T], Test on [T..T+k]
   - Train on [0..T+k], Test on [T+k..T+2k]
   - ...
   - Validates temporal generalization.

2. Noise Sensitivity Analysis:
   - Add Gaussian noise to input features.
   - Measure performance degradation.
   - Validates robustness to sensor errors.

3. Random Seed Stability:
   - Retrain model with different seeds.
   - Measure variance in RMSE.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green_nas.search_space import build_model_from_genome, Genome

# Configuration
CITY = 'dhaka' # Target city
FOLDS = 3
NOISE_LEVELS = [0.01, 0.05, 0.1] # 1%, 5%, 10% noise
SEEDS = [42, 101, 999]
DEVICE = torch.device('cpu')
BATCH_SIZE = 64
EPOCHS = 5 # Reduced for speed in validation script

def load_data(city):
    path = Path(f"data/processed/weather/{city}_processed.npz")
    data = np.load(path, allow_pickle=True)
    return data

def train_model(model, X_train, y_train, epochs=EPOCHS):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(DEVICE)
    model.train()
    
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(epochs):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
    return model

def evaluate(model, X_test, y_test):
    model.eval()
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    total_loss = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = criterion(out, y.unsqueeze(1))
            total_loss += loss.item()
    return np.sqrt(total_loss / len(loader))

def run_robustness_tests():
    print(f"🚀 Starting Robustness & Validation Tests for {CITY}")
    
    data = load_data(CITY)
    X_all = np.concatenate([data['X_train'], data['X_val'], data['X_test']])
    y_all = np.concatenate([data['y_train'], data['y_val'], data['y_test']])
    
    input_size = X_all.shape[2]
    
    # Best Architecture (Hybrid)
    genome = Genome({
        'model_type': 'hybrid',
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.0,
        'bidirectional': False
    })
    
    results = {}
    
    # 1. Rolling Origin Cross-Validation
    print("\n🔄 1. Running Rolling Origin Cross-Validation (3 Folds)")
    fold_size = len(X_all) // (FOLDS + 1)
    cv_rmses = []
    
    for i in range(FOLDS):
        train_end = fold_size * (i + 1)
        test_end = fold_size * (i + 2)
        
        X_train_fold = X_all[:train_end]
        y_train_fold = y_all[:train_end]
        X_test_fold = X_all[train_end:test_end]
        y_test_fold = y_all[train_end:test_end]
        
        print(f"   Fold {i+1}: Train [{0}:{train_end}], Test [{train_end}:{test_end}]")
        
        model = build_model_from_genome(genome, input_size)
        model = train_model(model, X_train_fold, y_train_fold)
        rmse = evaluate(model, X_test_fold, y_test_fold)
        cv_rmses.append(rmse)
        print(f"   Fold {i+1} RMSE: {rmse:.4f}")
        
    avg_cv_rmse = np.mean(cv_rmses)
    std_cv_rmse = np.std(cv_rmses)
    print(f"   ✅ Average CV RMSE: {avg_cv_rmse:.4f} ± {std_cv_rmse:.4f}")
    results['cv_rmse_mean'] = float(avg_cv_rmse)
    results['cv_rmse_std'] = float(std_cv_rmse)
    
    # 2. Noise Sensitivity
    print("\n🔊 2. Running Noise Sensitivity Analysis")
    # Train once on full train set
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    model = build_model_from_genome(genome, input_size)
    model = train_model(model, X_train, y_train)
    
    baseline_rmse = evaluate(model, X_test, y_test)
    print(f"   Baseline RMSE (0% Noise): {baseline_rmse:.4f}")
    
    noise_results = {'0.0': baseline_rmse}
    
    for noise_level in NOISE_LEVELS:
        # Add noise to Test set only
        noise = np.random.normal(0, noise_level, X_test.shape)
        X_test_noisy = X_test + noise
        
        rmse_noisy = evaluate(model, X_test_noisy, y_test)
        degradation = (rmse_noisy - baseline_rmse) / baseline_rmse * 100
        print(f"   Noise {noise_level*100}%: RMSE {rmse_noisy:.4f} (Degradation: {degradation:.1f}%)")
        noise_results[str(noise_level)] = float(rmse_noisy)
        
    results['noise_sensitivity'] = noise_results
    
    # 3. Seed Stability
    print("\n🌱 3. Running Seed Stability Test")
    seed_rmses = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = build_model_from_genome(genome, input_size)
        model = train_model(model, X_train, y_train)
        rmse = evaluate(model, X_test, y_test)
        seed_rmses.append(rmse)
        print(f"   Seed {seed}: RMSE {rmse:.4f}")
        
    avg_seed_rmse = np.mean(seed_rmses)
    std_seed_rmse = np.std(seed_rmses)
    print(f"   ✅ Stability: {avg_seed_rmse:.4f} ± {std_seed_rmse:.4f}")
    results['seed_stability_mean'] = float(avg_seed_rmse)
    results['seed_stability_std'] = float(std_seed_rmse)
    
    # Save
    with open("data/metadata/robustness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Robustness tests complete. Saved to data/metadata/robustness_results.json")

if __name__ == "__main__":
    run_robustness_tests()
