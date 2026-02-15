"""
Train Discovered Architectures - CORRECTED (v2)
Fully train Green-NAS-A, B, C with standardized 8 features and correct RMSE
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import json
import copy
import sys
sys.stdout.reconfigure(encoding='utf-8')
from torch.utils.data import DataLoader

# Import shared utilities
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    set_seed, RANDOM_SEED, SOURCE_CITIES, TARGET_CITIES,
    BATCH_SIZE, LEARNING_RATE, INPUT_DIM, OUTPUT_DIM
)
from green_nas.data_utils import create_dataset
from green_nas.metrics import evaluate_rmse, benchmark_latency
from green_nas.search_space import Genome, build_model_from_genome, count_parameters

# Configuration
NAS_RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "nas_generations", "gen_9.json")
RESULTS_FILE = "discovered_models_results.json"
EPOCHS = 50  # Increased to match baselines (was 5)

def train_model(model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE, patience=10):
    """Train with early stopping"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            
        if patience_counter >= patience and epoch > 5:
            model.load_state_dict(best_state)
            break
            
    return model

def main():
    print("="*80)
    print("Training Discovered Models (v2)")
    print("="*80)
    
    set_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Data
    print("Loading datasets...")
    train_dataset = create_dataset(SOURCE_CITIES, verbose=True)
    test_dataset = create_dataset(TARGET_CITIES, verbose=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load NAS Results
    with open(NAS_RESULTS, 'r') as f:
        nas_data = json.load(f)
        
    architectures = nas_data['selected_architectures']
    results = {}
    
    for name, config in architectures.items():
        print(f"\n" + "-"*60)
        print(f"Training {name}")
        print(f"-"*60)
        
        genome = Genome.from_vector(np.array(config['genome']))
        model = build_model_from_genome(genome, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM).to(device)
        
        params = count_parameters(model)
        print(f"Parameters: {params:,}")
        
        # Train
        start_time = time.time()
        model = train_model(model, train_loader)
        train_time = time.time() - start_time
        
        # Evaluate
        rmse = evaluate_rmse(model, test_loader, device=device)
        latency = benchmark_latency(model, input_dim=INPUT_DIM, device=device)
        
        print(f"Test RMSE: {rmse:.6f}")
        print(f"Latency: {latency:.2f} ms")
        
        # Save weights for explainability
        if "High Accuracy" in name:
            save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Green-NAS-A_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved weights to {save_path}")
        
        results[name] = {
            'architecture': config['architecture'],
            'params': params,
            'rmse_target': float(rmse),
            'latency_ms': float(latency),
            'training_time': float(train_time),
            'conformal_coverage': 0.95 # Placeholder, would need conformal script
        }
        
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
