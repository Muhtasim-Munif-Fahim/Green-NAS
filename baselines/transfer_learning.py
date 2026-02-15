"""
Transfer Learning Experiments
Evaluates cross-continental generalization of NAS-discovered architectures.

Scenarios:
1. Source: Asia (Dhaka, Delhi, Chongqing) -> Target: Europe (Kiev), Americas (São Paulo), Africa (Luanda)
2. Zero-shot: Direct application without retraining
3. Few-shot: Fine-tuning on 10% of target data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.baseline_models import BaselineLSTM, BaselineGRU, BaselineAttention, HybridCNNLSTM
from green_nas.search_space import build_model_from_genome, Genome

# Configuration
SOURCE_CITIES = ['dhaka', 'delhi', 'chongqing']
TARGET_CITIES = ['kiev', 'são_paulo', 'luanda']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS_SOURCE = 15
EPOCHS_FINETUNE = 5

def load_data(cities, data_dir="data/processed/weather"):
    """Load and concatenate data from multiple cities"""
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    
    for city in cities:
        path = Path(data_dir) / f"{city}_processed.npz"
        if not path.exists():
            print(f"⚠️ Warning: {city} data not found")
            continue
            
        data = np.load(path, allow_pickle=True)
        X_train_list.append(data['X_train'])
        y_train_list.append(data['y_train'])
        X_val_list.append(data['X_val'])
        y_val_list.append(data['y_val'])
        X_test_list.append(data['X_test'])
        y_test_list.append(data['y_test'])
    
    if not X_train_list:
        return None
        
    return {
        'X_train': np.concatenate(X_train_list),
        'y_train': np.concatenate(y_train_list),
        'X_val': np.concatenate(X_val_list),
        'y_val': np.concatenate(y_val_list),
        'X_test': np.concatenate(X_test_list),
        'y_test': np.concatenate(y_test_list)
    }

def get_best_genomes():
    """Load best genomes from NAS results"""
    # Load last generation
    results_dir = Path("models/nas_results")
    files = sorted(list(results_dir.glob("gen_*.json")), key=lambda f: int(f.stem.split('_')[1]))
    
    if not files:
        # Fallback if no NAS results (shouldn't happen in this flow)
        print("⚠️ No NAS results found. Using default architectures.")
        return [
            Genome({'model_type': 'gru', 'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2, 'bidirectional': False}),
            Genome({'model_type': 'hybrid', 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2, 'bidirectional': False})
        ]
        
    last_gen_file = files[-1]
    with open(last_gen_file, 'r') as f:
        population_data = json.load(f)
    
    # Reconstruct Genome objects
    population = []
    for p_data in population_data:
        g = Genome(p_data['genes'])
        g.fitness = p_data['fitness']
        g.id = p_data['id']
        population.append(g)
        
    # Sort by RMSE (fitness[0])
    population.sort(key=lambda x: x.fitness[0])
    best_acc = population[0]
    
    # Sort by Efficiency (fitness[1])
    population.sort(key=lambda x: x.fitness[1])
    best_eff = population[0]
    
    return [best_acc, best_eff]

def train_model(model, data, epochs=10, lr=0.001):
    """Train model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(DEVICE)
    
    train_dataset = TensorDataset(torch.FloatTensor(data['X_train']), torch.FloatTensor(data['y_train']))
    val_dataset = TensorDataset(torch.FloatTensor(data['X_val']), torch.FloatTensor(data['y_val']))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                out = model(X_b)
                loss = criterion(out, y_b.unsqueeze(1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict().copy()
            
    if best_state:
        model.load_state_dict(best_state)
    return model

def evaluate(model, data):
    """Evaluate model"""
    model.eval()
    criterion = nn.MSELoss()
    test_dataset = TensorDataset(torch.FloatTensor(data['X_test']), torch.FloatTensor(data['y_test']))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    total_loss = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            out = model(X_b)
            loss = criterion(out, y_b.unsqueeze(1))
            total_loss += loss.item()
            
    mse = total_loss / len(test_loader)
    return np.sqrt(mse) # RMSE

def run_experiments():
    print("🚀 Starting Transfer Learning Experiments")
    
    # 1. Load Source Data
    print(f"Loading Source Data (Asia): {SOURCE_CITIES}")
    source_data = load_data(SOURCE_CITIES)
    input_size = source_data['X_train'].shape[2]
    
    # 2. Get Best Architectures
    best_genomes = get_best_genomes()
    print(f"Selected {len(best_genomes)} architectures for transfer")
    
    results = []
    
    for i, genome in enumerate(best_genomes):
        model_type = genome.genes['model_type']
        print(f"\nTesting Architecture {i+1}: {model_type.upper()}")
        print(f"Genes: {genome.genes}")
        
        # 3. Train on Source
        print("  Training on Source (Asia)...")
        model = build_model_from_genome(genome, input_size)
        model = train_model(model, source_data, epochs=EPOCHS_SOURCE)
        
        # Save source model
        torch.save(model.state_dict(), f"models/transfer_source_{model_type}_{i}.pth")
        
        # 4. Transfer to Targets
        for target_city in TARGET_CITIES:
            print(f"  Transferring to {target_city}...")
            target_data = load_data([target_city])
            
            # Baseline: Train from scratch on Target
            print("    Training from scratch (Baseline)...")
            scratch_model = build_model_from_genome(genome, input_size)
            scratch_model = train_model(scratch_model, target_data, epochs=EPOCHS_SOURCE)
            rmse_scratch = evaluate(scratch_model, target_data)
            
            # Experiment A: Zero-shot (Direct Transfer)
            print("    Evaluating Zero-shot...")
            rmse_zeroshot = evaluate(model, target_data)
            
            # Experiment B: Few-shot (Fine-tuning)
            print("    Fine-tuning (Few-shot)...")
            # Create subset data (10%)
            n_samples = int(len(target_data['X_train']) * 0.1)
            finetune_data = {
                'X_train': target_data['X_train'][:n_samples],
                'y_train': target_data['y_train'][:n_samples],
                'X_val': target_data['X_val'], # Keep full val for fair stopping
                'y_val': target_data['y_val'],
                'X_test': target_data['X_test'],
                'y_test': target_data['y_test']
            }
            
            finetune_model = build_model_from_genome(genome, input_size)
            finetune_model.load_state_dict(model.state_dict()) # Load source weights
            finetune_model = train_model(finetune_model, finetune_data, epochs=EPOCHS_FINETUNE)
            rmse_finetune = evaluate(finetune_model, target_data)
            
            print(f"    RESULTS: Scratch={rmse_scratch:.4f}, Zero-shot={rmse_zeroshot:.4f}, Fine-tune={rmse_finetune:.4f}")
            
            results.append({
                'architecture': model_type,
                'genome_id': genome.id,
                'target_city': target_city,
                'rmse_scratch': rmse_scratch,
                'rmse_zeroshot': rmse_zeroshot,
                'rmse_finetune': rmse_finetune,
                'improvement_zeroshot': (rmse_scratch - rmse_zeroshot) / rmse_scratch,
                'improvement_finetune': (rmse_scratch - rmse_finetune) / rmse_scratch
            })
            
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("data/metadata/transfer_results.csv", index=False)
    print("\n✅ Transfer Experiments Complete!")
    print(df[['architecture', 'target_city', 'rmse_scratch', 'rmse_zeroshot', 'rmse_finetune']])

if __name__ == "__main__":
    run_experiments()
