"""
Transfer Learning Experiments
Comprehensive evaluation of Green-NAS-A's transfer learning capabilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import json
import copy
from torch.utils.data import DataLoader, TensorDataset, Subset
from scipy import stats

from green_nas.search_space import Genome, build_model_from_genome, count_parameters

# ==========================================
# Configuration
# ==========================================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "weather")
NAS_RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "nas_generations", "gen_9.json")
TRAINED_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "experiment_results.json")
RESULTS_FILE = "transfer_learning_results.json"
SEQ_LEN = 24
BATCH_SIZE = 64
FINE_TUNE_EPOCHS = 5
FROM_SCRATCH_EPOCHS = 10  # More epochs since starting from random
LEARNING_RATE = 0.0001
N_TRIALS = 3  # For statistical significance

# All 24 cities
ALL_CITIES = [
    "athens_greece_tier1.csv", "belgrade_serbia_tier3.csv", "buenos_aires_argentina_tier1.csv",
    "busan_south_korea_tier3.csv", "chengdu_china_tier3.csv", "chongqing_china_tier2.csv",
    "delhi_india_tier1.csv", "dhaka_bangladesh_tier1.csv", "harare_zimbabwe_tier3.csv",
    "kiev_ukraine_tier2.csv", "kolkata_india_tier1.csv", "lahore_pakistan_tier1.csv",
    "lima_peru_tier1.csv", "luanda_angola_tier3.csv", "lusaka_zambia_tier3.csv",
    "maputo_mozambique_tier3.csv", "mumbai_india_tier1.csv", "san_salvador_el_salvador_tier3.csv",
    "santiago_chile_tier1.csv", "sofia_bulgaria_tier2.csv", "são_paulo_brazil_tier1.csv",
    "windhoek_namibia_tier3.csv", "wuhan_china_tier3.csv", "zagreb_croatia_tier2.csv"
]

SOURCE_CITIES = ALL_CITIES[:18]  # 18 for pre-training
TARGET_CITIES = ALL_CITIES[18:]  # 6 for transfer testing

# Data percentages to test
DATA_PERCENTAGES = [0.01, 0.10, 0.50, 1.0]  # 1%, 10%, 50%, 100%

# ==========================================
# Data Loading (from train_discovered_models.py)
# ==========================================
def load_and_process_city(filename):
    """Load and preprocess city data."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None, None
    
    df = pd.read_csv(path)
    feature_cols = [
        'temperature_2m', 'relative_humidity_2m', 'precipitation',
        'pressure_msl', 'surface_pressure', 'cloud_cover',
        'wind_speed_10m', 'wind_direction_10m', 'shortwave_radiation'
    ]
    selected_cols = [c for c in feature_cols if c in df.columns]
    if not selected_cols:
        return None, None
    
    data = df[selected_cols].values.astype(np.float32)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    data_scaled = (data - min_val) / (max_val - min_val + 1e-6)
    
    X, y = [], []
    for i in range(len(data_scaled) - SEQ_LEN - 1):
        X.append(data_scaled[i : i+SEQ_LEN])
        y.append(data_scaled[i+SEQ_LEN, 0])
    
    return torch.tensor(np.array(X)), torch.tensor(np.array(y)).unsqueeze(1)


def get_dataset(cities):
    """Load multiple cities into a dataset."""
    all_X, all_y = [], []
    for city in cities:
        X, y = load_and_process_city(city)
        if X is not None:
            all_X.append(X)
            all_y.append(y)
    
    if not all_X:
        return None
    
    combined_X = torch.cat(all_X)
    combined_y = torch.cat(all_y)
    return TensorDataset(combined_X, combined_y)


def get_subset_loader(dataset, percentage, batch_size=BATCH_SIZE):
    """Get a random subset of the dataset."""
    total_size = len(dataset)
    subset_size = int(total_size * percentage)
    
    # Random subset
    indices = torch.randperm(total_size)[:subset_size].tolist()
    subset = Subset(dataset, indices)
    
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


# ==========================================
# Training Functions
# ==========================================
def train_model(model, train_loader, epochs, lr=LEARNING_RATE):
    """Train or fine-tune a model."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            
            if output.dim() == 2 and output.shape[1] == 1:
                output = output.squeeze(1)
            if y_batch.dim() == 2 and y_batch.shape[1] == 1:
                y_batch = y_batch.squeeze(1)
            
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
    
    return model


def evaluate_model(model, test_loader):
    """Evaluate model RMSE."""
    criterion = nn.MSELoss()
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(X_batch)
            
            if output.dim() == 2 and output.shape[1] == 1:
                output = output.squeeze(1)
            if y_batch.dim() == 2 and y_batch.shape[1] == 1:
                y_batch = y_batch.squeeze(1)
            
            loss = criterion(output, y_batch)
            total_loss += loss.item()
    
    rmse = np.sqrt(total_loss / len(test_loader))
    return rmse


# ==========================================
# Transfer Learning Experiments
# ==========================================
def run_transfer_learning():
    """
    Run comprehensive transfer learning experiments.
    
    For each data percentage (1%, 10%, 50%, 100%):
    1. Fine-tune pre-trained Green-NAS-A
    2. Train from scratch (baseline)
    3. Repeat N_TRIALS times
    4. Compute statistics
    """
    
    print("="*80)
    print("Transfer Learning Experiments - Green-NAS-A")
    print("="*80)
    
    # Load NAS results to get Green-NAS-A genome
    with open(NAS_RESULTS, 'r') as f:
        nas_results = json.load(f)
    
    green_nas_a = nas_results['selected_architectures']['Green-NAS-A (High Accuracy)']
    genome_vector = np.array(green_nas_a['genome'])
    genome = Genome.from_vector(genome_vector)
    
    print(f"\nModel: {green_nas_a['architecture']}")
    print(f"Parameters: {green_nas_a['params']:,}")
    
    # Load pre-trained weights (from full training)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Get datasets
    print("\nLoading datasets...")
    source_dataset = get_dataset(SOURCE_CITIES)
    target_dataset = get_dataset(TARGET_CITIES)
    
    # Split source for pre-training (already done, but we'll use for reference)
    train_size = int(0.8 * len(source_dataset))
    val_size = len(source_dataset) - train_size
    source_train, source_val = torch.utils.data.random_split(source_dataset, [train_size, val_size])
    
    # Full test loader
    test_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Source training: {len(source_train)} samples")
    print(f"Target full: {len(target_dataset)} samples")
    print(f"Target test: {len(test_loader.dataset)} samples")
    
    # Results storage
    results = {
        'config': {
            'model': green_nas_a['architecture'],
            'params': green_nas_a['params'],
            'source_cities': SOURCE_CITIES,
            'target_cities': TARGET_CITIES,
            'data_percentages': DATA_PERCENTAGES,
            'n_trials': N_TRIALS,
            'fine_tune_epochs': FINE_TUNE_EPOCHS,
            'from_scratch_epochs': FROM_SCRATCH_EPOCHS
        },
        'experiments': {}
    }
    
    # Zero-shot baseline
    print("\n" + "="*80)
    print("ZERO-SHOT BASELINE (No fine-tuning)")
    print("="*80)
    
    # Build and train pre-trained model
    print("Training source model...")
    source_train_loader = DataLoader(source_train, batch_size=BATCH_SIZE, shuffle=True)
    pretrained_model = build_model_from_genome(genome, input_dim=9, output_dim=1).to(device)
    pretrained_model = train_model(pretrained_model, source_train_loader, epochs=5)
    
    # Evaluate zero-shot
    zero_shot_rmse = evaluate_model(pretrained_model, test_loader)
    print(f"\nZero-Shot RMSE (pre-trained, no fine-tuning): {zero_shot_rmse:.4f}")
    
    results['zero_shot_rmse'] = float(zero_shot_rmse)
    
    # Save pre-trained weights for fine-tuning
    pretrained_state = copy.deepcopy(pretrained_model.state_dict())
    
    # Experiment for each data percentage
    for pct in DATA_PERCENTAGES:
        print("\n" + "="*80)
        print(f"EXPERIMENTS WITH {pct*100:.0f}% OF TARGET DATA")
        print("="*80)
        
        pct_key = f"{pct*100:.0f}%"
        results['experiments'][pct_key] = {
            'transfer': [],
            'from_scratch': []
        }
        
        for trial in range(N_TRIALS):
            print(f"\n--- Trial {trial+1}/{N_TRIALS} ---")
            
            # Get subset loader
            train_subset_loader = get_subset_loader(target_dataset, pct)
            print(f"Training samples: {len(train_subset_loader.dataset)}")
            
            # TRANSFER LEARNING: Fine-tune pre-trained model
            print("\n[Transfer] Fine-tuning pre-trained model...")
            transfer_model = build_model_from_genome(genome, input_dim=9, output_dim=1).to(device)
            transfer_model.load_state_dict(pretrained_state)
            
            start = time.time()
            transfer_model = train_model(transfer_model, train_subset_loader, epochs=FINE_TUNE_EPOCHS)
            transfer_time = time.time() - start
            
            transfer_rmse = evaluate_model(transfer_model, test_loader)
            print(f"Transfer RMSE: {transfer_rmse:.4f} ({transfer_time:.1f}s)")
            
            results['experiments'][pct_key]['transfer'].append({
                'rmse': float(transfer_rmse),
                'time': float(transfer_time)
            })
            
            
            # FROM SCRATCH: Train new model
            print("\n[Baseline] Training from scratch...")
            scratch_model = build_model_from_genome(genome, input_dim=9, output_dim=1).to(device)
            
            try:
                start = time.time()
                scratch_model = train_model(scratch_model, train_subset_loader, epochs=FROM_SCRATCH_EPOCHS)
                scratch_time = time.time() - start
                
                scratch_rmse = evaluate_model(scratch_model, test_loader)
                print(f"From-Scratch RMSE: {scratch_rmse:.4f} ({scratch_time:.1f}s)")
                
                results['experiments'][pct_key]['from_scratch'].append({
                    'rmse': float(scratch_rmse),
                    'time': float(scratch_time)
                })
                
                # Improvement
                improvement = ((scratch_rmse - transfer_rmse) / scratch_rmse) * 100
                print(f"Improvement: {improvement:.1f}%")
            except Exception as e:
                print(f"ERROR in from-scratch training: {e}")
                import traceback
                traceback.print_exc()
                # Use a fallback value
                results['experiments'][pct_key]['from_scratch'].append({
                    'rmse': float('inf'),
                    'time': 0.0
                })
        
        # Compute statistics
        transfer_rmses = [r['rmse'] for r in results['experiments'][pct_key]['transfer'] if r['rmse'] != float('inf')]
        scratch_rmses = [r['rmse'] for r in results['experiments'][pct_key]['from_scratch'] if r['rmse'] != float('inf')]
        
        if len(transfer_rmses) == 0 or len(scratch_rmses) == 0:
            print(f"\nWARNING: All trials failed for {pct*100:.0f}% data")
            continue
        
        # Mean and std
        transfer_mean = np.mean(transfer_rmses)
        transfer_std = np.std(transfer_rmses) if len(transfer_rmses) > 1 else 0.0
        scratch_mean = np.mean(scratch_rmses)
        scratch_std = np.std(scratch_rmses) if len(scratch_rmses) > 1 else 0.0
        
        # Statistical test (paired t-test) - only if we have >= 2 samples
        if len(transfer_rmses) >= 2 and len(scratch_rmses) >= 2:
            t_stat, p_value = stats.ttest_rel(scratch_rmses, transfer_rmses)
        else:
            t_stat, p_value = 0.0, 1.0  # Not enough samples
        
        # Overall improvement
        improvement = ((scratch_mean - transfer_mean) / scratch_mean) * 100
        
        results['experiments'][pct_key]['statistics'] = {
            'transfer_mean': float(transfer_mean),
            'transfer_std': float(transfer_std),
            'scratch_mean': float(scratch_mean),
            'scratch_std': float(scratch_std),
            'improvement_pct': float(improvement),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'n_successful_trials': len(transfer_rmses)
        }
        
        print(f"\n{'='*80}")
        print(f"SUMMARY FOR {pct*100:.0f}% DATA")
        print(f"{'='*80}")
        print(f"Successful trials: {len(transfer_rmses)}/{N_TRIALS}")
        print(f"Transfer:      {transfer_mean:.4f} ± {transfer_std:.4f}")
        print(f"From Scratch:  {scratch_mean:.4f} ± {scratch_std:.4f}")
        print(f"Improvement:   {improvement:.1f}%")
        if len(transfer_rmses) >= 2:
            print(f"P-value:       {p_value:.2e}")
            print(f"Significant:   {'YES' if p_value < 0.05 else 'NO'}")
        else:
            print(f"P-value:       N/A (need >= 2 trials)")
    
    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("TRANSFER LEARNING SUMMARY")
    print(f"{'='*80}")
    print(f"\nZero-Shot RMSE: {zero_shot_rmse:.4f}")
    print("\nFine-Tuning Results:")
    for pct in DATA_PERCENTAGES:
        pct_key = f"{pct*100:.0f}%"
        stats = results['experiments'][pct_key]['statistics']
        print(f"\n{pct*100:.0f}% Data:")
        print(f"  Transfer:    {stats['transfer_mean']:.4f} ± {stats['transfer_std']:.4f}")
        print(f"  From Scratch: {stats['scratch_mean']:.4f} ± {stats['scratch_std']:.4f}")
        print(f"  Improvement: {stats['improvement_pct']:.1f}%")
        print(f"  P-value:     {stats['p_value']:.2e}")
    
    return results


if __name__ == "__main__":
    results = run_transfer_learning()
