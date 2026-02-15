"""
Train Manual Baseline Models: GRU, Transformer, TCN
To complete the comparison table in the manuscript.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
from torch.utils.data import DataLoader, TensorDataset

# Set seed for reproducibility
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "weather")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "baseline_results.json")
LOOKBACK = 24
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Source cities (18) - same as used for NAS
SOURCE_CITIES = [
    "athens_greece_tier1.csv",
    "belgrade_serbia_tier3.csv",
    "buenos_aires_argentina_tier1.csv",
    "busan_south_korea_tier3.csv",
    "chengdu_china_tier3.csv",
    "chongqing_china_tier2.csv",
    "delhi_india_tier1.csv",
    "dhaka_bangladesh_tier1.csv",
    "harare_zimbabwe_tier3.csv",
    "kiev_ukraine_tier2.csv",
    "kolkata_india_tier1.csv",
    "lahore_pakistan_tier1.csv",
    "lima_peru_tier1.csv",
    "luanda_angola_tier3.csv",
    "lusaka_zambia_tier3.csv",
    "maputo_mozambique_tier3.csv",
    "mumbai_india_tier1.csv",
    "san_salvador_el_salvador_tier3.csv"
]

# Target cities (6) - held-out test set
TARGET_CITIES = [
    "santiago_chile_tier1.csv",
    "sofia_bulgaria_tier2.csv",
    "são_paulo_brazil_tier1.csv",
    "windhoek_namibia_tier3.csv",
    "wuhan_china_tier3.csv",
    "zagreb_croatia_tier2.csv"
]

# Model Definitions

class ManualGRU(nn.Module):
    """Standard 2-layer GRU baseline"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

class ManualTransformer(nn.Module):
    """Transformer baseline (4 layers, 4 heads)"""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, input_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

class ManualTCN(nn.Module):
    """Temporal Convolutional Network baseline"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, kernel_size=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            in_channels = input_dim if i == 0 else hidden_dim
            layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size, dilation=dilation, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        # Take last timestep
        x = x[:, :, -1]
        return self.fc(x)

# Data Loading
def load_data(city_files):
    X_all, y_all = [], []
    for city in city_files:
        path = os.path.join(DATA_DIR, city)
        df = pd.read_csv(path)
        
        # Select numeric features (same as in NAS training)
        feature_cols = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 
                       'surface_pressure', 'cloud_cover', 'wind_speed_10m', 
                       'wind_direction_10m', 'shortwave_radiation']
        data = df[feature_cols].values
        
        # MinMax scaling per city
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)
        
        # Create sequences
        for i in range(len(data) - LOOKBACK):
            X_all.append(data[i:i+LOOKBACK])
            y_all.append(data[i+LOOKBACK])
    
    return torch.FloatTensor(X_all), torch.FloatTensor(y_all)

# Training
def train_model(model, train_loader, epochs=20, lr=1e-3):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")
    
    return model

# Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
    
    rmse = np.sqrt(total_loss / len(test_loader))
    return rmse

# Benchmark latency
def benchmark_latency(model, input_dim, warmup=10, iters=100):
    model.eval()
    dummy_input = torch.randn(1, LOOKBACK, input_dim).to(DEVICE)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    avg_latency_ms = ((end - start) / iters) * 1000
    return avg_latency_ms

def main():
    print("="*80)
    print("Training Manual Baseline Models")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    X_train, y_train = load_data(SOURCE_CITIES)
    X_test, y_test = load_data(TARGET_CITIES)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    input_dim = X_train.shape[2]
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512)
    
    results = {}
    
    # 1. Manual GRU
    print("\n" + "="*80)
    print("Training Manual GRU (2 layers, 128 hidden)")
    print("="*80)
    model_gru = ManualGRU(input_dim, hidden_dim=128, num_layers=2)
    print(f"Parameters: {sum(p.numel() for p in model_gru.parameters()):,}")
    model_gru = train_model(model_gru, train_loader, epochs=20)
    rmse_gru = evaluate_model(model_gru, test_loader)
    latency_gru = benchmark_latency(model_gru, input_dim)
    params_gru = sum(p.numel() for p in model_gru.parameters())
    
    results['Manual_GRU'] = {
        'rmse': float(rmse_gru),
        'params': int(params_gru),
        'latency_ms': float(latency_gru),
        'composition': '2-Layer GRU (128 hidden)'
    }
    print(f"Test RMSE: {rmse_gru:.6f}")
    print(f"Latency: {latency_gru:.2f} ms")
    
    # 2. Transformer
    print("\n" + "="*80)
    print("Training Transformer (4 layers, 4 heads)")
    print("="*80)
    model_transformer = ManualTransformer(input_dim, d_model=64, nhead=4, num_layers=4)
    print(f"Parameters: {sum(p.numel() for p in model_transformer.parameters()):,}")
    model_transformer = train_model(model_transformer, train_loader, epochs=20)
    rmse_transformer = evaluate_model(model_transformer, test_loader)
    latency_transformer = benchmark_latency(model_transformer, input_dim)
    params_transformer = sum(p.numel() for p in model_transformer.parameters())
    
    results['Manual_Transformer'] = {
        'rmse': float(rmse_transformer),
        'params': int(params_transformer),
        'latency_ms': float(latency_transformer),
        'composition': '4-Layer Transformer (4 heads)'
    }
    print(f"Test RMSE: {rmse_transformer:.6f}")
    print(f"Latency: {latency_transformer:.2f} ms")
    
    # 3. TCN
    print("\n" + "="*80)
    print("Training TCN (3 layers, kernel=3)")
    print("="*80)
    model_tcn = ManualTCN(input_dim, hidden_dim=64, num_layers=3)
    print(f"Parameters: {sum(p.numel() for p in model_tcn.parameters()):,}")
    model_tcn = train_model(model_tcn, train_loader, epochs=20)
    rmse_tcn = evaluate_model(model_tcn, test_loader)
    latency_tcn = benchmark_latency(model_tcn, input_dim)
    params_tcn = sum(p.numel() for p in model_tcn.parameters())
    
    results['Manual_TCN'] = {
        'rmse': float(rmse_tcn),
        'params': int(params_tcn),
        'latency_ms': float(latency_tcn),
        'composition': '3-Layer TCN (64 hidden)'
    }
    print(f"Test RMSE: {rmse_tcn:.6f}")
    print(f"Latency: {latency_tcn:.2f} ms")
    
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("BASELINE TRAINING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("\nSummary:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  Params: {metrics['params']:,}")
        print(f"  Latency: {metrics['latency_ms']:.2f} ms")

if __name__ == "__main__":
    main()
