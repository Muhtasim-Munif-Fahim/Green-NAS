"""
Green AI Metrics: FLOPs & Latency
Quantifies computational efficiency to demonstrate suitability for edge deployment.

Metrics:
1. FLOPs (Floating Point Operations) - Estimated
2. Inference Latency (ms/sample) - Measured on CPU
3. Model Size (MB)
"""

import torch
import torch.nn as nn
import time
import numpy as np
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green_nas.search_space import build_model_from_genome, Genome

DEVICE = torch.device('cpu') # Measure on CPU to simulate edge device

def count_flops(model, input_shape):
    """
    Simple FLOP counter for standard layers.
    Input shape: (batch, seq_len, features)
    """
    total_flops = 0
    
    # Dummy input to trace
    x = torch.randn(input_shape)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # FLOPs = 2 * in * out (multiply + add)
            flops = 2 * module.in_features * module.out_features
            total_flops += flops
            
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            # FLOPs approx = 2 * 4 * (input + hidden) * hidden (for LSTM)
            # GRU is 3 gates instead of 4
            gates = 4 if isinstance(module, nn.LSTM) else 3
            input_sz = module.input_size
            hidden_sz = module.hidden_size
            
            # Per time step
            step_flops = 2 * gates * (input_sz + hidden_sz) * hidden_sz
            
            # Total = step * seq_len * num_layers * directions
            seq_len = input_shape[1]
            num_layers = module.num_layers
            dirs = 2 if module.bidirectional else 1
            
            total_flops += step_flops * seq_len * num_layers * dirs
            
        elif isinstance(module, nn.Conv1d):
            # FLOPs = 2 * kernel * in_ch * out_ch * length
            kernel = module.kernel_size[0]
            in_ch = module.in_channels
            out_ch = module.out_channels
            length = input_shape[1] # Approx (padding/stride affects this)
            
            total_flops += 2 * kernel * in_ch * out_ch * length
            
    return total_flops

def measure_latency(model, input_shape, runs=100):
    model.eval()
    model.to(DEVICE)
    input_data = torch.randn(input_shape).to(DEVICE)
    
    # Warmup
    for _ in range(10):
        _ = model(input_data)
        
    start = time.time()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(input_data)
    end = time.time()
    
    avg_latency_ms = ((end - start) / runs) * 1000
    return avg_latency_ms

def run_green_metrics():
    print("🚀 Starting Green AI Metrics Analysis")
    
    # Define Architectures to Compare
    # 1. Our Efficient Hybrid
    hybrid_genome = Genome({
        'model_type': 'hybrid',
        'hidden_size': 32,
        'num_layers': 2,
        'dropout': 0.0,
        'bidirectional': False
    })
    
    # 2. Standard GRU (Baseline)
    gru_genome = Genome({
        'model_type': 'gru',
        'hidden_size': 256,
        'num_layers': 3,
        'dropout': 0.0,
        'bidirectional': False
    })
    
    input_size = 8  # 8 standard weather features
    seq_len = 24
    batch_size = 1 # Single sample inference (Edge case)
    input_shape = (batch_size, seq_len, input_size)
    
    architectures = [
        ("Hybrid (Ours)", hybrid_genome),
        ("Standard GRU", gru_genome)
    ]
    
    results = []
    
    print(f"{'Model':<20} | {'Params':<10} | {'FLOPs (M)':<10} | {'Latency (ms)':<15} | {'Energy (uJ)':<15}")
    print("-" * 80)
    
    for name, genome in architectures:
        model = build_model_from_genome(genome, input_size)
        
        # Params
        params = sum(p.numel() for p in model.parameters())
        
        # FLOPs
        flops = count_flops(model, input_shape)
        flops_m = flops / 1e6
        
        # Latency
        latency = measure_latency(model, input_shape)
        
        # Energy Estimation (Rough proxy: 1pJ per FLOP is a common heuristic for mobile)
        # 1 pJ = 1e-12 J
        # Energy (uJ) = FLOPs * 1e-12 * 1e6 = FLOPs * 1e-6
        # This is very rough, but standard for "Green AI" comparisons without hardware meters
        energy_uj = flops * 1e-6 
        
        print(f"{name:<20} | {params:<10,} | {flops_m:<10.2f} | {latency:<15.2f} | {energy_uj:<15.2f}")
        
        results.append({
            "model": name,
            "params": params,
            "flops_M": flops_m,
            "latency_ms": latency,
            "energy_uj": energy_uj
        })
        
    # Save results
    with open("data/metadata/green_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved Green AI metrics to data/metadata/green_metrics.json")

if __name__ == "__main__":
    run_green_metrics()
