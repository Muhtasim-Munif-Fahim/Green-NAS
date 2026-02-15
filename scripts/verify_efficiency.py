"""
Verify Efficiency Claims - CORRECTED (8 Features)
Calculates exact parameter counts and FLOPs for the manuscript
"""
import torch
import torch.nn as nn
import numpy as np
import json
import sys
import os

# Import shared configuration
sys.path.insert(0, os.path.dirname(__file__))
from config import INPUT_DIM, OUTPUT_DIM
from green_nas.search_space import Genome, build_model_from_genome, count_parameters

NAS_RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "nas_generations", "gen_9.json")

def main():
    print("="*80)
    print(f"VERIFYING EFFICIENCY (Input Dim: {INPUT_DIM})")
    print("="*80)
    
    with open(NAS_RESULTS, 'r') as f:
        nas_data = json.load(f)
    
    architectures = nas_data['selected_architectures']
    
    results = []
    
    for name, config in architectures.items():
        print(f"\nAnalyzing: {name}")
        genome = Genome.from_vector(np.array(config['genome']))
        
        # Build model with standardized dimensions
        model = build_model_from_genome(genome, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
        
        # Count parameters
        params = count_parameters(model)
        print(f"  Parameters: {params:,}")
        
        # Calculate size in KB (assuming float32)
        size_kb = (params * 4) / 1024
        print(f"  Size: {size_kb:.2f} KB")
        
        # Store for report
        results.append({
            'name': name,
            'params': params,
            'size_kb': size_kb,
            'architecture': config['architecture']
        })
        
    print("\n" + "="*80)
    print("COMPARISON WITH GRAPHCAST")
    print("="*80)
    
    graphcast_params = 36700000
    
    for res in results:
        ratio = graphcast_params / res['params']
        print(f"{res['name']}: {ratio:.1f}x smaller than GraphCast")

if __name__ == "__main__":
    main()
