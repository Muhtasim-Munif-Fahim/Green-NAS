"""
Main NAS Execution Script
Runs NSGA-II evolutionary search to discover Pareto-optimal architectures
"""

import torch
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

from green_nas.search_space import Genome, build_model_from_genome, count_parameters, SEED_GENOMES, genome_to_string
from green_nas.nsga2 import Individual, fast_non_dominated_sort, crowding_distance_assignment, create_offspring, environmental_selection, extract_pareto_front
from green_nas.evaluator import FitnessEvaluator

# ==========================================
# Configuration
# ==========================================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "weather")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "nas_generations")
SEQ_LEN = 24
BATCH_SIZE = 64

# NAS Hyperparameters
POPULATION_SIZE = 20
GENERATIONS = 10
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.2
FAST_EPOCHS = 3  # For fitness evaluation

# Source cities for training (ALL 24 cities)
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
    "san_salvador_el_salvador_tier3.csv",
    "santiago_chile_tier1.csv",
    "sofia_bulgaria_tier2.csv",
    "são_paulo_brazil_tier1.csv",
    "windhoek_namibia_tier3.csv",
    "wuhan_china_tier3.csv",
    "zagreb_croatia_tier2.csv"
]

# ==========================================
# Data Loading
# ==========================================
def load_and_process_city(filename):
    """Load and preprocess city data."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {filename} not found")
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
    
    # Min-Max scaling
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    data_scaled = (data - min_val) / (max_val - min_val + 1e-6)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - SEQ_LEN - 1):
        X.append(data_scaled[i : i+SEQ_LEN])
        y.append(data_scaled[i+SEQ_LEN, 0])  # Predict temperature
    
    return torch.tensor(np.array(X)), torch.tensor(np.array(y)).unsqueeze(1)


def get_dataloader(cities, batch_size=BATCH_SIZE):
    """Load multiple cities into a single dataloader."""
    all_X, all_y = [], []
    for city in cities:
        X, y = load_and_process_city(city)
        if X is not None:
            all_X.append(X)
            all_y.append(y)
    
    if not all_X:
        return None, 0
    
    combined_X = torch.cat(all_X)
    combined_y = torch.cat(all_y)
    
    dataset = TensorDataset(combined_X, combined_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), combined_X.shape[2]


# ==========================================
# NAS Main Loop
# ==========================================
def run_nas():
    """Execute NSGA-II Neural Architecture Search."""
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"nas_log_{timestamp}.txt")
    
    def log(msg):
        """Log to both console and file."""
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
    
    log("="*80)
    log("Green-NAS: NSGA-II Neural Architecture Search")
    log("="*80)
    log(f"Start time: {datetime.now()}")
    log(f"Population size: {POPULATION_SIZE}")
    log(f"Generations: {GENERATIONS}")
    log(f"Crossover prob: {CROSSOVER_PROB}")
    log(f"Mutation prob: {MUTATION_PROB}")
    log("")
    
    # Load data
    log("Loading data...")
    train_loader, input_dim = get_dataloader(SOURCE_CITIES)
    if train_loader is None:
        log("ERROR: Could not load data!")
        return
    
    # Split into train/val
    full_dataset = train_loader.dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    log(f"Input dimension: {input_dim}")
    log(f"Train samples: {len(train_dataset)}")
    log(f"Val samples: {len(val_dataset)}")
    log("")
    
    # Initialize evaluator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"Device: {device}")
    
    evaluator = FitnessEvaluator(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        device=device,
        fast_epochs=FAST_EPOCHS
    )
    
    # Initialize population (seed with known good architectures)
    log("\n" + "="*80)
    log("INITIALIZATION")
    log("="*80)
    log("Creating initial population...")
    
    population = []
    
    # Add seed genomes
    for seed_genome in SEED_GENOMES:
        population.append(Individual(seed_genome))
    
    # Fill with random genomes
    while len(population) < POPULATION_SIZE:
        population.append(Individual(Genome()))
    
    log(f"Population size: {len(population)}")
    
    # Evaluate initial population
    log("\nEvaluating initial population...")
    eval_start = time.time()
    for i, ind in enumerate(population):
        ind.objectives = evaluator.evaluate(ind.genome)
        log(f"  [{i+1}/{len(population)}] {genome_to_string(ind.genome)}: RMSE={ind.objectives[0]:.4f}, Params={ind.objectives[1]:,}")
    eval_time = time.time() - eval_start
    log(f"Evaluation time: {eval_time:.2f}s ({eval_time/len(population):.2f}s per genome)")
    
    # Evolution loop
    log("\n" + "="*80)
    log("EVOLUTION")
    log("="*80)
    
    generation_history = []
    
    for gen in range(GENERATIONS):
        gen_start = time.time()
        log(f"\n--- Generation {gen+1}/{GENERATIONS} ---")
        
        # Create offspring
        offspring = create_offspring(
            population, 
            offspring_size=POPULATION_SIZE,
            crossover_prob=CROSSOVER_PROB,
            mutation_prob=MUTATION_PROB
        )
        
        # Evaluate offspring
        log("Evaluating offspring...")
        for i, ind in enumerate(offspring):
            ind.objectives = evaluator.evaluate(ind.genome)
        
        # Environmental selection
        combined = population + offspring
        population = environmental_selection(combined, POPULATION_SIZE)
        
        # Extract and log Pareto front
        pareto = extract_pareto_front(population)
        log(f"Pareto front size: {len(pareto)}")
        
        # Log best individuals
        pareto_sorted = sorted(pareto, key=lambda x: x.objectives[0])  # Sort by RMSE
        log("\nTop 3 Pareto-optimal architectures:")
        for i, ind in enumerate(pareto_sorted[:3]):
            log(f"  {i+1}. {genome_to_string(ind.genome)}")
            log(f"     RMSE={ind.objectives[0]:.4f}, Params={ind.objectives[1]:,}, Depth={ind.objectives[2]}")
        
        # Save generation stats
        gen_stats = {
            'generation': gen + 1,
            'pareto_size': len(pareto),
            'best_rmse': min(ind.objectives[0] for ind in population),
            'best_params': min(ind.objectives[1] for ind in population),
            'avg_rmse': np.mean([ind.objectives[0] for ind in population]),
            'time': time.time() - gen_start
        }
        generation_history.append(gen_stats)
        
        log(f"Generation time: {gen_stats['time']:.2f}s")
        log(f"Best RMSE: {gen_stats['best_rmse']:.4f}")
        log(f"Best Params: {gen_stats['best_params']:,}")
    
    # Final Pareto front
    log("\n" + "="*80)
    log("FINAL RESULTS")
    log("="*80)
    
    final_pareto = extract_pareto_front(population)
    log(f"\nFinal Pareto front size: {len(final_pareto)}")
    
    # Sort by objectives for selection
    pareto_by_rmse = sorted(final_pareto, key=lambda x: x.objectives[0])
    pareto_by_params = sorted(final_pareto, key=lambda x: x.objectives[1])
   
    # Select Green-NAS-A, B, C
    green_nas_a = pareto_by_rmse[0]  # Best accuracy
    green_nas_c = pareto_by_params[0]  # Best efficiency
    
    # Balanced: find point closest to "knee" of Pareto front
    # Simple heuristic: middle of sorted-by-RMSE list
    green_nas_b = pareto_by_rmse[len(pareto_by_rmse) // 2] if len(pareto_by_rmse) > 2 else pareto_by_rmse[0]
    
    selected = {
        'Green-NAS-A (High Accuracy)': green_nas_a,
        'Green-NAS-B (Balanced)': green_nas_b,
        'Green-NAS-C (High Efficiency)': green_nas_c
    }
    
    log("\nSelected Architectures:")
    for name, ind in selected.items():
        log(f"\n{name}:")
        log(f"  Architecture: {genome_to_string(ind.genome)}")
        log(f"  RMSE: {ind.objectives[0]:.4f}")
        log(f"  Parameters: {ind.objectives[1]:,}")
        log(f"  Depth: {ind.objectives[2]}")
    
    # Save results
    results = {
        'config': {
            'population_size': POPULATION_SIZE,
            'generations': GENERATIONS,
            'crossover_prob': CROSSOVER_PROB,
            'mutation_prob': MUTATION_PROB,
            'fast_epochs': FAST_EPOCHS
        },
        'selected_architectures': {
            name: {
                'genome': ind.genome.to_vector().tolist(),
                'architecture': genome_to_string(ind.genome),
                'rmse': float(ind.objectives[0]),
                'params': int(ind.objectives[1]),
                'depth': int(ind.objectives[2])
            }
            for name, ind in selected.items()
        },
        'final_pareto_front': [
            {
                'architecture': genome_to_string(ind.genome),
                'rmse': float(ind.objectives[0]),
                'params': int(ind.objectives[1]),
                'depth': int(ind.objectives[2])
            }
            for ind in final_pareto
        ],
        'generation_history': generation_history,
        'evaluator_stats': evaluator.get_stats()
    }
    
    results_file = os.path.join(RESULTS_DIR, f"nas_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    log(f"\nResults saved to: {results_file}")
    log(f"Total evaluations: {evaluator.get_stats()['total_evaluations']}")
    log(f"Cache hit rate: {evaluator.get_stats()['cache_hit_rate']:.2%}")
    log(f"\nNAS complete! Time: {datetime.now()}")
    
    return results


if __name__ == "__main__":
    run_nas()
