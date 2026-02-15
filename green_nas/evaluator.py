"""
NAS Fitness Evaluator
Fast training and evaluation of candidate architectures during NAS search.

Objectives:
    1. Minimize RMSE (accuracy)
    2. Minimize parameter count (efficiency)
    3. Minimize depth / num_layers (interpretability proxy)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import sys
from typing import Tuple, Dict
from torch.utils.data import DataLoader

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green_nas.search_space import Genome, build_model_from_genome, count_parameters


class FitnessEvaluator:
    """
    Evaluates genome fitness via fast proxy training.

    During NAS search, each candidate architecture is trained for only
    a few epochs on a subset of data to provide a rough fitness estimate.
    """

    def __init__(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 input_dim: int,
                 output_dim: int = 1,
                 device: str = 'cpu',
                 fast_epochs: int = 3,
                 cache_size: int = 100):
        """
        Args:
            train_loader: Training data
            val_loader: Validation data for RMSE
            input_dim: Number of input features (e.g. 8)
            output_dim: Number of output features (1 for proxy, 8 for full)
            device: 'cpu' or 'cuda'
            fast_epochs: Number of epochs for fast evaluation
            cache_size: Max cached evaluations
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.fast_epochs = fast_epochs

        # Cache to avoid re-evaluating identical genomes
        self.cache = {}
        self.cache_size = cache_size
        self.evaluations = 0

    def _genome_hash(self, genome: Genome) -> str:
        """Create a hashable string representation of a genome."""
        return str(sorted(genome.genes.items()))

    def fast_train(self, model: nn.Module) -> float:
        """
        Quick proxy training for fitness evaluation.
        Uses 50% of batches and fewer epochs for speed.

        Returns:
            Validation RMSE
        """
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(self.fast_epochs):
            total_loss = 0
            batch_count = 0

            for X_batch, y_batch in self.train_loader:
                # Use only 50% of batches for speed
                if np.random.random() > 0.5:
                    continue

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                output = model(X_batch)

                # Handle shape mismatch
                if output.dim() == 2 and y_batch.dim() == 1:
                    y_batch = y_batch.unsqueeze(1)

                loss = criterion(output, y_batch)
                loss.backward()

                # Gradient clipping to prevent divergence
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()
                batch_count += 1

            # Early stopping if loss is NaN
            if np.isnan(total_loss):
                return float('inf')

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                output = model(X_batch)

                if output.dim() == 2 and y_batch.dim() == 1:
                    y_batch = y_batch.unsqueeze(1)

                loss = criterion(output, y_batch)
                val_loss += loss.item()
                val_batches += 1

        rmse = np.sqrt(val_loss / val_batches) if val_batches > 0 else float('inf')
        return rmse

    def evaluate(self, genome: Genome) -> Tuple[float, int, int]:
        """
        Evaluate a genome and return multi-objective fitness.

        Returns:
            (RMSE, parameter_count, num_layers)
        """
        # Check cache
        genome_key = self._genome_hash(genome)
        if genome_key in self.cache:
            return self.cache[genome_key]

        try:
            # Build model
            model = build_model_from_genome(
                genome, self.input_dim, output_size=self.output_dim
            )

            # Count parameters
            params = count_parameters(model)

            # Skip models that are too large for edge deployment
            if params > 500_000:
                objectives = (1.0, params, genome.genes['num_layers'])
            else:
                rmse = self.fast_train(model)
                objectives = (rmse, params, genome.genes['num_layers'])

            # Cache result
            if len(self.cache) < self.cache_size:
                self.cache[genome_key] = objectives

            self.evaluations += 1
            return objectives

        except Exception as e:
            print(f"Evaluation failed for {genome}: {e}")
            return (1.0, 1_000_000, genome.genes['num_layers'])

    def batch_evaluate(self, genomes: list) -> list:
        """Evaluate multiple genomes."""
        return [self.evaluate(g) for g in genomes]

    def get_stats(self) -> Dict:
        """Return evaluation statistics."""
        return {
            'total_evaluations': self.evaluations,
            'cache_hits': len(self.cache),
            'cache_hit_rate': len(self.cache) / max(1, self.evaluations)
        }
