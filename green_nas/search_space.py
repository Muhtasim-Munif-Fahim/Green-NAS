"""
Neural Architecture Search Space Definition
Genome encoding and model building for Green-NAS

The Genome class encodes a candidate architecture as a dictionary of genes.
The build_model_from_genome function constructs a PyTorch model from a Genome.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import copy
import time
import os
import sys

# Ensure project root is on path for cross-module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.baseline_models import (
    BaselineLSTM, BaselineGRU, BaselineAttention, HybridCNNLSTM, SimpleMLP
)

# ==========================================
# Search Space Configuration
# ==========================================
SEARCH_SPACE = {
    'model_type': ['lstm', 'gru', 'attention', 'hybrid'],
    'hidden_size': [32, 64, 128, 256],
    'num_layers': [1, 2, 3, 4],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'bidirectional': [True, False]   # Only applicable to LSTM
}

# Derived constants
LAYER_TYPES = SEARCH_SPACE['model_type']
HIDDEN_DIMS = SEARCH_SPACE['hidden_size']
MAX_DEPTH = max(SEARCH_SPACE['num_layers'])
MIN_DEPTH = min(SEARCH_SPACE['num_layers'])
DROPOUT_RANGE = (min(SEARCH_SPACE['dropout']), max(SEARCH_SPACE['dropout']))
MAX_PARAMS = 500_000  # Edge device constraint


class Genome:
    """
    Represents a candidate architecture as a searchable genome.

    Genes (dict):
        model_type  : str   — one of 'lstm', 'gru', 'attention', 'hybrid'
        hidden_size : int   — one of 32, 64, 128, 256
        num_layers  : int   — 1 to 4
        dropout     : float — 0.0 to 0.5
        bidirectional: bool — only used when model_type == 'lstm'
    """

    def __init__(self, genes=None):
        if genes:
            self.genes = genes
        else:
            self.genes = self.random_genes()

        self.fitness = None   # (rmse, params, depth)
        self.rank = None
        self.crowding_distance = 0.0
        self.domination_count = 0
        self.dominated_solutions = []
        self.id = f"{int(time.time() * 1000)}_{random.randint(0, 9999)}"

    def random_genes(self):
        """Generate random genes from the search space."""
        genes = {}
        for key, values in SEARCH_SPACE.items():
            genes[key] = random.choice(values)

        # Constraint: bidirectional only applies to LSTM
        if genes['model_type'] != 'lstm':
            genes['bidirectional'] = False

        return genes

    def mutate(self):
        """Mutate one random gene."""
        key = random.choice(list(SEARCH_SPACE.keys()))
        self.genes[key] = random.choice(SEARCH_SPACE[key])

        # Re-apply constraints
        if self.genes['model_type'] != 'lstm':
            self.genes['bidirectional'] = False

        self.fitness = None
        self.id = f"{self.id}_mut"

    def validate(self):
        """Check if genome is structurally valid."""
        g = self.genes
        if g['model_type'] not in SEARCH_SPACE['model_type']:
            return False
        if g['hidden_size'] not in SEARCH_SPACE['hidden_size']:
            return False
        if g['num_layers'] not in SEARCH_SPACE['num_layers']:
            return False
        if g['dropout'] not in SEARCH_SPACE['dropout']:
            return False
        if g['model_type'] != 'lstm' and g['bidirectional']:
            return False
        return True

    def __repr__(self):
        return (f"Genome(type={self.genes['model_type']}, "
                f"hidden={self.genes['hidden_size']}, "
                f"layers={self.genes['num_layers']}, "
                f"dropout={self.genes['dropout']})")


def build_model_from_genome(genome, input_size, output_size=1):
    """
    Construct a PyTorch model from a Genome specification.

    Args:
        genome: Genome instance (uses genome.genes dict)
        input_size: Number of input features (e.g. 8)
        output_size: Number of output features (default 1)

    Returns:
        nn.Module instance
    """
    g = genome.genes

    if g['model_type'] == 'lstm':
        return BaselineLSTM(
            input_size=input_size,
            hidden_size=g['hidden_size'],
            num_layers=g['num_layers'],
            output_size=output_size,
            dropout=g['dropout'],
            bidirectional=g['bidirectional']
        )
    elif g['model_type'] == 'gru':
        return BaselineGRU(
            input_size=input_size,
            hidden_size=g['hidden_size'],
            num_layers=g['num_layers'],
            output_size=output_size,
            dropout=g['dropout']
        )
    elif g['model_type'] == 'attention':
        return BaselineAttention(
            input_size=input_size,
            d_model=g['hidden_size'],
            num_layers=g['num_layers'],
            output_size=output_size,
            dropout=g['dropout']
        )
    elif g['model_type'] == 'hybrid':
        return HybridCNNLSTM(
            input_size=input_size,
            lstm_hidden=g['hidden_size'],
            output_size=output_size,
            dropout=g['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {g['model_type']}")


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def genome_to_string(genome):
    """Convert genome to a human-readable string for logging."""
    g = genome.genes
    return f"{g['model_type'].upper()}-h{g['hidden_size']}-L{g['num_layers']}-d{g['dropout']}"


# ==========================================
# Seed Genomes (for seeding the initial population)
# ==========================================
SEED_GENOMES = [
    Genome(genes={'model_type': 'hybrid', 'hidden_size': 64,
                  'num_layers': 3, 'dropout': 0.0, 'bidirectional': False}),
    Genome(genes={'model_type': 'lstm', 'hidden_size': 64,
                  'num_layers': 2, 'dropout': 0.1, 'bidirectional': True}),
    Genome(genes={'model_type': 'gru', 'hidden_size': 128,
                  'num_layers': 2, 'dropout': 0.1, 'bidirectional': False}),
]
