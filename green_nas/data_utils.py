"""
Standard Data Loading Utilities
Ensures consistent preprocessing across all experiments
"""
import torch
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import TensorDataset
from config import STANDARD_FEATURES, SEQ_LEN, DATA_DIR

def load_city_data(filename, verbose=False):
    """
    Load and preprocess a single city's data.
    
    Args:
        filename: CSV filename (e.g., "delhi_india_tier1.csv")
        verbose: Print loading info
    
    Returns:
        X: torch.Tensor of shape (N, SEQ_LEN, INPUT_DIM)
        y: torch.Tensor of shape (N, INPUT_DIM)
    
    Raises:
        FileNotFoundError: If city file doesn't exist
        ValueError: If required features are missing
    """
    path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"City file not found: {path}")
    
    df = pd.read_csv(path)
    
    # Verify all features exist
    missing = [f for f in STANDARD_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in {filename}: {missing}")
    
    # Extract features in standard order
    data = df[STANDARD_FEATURES].values.astype(np.float32)
    
    # MinMax scaling (per city, to preserve relative differences)
    min_val = np.min(data, axis=0, keepdims=True)
    max_val = np.max(data, axis=0, keepdims=True)
    
    # Avoid division by zero for constant features
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    
    data_scaled = (data - min_val) / range_val
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - SEQ_LEN):
        X.append(data_scaled[i : i + SEQ_LEN])
        y.append(data_scaled[i + SEQ_LEN])  # ALL 8 features (multivariate)
    
    if verbose:
        print(f"  {filename}: {len(X):,} samples")
    
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))


def create_dataset(city_files, verbose=True):
    """
    Combine multiple cities into a single dataset.
    
    Args:
        city_files: List of CSV filenames
        verbose: Print loading progress
    
    Returns:
        TensorDataset with X (sequences) and y (targets)
    """
    if verbose:
        print(f"Loading {len(city_files)} cities...")
    
    all_X, all_y = [], []
    
    for city in city_files:
        X, y = load_city_data(city, verbose=verbose)
        all_X.append(X)
        all_y.append(y)
    
    X_combined = torch.cat(all_X, dim=0)
    y_combined = torch.cat(all_y, dim=0)
    
    if verbose:
        print(f"Total: {len(X_combined):,} samples")
        print(f"Shape: X={X_combined.shape}, y={y_combined.shape}")
    
    return TensorDataset(X_combined, y_combined)


def verify_data_consistency(dataset1, dataset2, name1="Dataset 1", name2="Dataset 2"):
    """
    Verify that two datasets have compatible dimensions.
    
    Args:
        dataset1, dataset2: TensorDatasets to compare
        name1, name2: Names for error messages
    
    Raises:
        ValueError: If dimensions don't match
    """
    X1, y1 = dataset1[0]
    X2, y2 = dataset2[0]
    
    if X1.shape != X2.shape:
        raise ValueError(f"X shape mismatch: {name1} {X1.shape} vs {name2} {X2.shape}")
    
    if y1.shape != y2.shape:
        raise ValueError(f"y shape mismatch: {name1} {y1.shape} vs {name2} {y2.shape}")
    
    print(f"✓ {name1} and {name2} have consistent dimensions")
    print(f"  X: {X1.shape}, y: {y1.shape}")
