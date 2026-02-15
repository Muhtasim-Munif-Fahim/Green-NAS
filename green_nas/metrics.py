"""
Evaluation Metrics for Weather Forecasting Models
"""
import torch
import torch.nn as nn
import numpy as np
import time

def evaluate_rmse(model, test_loader, device='cpu'):
    """
    Compute RMSE correctly (per-sample, not per-batch).
    """
    model.eval()
    all_squared_errors = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            
            # Compute squared errors per sample, averaged across features
            # Shape: (batch_size, num_features) -> (batch_size,)
            squared_errors_per_sample = ((y_pred - y_batch) ** 2).mean(dim=1)
            
            all_squared_errors.append(squared_errors_per_sample.cpu().numpy())
    
    # Concatenate all batches: total_samples
    all_squared_errors = np.concatenate(all_squared_errors)
    
    # RMSE = sqrt(mean(all squared errors))
    rmse = np.sqrt(np.mean(all_squared_errors))
    
    return float(rmse)


def evaluate_rmse_per_feature(model, test_loader, feature_names=None, device='cpu'):
    """
    Compute RMSE for each feature separately (for debugging).
    """
    model.eval()
    all_errors_per_feature = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            
            # Squared errors: (batch_size, num_features)
            squared_errors = ((y_pred - y_batch) ** 2).cpu().numpy()
            all_errors_per_feature.append(squared_errors)
    
    # Concatenate: (total_samples, num_features)
    all_errors = np.concatenate(all_errors_per_feature, axis=0)
    
    # RMSE per feature: (num_features,)
    rmse_per_feature = np.sqrt(np.mean(all_errors, axis=0))
    
    # Create dict
    if feature_names is None:
        return {f"Feature_{i}": float(rmse) for i, rmse in enumerate(rmse_per_feature)}
    else:
        return {name: float(rmse) for name, rmse in zip(feature_names, rmse_per_feature)}


def evaluate_mae(model, test_loader, device='cpu'):
    """
    Compute Mean Absolute Error (MAE) as an alternative metric.
    """
    model.eval()
    all_abs_errors = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            
            # Absolute errors per sample, averaged across features
            abs_errors_per_sample = torch.abs(y_pred - y_batch).mean(dim=1)
            
            all_abs_errors.append(abs_errors_per_sample.cpu().numpy())
    
    all_abs_errors = np.concatenate(all_abs_errors)
    mae = np.mean(all_abs_errors)
    
    return float(mae)

def benchmark_latency(model, input_dim, seq_len=24, warmup=10, iters=100, device='cpu'):
    """Benchmark inference latency"""
    model.eval()
    dummy_input = torch.randn(1, seq_len, input_dim).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Benchmark
    if torch.cuda.is_available() and device != 'cpu':
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_input)
    
    if torch.cuda.is_available() and device != 'cpu':
        torch.cuda.synchronize()
    
    end = time.time()
    avg_latency_ms = ((end - start) / iters) * 1000
    return avg_latency_ms
