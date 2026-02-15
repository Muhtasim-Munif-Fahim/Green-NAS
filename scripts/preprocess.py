"""
Weather Data Preprocessing Pipeline
Preprocesses 24-city ERA5 weather data for NAS training

Input: Raw CSV files from Open-Meteo API
Output: Normalized sequences ready for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import json
from datetime import datetime
import pickle


def load_city_data(city_name, data_dir="data/raw/weather"):
    """Load raw weather data for a city"""
    data_dir = Path(data_dir)
    
    # Find the city file
    pattern = f"{city_name.lower().replace(' ', '_')}_*.csv"
    files = list(data_dir.glob(pattern))
    
    if not files:
        print(f"  ❌ No data found for {city_name}")
        return None
    
    # Load the CSV
    df = pd.read_csv(files[0])
    
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    
    return df


def check_data_quality(df, city_name):
    """
    Check data quality and return statistics
    
    Returns:
        dict with quality metrics
    """
    total_records = len(df)
    
    # Check for missing values
    missing_pct = (df.isnull().sum() / total_records * 100).to_dict()
    
    # Check date range
    date_range = (df['time'].min(), df['time'].max())
    duration_days = (date_range[1] - date_range[0]).days
    
    # Expected records (5 years * 365 days * 24 hours)
    expected_records = 5 * 365 * 24
    coverage_pct = (total_records / expected_records) * 100
    
    # Check for outliers in key variables
    outliers = {}
    if 'temperature_2m' in df.columns:
        temp = df['temperature_2m'].dropna()
        outliers['temperature'] = len(temp[(temp < -50) | (temp > 60)])
    
    quality_score = coverage_pct / 100 * (1 - df['temperature_2m'].isnull().sum() / total_records)
    
    return {
        'city': city_name,
        'total_records': total_records,
        'expected_records': expected_records,
        'coverage_pct': coverage_pct,
        'duration_days': duration_days,
        'date_range': [str(date_range[0]), str(date_range[1])],
        'missing_pct': {k: round(v, 2) for k, v in missing_pct.items() if v > 0},
        'outliers': outliers,
        'quality_score': round(quality_score, 3)
    }


def engineer_features(df):
    """Create additional features"""
    df = df.copy()
    
    # Time features
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    df['day_of_year'] = df['time'].dt.dayofyear
    
    # Cyclical encoding for time
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Wind speed (if components available)
    if 'wind_speed_10m' in df.columns and df['wind_speed_10m'].isnull().all():
        # Wind speed not directly available, skip
        pass
    
    # Lagged features (previous hours)
    for lag in [1, 3, 6, 12, 24]:
        if 'temperature_2m' in df.columns:
            df[f'temp_lag_{lag}h'] = df['temperature_2m'].shift(lag)
    
    return df


def create_sequences(df, feature_cols, target_col='temperature_2m', 
                     sequence_length=24, forecast_horizon=1):
    """
    Create sequences for time series forecasting
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        target_col: Target variable to predict
        sequence_length: Number of past timesteps (24 = 1 day)
        forecast_horizon: Hours ahead to predict (1 = next hour)
    
    Returns:
        X, y arrays
    """
    X, y = [], []
    
    for i in range(len(df) - sequence_length - forecast_horizon + 1):
        # Input sequence
        seq = df[feature_cols].iloc[i:i+sequence_length].values
        
        # Target (future value)
        target = df[target_col].iloc[i+sequence_length+forecast_horizon-1]
        
        # Only add if no NaN
        if not np.isnan(seq).any() and not np.isnan(target):
            X.append(seq)
            y.append(target)
    
    return np.array(X), np.array(y)


def preprocess_city(city_name, output_dir="data/processed/weather"):
    """
    Complete preprocessing pipeline for one city
    
    Returns:
        dict with processing results
    """
    print(f"\n{'='*70}")
    print(f"Processing: {city_name}")
    print(f"{'='*70}")
    
    # Load data
    df = load_city_data(city_name)
    if df is None:
        return None
    
    print(f"  Loaded: {len(df):,} records")
    
    # Quality check
    quality = check_data_quality(df, city_name)
    print(f"  Quality Score: {quality['quality_score']:.3f}")
    print(f"  Coverage: {quality['coverage_pct']:.1f}%")
    print(f"  Date Range: {quality['date_range'][0]} to {quality['date_range'][1]}")
    
    # Feature engineering
    df = engineer_features(df)
    print(f"  Features engineered: {len(df.columns)} total columns")
    
    # Select features for modeling
    feature_cols = [
        'temperature_2m', 'relative_humidity_2m', 'precipitation',
        'pressure_msl', 'surface_pressure', 'cloud_cover',
        'wind_speed_10m', 'shortwave_radiation',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        # Add lagged features if available
    ]
    
    # Filter to only available features
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Drop rows with missing values in key features
    df_clean = df.dropna(subset=feature_cols)
    print(f"  After cleaning: {len(df_clean):,} records ({len(df_clean)/len(df)*100:.1f}%)")
    
    # Normalize features
    scaler = MinMaxScaler()
    df_clean[feature_cols] = scaler.fit_transform(df_clean[feature_cols])
    
    # Create sequences
    X, y = create_sequences(df_clean, feature_cols, target_col='temperature_2m',
                           sequence_length=24, forecast_horizon=1)
    
    print(f"  Sequences created: {len(X):,} samples")
    print(f"  Input shape: {X.shape}, Target shape: {y.shape}")
    
    # Train/Val/Test split (70/15/15)
    n = len(X)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    # Save processed data
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    city_file = city_name.lower().replace(' ', '_')
    output_path = output_dir / f"{city_file}_processed.npz"
    
    np.savez_compressed(
        output_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=feature_cols
    )
    
    # Save scaler
    scaler_path = output_dir / f"{city_file}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"  ✅ Saved to: {output_path}")
    
    return {
        'city': city_name,
        'quality': quality,
        'n_sequences': len(X),
        'n_features': len(feature_cols),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'output_file': str(output_path)
    }


def preprocess_all_cities():
    """Preprocess all downloaded cities"""
    # Load manifest to get list of cities
    manifest_file = "data/metadata/weather_download_manifest.json"
    with open(manifest_file) as f:
        manifest = json.load(f)
    
    cities = [c['city'] for c in manifest['cities']]
    
    print("=" * 70)
    print(f"PREPROCESSING {len(cities)} CITIES")
    print("=" * 70)
    
    results = []
    
    for city in cities:
        result = preprocess_city(city)
        if result:
            results.append(result)
    
    # Save preprocessing summary
    summary = {
        'preprocessing_date': datetime.now().isoformat(),
        'total_cities': len(results),
        'total_sequences': sum(r['n_sequences'] for r in results),
        'cities': results
    }
    
    summary_file = "data/metadata/preprocessing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total cities processed: {len(results)}")
    print(f"Total sequences: {sum(r['n_sequences'] for r in results):,}")
    print(f"Summary saved to: {summary_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess weather data")
    parser.add_argument('--city', help='Process specific city')
    parser.add_argument('--all', action='store_true', help='Process all cities')
    
    args = parser.parse_args()
    
    if args.city:
        preprocess_city(args.city)
    elif args.all:
        preprocess_all_cities()
    else:
        print("Please specify --city NAME or --all")
