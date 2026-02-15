"""
Shared Configuration for All Experiments
Ensures consistency across NAS, baselines, and transfer learning
"""
import random
import numpy as np
import torch

# ==========================================
# Reproducibility
# ==========================================
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to {seed}")

# ==========================================
# Standard Feature Set (8 features)
# ==========================================
STANDARD_FEATURES = [
    'temperature_2m',
    'relative_humidity_2m', 
    'precipitation',
    'surface_pressure',  # NOT pressure_msl (redundant with this)
    'cloud_cover',
    'wind_speed_10m',
    'wind_direction_10m',
    'shortwave_radiation'
]

import os

# ==========================================
# Data Configuration
# ==========================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "weather")
SEQ_LEN = 24  # 24-hour lookback window
INPUT_DIM = len(STANDARD_FEATURES)  # 8
OUTPUT_DIM = INPUT_DIM  # Predict all features (multivariate forecasting)

# ==========================================
# Training Configuration
# ==========================================
BATCH_SIZE = 128  # Standardized batch size
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50  # With early stopping

# ==========================================
# City Lists
# ==========================================
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

TARGET_CITIES = [
    "santiago_chile_tier1.csv",
    "sofia_bulgaria_tier2.csv",
    "são_paulo_brazil_tier1.csv",
    "windhoek_namibia_tier3.csv",
    "wuhan_china_tier3.csv",
    "zagreb_croatia_tier2.csv"
]

ALL_CITIES = SOURCE_CITIES + TARGET_CITIES
