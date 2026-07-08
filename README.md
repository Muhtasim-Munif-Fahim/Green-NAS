# 🌿 Green-NAS

**A Global-Scale Multi-Objective Neural Architecture Search for Robust and Efficient Edge-Native Weather Forecasting**

[![Paper](https://img.shields.io/badge/IEEE_QPAIN_2026-Paper-blue)](https://ieeexplore.ieee.org/document/11545925/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.00240-b31b1b)](https://arxiv.org/abs/2602.00240)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)

> **Accepted at IEEE QPAIN 2026** — 2nd International Conference on Quantum Photonics, Artificial Intelligence, and Networking, Chittagong, Bangladesh.

---

## Overview

Green-NAS is a multi-objective neural architecture search (NAS) framework that automatically discovers compact, efficient weather forecasting models suitable for edge deployment. Following **Green AI** principles, it minimizes computational cost and carbon footprint while maintaining competitive accuracy.

### Key Results

| Model | Parameters | RMSE | Inference | Size | Description |
|-------|-----------|------|-----------|------|-------------|
| Manual Transformer | 135K | 0.0974 | 1.89 ms | 527 KB | Hand-designed baseline |
| Manual GRU | 153K | 0.1004 | 0.29 ms | 598 KB | Hand-designed baseline |
| **Green-NAS-A** | **153K** | **0.0988** | 0.47 ms | 598 KB | 🏆 Best accuracy (GRU×2) |
| **Green-NAS-B** | **4.2K** | **0.0996** | 0.29 ms | 17 KB | ⚖️ Balanced (CNN-128) |
| **Green-NAS-C** | **1.1K** | **0.1019** | 0.33 ms | 4 KB | 🔋 IoT-ready (CNN-32) |
| GraphCast | 36.7M | — | — | ~1 GB | Global gridded (different task) |

- **Green-NAS-A** matches manual baselines with only 1.4% higher RMSE
- **Green-NAS-C** is **35,500× smaller** than GraphCast with IoT-deployable 4 KB footprint
- **Transfer learning** improves accuracy by **5.2%** even at 100% data availability

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Green-NAS Pipeline                │
├─────────────┬───────────────────┬───────────────────┤
│  Data Layer │   Search Engine   │   Evaluation      │
│             │                   │                   │
│ 24 Cities   │  NSGA-II          │  RMSE             │
│ 8 Features  │  Pop: 20          │  Parameter Count  │
│ 1.07M Samples│ Gen: 10          │  Interpretability │
│             │                   │                   │
│ Source (18) │  Search Space:    │  Pareto Front:    │
│ Target (6)  │  ├─ LSTM/GRU      │  20 Architectures │
│             │  ├─ 1D CNN        │                   │
│             │  ├─ Attention     │  Transfer Learn:  │
│             │  └─ MLP           │  18 → 6 cities    │
└─────────────┴───────────────────┴───────────────────┘
```

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/Muhtasim-Munif-Fahim/Green-NAS.git
cd Green-NAS
pip install -r requirements.txt
```

### 2. Download Weather Data

```bash
python scripts/download_data.py
python scripts/preprocess.py
```

Data is sourced from the [Open-Meteo Historical Weather API](https://open-meteo.com/) covering 24 cities across Tropical, Arid, Temperate, and Continental climate zones (2019–2024, hourly).

### 3. Run NAS

```bash
python scripts/run_nas.py
```

Runs NSGA-II with 20 individuals × 10 generations on the source cities. Results are saved to `results/nas_generations/`.

### 4. Train & Evaluate

```bash
# Train manual baselines
python scripts/train_baselines.py

# Train NAS-discovered architectures
python scripts/train_discovered.py

# Run transfer learning experiments
python scripts/run_transfer.py

# Verify efficiency metrics
python scripts/verify_efficiency.py
```

---

## Repository Structure

```
Green-NAS/
├── config.py                # Shared configuration (cities, features, hyperparams)
├── green_nas/               # Core NAS framework
│   ├── search_space.py      # Architecture search space definition
│   ├── nsga2.py             # NSGA-II multi-objective optimizer
│   ├── evaluator.py         # Genome evaluation pipeline
│   ├── models.py            # Green-NAS model definitions (A/B/C)
│   ├── data_utils.py        # Data loading and preprocessing
│   └── metrics.py           # RMSE, MAE, and custom metrics
├── baselines/               # Baseline model implementations
│   ├── baseline_models.py   # LSTM, GRU, Transformer, CNN, MLP
│   └── transfer_learning.py # Cross-city transfer learning
├── analysis/                # Post-hoc analysis tools
│   ├── robustness.py        # Conformal prediction (95% coverage)
│   ├── uncertainty.py       # Uncertainty quantification
│   ├── xai_analysis.py      # Permutation importance & SHAP
│   ├── statistical_tests.py # Paired t-tests, Wilcoxon tests
│   └── green_metrics.py     # Energy, latency, carbon metrics
├── visualization/           # Figure generation
│   ├── plot_pareto.py       # Pareto front visualization
│   ├── plot_transfer.py     # Transfer learning curves
│   └── plot_nas.py          # NAS evolution & architecture plots
├── scripts/                 # Runnable experiment scripts
│   ├── download_data.py     # Fetch weather data from Open-Meteo
│   ├── preprocess.py        # Min-Max scaling, windowing
│   ├── run_nas.py           # Execute full NAS search
│   ├── train_baselines.py   # Train manual baseline models
│   ├── train_discovered.py  # Train Pareto-optimal architectures
│   ├── run_transfer.py      # Transfer learning experiments
│   └── verify_efficiency.py # Measure latency, model size, FLOPs
├── results/                 # Experiment outputs
│   ├── nas_generations/     # Per-generation Pareto fronts (JSON)
│   ├── experiment_results.json
│   └── transfer_learning_results.json
└── figures/                 # Publication figures
    ├── pareto_front.png
    ├── transfer_curve.png
    └── feature_importance.png
```

---

## Dataset

| Split | Cities | Samples | Purpose |
|-------|--------|---------|---------|
| Source (Train) | 18 cities (Athens, Delhi, Dhaka, Mumbai, ...) | 757,000 | NAS search & pre-training |
| Target (Test) | 6 cities (Santiago, Sofia, São Paulo, ...) | 315,000 | Evaluation & transfer learning |

**Features:** Temperature, Relative Humidity, Precipitation, Surface Pressure, Cloud Cover, Wind Speed, Wind Direction, Shortwave Radiation

**Time span:** January 2019 – December 2024 (hourly)

---

## Hardware Requirements

All experiments were conducted on a consumer-grade workstation to demonstrate accessibility:

- **GPU:** NVIDIA RTX 3060 Ti (8 GB VRAM)
- **NAS Runtime:** ~24 hours (20 pop × 10 gen)
- **Inference:** Sub-millisecond on CPU

> **Note on Energy Estimation:** Energy values in `analysis/green_metrics.py` use the standard `1 pJ/FLOP` heuristic commonly used in Green AI literature. These are *estimates*, not hardware power measurements. For exact energy profiling, dedicated hardware meters (e.g., NVIDIA SMI, Intel RAPL) are recommended.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{fahim2026greenas,
  title     = {Green-{NAS}: A Global-Scale Multi-Objective Neural Architecture
               Search for Robust and Efficient Edge-Native Weather Forecasting},
  author    = {Fahim, Md Muhtasim Munif and Yesmin, Soyda Humyra and
               Islam, Saiful and Faruque, Md. Palash Bin and
               Salam, Md. A. and Uddin, Md. Mahfuz and
               Islam, Samiul and Ahmed, Tofayel and
               Binyamin, Md. and Karim, Md. Rezaul},
  booktitle = {2026 IEEE 2nd International Conference on Quantum Photonics,
               Artificial Intelligence, and Networking (QPAIN)},
  year      = {2026},
  address   = {Chittagong, Bangladesh},
  month     = apr
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

Weather data provided by the [Open-Meteo Historical Weather API](https://open-meteo.com/). We thank the University of Rajshahi for institutional support.
