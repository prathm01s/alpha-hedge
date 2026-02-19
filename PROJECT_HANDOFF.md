# Project Alpha-Hedge: Handoff Documentation

This document provides a detailed overview of the project structure, explaining the purpose of every file and folder in the repository.

## 1. Source Code (`src/`)
This directory contains the core Python scripts that power the application.

- **`pipeline.py`**: The master orchestrator script. It automates the entire benchmark workflow: generating datasets, training surrogate models, running hedging simulations, and generating visualizations for N independent datasets.
- **`pricing_engines.py`**: The mathematical core. Contains the classes for option pricing:
    - `BlackScholesEngine`: Analytical formula.
    - `HestonMCEngine`: GPU-accelerated Monte Carlo simulation for Stochastic Volatility.
    - `MertonJumpMCEngine`: GPU-accelerated Monte Carlo for Jump Diffusion.
    - `DeepSurrogateEngine`: The neural network wrapper that predicts prices and Greeks.
- **`hedge_simulator.py`**: The trading engine. Simulates a daily delta-hedging strategy over a specified horizon, tracking PnL, transaction costs, and "Greeks" attribution for different pricing models.
- **`train_surrogate.py`**: The machine learning trainer. Loads parquet data, preprocesses it (StandardScaler), and trains a PyTorch Deep Neural Network (DNN) to approximate Heston prices.
- **`generate_dataset.py`**: The data factory. Generates large-scale synthetic option pricing data (inputs: S, K, T, r, market params; output: Price) using the Monte Carlo engines.
- **`visualize_alpha_hedge.py`**: The visualization suite. Generates the project's key plots: Volatility Smile, Greeks Surface, Hedging Error Heatmaps, and PnL Scatter plots.
- **`run_generalization_matrix.py`**: The stress test. An advanced script that trains models on one "world" (e.g., Black-Scholes) and tests them on another (e.g., Heston) to compute a 3x3 Generalization Error Matrix.
- **`compare_engines.py`**: A sandbox script for side-by-side comparison of pricing engine outputs for single data points.

## 2. Benchmark Outputs

- **`benchmark_suite/`**: The primary output directory for the full-scale autonomous pipeline.
    - **`dataset_XX/`**: Contains artifacts for a specific independent dataset run.
        - `train_data.parquet`: The raw training data (100k samples).
        - `model.pth` / `scaler.joblib`: The trained surrogate model artifacts.
        - `run_XX/`: Results from specific simulation runs (1000 paths). Contains `hedging_results.csv` and all PNG plots.
    - **`generalization_matrix.png`**: The final heatmap showing cross-model performance.

- **`sim_run_YYYYMMDD_.../`**: Directories created by manual execution of `hedge_simulator.py` without an explicit output argument. Contains ad-hoc simulation results.

- **`visualizations/`**: Default output directory for `run_generalization_matrix.py` manually runs.

## 4. Models & Configuration

- **`models/`**: A directory storing trained model checkpoints (`.pth`) and Scikit-Learn scalers (`.joblib`). These are persistent artifacts used by the `DeepSurrogateEngine` during simulation.
- **`requirements.txt`**: List of Python dependencies (PyTorch, NumPy, Pandas, etc.) required to run the project.
- **`venv/`**: The Python virtual environment directory containing installed libraries.

## 5. Miscellaneous
- **`.gitignore`**: Specifies files that Git should ignore (e.g., large data files, virtual environments, pycache).
