"""
Project Alpha-Hedge: Advanced Visualization Suite

Generates:
  - vol_smile.png        (Implied Vol Smile from Heston)
  - all_greeks_surface.png (2x2 Delta/Gamma/Theta/Rho 3D surfaces)
  - error_heatmap.png    (Heston MC vs Deep Surrogate pricing error)
  - hedging_scatter.png  (Hedging PnL vs Moneyness scatter)

All functions accept dynamic model/scaler paths so the pipeline can point
them at per-dataset models.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for subprocess calls
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import torch
import os
import logging
import argparse
from pricing_engines import BlackScholesEngine, HestonMCEngine, DeepSurrogateEngine

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Globals (set in main or by caller)
OUTPUT_DIR = "visualizations"
MODEL_PATH = os.path.join('models', 'model_Heston.pth')
SCALER_PATH = os.path.join('models', 'scaler_Heston.joblib')


def plot_volatility_smile():
    logger.info("Generating Volatility Smile...")

    S0, T, r = 100.0, 0.5, 0.02
    v0, kappa, theta, sigma_v, rho = 0.04, 2.0, 0.04, 0.3, -0.7
    strikes = np.linspace(80, 120, 20)

    heston_engine = HestonMCEngine(n_paths=20_000, n_steps=50)
    bs_engine = BlackScholesEngine()

    implied_vols, moneyness = [], []
    for K in strikes:
        price = heston_engine.price(S0, K, T, r, v0, kappa, theta, sigma_v, rho)
        iv = bs_engine.implied_volatility(S0, K, T, r, price)
        implied_vols.append(iv)
        moneyness.append(K / S0)

    plt.figure(figsize=(8, 6))
    plt.plot(moneyness, implied_vols, marker='o', linestyle='-', color='purple', label='Heston Implied Vol')
    plt.title(f"Volatility Smile (Heston Model)\n$T={T}, \\rho={rho}, \\nu={sigma_v}$")
    plt.xlabel("Moneyness ($K/S$)")
    plt.ylabel(r"Implied Volatility ($\sigma_{BS}$)")
    plt.grid(True, alpha=0.5)
    plt.legend()

    path = os.path.join(OUTPUT_DIR, "vol_smile.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved {path}")


def plot_greeks_surface():
    logger.info("Generating 3D Multi-Greeks Surface...")

    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model not found at {MODEL_PATH}. Skipping Greeks Surface.")
        return

    dnn_engine = DeepSurrogateEngine(model_path=MODEL_PATH, scaler_path=SCALER_PATH)

    S_range = np.linspace(80, 120, 20)
    T_range = np.linspace(0.05, 1.0, 20)
    S_grid, T_grid = np.meshgrid(S_range, T_range)

    delta_surf = np.zeros_like(S_grid)
    gamma_surf = np.zeros_like(S_grid)
    theta_surf = np.zeros_like(S_grid)
    rho_surf   = np.zeros_like(S_grid)

    K, r, v0 = 100.0, 0.02, 0.04
    kappa, theta_param, sigma_v, rho = 2.0, 0.04, 0.3, -0.7

    for i in range(S_grid.shape[0]):
        for j in range(S_grid.shape[1]):
            greeks = dnn_engine.compute_greeks(
                S_grid[i, j], K, T_grid[i, j], r, v0, kappa, theta_param, sigma_v, rho
            )
            delta_surf[i, j] = greeks['delta']
            gamma_surf[i, j] = greeks['gamma']
            theta_surf[i, j] = greeks['theta']
            rho_surf[i, j]   = greeks['rho']

    fig = plt.figure(figsize=(18, 14))
    configs = [
        (1, delta_surf, 'viridis', r"Delta ($\Delta$)"),
        (2, gamma_surf, 'magma',   r"Gamma ($\Gamma$)"),
        (3, theta_surf, 'plasma',  r"Theta ($\Theta$)"),
        (4, rho_surf,   'cividis', r"Rho ($\rho$)"),
    ]
    for idx, Z, cmap, label in configs:
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        surf = ax.plot_surface(S_grid, T_grid, Z, cmap=cmap, edgecolor='none', alpha=0.8)
        ax.set_title(f"{label} Surface")
        ax.set_xlabel("Spot Price ($S$)")
        ax.set_ylabel("Time to Maturity ($T$)")
        ax.set_zlabel(label)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    fig.suptitle("Deep Surrogate Greeks Surface (Heston Model)", fontsize=16)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "all_greeks_surface.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved {path}")


def plot_error_heatmap():
    logger.info("Generating Error Heatmap...")

    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model not found at {MODEL_PATH}. Skipping Error Heatmap.")
        return

    heston_engine = HestonMCEngine(n_paths=10_000, n_steps=50)
    dnn_engine = DeepSurrogateEngine(model_path=MODEL_PATH, scaler_path=SCALER_PATH)

    S_vals  = np.linspace(80, 120, 10)
    v0_vals = np.linspace(0.01, 0.10, 10)
    error_matrix = np.zeros((len(v0_vals), len(S_vals)))

    K, T, r = 100.0, 0.25, 0.02
    kappa, theta, sigma_v, rho = 2.0, 0.04, 0.3, -0.7

    for i, v in enumerate(v0_vals):
        for j, s in enumerate(S_vals):
            p_true = heston_engine.price(s, K, T, r, v, kappa, theta, sigma_v, rho)
            p_pred = dnn_engine.price(s, K, T, r, v, kappa, theta, sigma_v, rho)
            error_matrix[i, j] = abs(p_true - p_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(error_matrix,
                xticklabels=np.round(S_vals, 1),
                yticklabels=np.round(v0_vals, 3),
                annot=True, fmt=".2f", cmap="magma_r",
                cbar_kws={'label': 'Absolute Pricing Error ($)'})
    plt.title("Pricing Error Heatmap: Heston MC vs Deep Surrogate")
    plt.xlabel("Spot Price ($S$)")
    plt.ylabel("Initial Variance ($v_0$)")

    path = os.path.join(OUTPUT_DIR, "error_heatmap.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved {path}")


def plot_hedging_scatter():
    logger.info("Generating Hedging PnL Scatter Plot...")

    csv_path = os.path.join(OUTPUT_DIR, 'hedging_results.csv')
    if not os.path.exists(csv_path):
        logger.warning(f"{csv_path} not found. Run hedge_simulator.py first.")
        return

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='Moneyness', y='Hedging_Error',
                    hue='Engine', alpha=0.5, style='Engine')
    plt.title("Hedging Error vs. Moneyness\n(Out-of-Distribution Robustness)")
    plt.xlabel("Moneyness ($S_T / K$)")
    plt.ylabel("Final Hedging PnL ($)")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "hedging_scatter.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha-Hedge Visualization Suite")
    parser.add_argument("--output_dir",  type=str, default="visualizations", help="Directory for outputs")
    parser.add_argument("--model_path",  type=str, default=None, help="Path to .pth model")
    parser.add_argument("--scaler_path", type=str, default=None, help="Path to .joblib scaler")
    args = parser.parse_args()

    OUTPUT_DIR  = args.output_dir
    MODEL_PATH  = args.model_path  or MODEL_PATH
    SCALER_PATH = args.scaler_path or SCALER_PATH

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_volatility_smile()
    plot_greeks_surface()
    plot_error_heatmap()
    plot_hedging_scatter()
