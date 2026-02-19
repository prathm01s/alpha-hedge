"""
Project Alpha-Hedge: The Quant Sandbox (Engine Comparison)

Standalone script to compare Price and Greeks across:
1. Black-Scholes (Analytical)
2. Heston Monte Carlo (Numerical)
3. Deep Surrogate (Neural Approximation)
"""

import numpy as np
import pandas as pd
import torch
import os
import logging
from pricing_engines import BlackScholesEngine, HestonMCEngine, DeepSurrogateEngine

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def run_comparison():
    print("\n" + "="*80)
    print("ALPHA-HEDGE ENGINE COMPARISON SANDBOX")
    print("="*80)

    # 1. Initialize Engines
    logger.info("Initializing Engines...")
    bs_engine = BlackScholesEngine()
    heston_engine = HestonMCEngine(n_paths=50_000, n_steps=100) # High precision for benchmark
    
    model_path = os.path.join('models', 'model_Heston.pth')
    scaler_path = os.path.join('models', 'scaler_Heston.joblib')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        dnn_engine = DeepSurrogateEngine(model_path=model_path, scaler_path=scaler_path)
    else:
        logger.warning("DNN Model not found! Comparisons will fail for Deep Surrogate.")
        return

    # 2. Market Parameters (ATM Option)
    S = 100.0
    K = 100.0
    T = 0.5
    r = 0.02
    
    # Heston Params
    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    sigma_v = 0.3
    rho = -0.7
    
    # BS Volatility Proxy
    sigma_bs = np.sqrt(v0)

    logger.info(f"Parameters: S={S}, K={K}, T={T}, r={r}, v0={v0}, kappa={kappa}, theta={theta}, sigma_v={sigma_v}, rho={rho}")

    # 3. Compute Values
    logger.info("Computing Prices and Greeks...")
    
    # --- Black-Scholes ---
    bs_greeks = bs_engine.compute_greeks(S, K, T, r, sigma_bs)
    bs_price = bs_engine.price(S, K, T, r, sigma_bs)
    
    # --- Heston MC ---
    # MC is expensive for Greeks, only Price and Delta (via bump) usually efficient
    # We will compute Price and Delta. Others N/A.
    start_mc = os.times().elapsed
    mc_price = heston_engine.price(S, K, T, r, v0, kappa, theta, sigma_v, rho)
    mc_delta = heston_engine.compute_delta(S, K, T, r, v0, kappa, theta, sigma_v, rho)
    
    # --- Deep Surrogate ---
    ds_greeks = dnn_engine.compute_greeks(S, K, T, r, v0, kappa, theta, sigma_v, rho)
    ds_price = dnn_engine.price(S, K, T, r, v0, kappa, theta, sigma_v, rho)
    
    # 4. Aggregate Results
    data = [
        {
            "Metric": "Price",
            "Black-Scholes": bs_price,
            "Heston MC": mc_price,
            "Deep Surrogate": ds_price,
            "Diff (DS - MC)": ds_price - mc_price
        },
        {
            "Metric": "Delta",
            "Black-Scholes": bs_greeks['delta'],
            "Heston MC": mc_delta,
            "Deep Surrogate": ds_greeks['delta'],
            "Diff (DS - MC)": ds_greeks['delta'] - mc_delta
        },
        {
            "Metric": "Gamma",
            "Black-Scholes": bs_greeks['gamma'],
            "Heston MC": "N/A",
            "Deep Surrogate": ds_greeks['gamma'],
            "Diff (DS - MC)": "N/A"
        },
        {
            "Metric": "Theta",
            "Black-Scholes": bs_greeks['theta'],
            "Heston MC": "N/A",
            "Deep Surrogate": ds_greeks['theta'],
            "Diff (DS - MC)": "N/A"
        },
        {
            "Metric": "Rho",
            "Black-Scholes": bs_greeks['rho'],
            "Heston MC": "N/A",
            "Deep Surrogate": ds_greeks['rho'],
            "Diff (DS - MC)": "N/A"
        }
    ]
    
    df = pd.DataFrame(data)
    
    # 5. Display Formatted Table
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(df.to_markdown(index=False, floatfmt=".6f"))
    print("="*80 + "\n")
    
    # Interpretation
    err_price = abs(ds_price - mc_price)
    err_delta = abs(ds_greeks['delta'] - mc_delta)
    
    if err_price < 0.05 and err_delta < 0.02:
        print("✅ SUCCESS: Deep Surrogate matches Heston MC within tolerances.")
    else:
        print("⚠️ WARNING: Significant deviation detected between Surrogate and MC.")

if __name__ == "__main__":
    run_comparison()
