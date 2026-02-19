"""
Script to generate synthetic training data for the Deep Neural Network Surrogate.
Uses sequential GPU execution with HestonMCEngine and TRUE BATCH VECTORIZATION.
"""
import torch
import numpy as np
import pandas as pd
import time
import os
import logging
from tqdm import tqdm
from pricing_engines import HestonMCEngine, BlackScholesEngine, MertonJumpMCEngine

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
# DEFAULT_N_SAMPLES = 100_000 # For full dataset generation
DEFAULT_N_SAMPLES = 10_000 # For testing/quicker runs
BATCH_SIZE = 200 # Number of samples to process in each batch

def generate_data(engine, n_samples, output_file):
    """
    Generates synthetic option pricing data using the specified pricing engine.

    Args:
        engine: An instance of a pricing engine (e.g., HestonMCEngine, BlackScholesEngine).
        n_samples (int): The number of samples to generate.
        output_file (str): The path to save the generated data (e.g., 'data.parquet').
    """
    engine_name = type(engine).__name__
    logger.info(f"Generating {n_samples} samples using {engine_name}...")

    # 1. Parameter Generation (Example for Heston, adjust for other engines)
    # These ranges should be carefully chosen to represent realistic market conditions
    # and cover the input space for the DNN surrogate.
    S = np.random.uniform(80, 120, n_samples) # Spot price
    K = np.random.uniform(80, 120, n_samples) # Strike price
    T = np.random.uniform(0.1, 2.0, n_samples) # Time to maturity (years)
    r = np.random.uniform(0.01, 0.05, n_samples) # Risk-free rate

    # Heston specific parameters
    v0 = np.random.uniform(0.01, 0.2, n_samples) # Initial variance
    kappa = np.random.uniform(1.0, 5.0, n_samples) # Mean reversion rate
    theta = np.random.uniform(0.01, 0.2, n_samples) # Long-run variance
    sigma_v = np.random.uniform(0.1, 1.0, n_samples) # Volatility of volatility
    rho = np.random.uniform(-0.9, -0.1, n_samples) # Correlation

    # Black-Scholes specific parameter (if used)
    sigma = np.random.uniform(0.1, 0.5, n_samples) # Volatility

    # Merton Jump Diffusion specific parameters (if used)
    lam = np.random.uniform(0.1, 1.0, n_samples) # Jump intensity
    mu_j = np.random.uniform(-0.5, 0.0, n_samples) # Mean jump size
    delta_j = np.random.uniform(0.1, 0.5, n_samples) # Std dev of jump size

    # 2. Pricing Loop (Batching)
    prices = np.zeros(n_samples)
    
    # Optimization: If BS, we can call all at once (if memory allows, but stick to batching for safety)
    if engine_name == 'BlackScholesEngine':
        # BS is CPU and lightweight usually, but consistency is good.
        # Let's batch it too to be safe.
        pass 
        
    start_time_pricing = time.time()
    
    # Loop in batches to avoid OOM
    for i in tqdm(range(0, n_samples, BATCH_SIZE), desc=f"Pricing Batches ({engine_name})"):
        end_idx = min(i + BATCH_SIZE, n_samples)
        
        # Extract Batch
        b_S = S[i:end_idx]
        b_K = K[i:end_idx]
        b_T = T[i:end_idx]
        b_r = r[i:end_idx]
        
        p = None
        
        if engine_name == 'BlackScholesEngine':
            # BS assumes volatility is constant, so we pass sqrt(v0).
            # The NN will have to learn to ignore kappa, theta, sigma_v, and rho.
            b_sigma = np.sqrt(v0[i:end_idx])
            p = engine.price(b_S, b_K, b_T, b_r, b_sigma)

        elif engine_name == 'HestonMCEngine':
            b_v0 = v0[i:end_idx]
            b_kappa = kappa[i:end_idx]
            b_theta = theta[i:end_idx]
            b_sigma_v = sigma_v[i:end_idx]
            b_rho = rho[i:end_idx]
            p = engine.price(b_S, b_K, b_T, b_r, b_v0, b_kappa, b_theta, b_sigma_v, b_rho)
            
        elif engine_name == 'MertonJumpMCEngine':
            # Merton uses sqrt(v0) for diffusion. 
            b_sigma = np.sqrt(v0[i:end_idx])
            # We fix the jump parameters. They act as "unobservable" market shocks 
            # that the 9-parameter Heston NN will struggle to explain.
            b_lam = np.full_like(b_sigma, 0.5)   # Jump intensity
            b_mu_j = np.full_like(b_sigma, -0.2) # Mean jump size
            b_delta_j = np.full_like(b_sigma, 0.2) # Jump volatility
            p = engine.price(b_S, b_K, b_T, b_r, b_sigma, b_lam, b_mu_j, b_delta_j)

        # Store (handle tensor output)
        if isinstance(p, torch.Tensor):
            prices[i:end_idx] = p.detach().cpu().numpy().flatten()
        else:
            prices[i:end_idx] = p
            
    # 3. Save to Parquet
    df = pd.DataFrame({
        'S': S, 'K': K, 'T': T, 'r': r,
        'v0': v0, 'kappa': kappa, 'theta': theta, 'sigma': sigma_v, 'rho': rho,
        'price': prices
    })
    
    # Ensure float32 for storage efficiency
    cols = df.columns
    df[cols] = df[cols].astype(np.float32)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    df.to_parquet(output_file, compression='snappy')
    end_time = time.time()
    logger.info(f"Dataset generated and saved to {output_file}. Rows: {len(df)}. Time: {end_time - start_time_pricing:.2f}s")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic option pricing data.")
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES, help="Number of samples to generate")
    parser.add_argument("--output_file", type=str, default="heston_training_data.parquet", help="Output parquet file path")
    parser.add_argument("--engine", type=str, default="Heston", choices=["BS", "Heston", "Merton"], help="Pricing Engine to use")
    
    args = parser.parse_args()
    
    if args.engine == "BS":
        engine = BlackScholesEngine()
    elif args.engine == "Merton":
        engine = MertonJumpMCEngine(n_paths=10_000, n_steps=50)
    else:
        # Heston Default
        engine = HestonMCEngine(n_paths=20_000, n_steps=50)

    generate_data(engine, args.n_samples, args.output_file)
