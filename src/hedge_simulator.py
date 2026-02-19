"""
Project Alpha-Hedge: Phase 3 - The Hedging Simulator

Simulates an event-driven daily Delta-Hedging loop across independent market paths
to evaluate the Hedging Error (PnL variance) and execution speed of:
1. Black-Scholes Engine
2. Heston Monte Carlo Engine
3. Deep Surrogate Engine
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
from tqdm import tqdm
import logging
import os
import torch
# Need to import tabulate for to_markdown? Pandas handles it if installed.
from pricing_engines import BlackScholesEngine, HestonMCEngine, DeepSurrogateEngine

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
N_PATHS = 1000          # Number of "Real World" market paths
REBALANCE_THRESHOLD = 0.02 # Band Hedging Threshold

T_HORIZON = 0.25        # 3 months
TRADING_DAYS = 63       # Approx trading days in 3 months
DT = T_HORIZON / TRADING_DAYS

# Heston Parameters for "Real World" Generation
S0 = 100.0
K = 100.0 # ATM Option
R_RISK_FREE = 0.02
V0 = 0.04
KAPPA = 2.0
THETA = 0.04
SIGMA_V = 0.3
RHO = -0.7

# Engine Configurations
BS_VOL = np.sqrt(THETA) # Naive volatility for Black-Scholes (using long-term mean)

def generate_heston_paths_full(n_paths, n_steps, T, r, v0, kappa, theta, sigma, rho):
    """
    Generate full Stock (S) and Variance (v) paths for the "Real World".
    Returns: S [n_steps+1, n_paths], v [n_steps+1, n_paths]
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    Z1 = np.random.normal(size=(n_steps, n_paths))
    Z2 = np.random.normal(size=(n_steps, n_paths))
    
    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    S = np.zeros((n_steps + 1, n_paths))
    v = np.zeros((n_steps + 1, n_paths))
    
    S[0] = S0
    v[0] = v0
    
    for t in range(n_steps):
        v_curr = v[t]
        S_curr = S[t]
        
        v_plus = np.maximum(v_curr, 0)
        
        v_next = v_curr + kappa * (theta - v_plus) * dt + sigma * np.sqrt(v_plus) * W2[t] * sqrt_dt
        S_next = S_curr + r * S_curr * dt + np.sqrt(v_plus) * S_curr * W1[t] * sqrt_dt
        
        v[t+1] = v_next
        S[t+1] = S_next
        
    return S, v

class HedgingPortfolio:
    """Tracks the PnL of a delta-hedging strategy, including Risk Attribution and Drawdowns."""
    def __init__(self, name):
        self.name = name
        self.cash = 0.0
        self.shares = 0.0
        self.pnl_history = []
        
        # Attribution Accumulators
        self.cum_delta_pnl = 0.0
        self.cum_theta_pnl = 0.0
        self.cum_rate_pnl = 0.0 # Interest earned
        
        # Risk Metrics
        self.max_equity = 0.0
        self.max_drawdown = 0.0 # Percentage or Absolute? Typically Absolute or % from peak. Let's use Absolute from peak.
        
    def inception(self, S, option_price, delta, initial_transaction_cost=0.0):
        # Short 1 Option (+Premium)
        # Buy Delta Shares (-Cost)
        # Equity = Cash + Shares*S - OptionPrice (Market Value)
        # At inception, Fair Value => Equity = 0 (minus trans cost)
        self.cash = option_price - (delta * S) - initial_transaction_cost
        self.shares = delta
        
        equity = self.cash + (self.shares * S) - option_price
        self.max_equity = max(0.0, equity) # Start at 0 usually
        
    def rebalance(self, S_prev, S_curr, new_delta, dt, r, prev_theta=0.0, transaction_cost=0.0):
        # Interest on Cash (Rate PnL)
        start_cash = self.cash
        self.cash *= np.exp(r * dt)
        interest_earned = self.cash - start_cash
        self.cum_rate_pnl += interest_earned
        
        # Delta PnL
        delta_pnl = self.shares * (S_curr - S_prev)
        self.cum_delta_pnl += delta_pnl
        
        # Theta PnL
        theta_pnl = -prev_theta * dt # Decay of liability is gain
        self.cum_theta_pnl += theta_pnl
        
        # Adjust Shares
        share_diff = new_delta - self.shares
        cost = share_diff * S_curr
        
        self.cash -= (cost + transaction_cost)
        self.shares = new_delta
        
        # Track Drawdown (Estimate Equity Proxy)
        # Equity ~= Cumulative PnL (since initial equity ~ 0)
        # Or rigorously: Cash + Shares*S - Option_Price. 
        # But we don't have option price here efficiently (engine call needed).
        # We can use Cumulative PnL as Equity proxy.
        current_pnl = self.cum_delta_pnl + self.cum_theta_pnl + self.cum_rate_pnl
        # Add trading costs? We are subtracting costs from cash.
        # Let's use Cash + Shares*S as "Asset Value" and approximate Liability?
        # Simpler: Drawdown on cumulative PnL.
        
        self.max_equity = max(self.max_equity, current_pnl)
        drawdown = self.max_equity - current_pnl
        self.max_drawdown = max(self.max_drawdown, drawdown)

    def settle(self, S_final, K, dt, r, prev_theta=0.0, transaction_cost=0.0):
        # Interest
        start_cash = self.cash
        self.cash *= np.exp(r * dt)
        self.cum_rate_pnl += (self.cash - start_cash)
        
        # Attribution (Final Step)
        # Missing S_prev here to do Delta PnL correctly for last step inside settle?
        # But we added manual update in loop before settle.
        # So settle just clears positions.
        
        # Liquidate Shares
        self.cash += self.shares * S_final - transaction_cost
        self.shares = 0
        
        # Pay Payoff
        payoff = max(S_final - K, 0.0)
        self.cash -= payoff
        
        # Final Drawdown Check
        current_pnl = self.cash # Since shares=0, payoff paid. Cash is final PnL.
        self.max_equity = max(self.max_equity, current_pnl)
        drawdown = self.max_equity - current_pnl
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        return self.cash
        



import argparse

def run_simulation(output_dir=None, model_path=None, scaler_path=None, n_paths=N_PATHS):
    # Create Output Directory
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"sim_run_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info(f"Output Directory: {output_dir}")

    # Resolve Model Paths (CLI overrides > defaults)
    if model_path is None:
        model_path = os.path.join('models', 'model_Heston.pth')
    if scaler_path is None:
        scaler_path = os.path.join('models', 'scaler_Heston.joblib')
    
    logger.info(f"Model Path: {model_path}")
    logger.info(f"Scaler Path: {scaler_path}")

    logger.info(f"Generating Real World Market Paths (N={n_paths})...")
    S_paths, v_paths = generate_heston_paths_full(
        n_paths, TRADING_DAYS, T_HORIZON, R_RISK_FREE, V0, KAPPA, THETA, SIGMA_V, RHO
    )
    logger.info(f"Paths Generated. Shape: {S_paths.shape}")

    # Initialize Engines
    logger.info("Initializing Pricing Engines...")
    bs_engine = BlackScholesEngine()
    heston_engine = HestonMCEngine(n_paths=10_000, n_steps=50) 
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        dnn_engine = DeepSurrogateEngine(model_path=model_path, scaler_path=scaler_path)
    else:
        logger.warning(f"DNN Model ({model_path}) or Scaler not found! Falling back to untrained.")
        dnn_engine = DeepSurrogateEngine()
    
    engines = [
        {'name': 'Black-Scholes', 'engine': bs_engine, 'color': 'red'},
        {'name': 'Heston MC', 'engine': heston_engine, 'color': 'green'},
        {'name': 'Deep Surrogate', 'engine': dnn_engine, 'color': 'blue'}
    ]
    
    results = {e['name']: [] for e in engines}
    drawdowns = {e['name']: [] for e in engines}
    execution_times = {e['name']: 0.0 for e in engines}
    attribution = {e['name']: {'delta': [], 'theta': [], 'rate': []} for e in engines}
    
    # Track Granular History for Path 0
    path0_S = None
    path0_deltas = {e['name']: [] for e in engines}
    path0_times = None
    
    # Track Final Data for Scatter Plot
    final_moneyness = []
    final_errors = {e['name']: [] for e in engines}

    logger.info("Starting Hedging Simulation Loop...")
    
    # Simulation Loop
    for i in tqdm(range(n_paths), desc="Simulating Paths"):
        # ... (Loop setup same as before) ...
        path_S = S_paths[:, i]
        path_v = v_paths[:, i]
        
        if i == 0:
            path0_S = path_S
            path0_times = np.linspace(0, T_HORIZON, TRADING_DAYS + 1)
        
        portfolios = {e['name']: HedgingPortfolio(e['name']) for e in engines}
        
        prev_S = path_S[0]
        prev_stats = {e['name']: {'theta': 0.0} for e in engines}
        
        # --- Day 0: Inception ---
        t = 0
        S_t = path_S[t]
        v_t = path_v[t]
        T_remaining = T_HORIZON
        
        for item in engines:
            name = item['name']
            engine = item['engine']
            
            start_time = time.time()
            if name == 'Black-Scholes':
                price = engine.price(S_t, K, T_remaining, R_RISK_FREE, BS_VOL)
                delta = engine.compute_delta(S_t, K, T_remaining, R_RISK_FREE, BS_VOL)
                theta = 0.0
            elif name == 'Deep Surrogate':
                greeks = engine.compute_greeks(S_t, K, T_remaining, R_RISK_FREE, v_t, KAPPA, THETA, SIGMA_V, RHO)
                price = engine.price(S_t, K, T_remaining, R_RISK_FREE, v_t, KAPPA, THETA, SIGMA_V, RHO)
                delta = greeks['delta']
                theta = greeks['theta']
            else: # Heston
                price = engine.price(S_t, K, T_remaining, R_RISK_FREE, v_t, KAPPA, THETA, SIGMA_V, RHO)
                delta = engine.compute_delta(S_t, K, T_remaining, R_RISK_FREE, v_t, KAPPA, THETA, SIGMA_V, RHO)
                theta = 0.0
            
            if torch.cuda.is_available() and name != 'Black-Scholes':
                torch.cuda.synchronize()
                
            execution_times[name] += (time.time() - start_time)
            
            portfolios[name].inception(S_t, price, delta)
            prev_stats[name]['theta'] = theta
            
            if i == 0:
                path0_deltas[name].append(delta)
            
        # --- Daily Rebalancing ---
        prev_S = S_t
        for day in range(1, TRADING_DAYS + 1):
            S_t = path_S[day]
            v_t = path_v[day]
            T_remaining = T_HORIZON - (day * DT)
            
            if T_remaining < 1e-5: T_remaining = 1e-5 
            
            for item in engines:
                name = item['name']
                engine = item['engine']
                
                # Check for maturity liquidation
                if day == TRADING_DAYS:
                    portfolios[name].cum_delta_pnl += portfolios[name].shares * (S_t - prev_S)
                    portfolios[name].cum_theta_pnl += -prev_stats[name]['theta'] * DT
                    
                    pnl = portfolios[name].settle(S_t, K, DT, R_RISK_FREE)
                    results[name].append(pnl)
                    drawdowns[name].append(portfolios[name].max_drawdown)
                    
                    final_errors[name].append(pnl)
                    
                    if i == 0:
                        path0_deltas[name].append(0.0)
                    continue
                
                start_time = time.time()
                theta = 0.0
                if name == 'Black-Scholes':
                    delta = engine.compute_delta(S_t, K, T_remaining, R_RISK_FREE, BS_VOL)
                elif name == 'Deep Surrogate':
                    greeks = engine.compute_greeks(S_t, K, T_remaining, R_RISK_FREE, v_t, KAPPA, THETA, SIGMA_V, RHO)
                    delta = greeks['delta']
                    theta = greeks['theta']
                else: # Heston
                    delta = engine.compute_delta(S_t, K, T_remaining, R_RISK_FREE, v_t, KAPPA, THETA, SIGMA_V, RHO)
                
                if torch.cuda.is_available() and name != 'Black-Scholes':
                    torch.cuda.synchronize()
                    
                execution_times[name] += (time.time() - start_time)
                
                # Band Hedging Logic
                current_shares = portfolios[name].shares
                if abs(delta - current_shares) > REBALANCE_THRESHOLD:
                    portfolios[name].rebalance(prev_S, S_t, delta, DT, R_RISK_FREE, prev_theta=prev_stats[name]['theta'])
                else:
                    # Skip Trade (Band Hedging), but accrue Interest/Theta
                    # Passing current_shares as new_delta results in 0 trades
                    portfolios[name].rebalance(prev_S, S_t, current_shares, DT, R_RISK_FREE, prev_theta=prev_stats[name]['theta'])
                
                prev_stats[name]['theta'] = theta
                if i == 0:
                    path0_deltas[name].append(delta)
            
            prev_S = S_t
        
        final_moneyness.append(path_S[-1] / K)
        
        engines_names = [e['name'] for e in engines]
        for name in engines_names:
            attribution[name]['delta'].append(portfolios[name].cum_delta_pnl)
            attribution[name]['theta'].append(portfolios[name].cum_theta_pnl)
            attribution[name]['rate'].append(portfolios[name].cum_rate_pnl)

    # Analysis & Visualization
    logger.info("Simulation Complete. Analyzing Results...")
    
    # Export Data for Visualization (Task 2 Requirement)
    visualization_data = []
    
    # We need lists aligned with Engines
    # final_moneyness is length N_PATHS (one per path)
    # final_errors[name] is length N_PATHS
    
    for item in engines:
        name = item['name']
        errs = final_errors[name]
        for mon, err in zip(final_moneyness, errs):
            visualization_data.append({
                'Engine': name,
                'Moneyness': mon,
                'Hedging_Error': err
            })
            
    df_viz = pd.DataFrame(visualization_data)
    df_viz.to_csv(os.path.join(output_dir, 'hedging_results.csv'), index=False)
    logger.info(f"Saved hedging_results.csv to {output_dir}")
    
    # 1. Histogram Plot
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    summary_stats = []
    
    for item in engines:
        name = item['name']
        pnls = np.array(results[name])
        dd_arr = np.array(drawdowns[name])
        
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        sharpe = mean_pnl / std_pnl if std_pnl > 1e-9 else 0.0
        avg_drawdown = np.mean(dd_arr)
        
        total_time = execution_times[name]
        
        mean_delta_attr = np.mean(attribution[name]['delta'])
        mean_theta_attr = np.mean(attribution[name]['theta'])
        mean_rate_attr = np.mean(attribution[name]['rate'])
        
        sns.histplot(pnls, kde=True, label=f"{name} (Std: {std_pnl:.4f})", color=item['color'], alpha=0.3)
        
        summary_stats.append({
            "Engine": name,
            "Mean PnL": mean_pnl,
            "Std PnL": std_pnl,
            "Sharpe Ratio": sharpe,
            "Avg Max Drawdown": avg_drawdown,
            "Time (s)": total_time,
            "Attr Delta": mean_delta_attr,
            "Attr Theta": mean_theta_attr,
            "Attr Rate": mean_rate_attr
        })
        
    plt.title(f"Hedging PnL Distributions (N={n_paths})\nBenchmark: BS vs Heston MC vs Deep Surrogate")
    plt.xlabel("Final PnL")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "hedging_results.png"))
    
    # 2. Granular Hedging Dynamics Plot (Path 0)
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(path0_times, path0_S, color='black', label='Asset Price ($S_t$)', linewidth=1.5)
    plt.title("Path 0: Asset Price Dynamics")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    for item in engines:
        plt.plot(path0_times, path0_deltas[item['name']], label=item['name'], color=item['color'], alpha=0.8)
    plt.title("Path 0: Hedging Strategy ($\Delta_t$)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "hedging_dynamics.png"))
    
    # 3. Moneyness vs Error Plot
    plt.figure(figsize=(10, 6))
    for item in engines:
        name = item['name']
        plt.scatter(final_moneyness, final_errors[name], label=name, color=item['color'], alpha=0.5, s=10)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title("Hedging Error vs Moneyness ($S_T / K$)")
    plt.xlabel("Moneyness ($S_T / K$)")
    plt.ylabel("Hedging PnL")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "moneyness_vs_error.png"))
    logger.info(f"Saved plots to {output_dir}")

    # Print Summary Table
    df_results = pd.DataFrame(summary_stats)
    print("\n" + "="*80)
    print("FINAL HEDGING PERFORMANCE REPORT (WITH ATTRIBUTION)")
    print("="*80)
    # Using to_string if to_markdown fails or tabulate not present, but requirement added
    try:
        print(df_results.to_markdown(index=False))
    except ImportError:
        print(df_results.to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha-Hedge Simulation")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    parser.add_argument("--model_path", type=str, default="models/model_Heston.pth", help="Path to trained .pth model")
    parser.add_argument("--scaler_path", type=str, default="models/scaler_Heston.joblib", help="Path to trained .joblib scaler")
    parser.add_argument("--n_paths", type=int, default=1000, help="Number of paths to simulate")

    args = parser.parse_args()

    # Run Simulation
    run_simulation(output_dir=args.output_dir, model_path=args.model_path, scaler_path=args.scaler_path, n_paths=args.n_paths)
