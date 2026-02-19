"""
Project Alpha-Hedge: Pricing Engines Module

This module defines the abstract base class `PricingEngine` and implements three concrete engines:
1. `BlackScholesEngine`: Closed-form analytical pricing.
2. `HestonMCEngine`: Monte Carlo simulation of the Heston Stochastic Volatility Model.
3. `DeepSurrogateEngine`: Deep Learning-based pricing using a Feedforward Neural Network.
"""

import abc
import logging
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
import torch
import torch.nn as nn
import joblib
from typing import Tuple, Optional, Dict

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PricingEngine(abc.ABC):
    """
    Abstract Base Class for Option Pricing Engines.
    Enforces implementation of price and delta computation.
    """

    @abc.abstractmethod
    def price(self, S: float, K: float, T: float, r: float, **kwargs) -> float:
        """Calculate the price of a European Call Option."""
        pass

    @abc.abstractmethod
    def compute_delta(self, S: float, K: float, T: float, r: float, **kwargs) -> float:
        """Calculate the Delta of a European Call Option."""
        pass

class BlackScholesEngine(PricingEngine):
    """
    Engine 1: Black-Scholes (The Baseline)
    Implements closed-form solutions for European Call Option Price and Delta.
    """

    def price(self, S, K, T, r, sigma, **kwargs):
        # Convert to numpy arrays for safe vectorization
        S, K, T, r, sigma = np.asarray(S), np.asarray(K), np.asarray(T), np.asarray(r), np.asarray(sigma)
        
        # Avoid division by zero by temporarily replacing T<=0 with a safe dummy value
        T_safe = np.where(T <= 0, 1.0, T)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * np.sqrt(T_safe))
        d2 = d1 - sigma * np.sqrt(T_safe)
        
        call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T_safe) * stats.norm.cdf(d2)
        
        # Apply intrinsic payoff where options are expired (T <= 0)
        payoff = np.maximum(S - K, 0.0)
        result = np.where(T <= 0, payoff, call_price)
        
        # Return scalar if input was scalar, else return array
        return result.item() if result.ndim == 0 else result

    def compute_greeks(self, S, K, T, r, sigma, **kwargs):
        """
        Calculate Analytical Greeks for Black-Scholes.
        Returns: {'delta', 'gamma', 'theta', 'rho'}
        """
        S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
        
        if T <= 0:
            delta = 1.0 if S > K else 0.0
            return {'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'rho': 0.0}
            
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        norm_pdf_d1 = stats.norm.pdf(d1)
        norm_cdf_d1 = stats.norm.cdf(d1)
        norm_cdf_d2 = stats.norm.cdf(d2)
        
        # Delta
        delta = norm_cdf_d1
        
        # Gamma
        gamma = norm_pdf_d1 / (S * sigma * sqrt_T)
        
        # Theta (Annual) - Standard logic for Long Call
        theta = -(S * norm_pdf_d1 * sigma) / (2 * sqrt_T) - r * K * np.exp(-r * T) * norm_cdf_d2
        
        # Rho
        rho = K * T * np.exp(-r * T) * norm_cdf_d2
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'rho': rho
        }

    def compute_delta(self, S, K, T, r, sigma, **kwargs):
        return self.compute_greeks(S, K, T, r, sigma)['delta']
        
    def implied_volatility(self, S: float, K: float, T: float, r: float, market_price: float) -> float:
        """
        Solve for Implied Volatility given a Market Price using Brent's method.
        """
        def objective(sigma):
            return self.price(S, K, T, r, sigma) - market_price
        
        try:
            # Search for vol between 1% and 500%
            iv = optimize.brentq(objective, 1e-4, 5.0)
            return iv
        except Exception:
            return np.nan

class HestonMCEngine(PricingEngine):
    """
    Engine 2: Heston Monte Carlo (The Ground Truth)
    Simulates Heston SDEs using Euler-Maruyama discretization with Full Truncation scheme.
    GPU-Accelerated using PyTorch with Batch Vectorization.
    """

    def __init__(self, n_paths: int = 10_000, n_steps: int = 100):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initialized HestonMCEngine with {n_paths} paths and {n_steps} steps on {self.device}.")

    def _simulate_paths(self, S0: torch.Tensor, T: torch.Tensor, r: torch.Tensor, v0: torch.Tensor, 
                       kappa: torch.Tensor, theta: torch.Tensor, sigma: torch.Tensor, rho: torch.Tensor,
                       Z1: torch.Tensor = None, Z2: torch.Tensor = None) -> torch.Tensor:
        """
        Simulate asset paths under Heston dynamics using PyTorch with Batch Support.
        Inputs are expected to be Tensors of shape (BATCH_SIZE, 1) or broadcastable.
        """
        batch_size = S0.shape[0]
        dt = T / self.n_steps
        sqrt_dt = torch.sqrt(dt)

        # Generate correlated Brownian motions (CRN support)
        if Z1 is None or Z2 is None:
            Z1 = torch.randn((batch_size, self.n_steps, self.n_paths), device=self.device)
            Z2 = torch.randn((batch_size, self.n_steps, self.n_paths), device=self.device)
        
        # Correlate Z2 with Z1
        rho_expanded = rho.unsqueeze(2) # (B, 1, 1)
        
        W1 = Z1
        W2 = rho_expanded * Z1 + torch.sqrt(1 - rho_expanded**2) * Z2
        
        # Initialize arrays
        S = S0.repeat(1, self.n_paths) 
        v = v0.repeat(1, self.n_paths) 
        
        for t in range(self.n_steps):
            W1_t = W1[:, t, :]
            W2_t = W2[:, t, :]
            
            v_plus = torch.maximum(v, torch.tensor(0.0, device=self.device))
            
            # Update Variance
            v_next = v + kappa * (theta - v_plus) * dt + sigma * torch.sqrt(v_plus) * W2_t * sqrt_dt
            
            # Update Asset Price
            S_next = S + r * S * dt + torch.sqrt(v_plus) * S * W1_t * sqrt_dt
            
            v = v_next
            S = S_next
            
        return S

    def price(self, S, K, T, r, v0, kappa, theta, sigma, rho, 
              Z1: torch.Tensor = None, Z2: torch.Tensor = None, **kwargs):
        """
        Calculate Heston Price using Monte Carlo (GPU).
        Handles both scalar inputs and batch Tensor inputs.
        """
        inputs = [S, K, T, r, v0, kappa, theta, sigma, rho]
        tensors = []
        is_batch = False
        
        for val in inputs:
            if isinstance(val, torch.Tensor):
                if val.ndim == 0:
                    tensors.append(val.reshape(1, 1))
                elif val.ndim == 1:
                    tensors.append(val.reshape(-1, 1))
                    is_batch = True
                else:
                    tensors.append(val)
                    is_batch = True
            elif isinstance(val, (float, int, np.float64, np.float32)):
                tensors.append(torch.tensor([[val]], dtype=torch.float32, device=self.device))
            elif isinstance(val, (list, np.ndarray)):
                tensors.append(torch.tensor(val, dtype=torch.float32, device=self.device).reshape(-1, 1))
                is_batch = True
            else:
                 tensors.append(torch.tensor([[float(val)]], dtype=torch.float32, device=self.device))

        S_t, K_t, T_t, r_t, v0_t, kappa_t, theta_t, sigma_t, rho_t = tensors
        
        terminal_S = self._simulate_paths(S_t, T_t, r_t, v0_t, kappa_t, theta_t, sigma_t, rho_t, Z1, Z2)
        
        payoffs = torch.maximum(terminal_S - K_t, torch.tensor(0.0, device=self.device))
        discount = torch.exp(-r_t * T_t)
        prices_batch = discount * torch.mean(payoffs, dim=1, keepdim=True)
        
        if is_batch:
            return prices_batch.flatten()
        else:
            return prices_batch.item()

    def compute_delta(self, S: float, K: float, T: float, r: float, 
                      v0: float, kappa: float, theta: float, sigma: float, rho: float, **kwargs) -> float:
        S = float(S)
        bump = 0.01 * S 
        if bump < 1e-4: bump = 1e-4
        
        # CRN for variance reduction
        Z1 = torch.randn((1, self.n_steps, self.n_paths), device=self.device)
        Z2 = torch.randn((1, self.n_steps, self.n_paths), device=self.device)

        price_up = self.price(S + bump, K, T, r, v0, kappa, theta, sigma, rho, Z1=Z1, Z2=Z2)
        price_down = self.price(S - bump, K, T, r, v0, kappa, theta, sigma, rho, Z1=Z1, Z2=Z2)
        
        delta = (price_up - price_down) / (2 * bump)
        return delta

class MertonJumpMCEngine(PricingEngine):
    """
    Engine 4: Merton Jump-Diffusion Monte Carlo
    SDE: dS/S = (r - lambda*k)dt + sigma*dW + (Y-1)dN
    """
    def __init__(self, n_paths: int = 10_000, n_steps: int = 100):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initialized MertonJumpMCEngine with {n_paths} paths on {self.device}.")

    def _simulate_paths(self, S0, T, r, sigma, lam, mu_j, delta_j) -> torch.Tensor:
        batch_size = S0.shape[0]
        dt = T / self.n_steps
        sqrt_dt = torch.sqrt(dt)

        # Expected percentage jump size k = E[Y] - 1 = exp(mu_j + 0.5*delta_j^2) - 1
        k = torch.exp(mu_j + 0.5 * delta_j**2) - 1

        S = S0.repeat(1, self.n_paths)
        
        # Drift adjustment for jumps to maintain risk-neutrality
        # drift = r - lambda * k - 0.5 * sigma^2
        drift = (r - lam * k - 0.5 * sigma**2) * dt

        for t in range(self.n_steps):
            # Diffusion part
            Z = torch.randn((batch_size, self.n_paths), device=self.device)
            
            # Jump part
            # Number of jumps in dt: Poisson(lambda * dt)
            # For small dt, assume approx Bernoulli (0 or 1 jump) or use torch.poisson
            # Poisson intensity vector
            intensity = lam * dt
            N = torch.poisson(intensity.expand(batch_size, self.n_paths))
            
            # Jump size Y ~ LogNormal(mu_j, delta_j)
            # ln(Y) ~ Normal(mu_j, delta_j)
            # M = ln(Y) sum for N jumps -> Normal(N * mu_j, sqrt(N) * delta_j)
            # So JumpFactor = exp(sum(ln Y))
            
            # Vectorized Jump Multiplier
            # If N=0, JumpFactor=1. If N=1, JumpFactor=Y.
            # We construct the jump log-return component.
            
            # M_t = Sum_{i=1}^N ln(Y_i)
            # M_t ~ Normal(N * mu_j, sqrt(N) * delta_j)
            
            ####M_t = torch.normal(mean=N * mu_j, std=torch.sqrt(N) * delta_j) # Note: sqrt(N) might be 0, torch handles this (std=0 -> sample=mean)
            # Reparameterization trick: avoids crashing when N=0 (std=0)
            Z_jump = torch.randn_like(N)
            M_t = N * mu_j + torch.sqrt(N) * delta_j * Z_jump
            # S_{t+1} = S_t * exp(drift + sigma*sqrt(dt)*Z + M_t)
            # (Note: Standard Geometric Brownian Motion solution form + Jumps)
            
            log_ret = drift + sigma * sqrt_dt * Z + M_t
            S = S * torch.exp(log_ret)

        return S

    def price(self, S, K, T, r, sigma, lam, mu_j, delta_j, **kwargs):
        # 1. Standardize Inputs
        inputs = [S, K, T, r, sigma, lam, mu_j, delta_j]
        tensors = []
        is_batch = False
        
        for val in inputs:
             if isinstance(val, torch.Tensor):
                if val.ndim == 0: tensors.append(val.reshape(1, 1))
                elif val.ndim == 1: 
                    tensors.append(val.reshape(-1, 1))
                    is_batch = True
                else: 
                    tensors.append(val)
                    is_batch = True
             elif isinstance(val, (float, int, np.float64, np.float32)):
                tensors.append(torch.tensor([[val]], dtype=torch.float32, device=self.device))
             elif isinstance(val, (list, np.ndarray)):
                tensors.append(torch.tensor(val, dtype=torch.float32, device=self.device).reshape(-1, 1))
                is_batch = True
             else:
                tensors.append(torch.tensor([[float(val)]], dtype=torch.float32, device=self.device))

        S_t, K_t, T_t, r_t, sigma_t, lam_t, mu_j_t, delta_j_t = tensors

        terminal_S = self._simulate_paths(S_t, T_t, r_t, sigma_t, lam_t, mu_j_t, delta_j_t)
        
        # Payoff
        payoffs = torch.maximum(terminal_S - K_t, torch.tensor(0.0, device=self.device))
        discount = torch.exp(-r_t * T_t)
        prices = discount * torch.mean(payoffs, dim=1, keepdim=True)
        
        if is_batch: return prices.flatten()
        else: return prices.item()

    def compute_delta(self, S, K, T, r, sigma, lam, mu_j, delta_j, **kwargs):
        # Finite Difference
        S = float(S)
        bump = 0.01 * S
        p_up = self.price(S + bump, K, T, r, sigma, lam, mu_j, delta_j)
        p_dn = self.price(S - bump, K, T, r, sigma, lam, mu_j, delta_j)
        return (p_up - p_dn) / (2 * bump)

class DeepSurrogateModel(nn.Module):
    """
    Standard Feedforward Neural Network for Option Pricing.
    """
    def __init__(self, input_dim: int = 9, hidden_dim: int = 64):
        super(DeepSurrogateModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus() # Ensure positive price
        )

    def forward(self, x):
        return self.net(x)

class DeepSurrogateEngine(PricingEngine):
    """
    Engine 3: Deep Neural Network (The AI Surrogate)
    Uses a trained PyTorch model to infer option prices and autograd for Gamma/Delta/Theta/Rho.
    """

    def __init__(self, model_path: str = None, scaler_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepSurrogateModel().to(self.device)
        self.model.eval()
        self.scaler = None
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded surrogate model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
        else:
            logger.warning("DeepSurrogateEngine initialized without a trained model path.")

        if scaler_path:
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded scaler from {scaler_path}")
            except Exception as e:
                logger.error(f"Failed to load scaler from {scaler_path}: {e}")

    def _prepare_input(self, S, K, T, r, v0, kappa, theta, sigma, rho):
        raw_features = np.array([[S, K, T, r, v0, kappa, theta, sigma, rho]])
        
        if self.scaler:
            features = self.scaler.transform(raw_features)
        else:
            features = raw_features
            
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        return x

    def price(self, S: float, K: float, T: float, r: float, 
              v0: float, kappa: float, theta: float, sigma: float, rho: float, **kwargs) -> float:
        x = self._prepare_input(S, K, T, r, v0, kappa, theta, sigma, rho)
        with torch.no_grad():
            price = self.model(x).item()
        return price

    def _bs_delta(self, S, K, T, r, v):
        sigma = np.sqrt(v)
        if T <= 0: return 1.0 if S > K else 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return stats.norm.cdf(d1)

    def compute_delta(self, S: float, K: float, T: float, r: float, 
                      v0: float, kappa: float, theta: float, sigma: float, rho: float, **kwargs) -> float:
        # Wrapper for simple Delta calls
        greeks = self.compute_greeks(S, K, T, r, v0, kappa, theta, sigma, rho)
        return greeks['delta']

    def compute_greeks(self, S: float, K: float, T: float, r: float,
                       v0: float, kappa: float, theta: float, sigma: float, rho: float) -> Dict[str, float]:
        """
        Calculate Higher-Order Greeks using PyTorch Autograd.
        Returns: {'delta', 'gamma', 'theta', 'rho'}
        """
        # Feature Mapping: [S, K, T, r, v0, kappa, theta, sigma, rho]
        # Indices:          0  1  2  3  4   5      6      7      8
        
        # Graceful Degradation near maturity
        if T < 0.05:
            # Fallback to Black-Scholes approx for Delta (others 0 for stability)
            bs_delta = self._bs_delta(S, K, T, r, v0)
            return {'delta': bs_delta, 'gamma': 0.0, 'theta': 0.0, 'rho': 0.0}

        # 1. Prepare Input with Gradient Tracking
        raw_inputs = np.array([S, K, T, r, v0, kappa, theta, sigma, rho], dtype=np.float32)
        
        if self.scaler:
            # Transform
            inputs_scaled_np = self.scaler.transform(raw_inputs.reshape(1, -1)).flatten()
            
            # Extract Scaling Factors (1 / std)
            # scaler.scale_ is the std deviation
            scale_S = 1.0 / self.scaler.scale_[0]
            scale_T = 1.0 / self.scaler.scale_[2]
            scale_r = 1.0 / self.scaler.scale_[3]
        else:
            inputs_scaled_np = raw_inputs
            scale_S = 1.0
            scale_T = 1.0
            scale_r = 1.0
            
        # Create Tensor requiring grad
        # We need to compute gradients w.r.t the input tensor elements.
        # Since input is a single tensor input to the model, we track gradients on x.
        x = torch.tensor(inputs_scaled_np, dtype=torch.float32, device=self.device, requires_grad=True)
        
        # 2. Forward Pass
        # Model expects batch dim (1, 9)
        price = self.model(x.unsqueeze(0))
        
        # 3. First Derivatives (create_graph=True for Gamma)
        # grad outputs shape (1, 9)
        grads = torch.autograd.grad(price, x, create_graph=True)[0]
        
        # Extract Scaled Gradients
        # dPrice/dx_S = grads[0]
        # True Delta = dPrice/dS = (dPrice/dx_S) * (dx_S/dS) = grads[0] * scale_S
        
        delta_model = grads[0]
        theta_model = grads[2] # dPrice/dx_T
        rho_model = grads[3]   # dPrice/dx_r
        
        delta = delta_model * scale_S
        theta_val = -(theta_model * scale_T) # Theta is -dC/dT
        rho_val = rho_model * scale_r
        
        # 4. Second Derivative (Gamma)
        # Gamma = dDelta/dS
        # We need gradient of (delta) w.r.t (x_S) again
        # d(Delta)/dS = d(delta_model * scale_S)/dS
        #             = scale_S * d(delta_model)/dS
        #             = scale_S * [ d(delta_model)/dx_S * dx_S/dS ]
        #             = scale_S * d(delta_model)/dx_S * scale_S
        #             = (scale_S^2) * d2Price/dx_S2
        
        gamma_model = torch.autograd.grad(delta_model, x, retain_graph=False)[0][0]
        gamma = gamma_model * (scale_S ** 2)
        
        return {
            'delta': float(delta.item()),
            'gamma': float(gamma.item()),
            'theta': float(theta_val.item()),
            'rho': float(rho_val.item())
        }

