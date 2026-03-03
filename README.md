# Project Alpha-Hedge: Deep Surrogate Option Pricing & Risk Simulation Architecture

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20Accelerated-ee4c2c.svg)](https://pytorch.org)
[![Quant](https://img.shields.io/badge/Domain-Quantitative%20Finance-005571.svg)]()

## 1. Project Idea & Executive Summary

**Project Alpha-Hedge** bridges the gap between traditional quantitative finance and modern deep learning. Complex Stochastic Differential Equations (SDEs) like the Heston Stochastic Volatility model or the Merton Jump-Diffusion model provide highly accurate representations of real-world market dynamics (e.g., volatility smiles and fat tails). However, they lack closed-form analytical solutions and rely on computationally expensive Monte Carlo simulations for pricing and Greek extraction.

This project solves the computational bottleneck by training a **Deep Neural Network Surrogate Model** to learn the pricing manifolds of these complex SDEs. Instead of performing standard finite-difference bumping to calculate sensitivities, the architecture leverages `torch.autograd` to extract instantaneous, algorithmic higher-order Greeks ($\Delta, \Gamma, \Theta, \rho$) directly from the neural network's computational graph. 

The robustness of the AI surrogate is empirically proven via a massive 63-day simulated event-driven Delta-hedging pipeline, cross-world generalization matrices, and institutional risk attribution metrics.

---

## 2. Core Mathematical Engines

The system abstracts the option pricing logic into an object-oriented hierarchy defined in `pricing_engines.py`, allowing identical simulator execution across diverse underlying mathematics.

### A. Black-Scholes Engine (The Baseline)
Implements fully vectorized, closed-form solutions for standard European Call Options. Assumes constant volatility and normal returns. Provides analytical calculations for $\Delta, \Gamma, \Theta,$ and $\rho$ to serve as a ground-truth benchmark against neural approximations. 

### B. Heston Monte Carlo Engine (The Stochastic World)
Simulates asset paths experiencing mean-reverting stochastic variance.
$$dv_t = \kappa(\theta - v_t)dt + \sigma_v \sqrt{v_t}dW_{2,t}$$
$$dS_t = rS_tdt + \sqrt{v_t}S_tdW_{1,t}$$
* **Implementation:** Employs Euler-Maruyama discretization with a Full Truncation scheme for negative variance prevention. GPU-accelerated utilizing multi-dimensional PyTorch tensors for rapid batch generation. 

### C. Merton Jump-Diffusion Engine (The Fat-Tail World)
Simulates asset paths under sudden market shocks via a Poisson process.
* **Implementation:** Integrates `torch.poisson` for discrete jump counters and log-normal jump distributions. A reparameterization trick (`torch.randn_like`) is used to construct the discrete jump log-return multipliers while preventing zero-variance CUDA crashes during intervals without jumps.

---

## 3. Deep Surrogate Architecture & Algorithmic Greeks

The `DeepSurrogateModel` is a GPU-bound PyTorch Feedforward Neural Network that accepts a unified 9-dimensional state space: $S, K, T, r, v_0, \kappa, \theta, \sigma_v, \rho$.

### The `Softplus` Innovation
Standard networks utilizing `ReLU` activations produce "wrinkled" gradient surfaces because the second derivative of a ReLU is zero everywhere. To ensure stable, actionable Greeks, this architecture exclusively employs `nn.Softplus` in hidden layers. This guarantees an infinitely differentiable pricing manifold.

### Autograd Sensitivity Extraction
Sensitivities are mathematically extracted using exact backpropagation computation:
1. **Delta ($\Delta$):** First-order derivative $\frac{\partial C}{\partial S}$ utilizing standard `torch.autograd.grad`.
2. **Gamma ($\Gamma$):** Second-order derivative $\frac{\partial^2 C}{\partial S^2}$ extracted by enforcing `create_graph=True` on the first backward pass, then differentiating the resulting Delta node.
3. **Feature Rescaling:** Since inputs are scaled via `StandardScaler`, the extracted gradients are correctly multiplied by inverse scale factors (e.g., multiplying by `scale_S ** 2` for Gamma).

---

## 4. The Hedging Simulator & Risk Analytics

The system features an institutional event-driven backtester (`hedge_simulator.py`) designed to evaluate realistic portfolio risk tracking over a 63-day options lifecycle across 1,000 generated real-world paths.

* **Band Hedging (Noise Reduction):** Implements a transaction threshold (`REBALANCE_THRESHOLD = 0.02`). The portfolio skips micro-trades caused by neural noise, solely executing when the requisite $\Delta$ drift surpasses the threshold.
* **Institutional Risk Attribution:** Breaks down total daily PnL into component drivers:
  * $\Delta$ PnL: Directional stock movement.
  * $\Theta$ PnL: Time decay of the liability.
  * Rate PnL: Risk-free compounding on the cash balance.
* **Performance Metrics:** Reports Sharpe Ratios and Average Maximum Drawdowns, establishing AI robustness versus the naive Black-Scholes baseline.

---

## 5. Model Generalization Matrix

To test for AI Model Misspecification Risk, `run_generalization_matrix.py` autonomously generates 1.5 million pricing data points mapping Black-Scholes, Heston, and Merton regimes. 
The pipeline trains three isolated neural architectures and cross-tests them inside a $3 \times 3$ grid to measure Mean Squared Error spikes when deployed out-of-distribution (e.g., exposing a Black-Scholes-trained AI to stochastic variance).

---

## 6. Execution & Automation Pipeline

The project supports a fully automated orchestration suite for statistical scale benchmarking:
**`python pipeline.py`**
1. Generates dynamically localized `parquet` datasets.
2. Trains a specialized Neural Surrogate for each generated dataset.
3. Executes `subprocess` calls to run isolated system simulations, ensuring perfectly scoped VRAM garbage collection and structured output handling (e.g., nested `run_01`, `run_02` directories).

### Specialized Scripts
* **`compare_engines.py`**: The Quant Sandbox. Instantly evaluates Point-in-Time analytical Greek exactness vs. Deep Surrogate Autograd approximations.
* **`visualize_alpha_hedge.py`**: Generates production-grade visualizations proving the Volatility Smile, 3D Multi-Greek capability, and error manifolds.
