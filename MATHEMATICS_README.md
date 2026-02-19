# Mathematics of Project Alpha-Hedge

This document details the mathematical models, stochastic differential equations (SDEs), and numerical methods **actually implemented** in the codebase.

---

## 1. `src/pricing_engines.py`

### A. Black-Scholes Engine (`BlackScholesEngine`)
**Model**: Geometric Brownian Motion.
$$ dS_t = r S_t dt + \sigma S_t dW_t $$

**Pricing**: Closed-form analytical solution (Call Option).
**Greeks**: Analytical $\Delta, \Gamma, \Theta, \rho$.

**Implied Volatility (Root Finding)**:
To inverse the Black-Scholes formula $C(\sigma) = C_{market}$, we use **Brent's Method** (`scipy.optimize.brentq`) to find the root of:
$$ f(\sigma) = C_{BS}(S, K, T, r, \sigma) - C_{market} = 0 $$
Search interval: $\sigma \in [0.01\%, 500\%]$.

### B. Heston Monte Carlo Engine (`HestonMCEngine`)
**Model**: Heston Stochastic Volatility.
$$ dS_t = r S_t dt + \sqrt{v_t} S_t dW_t^S $$
$$ dv_t = \kappa (\theta - v_t) dt + \sigma_v \sqrt{v_t} dW_t^v $$

**Numerical Scheme**:
We implement **Euler-Maruyama** for $S$ and **Full Truncation** for $v$ to handle negative variance.
1. **Variance**:
   $$ v_{t+\Delta t} = v_t + \kappa(\theta - v_t^+) \Delta t + \sigma_v \sqrt{v_t^+} \sqrt{\Delta t} Z_v $$
   where $v^+ = \max(v, 0)$.
2. **Asset** (Euler on Level):
   $$ S_{t+\Delta t} = S_t + r S_t \Delta t + \sqrt{v_t^+} S_t \sqrt{\Delta t} Z_S $$

### C. Merton Jump-Diffusion Engine (`MertonJumpMCEngine`)
**Model**: GBM + Compound Poisson Jumps.
$$ dS_t = (r - \lambda k) S_t dt + \sigma S_t dW_t + S_{t^-} (Y - 1) dN_t $$

**Numerical Scheme**:
We use **Exact Simulation** of log-turns (Geometric Brownian Motion + Jumps).
$$ S_{t+\Delta t} = S_t \exp\left( \text{drift} \cdot \Delta t + \sigma \sqrt{\Delta t} Z + M_t \right) $$
where $M_t = \sum_{i=1}^{N_t} \ln Y_i$ is the jump component.

### D. Deep Surrogate Engine (`DeepSurrogateEngine`)
**Approximation**: Feedforward Dense Neural Network ($9 \to 64 \to 64 \to 64 \to 1$).
**Greeks**: Computed via PyTorch **Automatic Differentiation** (Autograd).

**Hybrid Handling near Maturity**:
For $T < 0.05$ (close to expiry), the Autograd gradients can become unstable. The engine switches to a **Black-Scholes Approximation** to ensure numeric stability for $\Delta$.

---

## 2. `src/hedge_simulator.py`

**Hedging Logic**:
1. **Rebalancing Condition (Band Hedging)**:
   Trades are only executed if the target delta deviates from current holdings by a threshold:
   $$ |\Delta_{target} - \phi_{current}| > \epsilon_{threshold} \quad (\epsilon=0.02) $$
2. **Transaction Costs**:
   The `HedgingPortfolio` class supports linear transaction costs, though they are currently configured to **0.0** in the simulation loop.

**PnL Attribution**:
- **Delta PnL**: Gain from directional move $\phi \Delta S$.
- **Theta PnL**: Gain from time decay $-\Theta \Delta t$.
- **Rate PnL**: Interest earned on Cash position $B_t (e^{r \Delta t} - 1)$.
- **Hedging Error**: The residual unexplainable variance at maturity.

---

## 3. `src/train_surrogate.py`

**Optimization**:
- **Loss Function**: **Huber Loss** (Robust Regression), combines MSE and MAE to be less sensitive to MC noise outliers.
- **Preprocessing**: **StandardScaler** (Z-Score) applied to inputs to normalize ranges (e.g., $S \in [80, 120] \to [-1, 1]$).
- **Optimizer**: AdamW + ReduceLROnPlateau scheduler.

---

## 4. `src/generate_dataset.py`

**Distribution**:
Uniform LHS-like sampling over parameter space:
- $S, K \sim U(80, 120)$
- $v_0, \theta \sim U(0.01, 0.2)$
- $\kappa \sim U(1.0, 5.0)$
- $\rho \sim U(-0.9, -0.1)$

---

## 5. `src/run_generalization_matrix.py`

**Metric**: Mean Squared Error (MSE) of pricing.
**Evaluation**: Computes a $3 \times 3$ matrix of (Train World $\to$ Test World) performance.
