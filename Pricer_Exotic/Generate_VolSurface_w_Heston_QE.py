import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from tqdm import tqdm

# Heston model simulation
def heston_qe(S0, K, T, r, q, kappa, theta, xi, rho, v0, n_simulations=100000, N=500, option_type="call"):
    dt = T / N
    sqrt_dt = np.sqrt(dt)
    exp_kappa_dt = np.exp(-kappa * dt)

    S = np.zeros((n_simulations, N + 1))
    v = np.zeros((n_simulations, N + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    # Precompute constant factors
    one_minus_exp_kappa_dt = 1 - exp_kappa_dt
    kappa_dt = kappa * dt

    for t in range(1, N + 1):
        Z1 = np.random.normal(0, 1, n_simulations)
        Z2 = np.random.normal(0, 1, n_simulations)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2  # Correlation adjustment

        # Parameters for QE scheme
        m = theta + (v[:, t - 1] - theta) * exp_kappa_dt
        s2 = (v[:, t - 1] * xi**2 * one_minus_exp_kappa_dt) / (2 * kappa)
        psi = s2 / (m**2)

        # Ensure valid values
        m = np.maximum(m, 1e-8)
        s2 = np.maximum(s2, 1e-8)
        psi = np.maximum(psi, 1e-8)

        # Quadratic-Exponential scheme
        mask_uniform = psi <= 1.5
        mask_exponential = psi > 1.5

        b = np.zeros_like(psi)
        a = np.zeros_like(psi)
        b[mask_uniform] = (2 / psi[mask_uniform]) - 1 + np.sqrt(
            (2 / psi[mask_uniform]) * ((2 / psi[mask_uniform]) - 1))
        a[mask_uniform] = m[mask_uniform] / (1 + b[mask_uniform])
        U_uniform = np.random.uniform(0, 1, n_simulations)
        v[mask_uniform, t] = a[mask_uniform] * (b[mask_uniform] + 1 - U_uniform[mask_uniform]) ** (-1)

        p = np.zeros_like(psi)
        beta = np.zeros_like(psi)
        p[mask_exponential] = (psi[mask_exponential] - 1) / (psi[mask_exponential] + 1)
        beta[mask_exponential] = (1 - p[mask_exponential]) / m[mask_exponential]
        U_exponential = np.random.uniform(0, 1, n_simulations)
        v[mask_exponential, t] = np.where(U_exponential[mask_exponential] <= p[mask_exponential], 0,
            -np.log((1 - U_exponential[mask_exponential]) / (1 - p[mask_exponential])) / beta[mask_exponential])

        # Ensure non-negativity of variance
        v[:, t] = np.maximum(v[:, t], 1e-8)

        # Stock price update
        S[:, t] = S[:, t - 1] * np.exp((r - q - 0.5 * v[:, t - 1]) * dt + np.sqrt(v[:, t - 1]) * sqrt_dt * Z2)
        # Payoff at maturity
    if option_type == "call":
        payoff = np.maximum(S[:, -1] - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - S[:, -1], 0)
    else:
        raise ValueError("Invalid option type.")
    
    return np.mean(payoff) * np.exp(-r * T)

# Black-Scholes price calculation
def bs_price(S0, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

# Implied volatility calculation
def implied_volatility(price, S0, K, T, r, option_type="call"):
    def objective_function(sigma):
        return bs_price(S0, K, T, r, sigma, option_type) - price
    
    try:
        return brentq(objective_function, 1e-6, 5)  # Volatility bounds
    except ValueError:
        return np.nan

# Volatility surface generation
def generate_volatility_surface(S0, strikes, maturities, r, q, kappa, theta, xi, rho, v0, n_simulations=10000, N=252):
    vol_surface = np.zeros((len(strikes), len(maturities)))

    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            # Prix du call
            price_call = heston_qe(S0, K, T, r, q, kappa, theta, xi, rho, v0, n_simulations, N, option_type="call")
            # Prix du put
            price_put = heston_qe(S0, K, T, r, q, kappa, theta, xi, rho, v0, n_simulations, N, option_type="put")
            
            # Calcule la volatilité implicite, utilise call ou put en fonction du prix disponible
            if price_call > 0:
                vol_surface[i, j] = implied_volatility(price_call, S0, K, T, r, option_type="call")
            elif price_put > 0:
                vol_surface[i, j] = implied_volatility(price_put, S0, K, T, r, option_type="put")
            else:
                vol_surface[i, j] = np.nan  # Ignore si les deux prix sont invalides

    return vol_surface

# Parameters
S0 = 100
K = 100
T = 1.0
r = 0.05
q = 0.02
kappa = 1.5
theta = 0.04
xi = 0.2
rho = -0.5
v0 = 0.04
strikes = np.linspace(85, 115, 10)
maturities = np.linspace(0.5, 2.0, 5)

# Generate the volatility surface 
vol_surface = generate_volatility_surface(S0, strikes, maturities, r, q, kappa, theta, xi, rho, v0)

# Plot the volatility surface
X, Y = np.meshgrid(maturities, strikes)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, vol_surface, cmap='viridis')
ax.invert_xaxis()
ax.set_title("Surface de Volatilité Implicite (Modèle Heston)")
ax.set_xlabel("Maturity (T)")
ax.set_ylabel("Strikes (K)")
ax.set_zlabel("Volatilité")
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.show()
