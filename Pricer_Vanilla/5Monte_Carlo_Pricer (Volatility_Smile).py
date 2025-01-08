import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from scipy.optimize import brentq  
from scipy.stats import norm

    # Fonction pour calculer le prix d'un call 

def black_scholes_call(S, K, T, r, q, sigma):
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = ((np.log(S / K)) + (r - q + 0.5 * sigma ** 2) * T) / sigma * np.sqrt(T)
    d2 = d1 - sigma * np.sqrt(T)

    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility_call(price, S, K, T, r, q):
    def objective(sigma):
        return black_scholes_call(S, K, T, r, q, sigma) - price

    try: 
        return brentq(objective, 0.001, 5)
    except ValueError:
        return np.nan

# Simulation des trajectoires 

# Paramètres 

S0 = 100
K = 100
T = 1.0
r = 0.05
q = 0.02

N = 252
dt = T / N 
n_simulations = 10000

kappa = 0.2
theta = 0.04
xi = 0.5
rho = - 0.7
nu0 = 0.04

# Trajectoires
S = np.zeros((n_simulations, N + 1))
nu = np.zeros((n_simulations, N + 1))

S[:, 0] = S0
nu[:, 0] = nu0

for t in range(N):
    Z1 = np.random.normal(0, 1, n_simulations)
    Z2 = np.random.normal(0, 1, n_simulations)
    Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

    nu[:, t + 1] = nu[:,t] + kappa * (theta - nu[:, t]) * dt + xi * np.sqrt(np.maximum(nu[:, t],0)) * np.sqrt(dt) * Z2
    S[:, t + 1] = S[:, t] * np.exp((r - q - 0.5 * nu[:, t]) * dt + np.sqrt(np.maximum(nu[:, t], 0)) * np.sqrt(dt) * Z1)


# Calcul Surface de Vol

strikes = np.linspace(0.8 * S0, 1.2 * S0, 20)
times = np.linspace(0, T, 20)
volatility_surface = np.zeros((len(times), len(strikes)))

for t in range(N):
    Z1 = np.random.normal(0, 1, n_simulations)
    Z2 = np.random.normal(0, 1, n_simulations)
    Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

    nu[:, t + 1] = nu[:, t] + kappa * (theta - nu[:, t]) * dt + xi * np.sqrt(np.maximum(nu[:, t], 0)) * np.sqrt(dt) * Z2
    S[:, t + 1] = S[:, t] * np.exp((r - q - 0.5 * nu[:, t]) * dt + np.sqrt(np.maximum(nu[:, t], 0)) * np.sqrt(dt) * Z1)

# Calcul de la nappe de volatilité implicite
for i, T in enumerate(times): 
    for j, K in enumerate(strikes):
        index = min(int(T / dt), N)  # Utilisation correcte de T
        S_t = S[:, index]  # Utilisation des trajectoires simulées au temps T
        price_call = np.mean(np.maximum(S_t - K, 0)) * np.exp(-r * T)
        
        # Vérification des prix avant calcul de la volatilité implicite
        if price_call > 0:
            iv = implied_volatility_call(price_call, S0, K, T, r, q)
            if not np.isnan(iv):
                volatility_surface[i, j] = iv

# Visualisation
X, Y = np.meshgrid(strikes, times)
Z = volatility_surface

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap="viridis")

ax.set_title("Nappe de volatilité implicite (Modèle Heston)")
ax.set_xlabel("Strike (K)")
ax.set_ylabel("Temps (t)")
ax.set_zlabel("Volatilité Implicite")
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()