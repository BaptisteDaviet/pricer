import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.stats import norm

# Params Heston 
S0 = 100
K = 100
T = 1
r = 0.05
q = 0.02
N = 252
dt = T / N
n_simulations = 10000
kappa = 1.5
theta = 0.2
xi = 0.5
rho = - 0.7
nu0 = 0.1

# Matrices
St = np.zeros((n_simulations, N + 1))
nu = np.zeros((n_simulations, N + 1))

St[:, 0] = S0
nu[:, 0] = nu0

# CORR and EDS resolution
for t in range(1, N + 1):
    Zs = np.random.normal(0, 1, n_simulations)
    Zv = np.random.normal(0, 1, n_simulations)
    Zv = rho * Zs + np.sqrt(1 - rho ** 2) * Zv
    nu[:, t] = nu[:, t - 1] + kappa * (theta - nu[:, t - 1]) * dt + xi * np.sqrt(np.maximum(nu[:, t - 1], 0)) * np.sqrt(dt) * Zv
    St[:, t] = St[:, t - 1] * np.exp((r - q - 0.5 * nu[:, t - 1]) + np.sqrt(np.maximum(nu[:, t - 1], 0)) * np.sqrt(dt) * Zs)

option_type = input("call or put? : ").strip().lower()

def calculate_greeks(S0, K, T, r, nu0):
    d1 = (np.log(S0 / K) + (r - (nu0 ** 2) / 2) * T) / (nu0 * np.sqrt(T))
    d2 = d1 - nu0 * np.sqrt(T)
    gamma = norm.pdf(d1) / (S0 * nu0 * np.sqrt(T))
    vega = (S0 * norm.pdf(d1) * np.sqrt(T)) / 10
    if option_type == "call":
        delta =  norm.cdf(d1)
        theta = -(S0 * norm.pdf(d1) * nu0) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)    
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        delta =  norm.cdf(d1) - 1
        theta = -(S0 * norm.pdf(d1) * nu0) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)    
        rho = - K * T * np.exp(-r * T) * norm.cdf(-d2)
    theta_daily = theta / 252
    return delta, gamma, vega, theta_daily, rho

if option_type == "call":
    price = np.mean(np.maximum(St[:,-1] - K, 0)) * np.exp(-r * T)
elif option_type == "put":
    price = np.mean(np.maximum(K - St[:,-1], 0)) * np.exp(-r * T)

greeks = calculate_greeks(S0, K, T, r, nu0)

print(f"{option_type.capitalize()} price : ${price:.2f}")
print(f"Greeks {option_type.capitalize()} Option :")
print(f"Delta : {greeks[0]:.4f}")
print(f"Gamma : {greeks[1]:.4f}")
print(f"Vega : {greeks[2]:.4f}")
print(f"Theta : {greeks[3]:.4f}")
print(f"Rho : {greeks[4]:.4f}")

