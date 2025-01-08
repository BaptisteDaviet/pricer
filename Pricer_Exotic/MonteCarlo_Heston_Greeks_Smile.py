import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import RegularGridInterpolator

# volatility Smile
strikes = np.arange(50, 150, 5)
maturities = np.arange(0.5, 5.5, 0.5)

base_vol = np.sqrt(0.09)
smile_vol = 0.0001
skew_vol = - 0.001
time_effect_factor = 0.65

vol_surface = np.zeros((len(strikes), len(maturities)))

for i, K in enumerate(strikes):
    for j, T in enumerate(maturities):
        smile = smile_vol * ((K - 100) ** 2 ) * np.exp(- time_effect_factor * np.sqrt(T)) 
        skew = skew_vol * (K - 100)
        time_effect =  base_vol * (1 * time_effect_factor / np.sqrt(T))
        vol_surface[i, j] = smile + skew + time_effect

vol_interpolator = RegularGridInterpolator((strikes, maturities), vol_surface,  bounds_error=False, fill_value=None)

# Heston simulation
def simulate_Heston(S0, K, T, r, q, n_simulations, N, kappa, theta, xi, rho_correl):
    dt = T / N
    St = np.zeros((n_simulations, N + 1))
    nu = np.zeros((n_simulations, N + 1))
    St[:, 0] = S0
    nu_0 = vol_interpolator((K, T))
    nu[:, 0] = nu_0
    for t in range(1, N + 1):
        Zs = np.random.normal(0, 1, n_simulations)
        Zv = np.random.normal(0, 1, n_simulations)
        Zv = rho_correl * Zs + np.sqrt(1 - rho_correl**2) * Zv  
        nu[:, t] = np.maximum(
            nu[:, t - 1] + kappa * (theta - nu[:, t - 1]) * dt +
            xi * np.sqrt(np.maximum(nu[:, t - 1], 0)) * np.sqrt(dt) * Zv, 1e-8)
        St[:, t] = St[:, t - 1] * np.exp(
            (r - q - 0.5 * nu[:, t - 1]) * dt +
            np.sqrt(np.maximum(nu[:, t - 1], 0)) * np.sqrt(dt) * Zs)
         #scénarios de mouvement
        if t == int(0.3 * N):
            St[:, t] *= 0.995
        elif t == int(0.73 * N):
            St[:, t] *= 0.988
        
    return nu, St

# Calculate Greeks
def calculate_greeks(S, K, T, r, q, nu, option_type="call"):
    if S <= 0 or nu <= 0 or T <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    sqrt_T = np.sqrt(np.maximum(T, 1e-8))
    sqrt_nu = np.sqrt(np.maximum(nu, 1e-8))
    d1 = (np.log(S / K) + (r - 0.5 * nu) * T) / (sqrt_nu * sqrt_T)
    d2 = d1 - sqrt_nu * sqrt_T
    gamma = norm.pdf(d1) / (S * sqrt_nu * sqrt_T)
    vega = S * norm.pdf(d1) * sqrt_T / 100
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sqrt_nu) / (2 * sqrt_T) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 252
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sqrt_nu) / (2 * sqrt_T) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 252
        rho = - K * T * np.exp(-r * T) * norm.cdf(-d2)

    return delta, gamma, vega, theta, rho

# Dynamic calcul of Greeks
def calculate_dynamic_greeks(St, nu, K, T, r, q, option_type="call"):
    N = St.shape[1] - 1
    dt = T / N
    greeks_dynamic = {"delta": [], "gamma": [], "vega": [], "theta": [], "rho": []}
    for t in range(1, N + 1):
        S_t = St[:, t].mean()
        nu_t = nu[:, t].mean()
        time_to_maturity = T - t * dt
        delta, gamma, vega, theta, rho = calculate_greeks(S_t, K, time_to_maturity, r, q, nu_t, option_type)
        greeks_dynamic["delta"].append(delta)
        greeks_dynamic["gamma"].append(gamma)
        greeks_dynamic["vega"].append(vega)
        greeks_dynamic["theta"].append(theta)
        greeks_dynamic["rho"].append(rho)

    return greeks_dynamic

# Option Price
def option_price(St, K, r, T, option_type):
    N = St.shape[1] - 1
    if option_type == "call":
        price = np.mean(np.maximum(St[:, -1] - K, 0)) * np.exp(-r * T)
    elif option_type == "put":
        price = np.mean(np.maximum(K - St[:, -1], 0)) * np.exp(-r * T)
        
    return price

# Heston Parameters
S0 = 100       
K = 100
T = 3.0
r = 0.04         
q = 0.02
N = 252 * 3
kappa = 3.0
theta = 0.03
xi = 0.4    
rho_correl = - 0.7         
n_simulations = 100000

# Simulation
       
nu, St = simulate_Heston(S0, K, T, r, q, n_simulations, N, kappa, theta, xi, rho_correl)
greeks_dynamic = calculate_dynamic_greeks(St, nu, K, T, r, q, option_type)

price = option_price(St, K, r, T, option_type)
print(f"{option_type.capitalize()} Price : ${price:.2f}")

# PLOTS

time_points = np.linspace(0, T, N + 1)

figs, axs = plt.subplots(5, 1, figsize=(10, 12))
axs[0].plot(time_points[1:], greeks_dynamic["delta"], label="Delta", color="blue")
axs[0].set_title("Delta")
axs[1].plot(time_points[1:], greeks_dynamic["gamma"], label="Gamma", color="blue")
axs[1].set_title("Gamma")
axs[2].plot(time_points[1:], greeks_dynamic["vega"], label="Vega", color="blue")
axs[2].set_title("Vega")
axs[3].plot(time_points[1:], greeks_dynamic["theta"], label="Theta", color="blue")
axs[3].set_title("Theta")
axs[4].plot(time_points[1:], greeks_dynamic["rho"], label="Rho", color="blue")
axs[4].set_title("Rho")

for ax in axs:
    ax.legend()
    ax.grid()

plt.tight_layout()

min_price = np.min(St)
max_price = np.max(St)
y_ticks = np.arange(min_price - 50, max_price + 50, 100)

#visualisation nappe
X, Y = np.meshgrid(maturities, strikes)
fig = plt.figure(figsize=(13,9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, vol_surface, cmap='viridis')
ax.invert_xaxis()
ax.set_title("Surface de Volatilité")
ax.set_xlabel("Maturity (T)")
ax.set_ylabel("Strikes (K)")
ax.set_zlabel("Volatilité")

# S_t , v_t
plt.figure(figsize=(10, 6))
for i in range(50):
    plt.plot(np.linspace(0, T, N + 1), St[i,:], lw=0.5)
plt.ylabel("Prix $S_t$")
#plt.yticks(y_ticks)
plt.title("Generation Trajectoires de $S_t$")
plt.grid()


plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, T, N + 1), np.mean(nu, axis=0), label="Moyenne de ν_t", color="orange")  
plt.xlabel("Temps")
plt.ylabel("Variance ν_t")
plt.title("Évolution moyenne de ν_t")
plt.grid()
plt.legend()


plt.figure(figsize=(10,6))
plt.hist(St[:,-1], bins=50, color='blue', alpha=0.7, label="Distribution de $S_T$")
plt.title("Distribution des trajectoires de $S_T$ à maturité")
plt.xlabel("$S_T$")
plt.ylabel("Fréquence")
plt.legend()
plt.grid()


plt.show()