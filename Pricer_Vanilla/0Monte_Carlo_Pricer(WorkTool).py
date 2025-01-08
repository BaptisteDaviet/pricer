import numpy as np
import matplotlib.pyplot as plt


# Params Heston
S0 = 100  # Prix initial
K = 100  # Strike
T = 5.0   # Maturité (Y)
r = 0.05  # Taux sans risque
q = 0.02  # Taux de dividende 
N = 252 * 5  # Nbr pas de temps (1 par jour)
dt = T / N
n_simulations = 10000  # Nombre de simulations

# Params Variance Stochastique
kappa = 2.0  # Mean reversion, vitesse de retour à la moyenne 
theta = 0.2  # Niveau de variance à long terme
xi = 0.5  # Volatilité de la variance
rho = -0.7  # Corrélation entre les processus de Weiner (W1 prix et W2 var)
nu0 = 0.1  # Variance initiale (sigma^2)

# Initialisation des trajectoires
S = np.zeros((n_simulations, N + 1))  # Matrice de trajectoires prix
nu = np.zeros((n_simulations, N + 1))  # Matrice de trajectoires variance

# Conditions intiales
S[:, 0] = S0  # Fixe première col S0 
nu[:, 0] = nu0  # Fixe première col nu0


# Générer processus de Weiner corrélés : 
for t in range(N):
    Z1 = np.random.normal(0, 1, n_simulations)  # Prix
    Z2 = np.random.normal(0, 1, n_simulations)  # Variance 
    Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2  # Corrélation
# Methode d'Euler : approximation des solutions des EDS
    # Variance 
    nu[:, t + 1] = nu[:, t] + kappa * (theta - nu[:, t]) * dt + xi * np.sqrt(np.maximum(nu[:, t], 0)) * np.sqrt(dt) * Z2
    # Prix 
    S[:, t + 1] = S[:, t] * np.exp((r - q - 0.5 * nu[:, t]) * dt + np.sqrt(np.maximum(nu[:, t], 0)) * np.sqrt(dt) * Z1) 



# Afficher quelques trajectoires de prix
plt.figure(1, figsize=(10, 6))
for i in range(50):  # Afficher trajectoires
    plt.plot(np.linspace(0, T, N + 1), S[i, :], lw=1)
plt.title("Simulations des trajectoires du prix S_t (Modèle Heston)")
plt.xlabel("Temps") 
plt.ylabel("Prix S_t")
plt.grid()

# Afficher quelques trajectoires de variance
plt.figure(2, figsize=(10, 6))
for i in range(5):  # Afficher trajectoires
    plt.plot(np.linspace(0, T, N + 1), nu[i, :], lw=1)
plt.title("Simulations des trajectoires de la variance ν_t (Modèle Heston)")
plt.xlabel("Temps")
plt.ylabel("Variance ν_t")
plt.grid()

plt.show()

    # Calcul des payoffs
payoff_call = np.maximum(S[:, -1] - K, 0)
payoff_put = np.maximum(K - S[:, -1], 0)

    # Mean
mean_call = np.mean(payoff_call)
mean_put = np.mean(payoff_put)

    # Actualisation
price_call = mean_call * np.exp(-r * T) 
price_put = mean_put * np.exp(-r * T)

    
print(f"Prix du call : {price_call:.2f}")
print(f"Prix du put : {price_put:.2f}")
