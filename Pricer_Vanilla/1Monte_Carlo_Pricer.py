import numpy as np
import matplotlib.pyplot as plt

# Params
S0 = 100  # Prix initial
K = 100  # Strike
T = 1.0   # Maturité
r = 0.05  # Taux sans risque
sigma = 0.20  # Volatilité
n_simulations = 100000  # Nombre de simulations

# Simulate underlying evolution
def simulate_underlying(S0, T, r, sigma, n_simulations):
    Z = np.random.standard_normal(n_simulations)
    St = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    return St

# Payoff calculation
def calculate_payoff(St, K, option_type="call"):
    if option_type == "call":
        return np.maximum(0, St - K)
    elif option_type == "put":
        return np.maximum(0, K - St)    
    else:
        raise ValueError("Erreur option_type : utilisez 'call' ou 'put'.")

# Monte Carlo pricing
def pricing(S0, K, T, r, sigma, n_simulations):
    #Calcule le prix des options call et put en utilisant Monte Carlo.
    St = simulate_underlying(S0, T, r, sigma, n_simulations)
    call_payoff = calculate_payoff(St, K, "call")
    put_payoff = calculate_payoff(St, K, "put")
    call_price = np.exp(-r * T) * np.mean(call_payoff)
    put_price = np.exp(-r * T) * np.mean(put_payoff)
    return call_price, put_price

# Test du pricer
if __name__ == "__main__":
    # Calculer les prix
    call_price, put_price = pricing(S0, K, T, r, sigma, n_simulations)

    # Affichage des résultats
    print(f"Prix du Call : {call_price:.2f}")
    print(f"Prix du Put : {put_price:.2f}")

    St = simulate_underlying(S0, T, r, sigma, n_simulations)
    print("Average Forward Pice :", np.mean(St))

    # Visualisation des résultats
    plt.hist(St, bins=30, alpha=0.7, color="blue", edgecolor="black")
    plt.title("Distribution des prix simulés à maturité (S_T)")
    plt.xlabel("Prix simulés (S_T)")
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.show()

