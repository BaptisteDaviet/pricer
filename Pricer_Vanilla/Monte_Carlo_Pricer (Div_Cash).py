import numpy as np
import matplotlib.pyplot as plt

# Params
S0 = 100  # Prix initial
K = 100  # Strike
T = 1.0   # Maturité
r = 0.05  # Taux sans risque
sigma = 0.20  # Volatilité
n_simulations = 1000000  # Nombre de simulations
D = 0  # Div Cash
div_dates = [0.25, 0.5, 0.75]  # Dates de paiement des div en années

# Simulate underlying evolution
def simulate_underlying(S0, T, r, sigma, n_simulations, D, div_dates):
    n_steps = len(div_dates) 
    dt = T / n_steps
    St = np.full(n_simulations, S0)

    for _ in range(len(div_dates)) :
        Z = np.random.standard_normal(n_simulations)
        St *= np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        St -= D 
        St = np.maximum(0, St)
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
def pricing(S0, K, T, r, sigma, n_simulations, D, div_dates):
    """
    Calcule le prix des options call et put en utilisant Monte Carlo.
    """
    St = simulate_underlying(S0, T, r, sigma, n_simulations, D, div_dates)
    call_payoff = calculate_payoff(St, K, "call")
    put_payoff = calculate_payoff(St, K, "put")
    call_price = np.exp(-r * T) * np.mean(call_payoff)
    put_price = np.exp(-r * T) * np.mean(put_payoff)
    return call_price, put_price

# Test du pricer
if __name__ == "__main__":
    # Calculer les prix
    call_price, put_price = pricing(S0, K, T, r, sigma, n_simulations, D, div_dates)

    # Affichage des résultats
    print(f"Prix du Call : {call_price:.2f}")
    print(f"Prix du Put : {put_price:.2f}")

    St = simulate_underlying(S0, T, r, sigma, n_simulations, D, div_dates)
    print("Average Forward Price", np.mean(St))

    # Visualisation des résultats
    plt.hist(St, bins=30, alpha=0.7, color="blue", edgecolor="black")
    plt.title("Distribution des prix simulés à maturité (S_T)")
    plt.xlabel("Prix simulés (S_T)")
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.show()
