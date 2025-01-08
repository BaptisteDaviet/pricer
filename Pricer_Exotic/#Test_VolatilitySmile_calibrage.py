#Test_VolatilitySmile_calibrage
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

maturity_index = 5  # Choisir une maturité dans la nappe
plt.figure(figsize=(10, 6))
plt.plot(strikes, vol_surface[:, maturity_index], label=f"Smile à T={maturities[maturity_index]:.2f}")
plt.title("Smile de Volatilité")
plt.xlabel("Strike")
plt.ylabel("Volatilité")
plt.grid()
plt.legend()


# Visualisation de l'évolution de la volatilité pour un strike donné
strike_index = 10  # Choisir un strike dans la nappe
plt.figure(figsize=(10, 6))
plt.plot(maturities, vol_surface[strike_index, :], label=f"Vol à K={strikes[strike_index]:.2f}")
plt.title("Évolution de la Volatilité en Fonction de la Maturité")
plt.xlabel("Maturité")
plt.ylabel("Volatilité")
plt.grid()
plt.legend()
plt.show()