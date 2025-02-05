**Overview**

**A Python-based option pricer leveraging the Heston model to account for stochastic volatility, offering advanced risk analysis and visualization tools**

This option pricer implements the Heston model, a stochastic volatility model that more accurately captures market dynamics compared to Black-Scholes. The tool supports various option types and trading strategies, with built-in Monte Carlo simulations for pricing and risk analysis.

**Heston Model Implementation**

The Black-Scholes model assumes a fixed volatility, but in reality, market volatility varies over time and often exhibits patterns like:

**Volatility clustering:** Periods of high volatility tend to follow other high-volatility periods.

**Mean reversion:** Volatility tends to revert to a long-term average.

**Smile & Skew Effects:** The implied volatility observed in options markets often deviates from Black-Scholes' constant volatility assumption.
The Heston model captures these effects by modeling volatility as a random process, making it particularly useful for options on highly volatile assets or exotic derivatives, where sensitivity to volatility is more pronounced.
This pricer simulates thousands of stochastic volatility paths using Monte Carlo methods, providing more realistic option pricing and risk evaluation compared to traditional models.

Users can modify key Heston parameters to shape the market environment in which they wish to operate:
  - **κ (Mean reversion speed):** Determines how fast volatility returns to its long-term mean.
  - **θ (Long-term variance):** The level volatility tends to revert to over time.
  - **ξ (Volatility of volatility):** Measures the randomness of volatility changes.
  - **ρ (Correlation between price & volatility):** Determines the relationship between asset price movements and volatility.
  - **ν (Initial variance):** The starting level of variance.

Here are the pricer's current functions:
  - **Vanilla options** (call, put)
  - **Trading strategies** (spreads, straddles, strangles, butterfly)
  - **Exotic options** (digitals, barriers)
  - **Volatility Smile** (calculation and visualization of implied volatility across strikes)

And the key features:
- **Heston Model**: Monte Carlo simulation for option valuation under stochastic volatility.
- **Greeks Calculation**: Static and dynamic sensitivity analysis.
- **Volatility Smile**: Extraction and visualization of implied volatility for various strike prices.
- **Visualization**: Interactive plots for price trajectories and payoff structures.
- **Customizable Parameters**: Interface to adjust model parameters (kappa, theta, xi, rho, nu0).

**Future Developments**

In the longer term, the aim is to improve the pricer by adding several advanced features:

- Calculating and plotting Greeks for barrier options and trading strategies to improve risk sensitivity analysis.
- Enhanced Greeks Calculation (extend the calculation of Greeks to higher orders).
- Calculation and plotting of the volatility surface to better understand the dynamics of options on different maturities and strikes.

And the ultimate objective is to developp a **structured product pricing tool** enable to suppport various structured products by integrating their specific payoff and risk management features. And make **optimised hedging suggestions** based on market conditions, enabling more effective management of the risks associated with complex options and strategies.


To launch the pricer, run:
--> python projet_Pricer_v1.py

Ensure you have Python installed along with the following dependencies:
--> pip install numpy scipy matplotlib
