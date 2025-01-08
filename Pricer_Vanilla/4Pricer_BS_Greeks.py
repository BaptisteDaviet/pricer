    import numpy as np
    import pandas as pd 
    from scipy.stats import norm

    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2

    #price

    def d1_d2(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + (sigma**2) / 2) * T) / sigma * np.sqrt(T)
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def b_s(S, K, T, r, sigma, optiontype = "call"):
        d1, d2 = d1_d2(S, K, T, r, sigma)
        if optiontype == "call" :
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif optiontype == "put":
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


    callprice = b_s(S, K, T, r, sigma, optiontype = "call")
    putprice = b_s(S, K, T, r, sigma, optiontype = "put")

    #greeks

    def greeks(S, K, T, r, sigma, optiontype = "call"):
        d1, d2 = d1_d2(S, K, T, r, sigma)
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        if optiontype == "call":
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        elif optiontype == "put":
            rho = - K * T * np.exp(-r * T) * norm.cdf(-d2)
        return delta, gamma, vega, theta, rho    

    def display_results(optiontype, price, greeks) :
        if position == "buy":
            print(f"{optiontype.capitalize()} Price: ${price:.2f}")
            print(f"Greeks for {optiontype.capitalize()} Option:")
            print(f"Delta: {greeks[0]:.4f}")
            print(f"Gamma: {greeks[1]:.4f}")
            print(f"Vega: {greeks[2]:.4f}")
            print(f"Theta: {greeks[3]:.4f}")
            print(f"Rho: {greeks[4]:.4f}")
        elif position == "sell":
            print(f"{optiontype.capitalize()} Price: ${price:.2f}")
            print(f"Greeks for {optiontype.capitalize()} Option:")
            print(f"Delta: {-greeks[0]:.4f}")
            print(f"Gamma: {greeks[1]:.4f}")
            print(f"Vega: {-greeks[2]:.4f}")
            print(f"Theta: {-greeks[3]:.4f}")
            print(f"Rho: {-greeks[4]:.4f}")

    position = input("buy / sell :").strip().lower()
    if position not in ["buy", "sell"]:
        raise ValueError("invalid. Enter buy or sell")

    optiontype = input("Option Type (call or put) :").strip().lower()
    if optiontype not in ["call", "put"]:
        raise ValueError("invalid. Enter call or put")

    price = b_s(S, K, T, r, sigma, optiontype)
    greeks_values = greeks(S, K, T, r, sigma, optiontype)

    display_results(optiontype, price, greeks_values)

