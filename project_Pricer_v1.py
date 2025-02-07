import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

class MenuHandler:
    def __init__(self):
        self.heston_params = HestonParams()
        self.vanilla_option = VanilaOption (self.heston_params)
        self.trading_strategies = TradingStrategies(self.heston_params)
        self.exotic_options = ExoticOption(self.heston_params)
        self.volatility_smile = VolatilitySmile(self.heston_params)

    def menu_principal(self):
        while True:
            print("\nWelcome to the pricer!")
            print("\n1. Price Vanilla Options")
            print("2. Price Trading Strategies")
            print("3. Price Exotic Options")
            print("4. View Volatility Smile")
            print("5. View/ Modify Heston Parameters")
            print("6. Exit")
            print("Please enter Strikes in value.")
            
            choice = input("Select an option: ")

            if choice == "1":
                self.vanilla_option.menu_pricing_vanilla_options()
            elif choice == "2":
                self.trading_strategies.menu_pricing_trading_strategies()
            elif choice == "3":
                self.exotic_options.menu_pricing_exo_options()
            elif choice == "4":
                self.volatility_smile.menu_volatility_smile()
            elif choice == "5":
                self.heston_params.menu_heston_params()    
            elif choice == "6":    
                break
            else:
                print("Invalid choice. Please enter a valid number.")


class VanilaOption :
    def __init__(self, heston_params):
        self.heston_params = heston_params

    def menu_pricing_vanilla_options(self):
        option_type = input("Enter the option type (call/put): ").strip().lower()
        position = input("Do you want to buy or to sell the option? (buy/ sell): ").strip().lower()
        S0 = float(input("Enter the current price of the underlying (S): "))
        K = float(input("Enter the Strike price (K): "))
        T = float(input("Enter the maturity in years (T): "))
        r = float(input("Enter the free risk rate (r) [default 0.05]: ") or 0.05)
        q = float(input("Enter the annual dividend yield (q) [default 0.02]: ") or 0.02)

        price, _ = heston(self.heston_params, S0, K, T, r, q, option_type, n_simulations=100000)
        print(f"\nThe price of the {option_type} option is: ${price:.2f}")

        if input("Calculate Greeks? (y/n): ").strip().lower() == "y":
            self.calculate_greeks_vanilla(S0, K, T, r, q, option_type)
        if input ("Plot the trajectories of S? (y/n): ").strip().lower() == "y":
            print("Plotting the trajectories...")
            self.plot_trajectories(self.heston_params, S0, K, T, r, q, option_type)

        print("Plotting the option payoff...")
        plot_vanilla(self.heston_params, S0, K, T, r, q, option_type, position)

    def calculate_greeks_vanilla(self, S0, K, T, r, q, option_type):
        """ Fonction générique pour calculer et afficher les grecs """
        greeks_choice = input("Please Enter 1. for Static Greeks and 2. for Dynamic Greeks: ").strip().lower()
        if greeks_choice == "1":
            greeks,_ = calculate_greeks(S0, K, T, r, q, option_type, self.heston_params, option_variant="vanilla")
            print("\nStatic Greeks:")
            for g, val in greeks.items():
                print(f"{g}: {val:.4f}")
        elif greeks_choice == "2":
            S_range = np.linspace(0.1 * S0, 2 * S0, 125)
            greeks_dynamic = calculate_dynamic_greeks(S_range, S0, K, T, r, q, option_type, self.heston_params)
            print("\nDynamic Greeks Calculated. Plotting...")
            plot_dynamic_greeks(S_range, greeks_dynamic)
        else:
            print("Please Enter Valid Option.")
    
    def plot_trajectories(self, heston_params, S0, K, T, r, q, option_type, n_simulations=20000):
        _,S = heston(heston_params, S0, K, T, r, q, option_type, n_simulations) 
        N = 252
        time =  np.linspace(0, T, N + 1)

        plt.figure(figsize=(15,10))
        for i in range(45):
            plt.plot(time, S[i,:], linewidth=0.5)
        plt.xlabel("Time")
        plt.ylabel("Underlying Price (S)")
        plt.title("Random S trajectories under Heston Model")
        plt.grid()
        plt.show()

class TradingStrategies:
    def __init__(self, heston_params):
        self.heston_params = heston_params

    def menu_pricing_trading_strategies(self):
        while True:
            print("\nWhat do you want to price?")
            print("1. Call Spread / Put Spread")
            print("2. Straddle / Strangle")
            print("3. Butterfly")
            print("4. Exit")
            choice = input("Select an option: ")

            if choice == "1":
                self.pricing_spread()
            elif choice == "2":
                self.pricing_strangle()
            elif choice == "3":
                self.pricing_butterfly()
            elif choice == "4":
                break
            else:
                print("Invalid choice. Please enter a valid number.")

    def pricing_spread(self):
        spread_type = input("\nDo you want to price a Call Spread or a Put Sread? (call spread/ put spread): ").strip().lower()
        position = input("Do you want to buy or to sell the strategy? (buy/ sell): ").strip().lower()
        K1 = float(input("Enter the lower strike price (K1): "))
        K2 = float(input("Enter the higher strike price (K2): "))
        S0 = float(input("Enter the current price of the underlying (S): "))
        T = float(input("Enter the maturity in years (T): "))
        r = float(input("Enter the free risk rate (r) [default 0.05]: ") or 0.05)
        q = float(input("Enter the annual dividend yield (q) [default 0.02]: ") or 0.02)      
        
        if spread_type == "call spread":
            price_K1,_ = heston(self.heston_params, S0, K1, T, r, q, "call", n_simulations=20000)
            price_K2,_ = heston(self.heston_params, S0, K2, T, r, q, "call", n_simulations=20000)
        else:
            price_K1,_ = heston(self.heston_params, S0, K1, T, r, q, "put", n_simulations=20000)
            price_K2,_ = heston(self.heston_params, S0, K2, T, r, q, "put", n_simulations=20000)

        if spread_type == "call spread":
            price_spread = price_K1 - price_K2
        elif spread_type == "put spread":
            price_spread = price_K2 - price_K1
        else:
            print("Invalid Option. Please enter Call Spread or Put Spread.")

        """if position == "sell":
            price_spread *= -1"""

        print(f"The price of the {spread_type} strategy is ${price_spread:.2f}")
        print("Plotting the payoff...")
        plot_spread(S0, K1, K2, T, r, q, spread_type, position)

    def pricing_strangle(self):
        position = input("Do you want to buy or to sell the strategy? (buy/ sell): ").strip().lower()
        K1 = float(input("Enter the lower strike price (K1): "))
        K2 = float(input("Enter the higher strike price (K2): "))
        S0 = float(input("Enter the current price of the underlying (S): "))
        T = float(input("Enter the maturity in years (T): "))
        r = float(input("Enter the free risk rate (r) [default 0.05]: ") or 0.05)
        q = float(input("Enter the annual dividend yield (q) [default 0.02]: ") or 0.02)

        price_put,_ = heston(self.heston_params, S0, K1, T, r, q, "put", n_simulations=20000)
        price_call,_ = heston(self.heston_params, S0, K2, T, r, q, "call", n_simulations=20000)
        
        price = price_put + price_call

        if K1 == K2 :
            name = "Straddle"
        elif K1 != K2:
            name = "Strangle"

        print(f"The price of the {name} (with K1 = {K1} and K2 = {K2}) is ${price:.2f}")            
        print("Plotting the payoff...")
        plot_strangle(position, S0, K1, K2, T, r, q, name)

    def pricing_butterfly(self):
        strategy_type = input("Do you want to price a Call Butterfly or a Put Butterfly? (call butterfly/ put butterfly) ").strip().lower()
        position = input("Do you want to buy or to sell the strategy? (buy/ sell): ").strip().lower()
        K1 = float(input("Enter the lower strike price (K1): "))
        K2 = float(input("Enter the middle strike price (K2): "))
        K3 = K2 + (K2 - K1)
        print(f"Strike 1 = {K1}, Strike 2 = {K2} and Strike 3 = {K3}")
        qk3 = input(f"Ok with Strike 3 = {K3} ? (y/n) ")
        if qk3 == "y":
            K3 = K2 + (K2 - K1)
        elif qk3 == "n":
            K3 = float(input("Enter the higher strike price (K3): "))
        else : 
            print("Invalid choice. Please enter y or n ")
        S0 = float(input("Enter the current price of the underlying (S): "))
        T = float(input("Enter the maturity in years (T): "))
        r = float(input("Enter the free risk rate (r) [default 0.05]: ") or 0.05)
        q = float(input("Enter the annual dividend yield (q) [default 0.02]: ") or 0.02)

        if strategy_type == "call butterfly":
            price_K1,_ = heston(self.heston_params, S0, K1, T, r, q, "call", n_simulations=30000)
            price_K2,_ = heston(self.heston_params, S0, K2, T, r, q, "call", n_simulations=30000)
            price_K3,_ = heston(self.heston_params, S0, K3, T, r, q, "call", n_simulations=30000)
            price_butterfly = price_K1 - 2 * price_K2 + price_K3

        elif strategy_type == "put butterfly":
            price_K1,_ = heston(self.heston_params, S0, K1, T, r, q, "put", n_simulations=30000)
            price_K2,_ = heston(self.heston_params, S0, K2, T, r, q, "put", n_simulations=30000)
            price_K3,_ = heston(self.heston_params, S0, K3, T, r, q, "put", n_simulations=30000)
            price_butterfly = price_K1 - 2 * price_K2 + price_K3

        else : 
            print("Invalid choice. Please enter call butterfly or put butterfly.")
        
        price_butterfly = max(price_butterfly, 0)

        print(f"The price of the {strategy_type.capitalize()} strategy is ${price_butterfly:.2f}.")
        print("Plotting the payoff...")
        plot_butterfly(position, strategy_type, S0, K1, K2, K3, price_butterfly)


class ExoticOption: 
    def __init__(self, heston_params):
        self.heston_params = heston_params

    def menu_pricing_exo_options(self):
        while True :
            print("\nWhat do you want to price?")
            print("1. Digital Options")
            print("2. Barrier Options")
            print("3. Exit")
            choice = input("Please select an option: ")

            if choice == "1":
                self.pricing_digital_option()
            elif choice == "2":
                self.pricing_barrier_option()
            elif choice == "3":
                break
            else :
                print("Invalid choice. Please enter a valid number.")
    
    def pricing_digital_option(self):
        option_type = input("Enter the digit type (call/ put): ").strip().lower()
        position = input("Do you want to buy or to sell the digit? (buy/ sell): ").strip().lower()
        S0 = float(input("Enter the current price of the underlying (S): "))
        K = float(input("Enter the Strike price (K): "))
        T = float(input("Enter the maturity in years (T): "))
        r = float(input("Enter the free risk rate (r) [default 0.05]: ") or 0.05)
        q = float(input("Enter the annual dividend yield (q) [default 0.02]: ") or 0.02)
        
        _, d2 = calculate_greeks(S0, K, T, r, q, option_type, self.heston_params, option_variant="digital")

        if option_type == "call":
            digit_price = norm.pdf(d2) * np.exp(-r * T)
        elif option_type == "put":
            digit_price = (1 - norm.pdf(d2)) * np.exp(-r * T)
        
        print(f"Digital Option's price is ${digit_price:.2f}")

        if input("Calculate greeks ? (y/ n): ").strip().lower() == "y":
            self.calculate_greeks_digit( S0, K, T, r, q, option_type)
        print("Plotting the digital payoff...")
        plot_digit(self.heston_params, S0, K, T, r, q, option_type, position)

    def calculate_greeks_digit(self, S0, K, T, r, q, option_type):
        S_range = np.linspace(0.5 * S0, 1.5 * S0, 100)
        greeks_choice = input("Please Enter 1. for Static Greeks and 2. for Dynamic Greeks: ").strip().lower()
        if greeks_choice == "1":
            greeks,_ = calculate_greeks(S0, K, T, r, q, option_type, self.heston_params, option_variant="digital")
            for g, val in greeks.items():    
                print(f"{g}: {val:.4f}")
        elif greeks_choice == "2":
            greeks_dynamic_digit = calculate_dynamic_greeks_digit(S_range, S0, K, T, r, q, option_type, self.heston_params)
            print("\nDynamic Greeks Calculated. Plotting...")
            plot_dynamic_greeks_digit(S_range, greeks_dynamic_digit)
        else:
            print("Please Enter Valid Option.")

    def calculate_barrier_option(self, heston_params, S0, K, B, T, r, q, option_type, direction, barrier_type):
        _,S = heston(heston_params, S0, K, T, r, q, option_type, n_simulations=20000)

        if direction == "up":
            hit_barrier = np.any(S >= B, axis=1)
        elif direction == "down":
            hit_barrier = np.any(S <= B, axis=1) #tableau booléen true if barrier hit, else false 
        else:
            raise ValueError("Direction must be 'up' or 'down'.")

        if barrier_type == "ki":
            active_path = hit_barrier
        elif barrier_type == "ko":
            active_path = ~hit_barrier # inversement des true et false
        else:
            raise ValueError("Barrier type must be 'ki' or 'ko'.")

        if option_type == "call" : 
            payoff = np.maximum(S[:, -1] - K, 0) 
        elif option_type == "put":
            payoff = np.maximum(K - S[:, -1], 0)
        else:
            raise ValueError("Option type must be 'call or 'put'.")

        payoff = payoff[active_path]

        if len(payoff) > 0:
            price = np.mean(payoff) * np.exp(-r * T)
        else:
            0.0

        return price

    def pricing_barrier_option(self):
        print("\nBarrier Option Pricing")
        option_type = input("Enter option type (call/ put) :").strip().lower()
        barrier_type = input("Enter barrier type (knock-in/ knock-out) (enter: ki/ ko): ").strip().lower()
        direction = input("Barrier direction (up/ down) :").strip().lower()
        position = input("Do you want to buy or to sell the option (buy/ sell) :").strip().lower()
        S0 = float(input("Enter current price of the underlying (S): "))
        K = float(input("Enter the Strike price (K): "))
        B = float(input("Enter Barrier level (B): "))
        T = float(input("Enter the maturity in years (T): "))
        r_i = input("Enter the free risk rate (r) [default 0.05]: ").strip()
        r = float(r_i) if r_i else 0.05
        q_i = input("Enter the annual dividend yield (q) [default 0.02]: ").strip()
        q = float(q_i) if q_i else 0.02

        if (direction == "up" and B <= S0) or (direction == "down" and B >= S0):
            raise ValueError("La barrière doit être cohérente avec la direction choisie.")

        price = self.calculate_barrier_option(self.heston_params, S0, K, B, T, r, q, option_type, direction, barrier_type)
        print(f"{option_type.capitalize()} {direction.capitalize()} {barrier_type.capitalize()}'s price is ${price:.2f}")

        print("Plotting the option payoff...")
        plot_barrier_option(K, B, option_type, direction, barrier_type, position)


class VolatilitySmile:
    def __init__(self, heston_params):
        self.heston_params = heston_params

    def menu_volatility_smile(self):
        S0 = float(input("Enter the current price of the underlying (S): "))
        T = float(input("Enter the maturity in years (T): "))
        r = float(input("Enter the free risk rate (r) [default 0.05]: ") or 0.05)
        q = float(input("Enter the annual dividend yield (q) [default 0.02]: ") or 0.02)
        print("Plotting the volatility smile...")

        heston_params = HestonParams(
            kappa=self.heston_params.kappa,
            theta=self.heston_params.theta,
            xi=self.heston_params.xi,
            rho=self.heston_params.rho,
            nu0=self.heston_params.nu0,
            reset=False
        )

        K_range, call_vols, put_vols, avg_vols = self.calculate_volatility_smile(S0, T, r, q)    

        plt.figure()
        plt.plot(K_range, call_vols, label="Call Implied Volatility", color="blue", linestyle="--")
        plt.plot(K_range, put_vols, label="Put Implied Volatility", color="red", linestyle="--")
        plt.plot(K_range, avg_vols, label="Average Volatility Smile", color="black")
        plt.xlabel("Strike (K)")
        plt.ylabel("Implied Volatility")
        plt.title("Volatility Smile Based on Calls and Puts")
        plt.legend()
        plt.grid()
        plt.show()

    def b_s(self, S, K, T, r, q, sigma, option_type):
        d1 = ((np.log(S / K)) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            bs_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else :
            bs_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)        
        return bs_price

    def implied_volatility(self, price, S, K, T, r, q, option_type):
        def objective(sigma):
            return self.b_s(S, K, T, r, q, sigma, option_type) - price
        try:
            iv = brentq(objective, 0.0001, 10)
            return iv
        except ValueError:
            return np.nan
        
    def calculate_volatility_smile(self, S0, T, r, q):
        K_range = np.linspace(0.5 * S0, 2 * S0, 50)
        
        call_prices = []
        put_prices = []
        for K in K_range :
            call_price,_ = (heston(self.heston_params, S0, K, T, r, q, "call", n_simulations=30000))
            put_price,_ = (heston(self.heston_params, S0, K, T, r, q, "put", n_simulations=30000))
            call_prices.append(call_price)
            put_prices.append(put_price)

        call_vols = []
        put_vols = []
        for call_price, put_price, K in zip(call_prices, put_prices, K_range):
            call_vols.append(self.implied_volatility (call_price, S0, K, T, r, q, "call"))
            put_vols.append(self.implied_volatility(put_price, S0, K, T, r, q, "put"))
        
        avg_vols = []
        for call, put in zip (call_vols, put_vols):
            avg = (call + put) / 2
            avg_vols.append(avg)

        return K_range, call_vols, put_vols, avg_vols

import json

class HestonParams:
    def __init__(self, kappa=2.0, theta=0.09, xi=0.75, rho=-0.4, nu0=0.09, reset=False):
        
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.nu0 = nu0

        self.default_params = {
            "kappa": kappa,
            "theta": theta,
            "xi": xi,
            "rho": rho,
            "nu0": nu0
        }

        self.load_params()

        if reset:
            self.reset_params()

    def menu_heston_params(self):
        while True :   
            print("\nCurrent Heston Parameters:")
            self.show_params()

            print("\nOptions :")
            print("1. Modify parameters")
            print("2. Reset to default parameters")
            print("3. Help on parameters")
            print("0. Exit")

            choice = input("Select an option: ").strip()

            if choice == "1":
                self.modify_params()
            elif choice == "2":
                self.reset_params()
            elif choice == "3":
                self.help_params()
            elif choice == "0":
                break
            else :
                print("Invalid choice. Please enter a valid number.")

    def show_params(self):
        print(f"kappa : {self.kappa}")
        print(f"theta : {self.theta}")
        print(f"xi : {self.xi}")
        print(f"rho : {self.rho}")
        print(f"nu0 : {self.nu0}")

    def modify_params(self):
        print("Enter new values for Heston parameters: ")

        new_kappa = input(f"Enter new value for kappa (current {self.kappa}): ").strip()
        new_theta = input(f"Enter new value for theta (current {self.theta}): ").strip()
        new_xi = input(f"Enter new value for xi (current {self.xi}): ").strip()
        new_rho = input(f"Enter new value for rho (current {self.rho}): ").strip()
        new_nu0 = input(f"Enter new value for ν_0 (current {self.nu0}): ").strip()

        if new_kappa: self.kappa = float(new_kappa)
        if new_theta: self.theta = float(new_theta)
        if new_xi: self.xi = float(new_xi)
        if new_rho: self.rho = float(new_rho)
        if new_nu0: self.nu0 = float(new_nu0)

        self.save_params()

    def reset_params(self):
        self.kappa = self.default_params["kappa"]
        self.theta = self.default_params["theta"]
        self.xi = self.default_params["xi"]
        self.rho = self.default_params["rho"]
        self.nu0 = self.default_params["nu0"]
        print("Parameters have been reset to default values.")
        self.save_params()

    def save_params(self):
        params={
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
            "nu0": self.nu0
        }
        with open("heston_params.json", "w") as f:
            json.dump(params, f)
        print("Saved parameters")

    def load_params(self):
        try:
            with open("heston_params.json", "r") as f:
                params = json.load(f)
                self.kappa = params.get("kappa", self.default_params["kappa"])
                self.theta = params.get("theta", self.default_params["theta"])
                self.xi = params.get("xi", self.default_params["xi"])
                self.rho = params.get("rho", self.default_params["rho"])
                self.nu0 = params.get("nu0", self.default_params["nu0"])
            print("Parameters loaded")
        except FileNotFoundError:
            print("Using the default settings.")

    def help_params (self):
        print("\nHeston Model Parameter Help:")
        print("kappa: Speed of mean reversion (how quickly volatility returns to the long-term mean).")
        print("theta: Long-term variance (expected variance level over time).")
        print("xi: Volatility of volatility (how much volatility itself fluctuates).")
        print("rho: Correlation between asset price and its volatility (negative means price drops -> vol increases).")
        print("nu0: Initial variance of the asset.")


# Functions
def heston(heston_params, S0, K, T, r, q, option_type, n_simulations=20000):
    kappa, theta, xi, rho, nu0 = heston_params.kappa, heston_params.theta, heston_params.xi, heston_params.rho, heston_params.nu0
    N = 252
    dt = T / N

    S = np.zeros((n_simulations, N + 1))
    nu = np.zeros((n_simulations, N + 1))
    S[:, 0] =  S0 
    nu[:, 0] = nu0

    for t in range(N):
        Z1, Z2 = np.random.normal(0, 1, (2, n_simulations // 2))
        Z2 = Z1 * rho + np.sqrt(1 - rho**2) * Z2
        Z1_full = np.concatenate([Z1, -Z1])
        Z2_full = np.concatenate([Z2, -Z2])

        nu[:, t + 1] = np.maximum(nu[:, t] + kappa * (theta - nu[:, t]) * dt + xi * np.sqrt(nu[:, t]) * np.sqrt(dt) * Z2_full, 1e-6)
        S[:, t + 1] = S[:, t] * np.exp((r - q - 0.5 * nu[:, t]) * dt + np.sqrt(nu[:, t]) * np.sqrt(dt) * Z1_full)
        
    if option_type == "call" : 
        payoff = np.maximum(S[:, -1] - K, 0) 
    else :
        payoff = np.maximum(K - S[:, -1], 0)

    price = np.mean(payoff) * np.exp(-r * T)
    
    return price, S


def calculate_greeks(S0, K, T, r, q, option_type, heston_params, option_variant):
    sigma = max(np.sqrt(heston_params.nu0), 1e-6)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_variant == "vanilla":
        gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
        vega = S0 * norm.pdf(d1) * np.sqrt(T) / 100
        if option_type == "call":
            delta = norm.cdf(d1)
            theta = (-S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 252
            rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 10
        else:
            delta = norm.cdf(d1) - 1
            theta = (-S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 252
            rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 10

    elif option_variant == "digital":
        gamma = np.exp(-r * T) * ((norm.pdf(d2) * d2) / (
            S0 ** 2 * sigma ** 2 * T))
        vega = np.exp(-r * T) * norm.pdf(d2) * (d2 / sigma)       
        if option_type == "call":
            delta = np.exp(-r * T) * (
                norm.pdf(d2) / (S0 * sigma * np.sqrt(T)))
            theta = -r * np.exp(-r * T) * norm.pdf(d2) - (
                np.exp(-r * T) * (norm.pdf(d2) * d2) / (2 * T)
            )
            rho = T * np.exp(-r * T) * norm.pdf(d2)
        else :
            delta = - np.exp(-r * T) * (
                norm.pdf(d2) / (S0 * sigma * np.sqrt(T)))
            theta = -r * np.exp(-r * T) * (1 - norm.pdf(d2)) + (
                np.exp(-r * T) * (norm.pdf(d2) * d2) / (2 * T)
            )
            rho = - T * np.exp(-r * T) * (1 - norm.pdf(d2))

    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}, d2


def calculate_dynamic_greeks(S_range, S0, K, T, r, q, option_type, heston_params):
    greeks = {key: [] for key in ["Delta", "Gamma", "Theta", "Vega", "Rho"]}

    for S in S_range:
        result,_ = calculate_greeks(S, K, T, r, q, option_type, heston_params, option_variant="vanilla")

        for key in greeks:
            greeks[key].append(result.get(key, 0.0))

    return greeks

def calculate_dynamic_greeks_digit(S_range, S, K, T, r, q, option_type, heston_params):
    greeks = {key: [] for key in ["Delta", "Gamma", "Theta", "Vega", "Rho"]}

    for S in S_range:
        greeks_values, _ = calculate_greeks(S, K, T, r, q, option_type, heston_params, option_variant="digital")
        for key in greeks:
            greeks[key].append(greeks_values[key])
    
    return greeks

def plot_vanilla(heston_params, S, K, T, r, q, option_type, position):
    S_range = np.linspace(0.5 * S, 1.5 * S, 50)

    prices = []
    payoffs = []
    bs_prices = []

    premium,_ = heston(heston_params, S, K, T, r, q, option_type, n_simulations=8000)
    
    sigma = np.sqrt(heston_params.nu0)
    volatility_smile = VolatilitySmile(heston_params)

    for S_i in S_range:
        calculate_price,_ = heston(heston_params, S_i, K, T, r, q, option_type, n_simulations=8000)
        prices.append(calculate_price)
        bs_price = volatility_smile.b_s(S_i, K, T, r, q, sigma, option_type)
        bs_prices.append(bs_price)

        if option_type == "call":
            intrinsic_value = np.maximum(S_i - K, 0)
        elif option_type == "put":
            intrinsic_value = np.maximum(K - S_i, 0)
        else:
            raise ValueError("Invalid option type. Must be 'call' or 'put'.")

        if position == "buy":
            payoff_value = intrinsic_value - premium
        elif position == "sell":
            payoff_value = -intrinsic_value + premium
        else:
            raise ValueError("Invalid position. Must be 'buy' or 'sell'.")

        payoffs.append(payoff_value)
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(S_range, prices, label="Option Price (Monte-Carlo)", color="blue")
    plt.plot(S_range, payoffs, label="Option Payoff", color="green", linestyle="--")
    plt.plot(S_range, bs_prices, label="Option Price (Black_Scholes)", color="red", linestyle="--", linewidth=0.75)
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.axvline(K, color="red", linewidth=0.75, linestyle="--", label="Strike Level (K)")
    plt.title(f"{position.capitalize()} {option_type.capitalize()} Option Payoff")
    plt.xlabel("Sous-jacent (S)", fontsize=12)
    plt.ylabel("Prix / Payoff", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_digit(heston_params, S, K, T, r, q, option_type, position):
    S_range = np.linspace(0.1 * S, 2.5 * S, 200)

    digit_prices = []
    payoffs = []

    for S_i in S_range:

        _, d2 = calculate_greeks(S_i, K, T, r, q, option_type, heston_params, option_variant="digital")

        if option_type == "call":
            digit_price = np.exp(-r * T) * norm.cdf(d2)
            intrinsic_value = 1 if S_i > K else 0
        elif option_type == "put":
            digit_price = np.exp(-r * T) * (1 - norm.cdf(d2))
            intrinsic_value = 1 if S_i < K else 0
        else:
            raise ValueError("Invalid option type. Must be 'call' or 'put'.")

        premium = 0

        if position == "buy":
            payoff_value = intrinsic_value - premium
        elif position == "sell":
            payoff_value = premium - intrinsic_value
        else:
            raise ValueError("Invalid position. Must be 'buy' or 'sell'.")

        digit_prices.append(digit_price)
        payoffs.append(payoff_value)

    plt.figure()
    plt.plot(S_range, payoffs, label='Digital Option Payoff', color='black', linestyle='--')
    plt.plot(S_range, digit_prices, label='Digital Option Price', color='blue')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(K, color='red', linewidth=0.5, linestyle='--', label='Strike Level (K)')
    plt.title(f"{position.capitalize()} {option_type.capitalize()} Digital Option Payoff")
    plt.xlabel("Sous-jacent (S)")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid()

    plt.show()

    return digit_price

def plot_barrier_option(K, B, option_type, direction, barrier_type, position):
    S_range = np.linspace(0.5 * B, 1.5 * B, 200)
    payoffs =[]

    for S_t in S_range:
        if barrier_type == "ki":
            if direction == "up" and S_t >= B:
                if option_type == "call":
                    payoff = np.maximum(S_t - K, 0)
                else:
                    payoff = np.maximum(K - S_t, 0)
            elif direction == "down" and S_t <= B:
                if option_type == "call":
                    payoff = np.maximum(S_t - K, 0)
                else:
                    payoff = np.maximum(K - S_t, 0) 
            else:
                payoff = 0

        else:
            if direction == "up" and S_t >= B:
                payoff = 0
            elif direction == "down" and S_t <= B:
                payoff = 0
            else :
                if option_type == "call":
                    payoff = np.maximum(S_t - K, 0)
                else:
                    payoff = np.maximum(K - S_t, 0)

        if position == "sell":
            payoff = - payoff

        payoffs.append(payoff)

    plt.figure()
    plt.plot(S_range, payoffs, label='Barrier Option Payoff', color='blue')
    plt.axvline(x=B, color='black', linewidth=0.5, linestyle='--', label='Barrier Level (B)')
    plt.axvline(x=K, color='red', linewidth=0.5, linestyle='--', label='Strike Level (K)')
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.title(f"Option Barrier Payoff ({option_type.capitalize()} {direction.capitalize()} {barrier_type.capitalize()})")
    plt.xlabel("Underlying (S)")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid()

    plt.show()

def plot_spread(S0, K1, K2, T, r, q, spread_type, position):
    S_range = np.linspace(0.5 * S0, 1.5 * S0, 250)

    payoffs = []
    
    for S_i in S_range :
        if spread_type == "call spread":
            intrinsic_value = np.maximum(S_i - K1, 0) - np.maximum(S_i - K2, 0)
        elif spread_type == "put spread":
            intrinsic_value = np.maximum(K2 - S_i, 0) - np.maximum(K1 - S_i, 0)
        else:    
            raise ValueError("Invalid option type. Must be 'call spread' or 'put spread'.")
               
        if position == "buy": 
            payoff_value = intrinsic_value 
        elif position == "sell":
            payoff_value = -intrinsic_value 
        else :
            raise ValueError("Invalid option type. Must be 'buy' or 'sell'.")
            
        payoffs.append(payoff_value)

    plt.figure()
    plt.plot(S_range, payoffs, label=f"{spread_type.capitalize()} Payoff", color="green")
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.axvline(S0, color='blue', linestyle='--')
    plt.title(f"{spread_type.capitalize()} Payoff")
    plt.xlabel("Sous-jacent (S)")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_strangle(position, S0, K1, K2, T, r, q, name):
    S_min = min(K1, K2) * 0.5
    S_max = max(K1, K2) * 1.5
    S_range = np.linspace(S_min, S_max,200)

    payoff = []
    for S_i in S_range :    
        if position == "buy":
            payoff.append(np.maximum(K1 - S_i, 0) + np.maximum(S_i - K2, 0))
        elif position == "sell":
            payoff.append(-(np.maximum(K1 - S_i, 0) + np.maximum(S_i - K2, 0)))
        else :
            raise ValueError("Invalid option type.")

    plt.figure()
    plt.plot(S_range, payoff, label=f"{name.capitalize()} Payoff", color="black")   
    plt.axvline(S0, color="blue", linestyle="--", label="Spot Price")
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.title(f"{name.capitalize()} Payoff")
    plt.xlim(0.5 * S0, 1.5 * S0)
    plt.xlabel("Sous-jacent (S)")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_butterfly(position, strategy_type, S0, K1, K2, K3, price_butterfly):
    S_min = min(K1, K3) * 0.5
    S_max = max(K1, K3) * 1.5
    S_range = np.linspace(S_min, S_max,300)

    payoff =[]
    for S_i in S_range:
        if strategy_type == "call butterfly":
            payoff_value = (np.maximum(S_i - K1, 0) - 2 * np.maximum(S_i - K2, 0) + np.maximum(S_i - K3, 0))
        elif strategy_type == "put butterfly":
            payoff_value = (np.maximum(K1 - S_i, 0) - 2 * np.maximum(K2 - S_i, 0) + np.maximum(K3 - S_i, 0))
        else : 
            print("Invalid choice. Please enter strategy_type : call butterfly or put butterfly.")

        if position == "buy":
            payoff_value = + payoff_value - price_butterfly
        else :
            payoff_value = - payoff_value + price_butterfly    

        payoff.append(payoff_value)

    plt.figure()
    plt.plot(S_range, payoff, label=f"{strategy_type.capitalize()} Payoff", color="black")   
    plt.axvline(S0, color="blue", linestyle="--", label="Spot Price")
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"{strategy_type.capitalize()} Payoff")
    plt.xlim(0.5 * S0, 1.5 * S0)
    plt.xlabel("Sous-jacent (S)")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_dynamic_greeks(S_range, greeks_dynamic):
    """Affichage des grecs dynamiques organisés en groupes sur des graphiques distincts."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Tracé de Delta (axe de gauche)
    ax1.plot(S_range, greeks_dynamic["Delta"], label="Delta", color='blue')
    ax1.set_xlabel("Sous-jacent (S)")
    ax1.set_ylabel("Delta", color='blue')
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid()
    # Création d'un second axe y pour Gamma
    ax2 = ax1.twinx()
    ax2.plot(S_range, greeks_dynamic["Gamma"], label="Gamma", color='purple')
    ax2.set_ylabel("Gamma", color='purple')
    ax2.tick_params(axis='y', labelcolor="purple")
    # Ajout du titre
    plt.title("Delta & Gamma avec Axes Différents")
    # Ajustement de la mise en page
    fig.tight_layout()
    
    # Graphique 2 : Vega seul
    plt.figure(figsize=(10, 6))
    plt.plot(S_range, greeks_dynamic["Vega"], label="Vega", color="orange")
    plt.title("Vega")
    plt.xlabel("Sous-jacent (S)")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    # Graphique 3 : Theta seul
    plt.figure(figsize=(10, 6))
    plt.plot(S_range, greeks_dynamic["Theta"], label="Theta", color="red")
    plt.title("Theta")
    plt.xlabel("Sous-jacent (S)")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Graphique 4 : Rho seul
    plt.figure(figsize=(10, 6))
    plt.plot(S_range, greeks_dynamic["Rho"], label="Rho", color="green")
    plt.title("Rho")
    plt.xlabel("Sous-jacent (S)")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    plt.show()

def plot_dynamic_greeks_digit(S_range, greeks_dynamic_digit):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(S_range, greeks_dynamic_digit["Delta"], label="Delta", color='blue')
    ax1.set_xlabel("Sous-jacent (S)")
    ax1.set_ylabel("Delta", color='blue')
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.plot(S_range, greeks_dynamic_digit["Gamma"], label="Gamma", color='purple')
    ax2.set_ylabel("Gamma", color='purple')
    ax2.tick_params(axis='y', labelcolor="purple")
    plt.title("Delta & Gamma")
    fig.tight_layout()
    
    # Graphique 2 : Vega 
    plt.figure(figsize=(10, 6))
    plt.plot(S_range, greeks_dynamic_digit["Vega"], label="Vega", color="orange")
    plt.title("Vega")
    plt.xlabel("Sous-jacent (S)")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    # Graphique 3 : Theta 
    plt.figure(figsize=(10, 6))
    plt.plot(S_range, greeks_dynamic_digit["Theta"], label="Theta", color="red")
    plt.title("Theta")
    plt.xlabel("Sous-jacent (S)")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Graphique 4 : Rho 
    plt.figure(figsize=(10, 6))
    plt.plot(S_range, greeks_dynamic_digit["Rho"], label="Rho", color="green")
    plt.title("Rho")
    plt.xlabel("Sous-jacent (S)")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    plt.show()


if __name__ == "__main__":
    menu = MenuHandler()
    menu.menu_principal()
