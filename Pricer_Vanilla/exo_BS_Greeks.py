import pandas as pd 
import numpy as np
from scipy.stats import norm

S = 100
K = 100
T = 1 
r = 0.05
#d = 0.02
sigma = 0.2

def d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r - (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def calculates(S, K, T, r, sigma, optiontype = "call"):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    if optiontype == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif optiontype == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def greeks_calc(S, K, T, r, sigma, optiontype = "call"):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = -(S * sigma * norm.pdf(d1)) / (S * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.pdf(d2) 
    if optiontype == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif optiontype == "put":
       rho = - K * T * np.exp(-r * T) * norm.cdf(-d2)
    return delta, gamma, vega, theta, rho

price = calculates(S, K, T, r, sigma, optiontype = "call")
greeks = greeks_calc(S, K, T, r, sigma, optiontype = "call")

optiontype = input("call or put? ").strip().lower()

def results(optiontype, price, greeks):
    print(f"{optiontype.capitalize()} price : ${price:.2f} ")
    print (f"greeks for {optiontype} option : ")
    print(f"delta : {greeks[0]:.4f}")
    print(f"gamma : {greeks[1]:.4f}")
    print(f"vega : {greeks[2]:.4f}")
    print(f"theta : {greeks[3]:.4f}")
    print(f"rho : {greeks[4]:.4f}")

price = calculates(S, K, T, r, sigma, optiontype = "call")
greeks = greeks_calc(S, K, T, r, sigma, optiontype = "call")

results = results(optiontype, price, greeks)

