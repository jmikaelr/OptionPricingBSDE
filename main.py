from mclsq import *
import numpy as np
from scipy.stats import norm
import numpy.polynomial.polynomial as poly
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Option Pricing using BSDE and Longstaff Schwartz")
    
    parser.add_argument("--S", type=float, default=100.0, help="Stock price")
    parser.add_argument("--K", type=float, default=95.0, help="Strike price")
    parser.add_argument("--K2", type=float, default=105, help="Second strike price")
    parser.add_argument("--r", type=float, default=0.01, help="Rate")
    parser.add_argument("--R", type=float, default=0.01, help="Second rate")
    parser.add_argument("--mu", type=float, default=0.01, help="Drift term on stock")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility (sigma)")
    parser.add_argument("--T", type=float, default=0.25, help="Time to expiration (in years)")
    parser.add_argument("--N", type=int, default=52, help="Number of time steps")
    parser.add_argument("--M", type=int, default=10000, help="Number of Monte Carlo simulations")
    parser.add_argument("--L", type=float, default=0.025, help="Lower confidence alpha")
    parser.add_argument("--degree", type=int, default=3, help="Degree for regression")
    parser.add_argument("--samples", type=int, default=100, help="Number of sampels of solved BSDEs prices")
    parser.add_argument("--opt_payoff", type=str, choices=['call', 'put'],
                        default='call', help="Option payoff (either 'call' or 'put')")
    parser.add_argument("--opt_style", type=str, choices=['european',
                                                          'american',
                                                          'europeanspread',
                                                          'americanspread'],
                                                            
                        default='european', help="Option style (either 'european','american', 'europeanspread' or 'americanspread')")

    args = parser.parse_args()
    degrees = list(range(3, 20))
    if args.opt_style == 'european':
        euro_opt = BSDEOptionPricingEuropean(args.S, args.K, args.r, args.sigma,
                                            args.T, args.N, args.M, args.L,
                                             args.samples, args.opt_payoff,
                                             args.degree, args.mu)
        euro_opt.run()
        euro_opt.plot_and_show_table_by_degree(degrees)
        price = black_scholes(args.S, args.K, args.T, args.mu, args.sigma,
                                args.opt_payoff)
        print(f"European {args.opt_payoff} option price: {price:.4f}")
    elif args.opt_style == 'american':
        american_opt = BSDEOptionPricingAmerican(args.S, args.K, args.r,
                                                args.sigma, args.T, args.N,
                                                 args.M, args.L, args.samples,
                                                 args.opt_payoff, args.degree)
        american_opt.run()
        american_opt.plot_and_show_table_by_degree(degrees)
    elif args.opt_style == 'europeanspread':
        euro_opt_spread = BSDEOptionPricingEuropeanSpread(args.S, args.K, args.r, args.sigma,
                                            args.T, args.N, args.M, args.L,
                                             args.samples, args.opt_payoff,
                                             args.degree, args.mu, args.K2, args.R)
        euro_opt_spread.run()
        price = black_scholes_call_spread(args.S, args.K, args.K2, args.T, args.r, args.R, args.sigma)
                                
        print(f"European {args.opt_payoff} option spread price: {price:.4f}")
    elif args.opt_style == 'americanspread':
        american_opt_spread = BSDEOptionPricingAmericanSpread(args.S, args.K, args.r, args.sigma,
                                            args.T, args.N, args.M, args.L,
                                             args.samples, args.opt_payoff,
                                             args.degree, args.mu, args.K2, args.R)
        american_opt_spread.run()
        american_opt_spread.plot_and_show_table_by_degree(degrees)
    else:
        raise ValueError(f"Option style should be either european or american, not {args.opt_style}!")
    
def black_scholes(S, K, T, mu, sigma, opt_payoff):
    """ Calculates an european option using the analytical blach-scholes
    formula """
    N = norm.cdf
    d1 = (np.log(S/K) + (mu + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_payoff == 'call':
        return S * N(d1) - K * np.exp(-mu*T)* N(d2)
    else:
        return K*np.exp(-mu*T)*N(-d2) - S * N(-d1)

def black_scholes_call_spread(S, K1, K2, T, r, R, sigma):
    """Calculates a European call spread option price using the analytical Black-Scholes formula."""
    N = norm.cdf  

    d1_K1 = (np.log(S / K1) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2_K1 = d1_K1 - sigma * np.sqrt(T)
    
    d1_K2 = (np.log(S / K2) + (R + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2_K2 = d1_K2 - sigma * np.sqrt(T)

    call_price_K1 = S * N(d1_K1) - K1 * np.exp(-r * T) * N(d2_K1)
    call_price_K2 = S * N(d1_K2) - K2 * np.exp(-R * T) * N(d2_K2)
    
    call_spread_price = call_price_K1 - 2*call_price_K2

    return call_spread_price



if __name__ == '__main__':
    main()

