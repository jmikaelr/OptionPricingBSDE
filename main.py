from mclsq import *
import numpy as np
from scipy.stats import norm
import numpy.polynomial.polynomial as poly
import argparse

def main():
    parser = argparse.ArgumentParser(description="Option Pricing using BSDE and Longstaff Schwartz")
    
    parser.add_argument("--S", type=float, default=100.0, help="Stock price")
    parser.add_argument("--K", type=float, default=95.0, help="Strike price")
    parser.add_argument("--r", type=float, default=0.01, help="Rate")
    parser.add_argument("--mu", type=float, default=0.01, help="Mean return")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility (sigma)")
    parser.add_argument("--T", type=float, default=1, help="Time to expiration (in years)")
    parser.add_argument("--N", type=int, default=52, help="Number of time steps")
    parser.add_argument("--M", type=int, default=1000, help="Number of Monte Carlo simulations")
    parser.add_argument("--L", type=float, default=0.025, help="Lower confidence alpha")
    parser.add_argument("--degree", type=int, default=3, help="Degree for regression")
    parser.add_argument("--samples", type=int, default=100, help="Number of sampels of solved BSDEs prices")
    parser.add_argument("--opt_type", type=str, choices=['call', 'put'], default='call', help="Option type (either 'call' or 'put')")
    parser.add_argument("--opt_style", type=str, choices=['european',
                                                          'american'],
                        default='european', help="Option style (either 'european' or 'american')")
    args = parser.parse_args()
    
    if args.opt_style == 'european':
        euro_opt = BSDEOptionPricingEuropean(args.S, args.K, args.r, args.mu, args.sigma,
                                            args.T, args.N, args.M, args.L,
                                             args.samples, args.opt_type,
                                             args.degree)
        euro_opt.run()
        price = BS_CALL(args.S, args.K, args.T, args.r, args.sigma)
        print(f"European {args.opt_type} option price: {price:.2f}")
    elif args.opt_style == 'american':
        american_opt = BSDEOptionPricingAmerican(args.S, args.K, args.r,
                                                 args.mu,
                                                 args.sigma, args.T, args.N,
                                                 args.M, args.L, args.samples,
                                                 args.opt_type, args.degree)
        american_opt.run()
        price = longstaff_schwartz(args.S, args.K, args.r, args.sigma, args.T,
                                   args.N, args.M, args.opt_type)
        print(f"American {args.opt_type} option price: {price:.2f}")
    else:
        raise ValueError(f"Option style should be either european or american, not {args.opt_style}!")
    


def BS_CALL(S, K, T, r, sigma):
    N = norm.cdf
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

def longstaff_schwartz(S, K, r, sigma, T, N, M, option_type='put', seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    rand_values = np.random.randn(M, N)
    stock_paths = np.zeros_like(rand_values)
    stock_paths[:, 0] = S
    dt = T/N

    for t in range(1, N):
        stock_paths[:, t] = stock_paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand_values[:, t])
    
    num_paths, num_timesteps = stock_paths.shape

    if option_type == 'put':
        intrinsic_values = np.maximum(K - stock_paths, 0)
    elif option_type == 'call':
        intrinsic_values = np.maximum(stock_paths - K, 0)
    else:
        raise ValueError("Invalid option_type: choose either 'put' or 'call'")
        
    option_values = intrinsic_values[:, -1]

    for t in range(num_timesteps - 2, -1, -1):
        itm_indices = np.where((K - stock_paths[:, t]) * (-1 if option_type == 'call' else 1) > 0)
        stock_prices_itm = stock_paths[itm_indices, t]
        option_values_itm = option_values[itm_indices] * np.exp(-r * dt)
        
        X = poly.polyvander(stock_prices_itm.flatten(), 2)
        regression_coeffs = np.linalg.lstsq(X, option_values_itm, rcond=None)[0]
        continuation_values = np.dot(X, regression_coeffs)

        exercise_values = (stock_prices_itm - K) if option_type == 'call' else (K - stock_prices_itm)
        option_values[itm_indices] = np.where(exercise_values > continuation_values, exercise_values, option_values[itm_indices])

    option_price = np.mean(option_values)
    return option_price

if __name__ == '__main__':
    main()

