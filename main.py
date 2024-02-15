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
    parser.add_argument("--r", type=float, default=0.01, help="Rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility (sigma)")
    parser.add_argument("--T", type=float, default=1, help="Time to expiration (in years)")
    parser.add_argument("--N", type=int, default=252, help="Number of time steps")
    parser.add_argument("--M", type=int, default=100000, help="Number of Monte Carlo simulations")
    parser.add_argument("--L", type=float, default=0.025, help="Lower confidence alpha")
    parser.add_argument("--degree", type=int, default=3, help="Degree for regression")
    parser.add_argument("--samples", type=int, default=100, help="Number of sampels of solved BSDEs prices")
    parser.add_argument("--opt_payoff", type=str, choices=['call', 'put'],
                        default='call', help="Option payoff (either 'call' or 'put')")
    parser.add_argument("--opt_style", type=str, choices=['european',
                                                          'american'],
                        default='european', help="Option style (either 'european' or 'american')")
    args = parser.parse_args()

    if args.opt_style == 'european':
        euro_opt = BSDEOptionPricingEuropean(args.S, args.K, args.r, args.sigma,
                                            args.T, args.N, args.M, args.L,
                                             args.samples, args.opt_payoff,
                                             args.degree)
        euro_opt.run()
        price = black_scholes(args.S, args.K, args.T, args.r, args.sigma,
                              args.opt_payoff)
        print(f"European {args.opt_payoff} option price: {price:.2f}")
    elif args.opt_style == 'american':
        american_opt = BSDEOptionPricingAmerican(args.S, args.K, args.r,
                                                args.sigma, args.T, args.N,
                                                 args.M, args.L, args.samples,
                                                 args.opt_payoff, args.degree)
        american_opt.run()
        price = longstaff_schwartz(args.S, args.K, args.r, args.sigma, args.T,
                                   args.N, args.M, args.opt_payoff)
        print(f"American {args.opt_payoff} option price: {price:.2f}")
    else:
        raise ValueError(f"Option style should be either european or american, not {args.opt_style}!")
    


def black_scholes(S, K, T, r, sigma, opt_payoff):
    """ Calculates an european option using the analytical blach-scholes
    formula """
    N = norm.cdf
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_payoff == 'call':
        return S * N(d1) - K * np.exp(-r*T)* N(d2)
    else:
        return K*np.exp(-r*T)*N(-d2) - S * N(-d1)

def longstaff_schwartz(S, K, r, sigma, T, N, M, option_type='put', seed=None):
    if M < 1e4:
        M = int(1e5) 
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    discount_factor = np.exp(-r * dt)
    
    stock_paths = np.zeros((M, N+1))
    stock_paths[:, 0] = S
    for t in range(1, N+1):
        Z = np.random.normal(size=M)
        stock_paths[:, t] = stock_paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    if option_type == 'put':
        option_values = np.maximum(K - stock_paths, 0)
    elif option_type == 'call':
        option_values = np.maximum(stock_paths - K, 0)
    else:
        raise ValueError("Invalid option_type: choose either 'put' or 'call'")
    
    for t in range(N-1, 0, -1):
        itm_indices = np.where(option_values[:, t] > 0)[0]  # Indices where the option i
        stock_prices_itm = stock_paths[itm_indices, t]
        option_values_itm = option_values[itm_indices, t+1] * discount_factor  # Discounted one step forward
        
        # Step 4: Regression to estimate continuation value
        X = poly.polyvander(stock_prices_itm, 2)  # 2nd degree polynomial for basis functions
        regression_coeffs = np.linalg.lstsq(X, option_values_itm, rcond=None)[0]
        continuation_values = X @ regression_coeffs
        
        # Step 5: Decide whether to exercise early
        intrinsic_values = np.maximum(K - stock_prices_itm if option_type == 'put' else stock_prices_itm - K, 0)
        exercise = intrinsic_values > continuation_values
        
        # Update option values where early exercise is optimal
        option_values[itm_indices[exercise], t] = intrinsic_values[exercise]
        
        # Discount the option values from time t back to t-1
        option_values[:, t-1] = np.where(option_values[:, t] > 0, option_values[:, t], option_values[:, t-1])
        option_values[:, t-1] *= discount_factor  # Apply discounting
    
    # Step 6: Estimate the option price as the discounted mean of the option values at the first time step
    option_price = np.mean(option_values[:, 1]) * discount_factor  # Only discount once from t=1 to t=0
    return option_price


if __name__ == '__main__':
    main()

