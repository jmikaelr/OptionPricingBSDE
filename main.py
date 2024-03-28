from mclsq import *
import numpy as np
from scipy.stats import norm
import argparse

def main():
    parser = argparse.ArgumentParser(description="Option Pricing using BSDE and Longstaff Schwartz")
    
    parser.add_argument("--S", type=float, default=100.0, help="Stock price")
    parser.add_argument("--K", type=float, default=95.0, help="Strike price")
    parser.add_argument("--K2", type=float, default=None, help="Second strike price")
    parser.add_argument("--r", type=float, default=0.01, help="Rate")
    parser.add_argument("--R", type=float, default=None, help="Second rate")
    parser.add_argument("--mu", type=float, default=None, help="Drift term on stock")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility (sigma)")
    parser.add_argument("--T", type=float, default=0.25, help="Time to expiration (in years)")
    parser.add_argument("--N", type=int, default=252, help="Number of time steps")
    parser.add_argument("--M", type=int, default=100000, help="Number of Monte Carlo simulations")
    parser.add_argument("--L", type=float, default=0.025, help="confidence alpha")
    parser.add_argument("--degree", type=int, default=3, help="Degree for regression")
    parser.add_argument("--samples", type=int, default=10, help="Number of sampels of solved BSDEs prices")
    parser.add_argument("--opt_payoff", type=str, choices=['call', 'put'],
                        default='call', help="Option payoff (either 'call' or 'put')")
    parser.add_argument("--opt_style", type=str, choices=['european',
                                                          'american',
                                                          'europeanspread',
                                                          'americanspread'],
                                                            default='european', help="Option style (either 'european','american', 'europeanspread' or 'americanspread')")

    parser.add_argument("--plot", default=False, action='store_true', help="Optional to do figures and tables and save them") 
    parser.add_argument("--degrees", type=int, default=3, help="Up to what degree to plot and generate table to")
    parser.add_argument("--nofig", default=False, action='store_true', help="Optional to show plot or not. It plots by default.")
     
    args = parser.parse_args()
    if args.plot and args.degrees > args.degree and args.degrees < 30:
        degrees = list(range(2,args.degrees + 1))
    elif args.plot:
        degrees  = list(range(2, 5))
    if args.opt_style == 'european':
        euro_opt = BSDEOptionPricingEuropean(args.S, args.K, args.r, args.sigma,
                                            args.T, args.N, args.M, args.L,
                                             args.samples, args.opt_payoff,
                                             args.degree, args.mu)
        price = black_scholes(args.S, args.K, args.T, args.r, args.sigma, args.opt_payoff)
        if args.plot:
            euro_opt.plot_and_show_table_by_degree(degrees, args.opt_style, args.nofig, price)
        else:
            euro_opt.run()
            print(f"European {args.opt_payoff} option price: {price:.4f}")
    elif args.opt_style == 'american':
        american_opt = BSDEOptionPricingAmerican(args.S, args.K, args.r,
                                                args.sigma, args.T, args.N,
                                                 args.M, args.L, args.samples,
                                                 args.opt_payoff, args.degree)
        if args.plot:
            american_opt.plot_and_show_table_by_degree(degrees, args.opt_style, args.nofig)
        else:
            american_opt.run()

    elif args.opt_style == 'europeanspread':
        if args.K2 == None or args.R == None:
            raise ValueError('K2 and R must be set to a value!')
        euro_opt_spread = BSDEOptionPricingEuropeanSpread(args.S, args.K, args.r, args.sigma,
                                            args.T, args.N, args.M, args.L,
                                             args.samples, args.opt_payoff,
                                             args.degree, args.mu, args.K2, args.R)
        if args.plot:
            euro_opt_spread.plot_and_show_table_by_degree(degrees, args.opt_style, args.nofig)
        else:
            euro_opt_spread.run()
            price = black_scholes_call_spread(args.S, args.K, args.K2, args.T, args.r, args.R, args.sigma)
            print(f"European {args.opt_payoff} option spread price: {price:.4f}")
    elif args.opt_style == 'americanspread':
        if args.K2 == None or args.R == None:
            raise ValueError('K2 and R must be set to a value!')
        american_opt_spread = BSDEOptionPricingAmericanSpread(args.S, args.K, args.r, args.sigma,
                                            args.T, args.N, args.M, args.L,
                                             args.samples, args.opt_payoff,
                                             args.degree, args.mu, args.K2, args.R)
        if args.plot:
            american_opt_spread.plot_and_show_table_by_degree(degrees, args.opt_style, args.nofig)
        else:
            american_opt_spread.run()
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
    elif opt_payoff == 'put':
        return K*np.exp(-r*T)*N(-d2) - S * N(-d1)
    else:
        raise ValueError(f'Invalid payoff {opt_payoff}, it must be either "call" or "put"')


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

