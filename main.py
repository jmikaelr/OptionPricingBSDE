from bsde_models import *
import numpy as np
from scipy.stats import norm
import argparse
import cProfile
import pstats
import io


def main():
    parser = argparse.ArgumentParser(description="Option Pricing using BSDE") 
    
    parser.add_argument("--S", type=float, default=100.0, help="Stock price")
    parser.add_argument("--K", type=float, default=95.0, help="Strike price")
    parser.add_argument("--K2", type=float, default=105, help="Second strike price")
    parser.add_argument("--r", type=float, default=0.01, help="Rate")
    parser.add_argument("--R", type=float, default=0.01, help="Second rate")
    parser.add_argument("--dims", type=int, default=1, help="Number of risky assets (stocks), default is one.")
    parser.add_argument("--mu", type=float, default=0.01, help="Drift term on stock")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility (sigma)")
    parser.add_argument("--corr", type=float, default=0.5, help="Correlation (rho)")
    parser.add_argument("--T", type=float, default=0.25, help="Time to expiration (in years)")
    parser.add_argument("--N", type=int, default=20, help="Number of time steps")
    parser.add_argument("--M", type=int, default=32768, help="Number of Monte Carlo simulations")
    parser.add_argument("--L", type=float, default=0.025, help="confidence alpha")
    parser.add_argument("--delta", type=float, default=1, help="Length of hybercubes")
    parser.add_argument("--k", type=int, default=0, help="Polynomial degree")
    parser.add_argument("--H", type=float, default=60, help="The domain range which the hybercubes cover")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples of solved BSDEs prices")
    parser.add_argument("--opt_payoff", type=str, choices=['call', 'put'], default='call', help="Option payoff (either 'call' or 'put')")
    parser.add_argument("--opt_style", type=str, choices=['european', 'american', 'europeanspread', 'americanspread'], default='european', help="Option style")
    parser.add_argument("--nofig", action='store_true', help="Do not show plot. It plots by default.") 

    parser.add_argument("--plot_type", type=str, choices = ['N', 'M', 'deltas', 'samples'], default = None, help="What kind of plot, default is no plot.")
    parser.add_argument("--plot_values", type=str, help="Comma-separated list of values for the selected plot type (e.g., 10,20,30).")
    args = parser.parse_args()
    if args.plot_type == 'deltas':
        plot_values = [float(x) for x in args.plot_values.split(',')] if args.plot_values else None
    else:
        plot_values = [int(x) for x in args.plot_values.split(',')] if args.plot_values else None
    price = None
    if args.dims != 1:
        args.S = np.full(args.dims, args.S)
        args.mu = np.full(args.dims, args.mu)
        args.sigma = np.full(args.dims, args.sigma)
    if args.opt_style in ['european', 'americanspread', 'europeanspread', 'american']:
        if args.opt_style == 'european':
            option_pricing_obj = BSDEOptionPricingEuropean(args.S, args.mu, args.sigma, 
                                                           args.corr, args.K,
                                                           args.r, args.T, args.N, args.M, 
                                                           args.L, args.samples, args.dims, 
                                                           args.opt_payoff, args.H, 
                                                           args.delta, args.k)
            if args.dims != 1:
                pass
                #price = multi_dimensional_black_scholes(args.S, args.K, args.T, args.r, args.sigma, args.opt_payoff)
            else:
                price = black_scholes(args.S, args.K, args.T, args.r, args.sigma, args.opt_payoff)
        elif args.opt_style == 'american':
            option_pricing_obj = BSDEOptionPricingAmerican(args.S, args.mu, args.sigma, 
                                                           args.corr, args.K,
                                                           args.r, args.T, args.N, args.M, 
                                                           args.L, args.samples, args.dims, 
                                                           args.opt_payoff, args.H, 
                                                           args.delta, args.k)
        elif args.opt_style == 'europeanspread':
            option_pricing_obj = BSDEOptionPricingEuropeanSpread(args.S, args.mu, args.sigma, 
                                                                 args.corr, args.K, args.r, 
                                                                 args.T, args.N, 
                                                                 args.M, args.L, args.samples, 
                                                                 args.dims, args.opt_payoff, 
                                                                 args.H, args.delta, args.k,
                                                                 args.K2, args.R)
        elif args.opt_style == 'americanspread':
            option_pricing_obj = BSDEOptionPricingAmericanSpread(args.S, args.mu, args.sigma, 
                                                                 args.corr, args.K, args.r, 
                                                                 args.T, args.N, 
                                                                 args.M, args.L, args.samples, 
                                                                 args.dims, args.opt_payoff, 
                                                                 args.H, args.delta, args.k,
                                                                 args.K2, args.R)
        
        if args.plot_type in ["N","M","deltas","samples"]:
            if not plot_values:
                raise ValueError('No plot values to plot.')
            elif args.plot_type == "N": 
                print(f"Plotting by varying N with values: {args.plot_values}")
                option_pricing_obj.plot_and_show_table_by_N(plot_values, args.nofig, price)
            elif args.plot_type == "M":
                print(f"Plotting by varying M with values: {args.plot_values}")
                option_pricing_obj.plot_and_show_table_by_M(plot_values, args.nofig, price)
            elif args.plot_type == "deltas":
                print(f"Plotting by varying degrees with values: {args.plot_values}")
                option_pricing_obj.plot_and_show_table_by_deltas(plot_values, args.nofig, price)
            elif args.plot_type == "samples":
                print(f"Plotting by varying samples size with values: {args.plot_values}")
                option_pricing_obj.plot_and_show_table_by_samples(plot_values, args.nofig, price)
        else:  
            option_pricing_obj.solve()  
            print(price)
            if price:
                print(f"{args.opt_style.capitalize()} {args.opt_payoff} option price: {price}")
    else:
        raise ValueError('Invalid option: {args.opt_style}')

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


#def multi_dimensional_black_scholes(S, K, T, r, sigma, opt_payoff):
#    d = len(S)
#    ck = np.full(d, 1/d)
#    
#    b_tilde_squared = np.sum((sigma * ck)**2)
#    delta_tilde = np.sum(ck * ((b_tilde_squared / 2))) - (b_tilde_squared / 2)
#    
#    d1_tilde = (np.log(np.prod((S / K)**ck)) + (r - delta_tilde + (b_tilde_squared / 2)) * (T)) / (np.sqrt(b_tilde_squared) * np.sqrt(T))
#    d2_tilde = d1_tilde - np.sqrt(b_tilde_squared) * np.sqrt(T)
#    
#    if opt_payoff == 'call':
#        Y_t = np.exp(-delta_tilde * (T)) * np.prod(S**ck) * norm.cdf(d1_tilde) - np.exp(-r * (T)) * K * norm.cdf(d2_tilde)
#    elif opt_payoff == 'put':
#        Y_t = np.exp(-r * (T)) * K * norm.cdf(-d2_tilde) - np.exp(-delta_tilde * (T)) * np.prod(S**ck) * norm.cdf(-d1_tilde)
#    else:
#        raise ValueError(f'Invalid payoff {opt_payoff}, it must be either "call" or "put"')
#    
#    Zk_t = np.array([ck_i * np.exp(-delta_tilde * (T)) * np.prod(S**ck) * norm.cdf(d1_tilde) * sigma_i for ck_i, sigma_i in zip(ck, sigma)])
#    
#    return Y_t, Zk_t

def profile_solve():
    pr = cProfile.Profile()
    pr.enable()
    option_pricing_obj.solve()
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


if __name__ == '__main__':
#    option_pricing_obj = BSDEOptionPricingEuropean(
#        S0=100, mu=0.05, sigma=0.2, correlation=0.5, K=100, r=0.05, T=1,
#        N=5, M=10000, dims=1, option_payoff="call", samples = 1
#    )
#    profile_solve()
    main()


