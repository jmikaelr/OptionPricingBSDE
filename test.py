import numpy as np
from scipy.stats import norm

def black_scholes_multi(S, K, T, r, sigma, alpha, q, opt_payoff):
    """
    Calculates a multi-dimensional European option using the Black-Scholes formula.
    
    Parameters:
    S (array-like): Initial stock prices
    K (float): Strike price
    T (float): Time to maturity
    r (float): Risk-free rate
    sigma (array-like): Volatilities of the assets
    alpha (array-like): Weights of the assets
    q (array-like): Dividend yields of the assets
    opt_payoff (str): 'call' or 'put'
    
    Returns:
    float: Option price
    """
    S = np.asarray(S)
    sigma = np.asarray(sigma)
    alpha = np.asarray(alpha)
    q = np.asarray(q)
    
    # Calculate adjusted q_hat and sigma_hat
    sigma_hat_squared = np.sum((sigma * alpha) ** 2)
    sigma_hat = np.sqrt(sigma_hat_squared)
    q_hat = np.sum(alpha * (q + 0.5 * sigma ** 2)) - 0.5 * sigma_hat_squared
    
    # Calculate d_hat1 and d_hat2
    St_prod_alpha = np.prod(S ** alpha)
    d_hat1 = (np.log(St_prod_alpha / K) + (r - q_hat + 0.5 * sigma_hat_squared) * T) / (sigma_hat * np.sqrt(T))
    d_hat2 = d_hat1 - sigma_hat * np.sqrt(T)
    
    # Option price calculation
    N_d_hat1 = norm.cdf(d_hat1)
    N_d_hat2 = norm.cdf(d_hat2)
    exp_neg_q_hat_T = np.exp(-q_hat * T)
    exp_neg_r_T = np.exp(-r * T)
    
    if opt_payoff == 'call':
        price = exp_neg_q_hat_T * St_prod_alpha * N_d_hat1 - exp_neg_r_T * K * N_d_hat2
    elif opt_payoff == 'put':
        price = exp_neg_r_T * K * norm.cdf(-d_hat2) - exp_neg_q_hat_T * St_prod_alpha * norm.cdf(-d_hat1)
    else:
        raise ValueError("Invalid payoff type. Must be 'call' or 'put'.")
    
    return price

# Example usage
S = [100]
K = 95
T = 1
r = 0.01
sigma = [0.2]
alpha = [1]
q = [0.00, 0.00]
opt_payoff = 'call'

price = black_scholes_multi(S, K, T, r, sigma, alpha, q, opt_payoff)
print(f"Multi-dimensional European Option Price: {price:.4f}")

