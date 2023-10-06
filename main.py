from mclsq import *
import numpy as np
from scipy.stats import norm
import numpy.polynomial.polynomial as poly

def main():
    S = 100.0
    K = 95.0
    r = 0.01
    sigma = 0.2
    T = 0.25
    N = 25
    M = 8920
    opt_type = 'call'
    euro_opt = BSDEOptionPricingEuropean(S, K, r, sigma, T, N, M, 100, opt_type)
    #american_opt = BSDEOptionPricingAmerican(S, K, r, sigma, T, N, M, opt_type)
    euro_opt.run()
    print('European Option (Black Scholes): ' + str(BS_CALL(S,K,T,r,sigma)))
    #american_opt.run()
    #price = longstaff_schwartz(S, K, r, sigma, T, N, M, opt_type)
    #print(f"American {opt_type} option price: {price:.2f}")
    


def BS_CALL(S, K, T, r, sigma):
    N = norm.cdf
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

def longstaff_schwartz(S, K, r, sigma, T, N, M, option_type='put'):
    np.random.seed(0)
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
        
        for local_idx, fullpath_idx in enumerate(itm_indices[0]):
            exercise_value = (stock_paths[fullpath_idx, t] - K) if option_type == 'call' else (K - stock_paths[fullpath_idx, t])
            if exercise_value > continuation_values[local_idx]:
                option_values[fullpath_idx] = exercise_value

    option_price = np.mean(option_values * np.exp(-r * dt * (num_timesteps - 1)))

    return option_price


if __name__ == '__main__':
    main()
