from mclsq import *
import numpy as np
from scipy.stats import norm

def main():
    S = 100.0
    K = 95.0
    r = 0.06
    sigma = 0.2
    T = 1.0
    N = 252
    M = 1000
    opt_type = 'call'
    euro_opt = BSDEOptionPricingEuropean(S, K, r, sigma, T, N, M, opt_type)
    #american_opt = BSDEOptionPricingAmerican(S, K, r, sigma, T, N, M, opt_type)
    euro_opt.run()
    print('European Option (Black Scholes): ' + str(BS_CALL(S,K,T,r,sigma)))
    #american_opt.run()
    


def BS_CALL(S, K, T, r, sigma):
    N = norm.cdf
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

if __name__ == '__main__':
    main()
