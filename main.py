from mclsq import *

def main():
    S = 100.0
    K = 95.0
    r = 0.06
    sigma = 0.2
    T = 1.0
    N = 252
    M = 100000
    opt_type = 'call'
    obj1 = BSDEOptionPricingEuropean(S, K, r, sigma, T, N, M, opt_type)
    obj2 = BSDEOptionPricingAmerican(S, K, r, sigma, T, N, M, opt_type, 100)
    obj1.run()
    obj2.run()
    



if __name__ == '__main__':
    main()
