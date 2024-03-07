import matplotlib.pyplot as plt
from mclsq import * 

def plot_option_prices_by_degree(S0, K, r, sigma, T, N, M, lower, samples, mu, degrees, option_type='call'):
    prices = []
    errors = []
    
    for degree in degrees:
        bsde_solver = BSDEOptionPricingAmerican(S0, K, r, sigma, T, N, M, lower, samples, option_type, degree, mu)
        Y0_array, _ = bsde_solver._bsde_solver()
        est_Y0, std_Y0, CI_Y = bsde_solver._confidence_interval(Y0_array)
        prices.append(est_Y0)
        errors.append(std_Y0)
    
    plt.errorbar(degrees, prices, yerr=errors, fmt='-o', capsize=5, capthick=2, ecolor='red', label='Option Price')
    plt.xlabel('Degree of Laguerre Polynomials')
    plt.ylabel('Option Price')
    plt.title('Option Price vs. Degree of Laguerre Polynomials')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
S0 = 100
K = 100
r = 0.01
sigma = 0.2
T = 1
N = 50
M = 10000
lower = 0.025
samples = 100
mu = 0.01
degrees = [1, 2, 3, 4, 5]  # Different degrees to test
plot_option_prices_by_degree(S0, K, r, sigma, T, N, M, lower, samples, mu, degrees, option_type='call')

