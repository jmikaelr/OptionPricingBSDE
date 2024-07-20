import bsde_models
import numpy as np

S = 120
mu = 0.04
sigma = 0.4
K = 100
r = 0.04
T = 2
N = 100
M = 1000.2
H = 60
correlation = 0.5

bsde = bsde_models.BSDEOptionPricingEuropean(S, mu, sigma, correlation, K, r, T, N, M, H = H, dims = 1, samples = 1, delta = 5, option_payoff="put")

bsde.solve()


