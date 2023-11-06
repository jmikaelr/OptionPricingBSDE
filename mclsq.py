import time
import numpy as np
from scipy.linalg import cholesky
from scipy.stats import norm
from scipy.special import laguerre

LAGUERRE_BASIS = {
    n: [laguerre(i) for i in range(n+1)]
    for n in range(1, 11)
}

class BSDEOptionPricingEuropean:
    def __init__(self, S0, K, r, mu, sigma, T, N, M, lower = 0.025, samples = 100,
                 option_payoff="call", degree = 3):
        if not isinstance(S0, (int, float)) or S0 <= 0:
            raise ValueError('S0 must be positive.')
        if not isinstance(K, (int, float)) or K <= 0:
            raise ValueError('K must be positive.')
        if not isinstance(T, (int, float)) or T <= 0:
            raise ValueError('T must be positive.')
        if not isinstance(r, (int, float)):
            raise ValueError('r must be integer or float.')
        if not isinstance(mu, (int, float)):
            raise ValueError('mu must be integer or float.')
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError('sigma must be positive.')
        if not isinstance(N, int) or N <= 0:
            raise ValueError('N must be a positive integer.')
        if not isinstance(M, int) or M <= 0:
            raise ValueError('M must be a positive integer.')
        if not isinstance(degree, int) or degree < 0:
            raise ValueError('degree must be a non-negative integer.')
        if not isinstance(lower, float):
            raise ValueError('Lower confidence number msut be a float!')

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.mu = mu
        self.sigma = sigma
        self.lamb = (self.mu - self.r)/(self.sigma)
        self.N = N
        self.M = M
        self.option_payoff = self._get_opt_payoff(option_payoff)
        self.degree = degree
        self.dt = T / N
        self.samples = samples
        self.lower = lower

    def _get_opt_payoff(self, opt_payoff):
        """ Retrieves option payoff """
        if not isinstance(opt_payoff, str):
            raise TypeError('Option type should be a string!')
        if opt_payoff.lower() == 'call':
            return 'call'
        elif opt_payoff.lower() == 'put':
            return 'put'
        else:
            raise TypeError('Invalid option type! It should be call or put')

    def _payoff_func(self, S):
        """ Payoff function depending on the options payoff type """
        if self.option_payoff == "call":
            return np.maximum(S - self.K, 0)
        elif self.option_payoff == "put":
            return np.maximum(self.K - S, 0)
        else:
            raise ValueError(f"Invalid option type: {self.option_type}. Supported types are 'call' and 'put'.")

    def _generate_regression(self, S):
        """ Generates Laguerre polynomials up to degree self.degree """
        if self.degree > 10 or self.degree < 1:
            raise ValueError(f"Invalid degree on Polynomial basis, you chose: {self.degree}, choose between 1 and 10")
        
        basis_polynomials = LAGUERRE_BASIS[self.degree]
        Mk = np.array([pol(s) for s in S for pol in basis_polynomials]).reshape(S.shape[0], len(basis_polynomials))
        return Mk

    def _generate_stock_paths(self):
        """ Simulate Geometric Brownian Motion with a standard normal Brownian
        process """
        dW = np.random.normal(0, 1, (self.M, self.N+1))
        S = np.zeros((self.M, self.N+1))
        S[:,0] = self.S0
        for t in range(1, self.N+1):
            S[:, t] = S[:, t-1] * np.exp((self.r - 0.5 * self.sigma**2) * self.dt + 
                                         self.sigma * np.sqrt(self.dt) * dW[:, t])
        return S, dW

    def _bsde_solver(self):
        """ Solves the BSDE equation """
        Y0 = np.zeros(self.samples)

        for k in range(self.samples):
            S, dW = self._generate_stock_paths()
            Y = np.zeros((self.M, self.N + 1))

            Y[:, self.N] = self._payoff_func(S[:, self.N])

            for i in range(self.N, 0, -1):
                X = self._generate_regression(S[:, i])
                try:
                    beta = np.linalg.solve(X.T @ X, X.T @ Y[:, i])
                except np.linalg.LinAlgError as e:
                    print(f'Linear algebra error at time step {i}: {e}')
                    continue

                E = X @ beta
                discount_factor = np.exp(-self.r * self.dt)
                Y[:, i-1] = discount_factor * E 

            Y_est = np.mean(Y[:, 1])
            Y0[k] = Y_est

        return Y0


    def _confidence_interval(self, sample):
        mean_sample = np.mean(sample)
        std_sample = np.std(sample)
        upper = mean_sample + norm.ppf(1-self.lower/2) * std_sample/np.sqrt(self.M)
        lower = mean_sample - norm.ppf(1-self.lower/2) * std_sample/np.sqrt(self.M)
        CI = [lower, upper]
        return mean_sample, std_sample, CI

    def run(self):
        start_timer = time.time()
        Y0_array = self._bsde_solver()
        finished_time = time.time() - start_timer
        est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)
        print(f"\nBSDE solved in: {finished_time:.2f} seconds"
                f"\nEstimated option price: {est_Y0:.4f}"
                f"\nWith standard deviation: {std_Y0:.4f}"
                f"\nConfidence interval: {CI_Y}")

class BSDEOptionPricingAmerican(BSDEOptionPricingEuropean):

    def _bsde_solver(self):
        Y0 = np.zeros(self.samples)  # Will store the estimated option value at t=0 for each sample
        Z0 = np.zeros((self.samples, self.N))  # Will store the hedging strategy for each sample
        
        return Y0, Z0

        return Y0

