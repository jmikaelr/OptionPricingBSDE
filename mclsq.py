import time
import numpy as np
from scipy.linalg import cholesky
from scipy.stats import norm

class BSDEOptionPricingEuropean:
    def __init__(self, S0, K, r, sigma, T, N, M, samples = 100, option_type="call", degree = 3):
        if not isinstance(S0, (int, float)) or S0 <= 0:
            raise ValueError('S0 must be positive.')
        if not isinstance(K, (int, float)) or K <= 0:
            raise ValueError('K must be positive.')
        if not isinstance(T, (int, float)) or T <= 0:
            raise ValueError('T must be positive.')
        if not isinstance(r, (int, float)) or r < 0:
            raise ValueError('r must be non-negative.')
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError('sigma must be positive.')
        if not isinstance(N, int) or N <= 0:
            raise ValueError('N must be a positive integer.')
        if not isinstance(M, int) or M <= 0:
            raise ValueError('M must be a positive integer.')
        if not isinstance(degree, int) or degree < 0:
            raise ValueError('degree must be a non-negative integer.')

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.M = M
        self.option_type = self._get_opt_type(option_type)
        self.degree = degree
        self.dt = T / N
        self.samples = samples

    def _get_opt_type(self, opt_type):
        if not isinstance(opt_type, str):
            raise TypeError('Option type should be a string!')
        if opt_type.lower() == 'call':
            return 'call'
        elif opt_type.lower() == 'put':
            return 'put'
        else:
            raise TypeError('Invalid option type! It should be call or put')

    def _payoff_func(self, S):
        if self.option_type == "call":
            return np.maximum(S - self.K, 0)
        elif self.option_type == "put":
            return np.maximum(self.K - S, 0)
        else:
            raise ValueError(f"Invalid option type: {self.option_type}. Supported types are 'call' and 'put'.")

    def _generate_stock_paths(self):
        dW = np.random.normal(0, np.sqrt(self.dt), (self.M, self.N))
        S = np.zeros((self.M, self.N + 1))
        S[:, 0] = self.S0
        S[:, 1:] = np.exp(np.cumsum((self.r - 0.5 * self.sigma ** 2) * self.dt + self.sigma * dW, axis=1))
        S *= self.S0
        return S

    def _generate_regression(self, S):
        if self.degree == 3:
            pol = lambda x: [1, 1-x, (x**2 - 4*x + 2)/2, (-x**3 + 9*x**2 - 18*x + 6)/6]
        elif self.degree == 2:
            pol = lambda x: [1, 1-x, (x**2 - 4*x + 2)/2]
        elif self.degree == 1:
            pol = lambda x: [1, 1-x]
        else:
            raise ValueError(f"Invalid degree on Polynomial basis, you choosed: {self.degree}, choose between 1,2 or 3")
        Mk = np.zeros((S.shape[0], self.degree+1))
        for m in range(S.shape[0]):
            Mk[m, :] = pol(S[m])
        return Mk

    def _bsde_solver(self):
        Y0 = np.zeros(self.samples)
        Z_mean = np.zeros(self.N)

        for k in range(self.samples):
            S = self._generate_stock_paths()
            Y = np.zeros((self.M, self.N + 1))
            Z = np.zeros((self.M, self.N))  # Hedging ratios

            Y[:, self.N] = self._payoff_func(S[:, self.N])

            for i in range(self.N, 0, -1):
                X = self._generate_regression(S[:, i])
                XtX = cholesky(X.T @ X).T
                beta = np.linalg.solve(XtX, X.T @ Y[:, i])
                alpha = np.linalg.solve(XtX.T, beta)

                Z[:, i - 1] = alpha[1] 

                E = X @ alpha
                for _ in range(3):
                    Y[:, i-1] = E + (-self.r * Y[:, i-1]) * (self.T / self.N)

            Y_est = np.sum(Y[:, 1]) / self.M + (-self.r * np.mean(Y[:, 1])) * (self.T / self.N)
            Y0[k] = Y_est
            Z_mean += np.mean(Z, axis=0)
        Z_mean /= self.samples

        return Y0, Z_mean



    def _confidence_interval(self, sample):
        mean_sample = np.mean(sample)
        std_sample = np.std(sample)
        upper = mean_sample + norm.ppf(0.975) * std_sample/np.sqrt(self.M)
        lower = mean_sample + norm.ppf(0.025) * std_sample/np.sqrt(self.M)
        CI = [lower, upper]
        return mean_sample, std_sample, CI

    def run(self):
        start_timer = time.time()
        Y0_array, Z_array = self._bsde_solver()
        finished_time = time.time() - start_timer
        est_Y0, std_Y0, CI = self._confidence_interval(Y0_array)
        print(f"\nBSDE solved in: {finished_time:.2f} seconds\nEstimated option price: {est_Y0:.4f}\nWith standard deviation: {std_Y0:.4f}\nConfidence interval: {CI}")

class BSDEOptionPricingAmerican(BSDEOptionPricingEuropean):

    def _bsde_solver(self):
        Y0 = np.zeros(self.samples)
        Z_mean = np.zeros(self.N)

        for k in range(self.samples):
            S = self._generate_stock_paths()
            Y = np.zeros((self.M, self.N + 1))
            Z = np.zeros((self.M, self.N))  # Hedging ratios

            Y[:, self.N] = self._payoff_func(S[:, self.N])

            for i in range(self.N, 0, -1):
                X = self._generate_regression(S[:, i])
                XtX = cholesky(X.T @ X).T
                beta = np.linalg.solve(XtX, X.T @ Y[:, i])
                alpha = np.linalg.solve(XtX.T, beta)

                Z[:, i - 1] = alpha[1] 

                E = X @ alpha
                for _ in range(3):
                    continuation_value = E + (-self.r * Y[:, i-1]) * (self.T / self.N)
                    exercise_value = self._payoff_func(S[:, i-1])

                    Y[:, i-1] = np.maximum(continuation_value, exercise_value)

            Y_est = np.sum(Y[:, 1]) / self.M + (-self.r * np.mean(Y[:, 1])) * (self.T / self.N)
            Y0[k] = Y_est
            Z_mean += np.mean(Z, axis=0)
        Z_mean /= self.samples

        return Y0, Z_mean

