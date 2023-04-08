import numpy as np
import time

class BSDEOptionPricingEU:
    def __init__(self, S0, K, T, r, sigma, N, M, opt_type, degree = 3):
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
        self.opt_type = self._get_opt_type(opt_type)
        self.degree = degree
        self.dt = T / N

    def _get_opt_type(self, opt_type):
        if not isinstance(opt_type, str):
            raise TypeError('Option type should be a string!')
        if opt_type.lower() == 'call':
            return 'call'
        elif opt_type.lower() == 'put':
            return 'put'
        else:
            raise TypeError('Invalid option type! It should be call or put')


    def _generate_stock_paths(self):
        dW = np.random.normal(0, np.sqrt(self.dt), (self.M, self.N))
        S = np.empty((self.M, self.N + 1))
        S[:, 0] = self.S0
        for t in range(1, self.N + 1):
            S[:, t] = S[:, t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) *
                    self.dt + self.sigma * dW[:, t - 1])
        return S

    def _payoff_func(self, S_T):
        if self.opt_type == 'call':
            return np.maximum(S_T - self.K, 0)
        else:
            return np.maximum(self.K - S_T, 0)

    def _generate_regression_basis(self, S_t):
        X = np.zeros((S_t.shape[0], self.degree + 1))
        for i in range(self.degree + 1):
            X[:, i] = S_t ** i
        return X

    def _bsde_solver(self):
        S = self._generate_stock_paths()
        Y = self._payoff_func(S[:, -1])
        Z = np.zeros((self.M, self.N))

        for t in range(self.N - 1, 0, -1):
            dt = self.T / self.N
            discount_factor = np.exp(-self.r * dt)
            Y = Y * discount_factor

            X = self._generate_regression_basis(S[:, t])

            beta = np.linalg.lstsq(X, Y, rcond=None)[0]

            Y = np.sum(beta * X, axis=1)

            Z[:, t - 1] = (Y * (S[:, t] - S[:, t - 1])) / (self.sigma * 
                    S[:, t - 1])

        Y0 = np.mean(Y * np.exp(-self.r * dt))
        Z0 = np.mean(Z[:, 0])

        return Y0, Z0

        
    def run(self):
        start_timer = time.time()
        Y0, Z0 = self._bsde_solver()
        print(f'Runtime: {time.time() - start_timer:.3f}')
        print(f'Option price: {Y0:.5f}')
        print(f'Hedge ratio: {Z0:.5f}')



class ReflectedBSDEOptionPricing(BSDEOptionPricingEU):
    def __init__(self, S0, K, T, r, sigma, N, M, opt_type, lambda_=1.0):
        super().__init__(S0, K, T, r, sigma, N, M, opt_type)
        self.lambda_ = lambda_

    def _bsde_solver(self):
        S = self._generate_stock_paths()
        Y = self._payoff_func(S[:, -1])
        Z = np.zeros((self.M, self.N))
        K = np.zeros_like(Y)

        for t in range(self.N - 1, 0, -1):
            dt = self.T / self.N
            discount_factor = np.exp(-self.r * dt)
            Y = Y * discount_factor
            K = K * discount_factor

            X = self._generate_regression_basis(S[:, t])

            beta_Y = np.linalg.lstsq(X, Y, rcond=None)[0]
            beta_K = np.linalg.lstsq(X, K, rcond=None)[0]

            continuation_value = np.sum(beta_Y * X, axis=1)
            K_value = np.sum(beta_K * X, axis=1)

            early_exercise_value = self._payoff_func(S[:, t])

            exercise_indices = early_exercise_value > continuation_value
            Y[exercise_indices] = early_exercise_value[exercise_indices]

            K[exercise_indices] = np.maximum(continuation_value[exercise_indices] - 
                    early_exercise_value[exercise_indices], 
                    K[exercise_indices] - 
                    self.lambda_ * K_value[exercise_indices])


            Z[:, t - 1] = (Y * (S[:, t] - S[:, t - 1])) / (self.sigma * S[:, t - 1])

        Y0 = np.mean(Y * np.exp(-self.r * dt))

        itm_indices = (self._payoff_func(S[:, 1]) > 0)
        Z0 = np.mean(Z[itm_indices, 1])

        return Y0, Z0

