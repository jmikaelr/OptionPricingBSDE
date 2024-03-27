import time
import pandas as pd
import os
import numpy as np
from scipy.stats import norm
from main import black_scholes
import matplotlib.pyplot as plt

class BSDEOptionPricingEuropean:
    def __init__(self, S0, K, r, sigma, T, N, M, confidence_level = 0.025, samples = 100,
                 option_payoff="call", degree = 3, mu = 0.05, picard = 3):
        if not isinstance(S0, (int, float)) or S0 <= 0:
            raise ValueError('S0 must be positive.')
        if not isinstance(K, (int, float)) or K <= 0:
            raise ValueError('K must be positive.')
        if not isinstance(T, (int, float)) or T <= 0:
            raise ValueError('T must be positive.')
        if not isinstance(r, (int, float)):
            raise ValueError('r must be integer or float.')
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError('sigma must be positive.')
        if not isinstance(N, int) or N <= 0:
            raise ValueError('N must be a positive integer or float.')
        if not isinstance(M, int) or M <= 0:
            raise ValueError('M must be a positive integer.')
        if not isinstance(degree, int) or degree < 0:
            raise ValueError('degree must be a non-negative integer.')
        if not isinstance(confidence_level, float):
            raise ValueError('Lower confidence number msut be a float!')

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.mu = r if mu is None else mu
        self.sigma = sigma
        self.N = N
        self.M = M
        self.option_payoff = self._get_opt_payoff(option_payoff)
        self.degree = degree
        self.dt = T / N
        self.samples = samples
        self.confidence_level = confidence_level
        self.lamb = (mu - r)/(sigma) if mu is not None else 0
        self.picard = picard

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
            raise ValueError(f"Invalid option type: {self.option_payoff}. Supported types are 'call' and 'put'.")

    def _generate_regression(self, S):
        """ Generates Laguerre polynomials up to degree self.degree """
        if self.degree > 50 or self.degree < 1:
            raise ValueError(f"Invalid degree on Polynomial basis, you chose: {self.degree}, choose between 1 and 250")
            raise ValueError(f"Invalid degree on Polynomial basis, you chose: {self.degree}, choose between 1 and 20")
        S_expanded = np.expand_dims(S, axis=-1)   
        basis_polynomials = np.array([np.polynomial.laguerre.Laguerre.basis(deg)(S_expanded) for deg in range(self.degree + 1)])
        Mk = np.transpose(basis_polynomials, (1, 2, 0)).reshape(S.shape[0], self.degree + 1)
        return Mk

    def _generate_stock_paths(self):
        """Simulate Geometric Brownian Motion paths."""
        dt = self.T / self.N
        dW = np.random.normal(0, np.sqrt(dt), (self.M, self.N + 1))
        log_S = np.cumsum((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW, axis=1)
        S = self.S0 * np.exp(np.hstack([np.zeros((self.M, 1)), log_S]))
        return S, dW

    def _bsde_solver(self):
        """ Solves the BSDE equation for an european option using Cholesky Decomposition"""
        Y0_samples = np.zeros(self.samples)

        for k in range(self.samples):
            S, _ = self._generate_stock_paths()
            Y = np.zeros((self.M, self.N + 1))

            Y[:, -1] = self._payoff_func(S[:, -1])

            for i in range(self.N, 0, -1):
                X = self._generate_regression(S[:, i])
                A = X.T @ X
                alpha = np.linalg.lstsq(A, X.T @ Y[:, i], rcond=None)[0]
                continuation_value = X @ alpha
                Y[:, i-1] = continuation_value * np.exp(-self.r * self.dt)

            Y0_samples[k] = np.mean(Y[:, 0])

        return Y0_samples 

    def _confidence_interval(self, sample):
        """ Calculates the confidence interval with confidence_level """ 
        mean_sample = np.mean(sample)
        std_sample = np.std(sample, ddof=1)
        upper = mean_sample + norm.ppf(1-self.confidence_level/2) * std_sample/np.sqrt(self.samples)
        lower = mean_sample - norm.ppf(1-self.confidence_level/2) * std_sample/np.sqrt(self.samples)
        CI = [round(lower,4), round(upper,4)]
        return mean_sample, std_sample, CI

    def plot_and_show_table_by_degree(self, degrees, opt_style = 'european', bs_price = None, nofig = False):
        prices = []
        errors = []
        rows = []
        for degree in degrees:
            self.degree = degree
            Y0_array = self._bsde_solver()
            est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)
            prices.append(est_Y0)
            errors.append(std_Y0)
            rows.append([degree, est_Y0, std_Y0, CI_Y[0], CI_Y[1]])
            print(f'Done with degree {degree}.')
        mean_price = np.mean(prices)
                
        plt.errorbar(degrees, prices, yerr=errors, fmt='-o', capsize=5, capthick=2, ecolor='red', label='Option Price')
        
        plt.plot(degrees, [mean_price] * len(degrees), 'g--', label='Mean Price')
        if opt_style == 'european':
            plt.plot(degrees, [bs_price] * len(degrees), 'k-', label='Black-Scholes')
            plt.scatter(degrees, [bs_price] * len(degrees), color='black', marker='_', s=100)
        
        plt.scatter(degrees, [mean_price] * len(degrees), color='green', marker='x', s=100)
        
        tick_positions = np.linspace(degrees[0], degrees[-1], 5, dtype=int)
        plt.xticks(tick_positions, labels=[str(deg) for deg in tick_positions])

        plt.xlabel('Degree of Laguerre Polynomials')
        plt.ylabel('Option Price')
        plt.title('Option Price vs. Degree of Laguerre Polynomials')
        plt.legend()
        plt.grid(True)
        plot_directory = './plots'
        if not os.path.exists(plot_directory):
            os.mkdir('plots')

        table_directory = './tables'
        if not os.path.exists(table_directory):
            os.mkdir(table_directory)

        match opt_style:
            case 'european':
                plot_name = f'{opt_style}_{self.option_payoff}_{self.N}-{self.M}_{len(degrees)+1}.png'
                table_name = f'{opt_style}_{self.option_payoff}_{self.N}-{self.M}_{len(degrees)+1}.csv'
            case 'american':
                plot_name = f'{opt_style}_{self.option_payoff}_{self.N}-{self.M}_{len(degrees)+1}.png'
                table_name = f'{opt_style}_{self.option_payoff}_{self.N}-{self.M}_{len(degrees)+1}.csv'
            case 'europeanspread':
                plot_name = f'{opt_style}_{self.option_payoff}_{self.N}-{self.M}_{len(degrees)+1}.png'
                table_name = f'{opt_style}_{self.option_payoff}_{self.N}-{self.M}_{len(degrees)+1}.csv'
            case 'americanspread':
                plot_name = f'{opt_style}_{self.option_payoff}_{self.N}-{self.M}_{len(degrees)+1}.png'
                table_name = f'{opt_style}_{self.option_payoff}_{self.N}-{self.M}_{len(degrees)+1}.csv'
            case _:
                raise ValueError('Invalid option style')

        plot_path = os.path.join(plot_directory, plot_name)
        plt.savefig(plot_path)
        if not nofig:
            plt.show()
        

        df = pd.DataFrame(rows, columns=['Degree', 'Estimated Price', 'Std. Deviation', 'CI Lower Bound', 'CI Upper Bound'])
        table_path = os.path.join(table_directory, table_name)
        df.to_csv(table_path, index=False)
        print(df)
        print(f"Mean Price across all degrees: {mean_price:.4f}")
        print(f"European {self.option_payoff} option price: {bs_price:.4f}")

    def run(self):
        """ Method called to run the program and solve the BSDE """
        start_timer = time.time()
        Y0_array = self._bsde_solver()
        finished_time = time.time() - start_timer
        est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)
        print(f"\nBSDE solved in: {finished_time:.2f} seconds"
                f"\nEstimated option price: {est_Y0:.4f}"
                f"\nWith standard deviation: {std_Y0:.4f}"
                f"\nConfidence interval: {CI_Y}"
              )

class BSDEOptionPricingAmerican(BSDEOptionPricingEuropean):

    def _bsde_solver(self):
        """ Solves the BSDE equation for an european option using Cholesky Decomposition"""
        Y0_samples = np.zeros(self.samples)

        for k in range(self.samples):
            S, _ = self._generate_stock_paths()
            Y = np.zeros((self.M, self.N + 1))

            Y[:, -1] = self._payoff_func(S[:, -1])

            for i in range(self.N, 0, -1):
                X = self._generate_regression(S[:, i])
                A = X.T @ X
                alpha = np.linalg.lstsq(A, X.T @ Y[:, i], rcond=None)[0]
                continuation_value = X @ alpha
                exercise_value = self._payoff_func(S[:, i])
                Y[:, i-1] = np.maximum(continuation_value * np.exp(-self.r * self.dt), exercise_value)
                

            Y0_samples[k] = np.mean(Y[:, 0])

        return Y0_samples


class BSDEOptionPricingEuropeanSpread(BSDEOptionPricingEuropean):
    def __init__(self, S0, K, r, sigma, T, N, M, confidence_level, samples, option_payoff, degree, mu, K2, R):
        super().__init__(S0, K, r, sigma, T, N, M, confidence_level, samples, option_payoff, degree, mu)
        self.K2 = 105 if K2 is None else K2 
        self.R = 0.06 if R is None else R 

    def _payoff_func(self, S):
        if self.option_payoff == "call":
            return np.maximum(S - self.K, 0) - 2*np.maximum(S - self.K2, 0) 
        else:
            raise ValueError(f"Invalid option type: {self.option_payoff}. Supported types are 'call'.")

    def _bsde_solver(self):
        """ Solves the BSDE equation for an european option using Cholesky Decomposition"""
        Y0_samples = np.zeros(self.samples)

        for k in range(self.samples):
            S, dW = self._generate_stock_paths()
            Y = np.zeros((self.M, self.N + 1))
            Z = np.zeros((self.M, self.N + 1))

            Y[:, -1] = self._payoff_func(S[:, -1])

            for i in range(self.N, 0, -1):
                X = self._generate_regression(S[:, i])
                A = X.T @ X
                alpha_y = np.linalg.lstsq(A, X.T @ Y[:, i], rcond=None)[0]
                alpha_z = np.linalg.lstsq(A, X.T @ (Y[:, i] * dW[:, i]), rcond=None)[0]
                E = X @ alpha_y
                Z[:, i] = X @ alpha_z 
                for _ in range(self.picard):
                    Y[:, i-1] = E + (-self.r*Y[:, i] + self.lamb*Z[:, i] - (self.R - self.r)*np.min(Y[:, i] -Z[:, i]/self.sigma, 0))*self.dt

            Y0_samples[k] = np.mean(Y[:, 0])

        return Y0_samples


class BSDEOptionPricingAmericanSpread(BSDEOptionPricingAmerican):
    def __init__(self, S0, K, r, sigma, T, N, M, confidence_level, samples, option_payoff, degree, mu, K2, R):
        super().__init__(S0, K, r, sigma, T, N, M, confidence_level, samples, option_payoff, degree, mu)
        self.K2 = 105 if K2 is None else K2 
        self.R = 0.06 if R is None else R 

    def _payoff_func(self, S):
        if self.option_payoff == "call":
            return np.maximum(S - self.K, 0) - np.maximum(S - self.K2, 0) 
        else:
            raise ValueError(f"Invalid option type: {self.option_payoff}. Supported types are 'call'.")

    def _bsde_solver(self):
        """ Solves the BSDE equation for an european option using Cholesky Decomposition"""
        Y0_samples = np.zeros(self.samples)

        for k in range(self.samples):
            S, dW = self._generate_stock_paths()
            Y = np.zeros((self.M, self.N + 1))
            Z = np.zeros((self.M, self.N + 1))

            Y[:, -1] = self._payoff_func(S[:, -1])

            for i in range(self.N, 0, -1):
                X = self._generate_regression(S[:, i])
                A = X.T @ X
                alpha_y = np.linalg.lstsq(A, X.T @ Y[:, i], rcond=None)[0]
                alpha_z = np.linalg.lstsq(A, X.T @ (Y[:, i] * dW[:, i]), rcond=None)[0]
                E = X @ alpha_y
                Z[:, i] = X @ alpha_z 
                exercise_value = self._payoff_func(S[:, i])
                for _ in range(self.picard):
                    Y[:, i-1] = E + (-self.r*Y[:, i] + self.lamb*Z[:, i] - (self.R - self.r)*np.min(Y[:, i] -Z[:, i]/self.sigma, 0))*self.dt
                Y[:, i-1] = np.maximum(exercise_value, Y[:, i-1])
                
            Y0_samples[k] = np.mean(Y[:, 0])

        return Y0_samples


