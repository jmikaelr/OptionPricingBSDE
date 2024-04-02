import time
import yaml
import inspect
import pandas as pd
import os
import numpy as np
from scipy.stats import norm
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
        self._opt_style = 'european'

        self._load_configs()

    @property
    def opt_style(self):
        return self._opt_style

    def _load_configs(self):
        with open('configs.yaml', 'r') as config_file:
            self._configs = yaml.safe_load(config_file)

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
            raise ValueError(f"Invalid degree on Polynomial basis, you chose: {self.degree}, choose between 1 and 50")
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

    def plot_and_show_table_by_N(self, N_values, nofig=False, bs_price=None):
        function_name = inspect.currentframe().f_code.co_name
        prices, errors, rows = [], [], []

        for N_val in N_values:
            self.N= N_val
            Y0_array = self._bsde_solver()
            est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)
            prices.append(est_Y0)
            errors.append(std_Y0)
            rows.append([N_val, est_Y0, std_Y0, CI_Y[0], CI_Y[1]])
            print(f'Done with degree: {N_val}.')

        mean_price = np.mean(prices)
        self._generate_plot(N_values, prices, errors, mean_price, bs_price, function_name, nofig)

    def plot_and_show_table_by_M(self, M_values, nofig=False, bs_price=None):
        function_name = inspect.currentframe().f_code.co_name
        prices, errors, rows = [], [], []

        for M_val in M_values:
            self.M = M_val
            Y0_array = self._bsde_solver()
            est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)
            prices.append(est_Y0)
            errors.append(std_Y0)
            rows.append([M_val, est_Y0, std_Y0, CI_Y[0], CI_Y[1]])
            print(f'Done with M_val: {M_val}.')

        mean_price = np.mean(prices)
        self._generate_plot(M_values, prices, errors, mean_price, bs_price, function_name, nofig)

    def plot_and_show_table_by_degree(self, degrees, nofig=False, bs_price=None):
        function_name = inspect.currentframe().f_code.co_name
        prices, errors, rows = [], [], []

        for degree in degrees:
            self.degree = degree
            Y0_array = self._bsde_solver()
            est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)
            prices.append(est_Y0)
            errors.append(std_Y0)
            rows.append([degree, est_Y0, std_Y0, CI_Y[0], CI_Y[1]])
            print(f'Done with degree: {degree}.')

        mean_price = np.mean(prices)
        self._generate_plot(degrees, prices, errors, mean_price, bs_price, function_name, nofig)

    def plot_and_show_table_by_samples(self, samples, nofig=False, bs_price=None):
        function_name = inspect.currentframe().f_code.co_name
        prices, errors, rows = [], [], []

        for sample in samples:
            self.samples = sample
            Y0_array = self._bsde_solver()
            est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)
            if sample == 1:
                std_Y0 = 0
                CI_Y = (0, 0)
            prices.append(est_Y0)
            errors.append(std_Y0)
            rows.append([sample, est_Y0, std_Y0, CI_Y[0], CI_Y[1]])
            print(f'Done with sample: {sample}.')

        mean_price = np.mean(prices)
        self._generate_plot(samples, prices, errors, mean_price, bs_price, function_name, nofig)
        self._generate_table(rows, function_name)

    def _generate_plot(self, x_values, y_values, y_errors, mean_price, bs_price, function_name, nofig):
        plot_config = self._configs['plot_config'][function_name]

        plt.errorbar(x_values, y_values, yerr=y_errors, fmt='-o', capsize=5, capthick=2, ecolor='red', label='Option Price')
        plt.plot(x_values, [mean_price] * len(x_values), 'g--', label='Mean Price')

        if bs_price is not None:
            plt.plot(x_values, [bs_price] * len(x_values), 'k-', label='Black-Scholes Price')
            plt.scatter(x_values, [bs_price] * len(x_values), color='black', marker='_', s=100)

        plt.scatter(x_values, [mean_price] * len(x_values), color='green', marker='x', s=100)
        plt.xlabel(plot_config.get('xlabel'))
        plt.ylabel(plot_config.get('ylabel'))
        plt.title(plot_config.get('title').format(opt_style=self._opt_style.capitalize(), option_payoff=self.option_payoff.capitalize()))
        plt.legend()
        if plot_config.get('show_legend', False):
            plt.legend(loc=plot_config.get('legend_location', 'best'))
        plt.grid(plot_config.get('grid', True))

        plot_directory = os.path.join(self._configs['general_settings']['plot_directory'], function_name)
        os.makedirs(plot_directory, exist_ok=True)

        plot_name = plot_config['plot_name_template'].format(
            opt_style=self.opt_style,
            opt_payoff=self.option_payoff,
            N=self.N,
            M=self.M,
            degree=self.degree,
            min_value=min(x_values),
            max_value=max(x_values)
        )

        plot_path = os.path.join(plot_directory, plot_name)
        plt.savefig(plot_path)
        if not nofig:
            plt.show()
        plt.close()

    def _generate_table(self, rows, function_name):
        table_directory = os.path.join(self._configs['general_settings']['table_directory'], function_name)
        os.makedirs(table_directory, exist_ok=True)
        
        first_elements = [row[0] for row in rows]
        table_name = self._configs['plot_config'][function_name]['table_name_template'].format(
            opt_style=self.opt_style,
            opt_payoff=self.option_payoff,
            N=self.N,
            M=self.M,
            degree=self.degree,
            min_value= min(first_elements),
            max_value= max(first_elements)
        )
        
        table_path = os.path.join(table_directory, table_name)
        df = pd.DataFrame(rows, columns=['Degree', 'Estimated Price', 'Std. Deviation', 'CI Lower Bound', 'CI Upper Bound'])
        df.to_csv(table_path, index=False)
        print(f"Table saved to {table_path}")

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
    def __init__(self, S0, K, r, sigma, T, N, M, confidence_level=0.025, samples=100, 
                 option_payoff="call", degree=3, mu=0.05, picard=3):
        super().__init__(S0, K, r, sigma, T, N, M, confidence_level, samples, 
                         option_payoff, degree, mu, picard)
        self._opt_style = 'american'

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
        self._opt_style = 'europeanspread'

    def _payoff_func(self, S):
        if self.option_payoff == "call":
            return np.maximum(S - self.K, 0) - 2*np.maximum(S - self.K2, 0) 
        else:
            raise ValueError(f"Invalid option type: {self.option_payoff}. Supported types are 'call'.")

    def _bsde_solver(self):
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


class BSDEOptionPricingAmericanSpread(BSDEOptionPricingEuropean):
    def __init__(self, S0, K, r, sigma, T, N, M, confidence_level, samples, option_payoff, degree, mu, K2, R):
        super().__init__(S0, K, r, sigma, T, N, M, confidence_level, samples, option_payoff, degree, mu)
        self.K2 = 105 if K2 is None else K2 
        self.R = 0.06 if R is None else R 
        self._opt_style = 'americanspread'

    def _payoff_func(self, S):
        if self.option_payoff == "call":
            return np.maximum(S - self.K, 0) - 2*np.maximum(S - self.K2, 0) 
        else:
            raise ValueError(f"Invalid option type: {self.option_payoff}. Supported types are 'call'.")

    def _bsde_solver(self):
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
                Y[:, i-1] = np.maximum(self._payoff_func(S[:, i]), Y[:, i-1])
            Y0_samples[k] = np.mean(Y[:, 0])

        return Y0_samples

