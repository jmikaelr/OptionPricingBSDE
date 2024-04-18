import time
import yaml
import inspect
import pandas as pd
import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

np.random.seed(0)

class BSDEOptionPricingEuropean:
    def __init__(self, S0, K, r, sigma, T, N, M, confidence_level = 0.025, samples = 100,
                 mu = 0.05, option_payoff="call", degree_y = 4, degree_z = 2, picard = 3):
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
        if not isinstance(degree_y, int) or degree_y < 0:
            raise ValueError('degree must be a non-negative integer.')
        if not isinstance(degree_z, int) or degree_z < 0:
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
        self.degree_y = degree_y
        self.degree_z = degree_z
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
        """ Generates Laguerre polynomials up to degree self.degree_y """
        if self.degree_y > 50 or self.degree_y < 1:
            raise ValueError(f"Invalid degree on Polynomial basis, you chose: {self.degree_y}, choose between 1 and 50")
        S_expanded = np.expand_dims(S, axis=-1)
        basis_polynomials_y = np.array([np.polynomial.laguerre.Laguerre.basis(deg)(S_expanded) for deg in range(self.degree_y + 1)])
        basis_polynomials_z = np.array([np.polynomial.laguerre.Laguerre.basis(deg)(S_expanded) for deg in range(self.degree_z + 1)])
        p_y = np.transpose(basis_polynomials_y, (1, 2, 0)).reshape(S.shape[0], self.degree_y + 1)
        p_z = np.transpose(basis_polynomials_z, (1, 2, 0)).reshape(S.shape[0], self.degree_z + 1)
        return p_y, p_z

    def _generate_stock_paths(self):
        """Simulate Geometric Brownian Motion """ 
        dt = self.T / self.N
        dW = np.random.normal(0, np.sqrt(dt), (self.M, self.N))
        S = np.empty((self.M, self.N+1))
        S[:, 0] = self.S0  
        log_S = np.cumsum((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW, axis=1, out=S[:, 1:])
        S[:, 1:] = self.S0 * np.exp(log_S)
        return S, dW

    def _bsde_solver(self):
        """ Solves the BSDE equation for an european option using """
        Y0_samples = np.zeros(self.samples)
        Z0_samples = np.zeros(self.samples)

        for k in range(self.samples):
            S, dW = self._generate_stock_paths()
            Y = np.zeros((self.M, self.N))
            Z = np.zeros((self.M, self.N))

            Y[:, -1] = self._payoff_func(S[:, -1])

            for i in range(self.N - 2, -1, -1):
                X_y, X_z = self._generate_regression(S[:, i])
                A_z = X_z.T @ X_z
                A_y = X_y.T @ X_y
                Y_prev = Y[:, i+1]
                b_z = X_z.T @ (Y_prev * dW[:, i])
                b_y = X_y.T @ Y_prev
                alpha_z, _, _, _ = np.linalg.lstsq(A_z, b_z, rcond=None)
                alpha_y, _, _, _ = np.linalg.lstsq(A_y, b_y, rcond=None)
                Z[:, i] = (X_z @ alpha_z) / self.dt
                Y[:, i] = X_y @ alpha_y 
                Z_lamb = Z[: ,i]*self.lamb 
                for _ in range(self.picard):
                    Y[:, i] = Y_prev - (Z_lamb + self.r*Y[:, i])*self.dt

            Y0_samples[k] = np.mean(Y[:, 0])
            Z0_samples[k] = np.mean(Z[:, 0])

        return Y0_samples, Z0_samples

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
            self.N = N_val
            Y0_array, _ = self._bsde_solver()
            est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)
            prices.append(est_Y0)
            errors.append(std_Y0)
            rows.append([N_val, est_Y0, std_Y0, CI_Y[0], CI_Y[1]])
            print(f'Done with N: {N_val}.')

        self._generate_plot(N_values, prices, errors, function_name, bs_price, nofig)
        self._generate_table(rows, function_name, prices)

    def plot_and_show_table_by_M(self, M_values, nofig=False, bs_price=None):
        function_name = inspect.currentframe().f_code.co_name
        prices, errors, rows = [], [], []

        for M_val in M_values:
            self.M = M_val
            Y0_array, _ = self._bsde_solver()
            est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)
            prices.append(est_Y0)
            errors.append(std_Y0)
            rows.append([M_val, est_Y0, std_Y0, CI_Y[0], CI_Y[1]])
            print(f'Done with M_val: {M_val}.')

        self._generate_plot(M_values, prices, errors, function_name, bs_price, nofig)
        self._generate_table(rows, function_name, prices)

    def plot_and_show_table_by_degree(self, degrees, nofig=False, bs_price=None):
        function_name = inspect.currentframe().f_code.co_name
        prices, errors, rows = [], [], []

        for degree in degrees:
            self.degree_y = degree
            Y0_array, _ = self._bsde_solver()
            est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)
            prices.append(est_Y0)
            errors.append(std_Y0)
            rows.append([degree, est_Y0, std_Y0, CI_Y[0], CI_Y[1]])
            print(f'Done with degree: {degree}.')

        self._generate_plot(degrees, prices, errors, function_name, bs_price, nofig)
        self._generate_table(rows, function_name, prices)

    def plot_and_show_table_by_samples(self, samples, nofig=False, bs_price=None):
        function_name = inspect.currentframe().f_code.co_name
        prices, errors, rows = [], [], []

        for sample in samples:
            self.samples = sample
            Y0_array, _ = self._bsde_solver()
            est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)
            if sample == 1:
                std_Y0 = 0
                CI_Y = (0, 0)
            prices.append(est_Y0)
            errors.append(std_Y0)
            rows.append([sample, est_Y0, std_Y0, CI_Y[0], CI_Y[1]])
            print(f'Done with sample: {sample}.')

        self._generate_plot(samples, prices, errors, function_name, bs_price, nofig)
        self._generate_table(rows, function_name, prices)

    def _generate_plot(self, x_values, y_values, y_errors, function_name, bs_price, nofig):
        plot_config = self._configs['plot_config'][function_name]

        running_totals = np.cumsum(y_values)
        running_average = running_totals / np.arange(1, len(y_values) + 1)

        plt.errorbar(x_values, y_values, yerr=y_errors, fmt='-o', capsize=5, capthick=2, ecolor='red', label='Option Price')
        plt.plot(x_values, running_average, 'b--', label='Running Average Price')

        if bs_price is not None:
            plt.plot(x_values, [bs_price] * len(x_values), 'k-', label='Black-Scholes Price')
            plt.scatter(x_values, [bs_price] * len(x_values), color='black', marker='_', s=100)

        plt.xlabel(plot_config.get('xlabel'))
        plt.ylabel(plot_config.get('ylabel'))
        plt.title(plot_config.get('title').format(opt_style=self._opt_style.capitalize(), option_payoff=self.option_payoff.capitalize()))
        plt.legend(loc=plot_config.get('legend_location', 'best'))
        plt.grid(plot_config.get('grid', True))

        plot_directory = os.path.join(self._configs['general_settings']['plot_directory'], function_name)
        os.makedirs(plot_directory, exist_ok=True)

        plot_name = plot_config['plot_name_template'].format(
            opt_style=self.opt_style,
            opt_payoff=self.option_payoff,
            N=self.N,
            M=self.M,
            degree=self.degree_y,
            min_value=min(x_values),
            max_value=max(x_values)
        )

        plot_path = os.path.join(plot_directory, plot_name)
        plt.savefig(plot_path)
        if not nofig:
            plt.show()
        plt.close()

    def _generate_table(self, rows, function_name, prices):
        table_directory = os.path.join(self._configs['general_settings']['table_directory'], function_name)
        os.makedirs(table_directory, exist_ok=True)

        first_elements = [row[0] for row in rows]

        running_totals = np.cumsum(prices)
        running_average = running_totals / np.arange(1, len(prices) + 1)

        for i, row in enumerate(rows):
            row.append(running_average[i])

        table_name = self._configs['plot_config'][function_name]['table_name_template'].format(
            opt_style=self.opt_style,
            opt_payoff=self.option_payoff,
            N=self.N,
            M=self.M,
            degree=self.degree_y,
            min_value=min(first_elements),
            max_value=max(first_elements)
        )

        table_path = os.path.join(table_directory, table_name)
        
        df = pd.DataFrame(rows, columns=[function_name.split('_')[-1], 'Estimated Price', 'Std. Deviation', 'CI Lower Bound', 'CI Upper Bound', 'Running Average Price'])
        df.to_csv(table_path, index=False)
        print(f"Table saved to {table_path}")

    def run(self):
        """ Method called to run the program and solve the BSDE """
        start_timer = time.time()
        Y0_array, Z0_array = self._bsde_solver()
        finished_time = time.time() - start_timer
        est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)
        est_Z0, std_Z0, CI_Z = self._confidence_interval(Z0_array)
        print(f"BSDE solved in: {finished_time:.2f} seconds\n"
                f"\nEstimated option price (Y\u2080): {est_Y0:.4f}"
                f"\nStandard Deviation: {std_Y0:.4f}"
                f"\nConfidence Interval: {CI_Y}\n"
                f"\nEstimated (Z\u2080): {est_Z0:.4f}"
                f"\nStandard Deviation (Z\u2080): {std_Z0:.4f}"
                f"\nConfidence Interval: {CI_Z}\n"
              )

    def __repr__(self):
        return (f"{self.__class__.__name__}(\n"
                f"  S0={self.S0},\n"
                f"  K={self.K},\n"
                f"  r={self.r},\n"
                f"  sigma={self.sigma},\n"
                f"  T={self.T},\n"
                f"  N={self.N},\n"
                f"  M={self.M},\n"
                f"  confidence_level={self.confidence_level},\n"
                f"  samples={self.samples},\n"
                f"  mu={self.mu},\n"
                f"  option_payoff='{self.option_payoff}',\n"
                f"  degree_y={self.degree_y},\n"
                f"  degree_z={self.degree_z},\n"
                f"  picard={self.picard}\n)")

class BSDEOptionPricingAmerican(BSDEOptionPricingEuropean):
    def __init__(self, S0, K, r, sigma, T, N, M, confidence_level=0.025, samples=100, 
                mu=0.05, option_payoff="call", degree_y=3, degree_z=3, picard=3):
        super().__init__(S0, K, r, sigma, T, N, M, confidence_level, samples, mu, 
                         option_payoff, degree_y, degree_z, picard)
        self._opt_style = 'american'

    def _bsde_solver(self):
        """ Solves the BSDE equation for an european option using """
        Y0_samples = np.zeros(self.samples)
        Z0_samples = np.zeros(self.samples)

        for k in range(self.samples):
            S, dW = self._generate_stock_paths()
            Y = np.zeros((self.M, self.N))
            Z = np.zeros((self.M, self.N))

            exercise_values = self._payoff_func(S)
            Y[:, -1] = exercise_values[:, -1] 

            for i in range(self.N - 2, -1, -1):
                X_y, X_z = self._generate_regression(S[:, i])
                A_z = X_z.T @ X_z
                A_y = X_y.T @ X_y
                Y_prev = Y[:, i+1]
                b_z = X_z.T @ (Y_prev * dW[:, i])
                b_y = X_y.T @ Y_prev
                alpha_z, _, _, _ = np.linalg.lstsq(A_z, b_z, rcond=None)
                alpha_y, _, _, _ = np.linalg.lstsq(A_y, b_y, rcond=None)
                Z[:, i] = (X_z @ alpha_z) / self.dt
                Y[:, i] = X_y @ alpha_y 
                Z_lamb = Z[: ,i]*self.lamb 
                #for _ in range(self.picard):
                    #Y[:, i] = Y_prev - (Z_lamb + self.r*Y[:, i])*self.dt

                Y[:, i] = np.maximum(Y[:, i], exercise_values[:, i])

            Y0_samples[k] = np.mean(Y[:, 0])
            Z0_samples[k] = np.mean(Z[:, 0])

        return Y0_samples, Z0_samples

    def __repr__(self):
        base_repr = super().__repr__()

        return base_repr[:-1] + f" opt_style='{self._opt_style}'\n)"


class BSDEOptionPricingEuropeanSpread(BSDEOptionPricingEuropean):
    def __init__(self, S0, K, r, sigma, T, N, M, confidence_level, samples, mu, option_payoff, degree_y, degree_z, picard, K2, R):
        super().__init__(S0, K, r, sigma, T, N, M, confidence_level, samples, mu, option_payoff, degree_y, degree_z, picard)
        self.K2 = 105 if K2 is None else K2 
        self.R = 0.06 if R is None else R 
        self._opt_style = 'europeanspread'

    def _payoff_func(self, S):
        if self.option_payoff == "call":
            return (np.maximum(S - self.K, 0) - 2*np.maximum(S - self.K2, 0))
        else:
            raise ValueError(f"Invalid option type: {self.option_payoff}. Supported types are 'call'.")

    def _bsde_solver(self):
        Y0_samples = np.zeros(self.samples)
        Z0_samples = np.zeros(self.samples)

        for k in range(self.samples):
            S, dW = self._generate_stock_paths()
            Y = np.zeros((self.M, self.N))
            Z = np.zeros((self.M, self.N))
            E = np.zeros((self.M, self.N))

            Y[:, -1] = self._payoff_func(S[:, -1])

#            for i in range(self.N - 1, -1, -1):
#                X_y, X_z = self._generate_regression(S[:, i])
#                L_z = np.linalg.cholesky(X_z.T @ X_z)
#                L_y = np.linalg.cholesky(X_y.T @ X_y)
#                z_z = np.linalg.solve(L_z, X_z.T @ (Y[:, i] * dW[:, i]))
#                z_y = np.linalg.solve(L_y, X_y.T @ Y[:, i])
#                alpha_z = np.linalg.lstsq(L_z.T, z_z, rcond=None)[0]
#                alpha_y = np.linalg.lstsq(L_y.T, z_y, rcond=None)[0]
#                print(alpha_y)
#                Z[:, i] = (X_z @ alpha_z) / self.dt 
#                E[:, i] = X_y @ alpha_y
#                if i == self.N - 2:
#                    return
#                for _ in range(self.picard):
#                    Y[:, i] = E[:, i] + (-self.r*Y[:, i] - Z[:, i] * self.lamb + (self.R-self.r)*np.minimum(Y[:, i] - Z[:, i], 0)) * self.dt
            for i in range(self.N - 2, -1, -1):
                X_y, X_z = self._generate_regression(S[:, i])

                A_z = X_z.T @ X_z
                A_y = X_y.T @ X_y
                
                b_z = X_z.T @ (Y[:, i+1] * dW[:, i])
                b_y = X_y.T @ Y[:, i+1]

                # Directly compute the regression coefficients using least squares
                alpha_z, _, _, _ = np.linalg.lstsq(A_z, b_z, rcond=None)
                alpha_y, _, _, _ = np.linalg.lstsq(A_y, b_y, rcond=None)

                Z[:, i] = (X_z @ alpha_z) / self.dt
                E[:, i] = X_y @ alpha_y
                #print("E[:, i]:", E[:, i])
                for _ in range(self.picard):
                    Y[:, i] = E[:, i] + (-self.r*Y[:, i] - Z[:, i] * self.lamb + (self.R-self.r)*np.minimum(Y[:, i] - Z[:, i]/self.sigma, 0)) * self.dt

                
            Y0_samples[k] = np.mean(Y[:, 0])
            Z0_samples[k] = np.mean(Z[:, 0])               

        return Y0_samples, Z0_samples

    def __repr__(self):
        base_repr = super().__repr__()
        return base_repr[:-1] + f"  opt_style='{self._opt_style}'\n  K2='{self.K2}'\n  R='{self.R}'\n)"


class BSDEOptionPricingAmericanSpread(BSDEOptionPricingEuropean):
    def __init__(self, S0, K, r, sigma, T, N, M, confidence_level, samples, mu, option_payoff, degree_y, degree_z, picard, K2, R):
        super().__init__(S0, K, r, sigma, T, N, M, confidence_level, samples, mu, option_payoff, degree_y, degree_z, picard)
        self.K2 = 105 if K2 is None else K2 
        self.R = 0.06 if R is None else R 
        self.K2 = 105 if K2 is None else K2 
        self.R = 0.06 if R is None else R 
        self._opt_style = 'americanspread'

    def _payoff_func(self, S):
        if self.option_payoff == "call":
            return (np.maximum(S - self.K, 0) - 2*np.maximum(S - self.K2, 0))
        else:
            raise ValueError(f"Invalid option type: {self.option_payoff}. Supported types are 'call'.")

    def _bsde_solver(self):
        Y0_samples = np.zeros(self.samples)
        Z0_samples = np.zeros(self.samples)

        for k in range(self.samples):
            S, dW = self._generate_stock_paths()
            Y = np.zeros((self.M, self.N))
            Z = np.zeros((self.M, self.N))

            exercise_values = self._payoff_func(S)
            Y[:, self.N - 1] = exercise_values[:, self.N - 1]

            for i in range(self.N - 2, -1, -1):
                X_y, X_z = self._generate_regression(S[:, i])
                A_z = X_z.T @ X_z
                A_y = X_y.T @ X_y
                Y_prev = Y[:, i+1]
                b_z = X_z.T @ (Y_prev * dW[:, i])
                b_y = X_y.T @ Y_prev
                alpha_z, _, _, _ = np.linalg.lstsq(A_z, b_z, rcond=None)
                alpha_y, _, _, _ = np.linalg.lstsq(A_y, b_y, rcond=None)
                Z[:, i] = (X_z @ alpha_z) / self.dt
                Y[:, i] = X_y @ alpha_y 
                Z_lamb = Z[: ,i]*self.lamb 
                for _ in range(self.picard):
                    Y[:, i] = (Y_prev - (Y[:, i]*self.r + Z_lamb - 
                        (self.R-self.r)*np.minimum((Y[:, i] - (Z[:, i]/self.sigma)), 0)) * self.dt)

                Y[:, i] = np.maximum(Y[:, i], exercise_values[:, i])

            Y0_samples[k] = np.mean(Y[:, 0])
            Z0_samples[k] = np.mean(Z[:, 0])

        return Y0_samples, Z0_samples

    def __repr__(self):
        base_repr = super().__repr__()
        return base_repr[:-1] + f"  opt_style='{self._opt_style}'\n  K2='{self.K2}'\n  R='{self.R}'\n)"
