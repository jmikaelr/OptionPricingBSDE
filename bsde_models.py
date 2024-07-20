import time
import itertools
import yaml
import pandas as pd
import os
import numpy as np
from scipy.stats import norm
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt

class BSDEOptionPricingEuropean:
    def __init__(self, S0, mu, sigma, correlation, K, r, T, N, M,  confidence_level = 0.025, 
                 samples = 50, dims = 1, option_payoff = "call", H = 60, delta= 1, k = 0):
        self._validation_check(S0, mu, sigma, K, r, T, N, M, 
                         confidence_level, samples, dims, 
                         option_payoff, H, delta) 
        self.S0 = np.float64(S0) 
        self.mu = np.float64(mu) 
        self.sigma = np.float64(sigma)
        self.K = np.float16(K)
        self.r = np.float64(r)
        self.T = np.float16(T)
        self.N = np.uint32(N)
        self.M = np.uint32(M)
        self.dt = np.float64(T/N)
        self.confidence_level = np.float32(confidence_level)
        self.samples = np.uint16(samples)
        self.dims = np.uint8(dims)
        self.option_payoff = option_payoff
        self.H = np.uint8(H)
        self.delta = np.uint8(delta)
        self.num_cubes_per_dim = int(2 * H // delta)
        self.num_basis_per_cube = (k + 1) ** dims
        self.dim_phi = self.num_cubes_per_dim ** dims * self.num_basis_per_cube
        self.cube_ranges = ([range(int((S0 - H) // delta), 
                                   int((S0 + H) // delta)) for _ in range(dims)])
        self.cube_indices = list(itertools.product(*self.cube_ranges))
        self.k = np.uint8(k)
        self.lamb = np.float64((mu - r)/sigma)
        self.correlation = np.float32(np.full((1, 1), 1) if self.dims == 1 
            else self._construct_correlation_matrix(correlation))
        self._opt_style = 'european'

    def _validation_check(self, S0, mu, sigma, K, r, T, N, M,
                          confidence_level, samples, dims,
                          option_payoff, H, delta):
        if not isinstance(S0, (int, float)) or S0 <= 0:
            raise ValueError('S must be positive number.')
        if not isinstance(mu, (int, float)):
            raise ValueError('mu must be a number.')
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError('Sigma must be positive number.')
        if not isinstance(K, (int, float)) or K <= 0:
            raise ValueError('K must be positive number.')
        if not isinstance(T, (int, float)) or T <= 0:
            raise ValueError('T must be positive number.')
        if not isinstance(r, (int, float)):
            raise ValueError('r must be a number.')
        if not isinstance(N, int) or N <= 0:
            raise ValueError('N must be a positive integer.')
        if not isinstance(M, int) or M <= 0:
            raise ValueError('M must be a positive integer.')
        if not isinstance(delta, (int, float)) or delta <= 0:
            raise ValueError('delta must be an integer or float greater than zero')
        if not (isinstance(confidence_level, float) 
            or confidence_level < 0 or confidence_level > 1):
            raise ValueError('Lower confidence number must be a float within [0,1]!') 
        if not isinstance(samples, int) or samples < 1:
            raise ValueError('Samples must be an integer greater or equal to 1!') 
        if not isinstance(dims, int) or dims < 1:
            raise ValueError('Dimensions must be an integer greater than 1!')
        if not isinstance(option_payoff, str):
            raise ValueError('The option payoff must be a string! Either "call" or "put"')
        if not isinstance(delta, (int, float)) or delta <= 0:
            raise ValueError('Delta must be a positive number')
        if not isinstance(H, (int, float)) or H < 0:
            raise ValueError('Domain range must be a positive number')

    @property
    def opt_style(self):
        """ Not meant for user to change this, attribute given by class """
        return self._opt_style

    def _construct_correlation_matrix(self, correlation):
        d = self.dims
        corr_matrix = np.ones([d,d]) * correlation
        np.fill_diagonal(corr_matrix, 1)
        return corr_matrix

    def _load_configs(self):
        """ Load configs for plotting and generating tables """
        """ Loaded only when calling the plot method """
        with open('configs.yaml', 'r') as config_file:
            return yaml.safe_load(config_file)

    def _get_opt_payoff(self, opt_payoff):
        """ Retrieves the option payoff from user input """
        if not isinstance(opt_payoff, str):
            raise TypeError('Option type should be a string!')
        if opt_payoff.lower() == 'call':
            return 'call'
        elif opt_payoff.lower() == 'put':
            return 'put'
        else:
            raise TypeError('Invalid option type! It should be call or put')

    def _payoff_func(self, S):
        if self.option_payoff == "call":
            payoff = np.maximum(np.max(S, axis=0) - self.K, 0)
        elif self.option_payoff == "put":
            payoff = np.maximum(self.K - np.max(S, axis=0), 0)
        else:
            raise ValueError("Unsupported option payoff type.")
        payoff = np.expand_dims(payoff, axis=1)
        return payoff

    def _generate_stock_paths(self):
        """ S is d-dimensional forward component (Euler Scheme) """
        """ With M simulations and N steps S will be a dxMxN matrix """
        dw = np.random.normal(0, np.sqrt(self.dt), (self.dims, self.M, self.N)) 
        L = np.linalg.cholesky(self.correlation)
        correlated_dw = np.einsum('ij,jkl->ikl', L, dw)
        
        S = np.empty((self.dims, self.M, self.N+1))
        
        for dim in range(self.dims):
            S[dim, :, 0] = self.S0
            
            log_S = np.cumsum(
                (self.mu - 0.5 * self.sigma**2) * self.dt + 
                self.sigma * correlated_dw[dim, :, :], 
                axis=1
            )
            
            S[dim, :, 1:] = self.S0 * np.exp(log_S)
        
        return S, correlated_dw 

    def _generate_hypercube_basis(self, S):
        H = self.H 
        k = self.k
        delta = self.delta
        dims = self.dims
        dim_phi = self.dim_phi
        cube_indices = self.cube_indices
        num_basis_per_cube = self.num_basis_per_cube
        M, N = S.shape[1], S.shape[2]  
        indicators_list = []

        cube_min_vals = [np.maximum(np.array([cube_index[dim] * delta for dim in range(dims)]), 
                                    0) for cube_index in cube_indices]
        cube_max_vals = [cube_min + delta for cube_min in cube_min_vals]

        poly_degrees_list = np.array(list(itertools.product(range(k + 1), repeat=dims)))

        for t in range(N):
            indicators = np.zeros((M, dim_phi))
            for i, (cube_min, cube_max) in enumerate(zip(cube_min_vals, cube_max_vals)):
                indicator = (np.logical_and.reduce((S[:, :, t] >= cube_min[:, np.newaxis]) & 
                        (S[:, :, t] < cube_max[:, np.newaxis]), axis=0))
                if not np.any(indicator):
                    continue

                poly_values = np.ones((dims, M))
                for dim in range(dims):
                    for deg in range(k + 1):
                        poly_values[dim] *= np.where(indicator, S[dim, :, t] ** deg, 1)

                for poly_idx, poly_degrees in enumerate(poly_degrees_list):
                    basis_index = i * num_basis_per_cube + poly_idx
                    polynomial_values = (np.prod([S[dim, :, t] ** poly_degrees[dim] 
                        for dim in range(dims)], axis=0))
                    indicators[:, basis_index] = indicator * polynomial_values
            indicators_list.append(sp.csr_matrix(indicators))
        
        return indicators_list

    def _generate_Z(self, p_li, A, Y_plus, dw):
        """ Generates the conditional expectation for Z at time t_i """
        b_z =  p_li.T @ (Y_plus * dw.T)
        Z = np.zeros((p_li.shape[0], self.dims))
        for d in range(self.dims):
            alpha_z = splinalg.lsqr(A, b_z[:, d].toarray() if sp.issparse(b_z[:, d]) else 
                                    b_z[:, d], atol=0,btol=0,conlim=0)[0]
            Z[:, d] = (p_li @ alpha_z)
        return Z / self.dt
       
    def _generate_Y(self, p_li, A, Y_plus, Z):
        """ Generates the conditional expectation for Y at time t_i """
        opr = (Y_plus + self.dt * self._driver(Y_plus, Z))
        b_y = p_li.T @ opr
        alpha_y = np.expand_dims(splinalg.lsqr(A, b_y, atol=0,btol=0,conlim=0)[0], axis = 1)
        return (p_li @ alpha_y) 

    def _driver(self, Y_plus, Z):
        """ Returns the driver in the BSDE with same interest for lending and borrowing """
        sum_term = 0
        for dim in range(self.dims):
            ((self.mu - self.r)/self.sigma) * Z[:, dim]
        return -(self.r * Y_plus + sum_term) 

    def _bsde_solver(self):
        """ Solves the backward stochastic differential equation to estimate option prices. """
        domain_out_of_range = False
        Y0_samples = np.zeros(self.samples) 
        Z0_samples = np.zeros((self.dims, self.samples))
        
        for k in range(self.samples):
            sample_time = time.time()
            S, dw = self._generate_stock_paths()
            if ((np.max(S) > (np.max(self.S0) + self.H)) or 
                (np.min(S) < (np.min(self.S0) - self.H))):
                domain_out_of_range = True
            Y_plus = self._payoff_func(S[:, :, -1])
            Y = np.zeros(self.M)
            Z = np.zeros((self.M, self.dims))
            p_li_list = self._generate_hypercube_basis(S)
            for i in range(self.N - 1, -1, -1):
                p_li = p_li_list[i] 
                A = p_li.T @ p_li
                Z = self._generate_Z(p_li, A, Y_plus, dw[:, :, i])
                Y = self._generate_Y(p_li, A, Y_plus, Z)
                Y_plus = Y
                del p_li, A
             
            Y0_samples[k] = np.mean(Y_plus)
            Z0_samples[:, k] = np.mean(Z, axis=0)
            sample_finish_time = time.time() - sample_time
            print(f"Done with sample: {k+1} in {sample_finish_time} seconds")
            del p_li_list, S, dw, Z, Y, Y_plus
        if domain_out_of_range:
            print("Domain possibly out of range, consider increasing H!")
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
        """ Plots and show tables by varying N """
        function_name = inspect.currentframe().f_code.co_name
        Y, Z, Y_errors, Z_errors, rows = [], [], [], [], []

        for N_val in N_values:
            self.N = N_val
            self.dt = self.T / self.N
            Y0_array, Z0_array = self._bsde_solver()
            est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)
            est_Z0, std_Z0, CI_Z = self._confidence_interval(Z0_array)

            Y.append(est_Y0)
            Z.append(est_Z0)
            Y_errors.append(std_Y0)
            Z_errors.append(std_Z0)
            rows.append([N_val, est_Y0, std_Y0, CI_Y[0], CI_Y[1], est_Z0, std_Z0, 
                         CI_Z[0], CI_Z[1]])
            print(f'Done with N: {N_val}.')

        self._generate_plot(N_values, Y, Y_errors, function_name, bs_price, nofig)
        self._generate_table(rows, function_name)

    def plot_and_show_table_by_M(self, M_values, nofig=False, bs_price=None):
        """ Plots and show tables by varying M """
        function_name = inspect.currentframe().f_code.co_name
        Y, Z, Y_errors, Z_errors, rows = [], [], [], [], []

        for M_val in M_values:
            self.M = M_val
            Y0_array, Z0_array = self._bsde_solver()
            est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)
            est_Z0, std_Z0, CI_Z = self._confidence_interval(Z0_array)

            Y.append(est_Y0)
            Z.append(est_Z0)
            Y_errors.append(std_Y0)
            Z_errors.append(std_Z0)
            rows.append([M_val, est_Y0, std_Y0, CI_Y[0], CI_Y[1], est_Z0, std_Z0, 
                         CI_Z[0], CI_Z[1]])
            print(f'Done with M_val: {M_val}.')

        self._generate_plot(M_values, Y, Y_errors, function_name, bs_price, nofig)
        self._generate_table(rows, function_name)

    def plot_and_show_table_by_deltas(self, deltas, nofig=False, bs_price=None):
        """ Plots and show tables by varying degrees """
        function_name = inspect.currentframe().f_code.co_name
        Y, Z, Y_errors, Z_errors, rows = [], [], [], [], []

        for delta in deltas:
            self.delta = delta 
            Y0_array, Z0_array = self._bsde_solver()
            est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)
            est_Z0, std_Z0, CI_Z = self._confidence_interval(Z0_array)

            Y.append(est_Y0)
            Z.append(est_Z0)
            Y_errors.append(std_Y0)
            Z_errors.append(std_Z0)
            rows.append([delta, est_Y0, std_Y0, CI_Y[0], CI_Y[1], est_Z0, std_Z0, 
                         CI_Z[0], CI_Z[1]])
            print(f'Done with delta: {delta}.')

        self._generate_plot(deltas, Y, Y_errors, function_name, bs_price, nofig)
        self._generate_table(rows, function_name)

    def plot_and_show_table_by_samples(self, samples, nofig=False, bs_price=None):
        """ Plots and show tables by varying samples """
        function_name = inspect.currentframe().f_code.co_name
        Y, Z, Y_errors, Z_errors, rows = [], [], [], [], []

        for sample in samples:
            self.samples = sample
            Y0_array, Z0_array = self._bsde_solver()
            est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)
            est_Z0, std_Z0, CI_Z = self._confidence_interval(Z0_array)

            Y.append(est_Y0)
            Z.append(est_Z0)
            Y_errors.append(std_Y0)
            Z_errors.append(std_Z0)
            rows.append([sample, est_Y0, std_Y0, CI_Y[0], CI_Y[1], est_Z0, std_Z0, 
                         CI_Z[0], CI_Z[1]])
            print(f'Done with sample: {sample}.')

        self._generate_plot(samples, Y, Y_errors, function_name, bs_price, nofig)
        self._generate_table(rows, function_name)

    def _generate_plot(self, x_values, y_values, y_errors, function_name, bs_price, nofig):
        """ Generates the plot with the configs in the yaml file """
        configs = self._load_configs()
        plot_config = configs['plot_config'][function_name]

        plt.errorbar(x_values, y_values, yerr=y_errors, fmt='-o', capsize=5, capthick=2, ecolor='red', label='Option Price')

        if bs_price is not None:
            plt.plot(x_values, [bs_price] * len(x_values), 'k-', label='Black-Scholes Price')
            plt.scatter(x_values, [bs_price] * len(x_values), color='black', marker='_', s=100)

        plt.xlabel(plot_config.get('xlabel'))
        plt.ylabel(plot_config.get('ylabel'))
        plt.title(plot_config.get('title').format(opt_style=self._opt_style.capitalize(), 
                                                  option_payoff=self.option_payoff.capitalize()))
        plt.legend(loc=plot_config.get('legend_location', 'best'))
        plt.grid(plot_config.get('grid', True))

        plot_directory = os.path.join(configs['general_settings']['plot_directory'], 
                                      function_name)
        os.makedirs(plot_directory, exist_ok=True)

        plot_name = plot_config['plot_name_template'].format(
            opt_style=self.opt_style,
            opt_payoff=self.option_payoff,
            N=self.N,
            M=self.M,
            delta=self.delta,
            min_value=min(x_values),
            max_value=max(x_values)
        )

        plot_path = os.path.join(plot_directory, plot_name)
        plt.savefig(plot_path)
        if not nofig:
            plt.show()
        plt.close()

    def _generate_table(self, rows, function_name):
        """ Generates the table with the configs in the yaml file """
        configs = self._load_configs()
        table_directory = os.path.join(configs['general_settings']['table_directory'], 
                                       function_name)
        os.makedirs(table_directory, exist_ok=True)

        first_elements = [row[0] for row in rows]
        
        columns = (
            [function_name.split('_')[-1].capitalize(), 
             'Estimated Price Y0', 'Std. Deviation Y0',
            'CI Lower Bound Y0', 'CI Upper Bound Y0','Estimated Volatility Z0', 
            'Std. Deviation Z0', 'CI Lower Bound Z0', 'CI Upper Bound Z0']
        )

        table_name = configs['plot_config'][function_name]['table_name_template'].format(
            opt_style=self.opt_style,
            opt_payoff=self.option_payoff,
            N=self.N,
            M=self.M,
            delta=self.delta,
            min_value=min(first_elements),
            max_value=max(first_elements)
        )

        table_path = os.path.join(table_directory, table_name)
        df = pd.DataFrame(rows, columns=columns)
        df_rounded = df.round(4)
        df_rounded.to_csv(table_path, index=False)
        print(f"Table saved to {table_path}")

    def solve(self):
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
                f"  rho={self.correlation},\n"
                f"  T={self.T},\n"
                f"  N={self.N},\n"
                f"  M={self.M},\n"
                f"  confidence_level={self.confidence_level},\n"
                f"  samples={self.samples},\n"
                f"  mu={self.mu},\n"
                f"  option_payoff={self.option_payoff},\n"
                f"  domain range={self.H},\n"
                f"  delta={self.delta},\n)"
        )

class BSDEOptionPricingAmerican(BSDEOptionPricingEuropean):
    def __init__(self, S0, mu, sigma, correlation, K, r, T, N, M, confidence_level = 0.025, 
                 samples = 50, dims = 1, option_payoff = "call", H = 60, delta = 1, k=0): 
        super().__init__(S0, mu, sigma, correlation, K, r, T, N, M, confidence_level, 
                         samples, dims, option_payoff, H, delta, k)
        self._opt_style = 'american'

    def _bsde_solver(self):
        """ Solves the backward stochastic differential equation to estimate option prices. """
        domain_out_of_range = False
        Y0_samples = np.zeros(self.samples)
        Z0_samples = np.zeros((self.dims, self.samples))

        for k in range(self.samples):
            sample_time = time.time()
            S, dw = self._generate_stock_paths()
            if ((np.max(S) > (np.max(self.S0) + self.H)) or 
                (np.min(S) < (np.min(self.S0) - self.H))):
                domain_out_of_range = True
            Y_plus = self._payoff_func(S[:, :, -1])
            Y = np.zeros(self.M)
            Z = np.zeros((self.M, self.dims))
            p_li_list = self._generate_hypercube_basis(S)
            for i in range(self.N - 1, -1, -1):
                p_li = p_li_list[i] 
                A = p_li.T @ p_li
                Z = self._generate_Z(p_li, A, Y_plus, dw[:, :, i])
                Y = self._generate_Y(p_li, A, Y_plus, Z)
                exercise_values = self._payoff_func(S[:, :, i])
                Y_plus = np.maximum(Y, exercise_values)
                del p_li, A
             
            Y0_samples[k] = np.mean(Y_plus)
            Z0_samples[:, k] = np.mean(Z, axis=0)
            sample_finish_time = time.time() - sample_time
            print(f"Done with sample: {k+1} in {sample_finish_time} seconds")
            del p_li_list, S, dw, Z, Y, Y_plus
        if domain_out_of_range:
            print("Domain possibly out of range, consider increasing H!")
        return Y0_samples, Z0_samples

    def __repr__(self):
        base_repr = super().__repr__()
        return base_repr[:-1] + f" opt_style='{self._opt_style}'\n)"

class BSDEOptionPricingEuropeanSpread(BSDEOptionPricingEuropean):
    def __init__(self, S0, mu, sigma, correlation, K, r, T, N, M, 
                 confidence_level, samples, dims, option_payoff, domain, delta, k, K2, R):
        super().__init__(S0, mu, sigma, correlation, K, r, T, N, M, 
                         confidence_level, samples, dims, option_payoff, domain, delta, k) 
        if K2 is None:
            raise ValueError('K2 must be specified for a spread option.')
        if K2 <= K:
            raise ValueError('K2 must be greater than K for a spread to make sense.')
        if R is None:
            raise ValueError('No second interest rate given.')
        self.K2 = K2 
        self.R = R 
        self._opt_style = 'europeanspread'

    def _payoff_func(self, S):
        if self.option_payoff == "call":
            payoff = (np.maximum(np.max(S, axis=0) - self.K, 0) - 
                    2*np.maximum(np.max(S, axis=0) - self.K2, 0))
        elif self.option_payoff == "put":
            payoff = (np.maximum(self.K - np.max(S, axis=0) , 0) - 
                    2*np.maximum(self.K2 - np.max(S, axis=0) , 0))

        else:
            raise ValueError(f"Unsupported option payoff type {self.option_payoff}.")
        payoff = np.expand_dims(payoff, axis=1)
        return payoff

    def _driver(self, Y_plus, Z):
        """ Returns the driver in the BSDE for different interest rates for borrowing and lending """
        Z_sum = np.sum(Z, axis=1, keepdims=True)
        
        term1 = Y_plus* self.R
        term2 = Z_sum * self.lamb
        term3 = (self.R - self.r) * np.minimum(Y_plus - Z_sum/self.sigma, 0)

        result = - term1 - term2 + term3
        return result


    def _bsde_solver(self):
        """ Solves the backward stochastic differential equation to estimate option prices. """
        domain_out_of_range = False
        Y0_samples = np.zeros(self.samples) 
        Z0_samples = np.zeros((self.dims, self.samples))
        
        for k in range(self.samples):
            sample_time = time.time()
            S, dw = self._generate_stock_paths()
            if ((np.max(S) > (np.max(self.S0) + self.H)) or 
                (np.min(S) < (np.min(self.S0) - self.H))):
                domain_out_of_range = True
            Y_plus = self._payoff_func(S[:, :, -1])
            Y = np.zeros(self.M)
            Z = np.zeros((self.M, self.dims))
            p_li_list = self._generate_hypercube_basis(S)
            for i in range(self.N - 1, -1, -1):
                p_li = p_li_list[i] 
                A = p_li.T @ p_li
                Z = self._generate_Z(p_li, A, Y_plus, dw[:, :, i])
                Y = self._generate_Y(p_li, A, Y_plus, Z)
                Y_plus = Y
                del p_li, A
             
            Y0_samples[k] = np.mean(Y_plus)
            Z0_samples[:, k] = np.mean(Z, axis=0)
            sample_finish_time = time.time() - sample_time
            print(f"Done with sample: {k+1} in {sample_finish_time} seconds")
            del p_li_list, S, dw, Z, Y, Y_plus
        if domain_out_of_range:
            print("Domain possibly out of range, consider increasing H!")
        return Y0_samples, Z0_samples   

    def __repr__(self):
        base_repr = super().__repr__()
        return base_repr[:-1] + f"\n  opt_style='{self._opt_style}'\n  K2='{self.K2}'\n  R='{self.R}'\n)"

class BSDEOptionPricingAmericanSpread(BSDEOptionPricingEuropeanSpread):
    def __init__(self, S0, mu, sigma, correlation, K, r, T, N, M, 
                 confidence_level, samples, dims, option_payoff, domain, delta, k, K2, R):
        super().__init__(S0, mu, sigma, correlation, K, r, T, N, M, 
                         confidence_level, samples, dims, option_payoff, domain, delta, k, K2, R)
        self._opt_style = 'americanspread'

    def _bsde_solver(self):
        """ Solves the backward stochastic differential equation to estimate option prices. """
        domain_out_of_range = False
        Y0_samples = np.zeros(self.samples)
        Z0_samples = np.zeros((self.dims, self.samples))

        for k in range(self.samples):
            sample_time = time.time()
            S, dw = self._generate_stock_paths()
            if ((np.max(S) > (np.max(self.S0) + self.H)) or 
                (np.min(S) < (np.min(self.S0) - self.H))):
                domain_out_of_range = True
            Y_plus = self._payoff_func(S[:, :, -1])
            Y = np.zeros(self.M)
            Z = np.zeros((self.M, self.dims))
            p_li_list = self._generate_hypercube_basis(S)
            for i in range(self.N - 1, -1, -1):
                p_li = p_li_list[i] 
                A = p_li.T @ p_li
                Z = self._generate_Z(p_li, A, Y_plus, dw[:, :, i])
                Y = self._generate_Y(p_li, A, Y_plus, Z)
                exercise_values = self._payoff_func(S[:, :, i])
                Y_plus = np.maximum(Y, exercise_values)
                del p_li, A
             
            Y0_samples[k] = np.mean(Y_plus)
            Z0_samples[:, k] = np.mean(Z, axis=0)
            sample_finish_time = time.time() - sample_time
            print(f"Done with sample: {k+1} in {sample_finish_time} seconds")
            del p_li_list, S, dw, Z, Y, Y_plus
        if domain_out_of_range:
            print("Domain possibly out of range, consider increasing H!")
        return Y0_samples, Z0_samples

    def __repr__(self):
        base_repr = super().__repr__()
        return base_repr[:-1] + f"\n  opt_style='{self._opt_style}'\n  K2='{self.K2}'\n  R='{self.R}'\n)"


