import time
import sys
import itertools
import yaml
import inspect
import pandas as pd
import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class BSDEOptionPricingEuropean:
    """ 
    Constructs an instance to price a vanilla European option.
    Supports both put and call options with Laguerre or hybercube regression bases.

    Attributes:
        S0 (float): Initial stock price.
        K (float): Strike price of the option.
        r (float): Risk-free rate.
        sigma (float): Volatility of the stock.
        T (float): Time to maturity of the option.
        N (int): Number of time steps in the discretization.
        M (int): Number of simulated paths.
        confidence_level (float): The alpha value for the confidence interval calculation.
        samples (int): Number of BSDE samples to estimate the price.
        mu (float): Drift rate of the stock, defaults to r if not provided.
        option_payoff (str): Type of the option ('call' or 'put').
        domain (list[int]): The domain which the hybercubes cover.
        delta (float): The length of the hybercubes.
    """
    def __init__(self, S0, mu, sigma, K, r, T, N, M,  confidence_level = 0.025, 
                 samples = 50, dims = 1, option_payoff="call", domain=[40,180], delta= 1, weights=None):
        self._validation_check(S0, mu, sigma, K, r, T, N, M, 
                         confidence_level, samples, dims, 
                         option_payoff, domain, delta, weights) 
        self.S0 = self._initialize_array(S0, dims) 
        self.mu = self._initialize_array(mu, dims)
        self.sigma = self._initialize_matrix(sigma, dims, dims) 
        self.K = np.float32(K)
        self.r = np.float32(r)
        self.T = np.float32(T)
        self.N = np.int32(N)
        self.M = np.int32(M)
        self.dt = np.float32(T/N)
        self.confidence_level = np.float32(confidence_level)
        self.samples = np.int32(samples)
        self.dims = np.int32(dims)
        self.option_payoff = option_payoff
        self.domain = domain
        self.delta = delta
        self.k = 0
        self.weights = np.ones(self.dims) / self.dims if weights == None else weights
        self.sum_driver = self._build_sum_driver()

    def _validation_check(self, S0, mu, sigma, K, r, T, N, M,
                          confidence_level, samples, dims,
                          option_payoff, domain, delta, weights):
        if dims > 1:
            if not isinstance(S0, np.ndarray):
                raise ValueError('S0 must be a numpy array when dims > 1.')
            if not isinstance(mu, np.ndarray):
                raise ValueError('mu must be a numpy array when dims > 1.')
            if not isinstance(sigma, np.ndarray):
                raise ValueError('sigma must be a numpy array when dims > 1.')

        else:
            if not isinstance(S0, (int, float)):
                raise ValueError('S0 must be an integer or float when dims = 1.')
            if not isinstance(mu, (int, float)):
                raise ValueError('mu must be an integer or float when dims = 1.')
            if not isinstance(sigma, (int, float)):
                raise ValueError('sigma must be an integer or float when dims = 1.')
        if not isinstance(K, (int, float)) or K <= 0:
            raise ValueError('K must be positive.')
        if not isinstance(T, (int, float)) or T <= 0:
            raise ValueError('T must be positive.')
        if not isinstance(r, (int, float)):
            raise ValueError('r must be an integer or float.')
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
        if not isinstance(domain, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in domain):
            raise ValueError('Domain must be a list of tuples, each representing the boundaries of a hypercube!')
        if not len(domain) == dims:
            raise ValueError('The domain must match the amount of dimensions')
        return
        if not isinstance(weights, list) or not all(isinstance(item, (int, float)) and 0 <= item <= 1 for item in weights):
            raise ValueError('Weights must be a list of integers or floats between 0 and 1')
        if not len(weights) == dims:
            raise ValueError('Every stock must be given a weight!')


    def _is_valid(self, param, check_positive=False):
        if isinstance(param, (int, float)):
            return not check_positive or param > 0
        if isinstance(param, list) and all(isinstance(item, (int, float)) for item in param):
            return not check_positive or all(item > 0 for item in param)
        return False

    def _initialize_array(self, values, d):
        if d > 1:
            if isinstance(values, np.ndarray):
                if values.shape == (d,):
                    return values
                else:
                    raise ValueError(f'Array must be of shape ({d},)')
            else:
                raise ValueError('Values must be a numpy array when dims > 1.')
        else:
            if isinstance(values, (int, float)):
                return np.full(d, values)
            elif isinstance(values, list):
                return np.array(values)
            else:
                raise ValueError('Values must be an integer or float when dims = 1.')

    def _initialize_matrix(self, values, rows, cols):
        if rows > 1 and cols > 1:
            if isinstance(values, np.ndarray):
                if values.shape == (rows, cols):
                    return values
                else:
                    raise ValueError(f'Sigma must be of shape ({rows}, {cols})')
            else:
                raise ValueError('Sigma must be a numpy array when dims > 1.')
        else:
            if isinstance(values, (int, float)):
                return np.full((rows, cols), values)
            elif isinstance(values, list):
                return np.array(values).reshape((rows, cols))
            else:
                raise ValueError('Sigma must be an integer or float when dims = 1.')

    @property
    def opt_style(self):
        """ Not meant for user to change this, attribute given by class """
        return self._opt_style

    def _load_configs(self):
        """ Load configs for plotting and generating tables """
        """ Loaded only when calling the plot method """
        with open('configs.yaml', 'r') as config_file:
            return yaml.safe_load(config_file)

    def _build_sum_driver(self):
        return np.sum((self.mu - self.r) / self.sigma.diagonal())

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
        return np.mean(payoff)


    def _generate_stock_paths(self):
        """ S is d-dimensinoal forward component (Euler Scheme) """
        """ With M simulations and N steps S will be a dxMxN matrix """
        dw = np.random.normal(0, np.sqrt(self.dt), (self.dims, self.M, self.N-1))
        S = np.empty((self.dims, self.M, self.N))
        for dim in range(self.dims):
            S[dim, :, 0] = self.S0[dim]
            log_S = np.cumsum((self.mu[dim] - 0.5 * self.sigma[dim][dim]**2) * self.dt + self.sigma[dim][dim] * dw[dim, :, :], axis=1) 
            S[dim, :, 1:] = self.S0[dim] * np.exp(log_S)
        
        return S, dw

    def _generate_hypercube_basis(self, S):
        H = 70
        num_cubes_per_dim = int(2 * H // self.delta)
        num_basis_per_cube = (self.k + 1) ** self.dims

        M = S.shape[1]
        dim_phi = num_cubes_per_dim ** self.dims * num_basis_per_cube
        indicators = np.zeros((M, dim_phi))

        # Generate cube ranges for each dimension
        cube_ranges = [range(int((self.S0[dim] - H) // self.delta), int((self.S0[dim] + H) // self.delta)) for dim in range(self.dims)]
        cube_indices = list(itertools.product(*cube_ranges))

        for i, cube_index in enumerate(cube_indices):
            # Calculate cube boundaries
            cube_min = np.array([cube_index[dim] * self.delta for dim in range(self.dims)])
            cube_max = cube_min + self.delta
            
            # Calculate indicator using broadcasting
            indicator = np.all((S >= cube_min[:, np.newaxis]) & (S < cube_max[:, np.newaxis]), axis=0)

            if not np.any(indicator):
                continue

            # Compute polynomial values for the current hypercube
            for poly_idx, poly_degrees in enumerate(itertools.product(range(self.k + 1), repeat=self.dims)):
                basis_index = i * num_basis_per_cube + poly_idx
                polynomial_values = np.prod([S[dim, :] ** poly_degrees[dim] for dim in range(self.dims)], axis=0)
                indicators[:, basis_index] = indicator * polynomial_values
        #indices_with_ones = np.argwhere(indicators == 1)
        #print("Indices where indicators are 1:\n", indices_with_ones)

        return indicators

    def _generate_Z(self, p_li, A, Y_plus, dw):
        """ Generates the conditional expectation for Z at time t_i """
        b_z =  p_li.T @ (Y_plus * dw.T)
        alpha_z, _, _, _ = np.linalg.lstsq(A, b_z, rcond=None)
        return (p_li @ alpha_z) / self.dt

        
    def _generate_Y(self, p_li, A, Y_plus, Z):
        """ Generates the conditional expectation for Y at time t_i """
        opr = (Y_plus + self.dt * self._driver(Y_plus, Z))
        b_y = p_li.T @ opr
        alpha_y, _, _, _ = np.linalg.lstsq(A, b_y, rcond=None) 
        return (p_li @ alpha_y)

    def _driver(self, Y_plus, Z):
        """ Returns the driver in the BSDE with same interest for lending and borrowing """
        return -(Z * self.sum_driver + self.r * Y_plus)


    def _bsde_solver(self):
        """ Solves the backward stochastic differential equation to estimate option prices. """
        Y0_samples = np.zeros(self.samples)
        Z0_samples = np.zeros((self.dims, self.samples))

        epsilon = 1e-6

        for k in range(self.samples):
            S, dw = self._generate_stock_paths()
            Y_plus = self._payoff_func(S[:, :, -1])
            Y = 0
            Z = np.zeros(self.dims)
            for i in range(self.N - 1, -1, -1):
                p_li = self._generate_hypercube_basis(S[:, :, i])
                A = p_li.T @ p_li
                A_reg = A + epsilon * np.eye(A.shape[0])
                Z = self._generate_Z(p_li, A, Y_plus, dw[:, :, i - 1])
                Y = self._generate_Y(p_li, A, Y_plus, Z)
                Y_plus = np.mean(Y)
             
            Y0_samples[k] = Y_plus
            Z0_samples[:, k] = np.mean(Z, axis=0)

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
            rows.append([N_val, est_Y0, std_Y0, CI_Y[0], CI_Y[1], est_Z0, std_Z0, CI_Z[0], CI_Z[1]])
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
            rows.append([M_val, est_Y0, std_Y0, CI_Y[0], CI_Y[1], est_Z0, std_Z0, CI_Z[0], CI_Z[1]])
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
            rows.append([delta, est_Y0, std_Y0, CI_Y[0], CI_Y[1], est_Z0, std_Z0, CI_Z[0], CI_Z[1]])
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
            rows.append([sample, est_Y0, std_Y0, CI_Y[0], CI_Y[1], est_Z0, std_Z0, CI_Z[0], CI_Z[1]])
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
        plt.title(plot_config.get('title').format(opt_style=self._opt_style.capitalize(), option_payoff=self.option_payoff.capitalize()))
        plt.legend(loc=plot_config.get('legend_location', 'best'))
        plt.grid(plot_config.get('grid', True))

        plot_directory = os.path.join(configs['general_settings']['plot_directory'], function_name)
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
        table_directory = os.path.join(configs['general_settings']['table_directory'], function_name)
        os.makedirs(table_directory, exist_ok=True)

        first_elements = [row[0] for row in rows]
        
        columns = (
            [function_name.split('_')[-1].capitalize(), 'Estimated Price Y0', 'Std. Deviation Y0',
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
                f"  domain={self.domain},\n"
                f"  delta={self.delta},\n)"
        )


class BSDEOptionPricingAmerican(BSDEOptionPricingEuropean):
    def __init__(self, S0, mu, sigma, K, r, T, N, M, confidence_level, samples, 
                 dims, option_payoff, domain, delta): 
        super().__init__(S0, mu, sigma, K, r, T, N, M, confidence_level, samples, dims, 
                         option_payoff, domain, delta)
        self._opt_style = 'american'

    def _bsde_solver(self):
        """ Solves the backward stochastic differential equation to estimate option prices. """
        Y0_samples = np.zeros(self.samples)
        Z0_samples = np.zeros(self.samples)

        for k in range(self.samples):
            S, dW = self._generate_stock_paths()
            Y = np.zeros((self.M, self.N + 1))
            Z = np.zeros((self.M, self.N))

            exercise_values = self._payoff_func(S)
            Y[:, -1] = exercise_values[:, -1] 

            for i in range(self.N - 1, -1, -1):
                p_li = self._generate_hypercube_basis(S[:, [i]])
                A = p_li.T @ p_li
                Z[:, i] = self._generate_Z(p_li, A, Y[:, i+1], dW[:, i])
                Y[:, i] = self._generate_Y(p_li, A, Y[:, i+1], Z[:, i])
                Y[:, i] = np.maximum(Y[:, i], exercise_values[:, i])

            Y0_samples[k] = np.mean(Y[:, 0])
            Z0_samples[k] = np.mean(Z[:, 0])

        return Y0_samples, Z0_samples

    def __repr__(self):
        base_repr = super().__repr__()
        return base_repr[:-1] + f" opt_style='{self._opt_style}'\n)"


class BSDEOptionPricingEuropeanSpread(BSDEOptionPricingEuropean):
    def __init__(self, S0, mu, sigma, K, r, T, N, M, confidence_level, samples, dims, option_payoff, domain, delta, K2, R):
        super().__init__(S0, mu, sigma, K, r, T, N, M, confidence_level, samples, dims, option_payoff, domain, delta) 
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
            return (np.maximum(S - self.K, 0) - 2*np.maximum(S - self.K2, 0))
        else:
            raise ValueError(f"Invalid option type: {self.option_payoff}. Supported types are 'call'.")

    def _driver(self, Y_plus, Z):
        """ Returns the driver in the BSDE for different interest rates for borrowing and lending """
        return -(Y_plus*self.R + Z * self.lamb - (self.R-self.r)*np.minimum(Y_plus - Z/self.sigma, 0)) 

    def _bsde_solver(self):
        """ Solves the backward stochastic differential equation to estimate option prices. """
        Y0_samples = np.zeros(self.samples)
        Z0_samples = np.zeros(self.samples)

        for k in range(self.samples):
            S, dW = self._generate_stock_paths()
            Y = np.zeros((self.M, self.N + 1))
            Z = np.zeros((self.M, self.N))
            Y[:, -1] = self._payoff_func(S[:, -1])

            for i in range(self.N - 1, -1, -1): 
                p_li = self._generate_hypercube_basis(S[:, [i]])
                A = p_li.T @ p_li
                Z[:, i] = self._generate_Z(p_li, A, Y[:, i+1], dW[:, i])
                Y[:, i] = self._generate_Y(p_li, A, Y[:, i+1], Z[:, i])

            Y0_samples[k] = np.mean(Y[:, 0])
            Z0_samples[k] = np.mean(Z[:, 0])

        return Y0_samples, Z0_samples

    def __repr__(self):
        base_repr = super().__repr__()
        return base_repr[:-1] + f"\n  opt_style='{self._opt_style}'\n  K2='{self.K2}'\n  R='{self.R}'\n)"


class BSDEOptionPricingAmericanSpread(BSDEOptionPricingEuropeanSpread):
    def __init__(self, S0, mu, sigma, K, r, T, N, M, confidence_level, samples, dims, option_payoff, domain, delta, K2, R):
        super().__init__(S0, mu, sigma, K, r, T, N, M, confidence_level, samples, dims, option_payoff, domain, delta, K2, R)
        self._opt_style = 'americanspread'

    def _bsde_solver(self):
        """ Solves the backward stochastic differential equation to estimate option prices. """
        Y0_samples = np.zeros(self.samples)
        Z0_samples = np.zeros(self.samples)

        for k in range(self.samples):
            S, dW = self._generate_stock_paths()
            Y = np.zeros((self.M, self.N + 1))
            Z = np.zeros((self.M, self.N))

            exercise_values = self._payoff_func(S)
            Y[:, -1] = self._payoff_func(S[:, -1])

            for i in range(self.N - 1, -1, -1): 
                p_li = self._generate_hypercube_basis(S[:, [i]])
                A = p_li.T @ p_li
                Z[:, i] = self._generate_Z(p_li, A, Y[:, i+1], dW[:, i])
                Y[:, i] = self._generate_Y(p_li, A, Y[:, i+1], Z[:, i])
                Y[:, i] = np.maximum(Y[:, i], exercise_values[:, i])

            Y0_samples[k] = np.mean(Y[:, 0])
            Z0_samples[k] = np.mean(Z[:, 0])

        return Y0_samples, Z0_samples
