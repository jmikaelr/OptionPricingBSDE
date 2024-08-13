import time
import psutil
from joblib import Parallel, delayed
import itertools
import yaml
import pandas as pd
import os
import numpy as np
from scipy.stats import norm
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsqr
import inspect
from threading import Thread
import torch

class BSDEOptionPricingEuropean:
    # Constructor: Initializes parameters for the BSDE solver for European options
    def __init__(self, S0, mu, sigma, correlation, K, r, div, T, N, M, confidence_level=0.025, 
                 samples=50, dims=1, option_payoff="call", H=60, delta=1, k=0):
        # Validates the input parameters to ensure they are within expected ranges
        self._validation_check(S0, mu, sigma, K, r, div, T, N, M, 
                         confidence_level, samples, dims, 
                         option_payoff, H, delta)
        # Assigning input parameters to instance variables
        self.S0 = np.float64(S0) 
        self.mu = np.float64(mu) 
        self.sigma = np.float64(sigma)
        self.K = np.float64(K)
        self.r = np.float64(r)
        self.dividend = np.float64(div)
        self.T = np.float64(T)
        self.N = np.uint32(N)  # Number of time steps
        self.M = np.uint32(M)  # Number of Monte Carlo simulations
        self.dt = np.float64(T/N)  # Time step size
        self.confidence_level = np.float32(confidence_level)
        self.samples = np.uint16(samples)  # Number of samples for estimating expectations
        self.dims = np.uint8(dims)  # Dimensionality of the problem (e.g., number of underlying assets)
        self.option_payoff = option_payoff  # Type of option ("call" or "put")
        self.H = np.uint16(H)  # Domain size for hypercubes
        self.delta = np.uint8(delta)  # Size of hypercubes
        self.num_cubes_per_dim = int(2 * H // delta)  # Number of hypercubes in each dimension
        self.num_basis_per_cube = (k + 1) ** dims  # Number of basis functions per hypercube
        self.dim_phi = self.num_cubes_per_dim ** dims * self.num_basis_per_cube  # Total number of basis functions
        # Determine ranges for hypercubes based on the initial stock price and domain size
        self.cube_ranges = ([range(int((S0 - H) // delta), 
                                   int((S0 + H) // delta)) for _ in range(dims)])
        self.cube_indices = list(itertools.product(*self.cube_ranges))
        self.k = np.uint8(k)  # Degree of polynomial basis functions
        self.lamb = np.float64((mu - r) / sigma)  # Drift term for the BSDE driver
        # Correlation matrix for multidimensional problems
        self.correlation = np.float64(np.full((1, 1), 1) if self.dims == 1 
            else self._construct_correlation_matrix(correlation))
        self._opt_style = 'european'  # Option style: European option

    # Validates input parameters to ensure they are within expected ranges
    def _validation_check(self, S0, mu, sigma, K, r, div, T, N, M,
                          confidence_level, samples, dims,
                          option_payoff, H, delta):
        # Perform various checks to ensure input values are valid
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
        if not isinstance(div, (int, float)):
            raise ValueError('dividend must be a number.')
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

    # Property to access the option style
    @property
    def opt_style(self):
        """ Not meant for user to change this, attribute given by class """
        return self._opt_style

    # Constructs the correlation matrix for multiple underlying assets
    def _construct_correlation_matrix(self, correlation):
        d = self.dims
        corr_matrix = np.ones([d,d]) * correlation
        np.fill_diagonal(corr_matrix, 1)
        return corr_matrix

    # Loads configuration settings for plots and tables from a YAML file
    def _load_configs(self):
        """ Load configs for plotting and generating tables """
        """ Loaded only when calling the plot method """
        with open('configs.yaml', 'r') as config_file:
            return yaml.safe_load(config_file)

    # Determines the option payoff type ("call" or "put") based on user input
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

    # Defines the payoff function for the option, used in the BSDE
    def _payoff_func(self, S):
        K = self.K
        if self.option_payoff == "call":
            payoff = np.maximum(np.max(S, axis=0) - K, 0)
        elif self.option_payoff == "put":
            payoff = np.maximum(K - np.max(S, axis=0), 0)
        else:
            raise ValueError("Unsupported option payoff type.")
        payoff = np.expand_dims(payoff, axis=1)
        return payoff

    # Generates simulated stock price paths using the Euler scheme for Geometric Brownian Motion (GBM)
    def _generate_stock_paths(self):
        """ In order to generate same samples as in the optimal stopping code utilising DNN """
        d = self.dims
        M = self.M
        N = self.N
        T = self.T
        r = self.r
        div = self.dividend
        sigma = self.sigma
        S0 = self.S0
        """ Simulates the stock price that follows a GBM with Euler scheme """
        correlation = self.correlation
        dt = T/N
        # Create the correlation matrix and simulate correlated Brownian motions
        q = np.ones([d, d], dtype=np.float32) * correlation
        np.fill_diagonal(q, 1.)
        q = torch.tensor(q, dtype = torch.float32).transpose(0,1)
        l = torch.linalg.cholesky(q)
        dw = torch.matmul(torch.randn(M * N, d) * np.sqrt(T / N), l)
        dw = dw.view(M, N, d).permute(0, 2, 1)
        w = torch.cumsum(dw, dim=2)  # Cumulative sum to simulate the Brownian paths
        t = torch.tensor(np.linspace(start=dt, stop=T , num=N, endpoint=True), dtype=torch.float32)
        # Stock prices are simulated using the Euler-Maruyama method
        s = torch.exp((r - div - sigma ** 2 / 2) * t + sigma * w) * S0
        s = s.permute(1,0,2)
        dw = dw.permute(1,0,2)
        
        return s.numpy(), dw.numpy()

    # Generates the basis functions for each hypercube in the domain for regression
    def _generate_hypercube_basis(self, S):
        k = self.k
        delta = self.delta
        dim_phi = self.dim_phi
        cube_indices = self.cube_indices
        num_basis_per_cube = self.num_basis_per_cube
        dims, M, N = S.shape
        indicators_list = []

        # Calculate the minimum and maximum values for each cube
        cube_min_vals = [np.maximum(np.array([cube_index[dim] * delta for dim in range(dims)]), 
                                    0) for cube_index in cube_indices]
        cube_max_vals = [cube_min + delta for cube_min in cube_min_vals]

        # Generate polynomial degrees for the basis functions
        poly_degrees_list = np.array(list(itertools.product(range(k + 1), repeat=dims)))

        for t in range(N):
            indicators = np.zeros((M, dim_phi))
            for i, (cube_min, cube_max) in enumerate(zip(cube_min_vals, cube_max_vals)):
                # Determine which stock paths fall within each cube
                indicator = (np.logical_and.reduce((S[:, :, t] >= cube_min[:, np.newaxis]) & 
                        (S[:, :, t] < cube_max[:, np.newaxis]), axis=0))
                if not np.any(indicator):
                    continue

                poly_values = np.ones((dims, M))
                for dim in range(dims):
                    for deg in range(k + 1):
                        poly_values[dim] *= np.where(indicator, S[dim, :, t] ** deg, 1)

                # Generate the basis functions for each polynomial degree
                for poly_idx, poly_degrees in enumerate(poly_degrees_list):
                    basis_index = i * num_basis_per_cube + poly_idx
                    polynomial_values = (np.prod([S[dim, :, t] ** poly_degrees[dim] 
                        for dim in range(dims)], axis=0))
                    indicators[:, basis_index] = indicator * polynomial_values

            indicators_list.append(sp.coo_matrix(indicators))  # Store the sparse matrix representation
        return indicators_list

    # Solves for the Z component of the BSDE at time t_i
    def _generate_Z(self, p_li, A, Y_plus, dw):
        """ Generates the conditional expectation for Z at time t_i """
        b_z =  p_li.T @ (Y_plus * dw.T)  # Regression target for Z
        alpha_z = self._solve_lsqr(A, b_z)  # Solve the least squares problem to find Z
        Z = (p_li @ alpha_z)  # Compute the regression estimate of Z
        return Z / self.dt  # Scale by the time step

    # Solves for the Y component of the BSDE at time t_i
    def _generate_Y(self, p_li, A, Y_plus, Z):
        """ Generates the conditional expectation for Y at time t_i """
        opr = (Y_plus + self.dt * self._driver(Y_plus, Z))  # BSDE driver term
        b_y = p_li.T @ opr  # Regression target for Y
        alpha_y = self._solve_lsqr(A, b_y)  # Solve the least squares problem to find Y
        return (p_li @ alpha_y)  # Compute the regression estimate of Y

    # Solves the linear system using least squares for sparse matrices
    def _solve_lsqr(self, A, b_y):
        """ Solve least-squares problem for each column of b_y individually """
        results = []
        for i in range(b_y.shape[1]):
            b_col = b_y[:, i]
            alpha_col = lsqr(A, b_col, atol=0, btol=0, conlim=0)[0]
            results.append(alpha_col)
        return np.stack(results, axis=1)

    # Defines the driver function in the BSDE, typically related to the risk-neutral measure
    def _driver(self, Y_plus, Z):
        """ Returns the driver in the BSDE with same interest for lending and borrowing """
        driver = -(self.r * Y_plus + self.lamb * Z)  # Standard BSDE driver with same lending and borrowing rate
        return driver 

    # Task executed for each sample to estimate Y_0 and Z_0 using the BSDE solver
    def _sample_task(self, sample_id):
        sample_time = time.time()
        S, dw = self._generate_stock_paths()  # Simulate stock price paths
        domain_out_of_range = ((np.max(S) > (self.S0 + self.H)) or 
                               (np.min(S) < (self.S0 - self.H)))  # Check if any stock path falls outside the domain
        Y_plus = self._payoff_func(S[:, :, -1])  # Calculate the payoff at maturity
        Y = np.zeros(self.M)
        Z = np.zeros((self.M, self.dims))
        p_li_list = self._generate_hypercube_basis(S)  # Generate the basis functions for regression
        for i in range(self.N - 1, -1, -1):
            p_li = p_li_list[i] 
            A = p_li.T @ p_li  # Compute the Gram matrix for least squares
            Z = self._generate_Z(p_li, A, Y_plus, dw[:, :, i])  # Estimate Z using regression
            Y = self._generate_Y(p_li, A, Y_plus, Z)  # Estimate Y using regression
            Y_plus = Y  # Update Y_plus for the next step
            del p_li, A  # Clean up memory

        Y0_sample = np.mean(Y_plus)  # Estimate Y_0 as the mean of the last Y_plus
        Z0_sample = np.mean(Z, axis=0)  # Estimate Z_0 as the mean of the last Z
        sample_finish_time = time.time() - sample_time
        print(f"Done with sample: {sample_id+1} in {sample_finish_time} seconds")
        return Y0_sample, Z0_sample, domain_out_of_range

    # Solves the BSDE by averaging results from multiple samples
    def _bsde_solver(self):
        """ Solves the backward stochastic differential equation to estimate option prices. """
        results = Parallel(n_jobs=-1)(delayed(self._sample_task)(k) for k in range(self.samples))
        
        Y0_samples = np.zeros(self.samples)
        Z0_samples = np.zeros((self.dims, self.samples))
        domain_out_of_range = False

        for k, result in enumerate(results):
            Y0_samples[k] = result[0]
            Z0_samples[:, k] = result[1]
            if result[2]:
                domain_out_of_range = True

        if domain_out_of_range:
            print("Domain possibly out of range, consider increasing H!")
            
        return Y0_samples, Z0_samples

    # Calculates the confidence interval for the estimated Y_0 or Z_0
    def _confidence_interval(self, sample):
        """ Calculates the confidence interval with confidence_level """ 
        confidence_level = self.confidence_level
        samples = self.samples
        mean_sample = np.mean(sample)
        std_sample = np.std(sample, ddof=1)
        upper = mean_sample + norm.ppf(1-confidence_level/2) * std_sample/np.sqrt(samples)
        lower = mean_sample - norm.ppf(1-confidence_level/2) * std_sample/np.sqrt(samples)
        CI = [round(lower,4), round(upper,4)]
        return mean_sample, std_sample, CI

    # Plotting results by varying N (number of time steps)
    def plot_and_show_table_by_N(self, N_values, nofig=False, bs_price=None):
        """ Plots and show tables by varying M """
        bs_price = 19.9924  # Reference price for comparison
        function_name = inspect.currentframe().f_code.co_name
        Y, Z, Y_errors, Z_errors, rows, computation_times, memory_usages, errors = [], [], [], [], [], [], [], []

        for N_val in N_values:
            memory_monitor = MemoryMonitor()
            memory_monitor.start()
            start_time = time.time()
            self.N = N_val
            self.dt = self.T / N_val

            Y0_array, Z0_array = self._bsde_solver()  # Solve the BSDE
            est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)  # Calculate confidence interval for Y_0
            est_Z0, std_Z0, CI_Z = self._confidence_interval(Z0_array)  # Calculate confidence interval for Z_0

            if bs_price:
                error = np.mean((Y0_array - bs_price) **2)  # Calculate the error with respect to the reference price

            end_time = time.time()
            computation_time = end_time - start_time

            memory_monitor.stop()
            memory_monitor.join()
            memory_usage_megabytes = memory_monitor.peak_memory

            Y.append(est_Y0)
            Z.append(est_Z0)
            Y_errors.append(std_Y0)
            Z_errors.append(std_Z0)
            if bs_price:
                errors.append(error)
            
            computation_times.append(computation_time)
            memory_usages.append(memory_usage_megabytes)

            rows.append([N_val, est_Y0, std_Y0, CI_Y[0], CI_Y[1], est_Z0, std_Z0, 
                         CI_Z[0], CI_Z[1], error, computation_time, memory_usage_megabytes])
            print(f'Done with N_val: {N_val}.')

        memory_usages.reverse()  # Reverse memory usage for plotting

        if not nofig:
            self._generate_plot(N_values, Y, Y_errors, function_name, bs_price)
            self._plot_computation_times(N_values, computation_times, function_name)
            if bs_price:
                self._plot_convergence(N_values, errors, function_name)

        self._generate_table(rows, function_name)

    # Plotting results by varying M (number of Monte Carlo simulations)
    def plot_and_show_table_by_M(self, M_values, nofig=False, bs_price=None):
        """ Plots and show tables by varying M """
        bs_price = 23.905  # Reference price for comparison

        function_name = inspect.currentframe().f_code.co_name
        Y, Z, Y_errors, Z_errors, rows, computation_times, memory_usages, errors = [], [], [], [], [], [], [], []

        for M_val in M_values:
            memory_monitor = MemoryMonitor()
            memory_monitor.start()
            start_time = time.time()
            self.M = M_val

            Y0_array, Z0_array = self._bsde_solver()  # Solve the BSDE
            est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)  # Calculate confidence interval for Y_0
            est_Z0, std_Z0, CI_Z = self._confidence_interval(Z0_array)  # Calculate confidence interval for Z_0

            error = np.mean((Y0_array - bs_price) **2)  # Calculate the error with respect to the reference price

            end_time = time.time()
            computation_time = end_time - start_time

            memory_monitor.stop()
            memory_monitor.join()
            memory_usage_megabytes = memory_monitor.peak_memory

            Y.append(est_Y0)
            Z.append(est_Z0)
            Y_errors.append(std_Y0)
            Z_errors.append(std_Z0)
            errors.append(error)
            
            computation_times.append(computation_time)
            memory_usages.append(memory_usage_megabytes)

            rows.append([M_val, est_Y0, std_Y0, CI_Y[0], CI_Y[1], est_Z0, std_Z0, 
                         CI_Z[0], CI_Z[1], error, computation_time, memory_usage_megabytes])
            print(f'Done with M_val: {M_val}.')

        memory_usages.reverse()  # Reverse memory usage for plotting

        if not nofig:
            self._generate_plot(M_values, Y, Y_errors, function_name, bs_price)
            self._plot_computation_times(M_values, computation_times, function_name)
            self._plot_convergence(M_values, errors, function_name)

        self._generate_table(rows, function_name)

    # Generates and saves plots based on the configurations in the YAML file
    def _generate_plot(self, x_values, y_values, y_errors, function_name, bs_price):
        """ Generates the plot with the configs in the yaml file """
        configs = self._load_configs()
        plot_config = configs['plot_config'][function_name]

        plt.errorbar(x_values, y_values, yerr=y_errors, fmt='-o', capsize=5, capthick=2, ecolor='red', label='Option Price')

        if bs_price is not None:
            plt.plot(x_values, [bs_price] * len(x_values), 'k-', label='Reference Price (2.95)')
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
            S_0=self.S0,
            mu=self.mu,
            sigma=self.sigma,
            T=self.T,
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
        plt.show()
        plt.close()

    # Plots the convergence rate versus the number of time steps or Monte Carlo simulations
    def _plot_convergence(self, x_values, errors, function_name):
        configs = self._load_configs()
        plot_config = configs['plot_config'][function_name]
        name = function_name.split("_")[-1]
        plt.figure()
        plt.xlabel(plot_config.get('xlabel'))
        plt.ylabel('Mean Squared Error')
        plt.plot(x_values, errors, label='Convergence Rate')
        plt.title(f'Convergence Rate vs. Number of {name}')
        plt.grid(True)

        plot_directory = os.path.join('plots', function_name)
        os.makedirs(plot_directory, exist_ok=True)

        plot_name = f'convergence_vs_{name}_{self.opt_style}_{self.option_payoff}.png'
        plot_path = os.path.join(plot_directory, plot_name)
        plt.savefig(plot_path)
        plt.show()
        plt.close()

    # Plots memory usage against convergence error
    def _plot_memory_usage_vs_convergence_error(self, memory_usages, errors, function_name):
        plot_name = (function_name.split("_")[-1]).capitalize()
        """ Plots the memory usage against the convergence rate"""
        plt.figure()
        plt.plot(memory_usages, errors, 'o-', label='Memory Usage - Convergence Rate')
        plt.xlabel('Memory Usage (MB)')
        plt.ylabel('Convergence Rate')
        plt.title(f'Convergence Rate vs. Memory Usage')
        plt.legend(loc='best')
        plt.grid(True)

        plot_directory = os.path.join('plots', function_name)
        os.makedirs(plot_directory, exist_ok=True)

        plot_name = f'memory_usage_vs_convergence_{plot_name.lower()}{self.opt_style}_{self.option_payoff}.png'
        plot_path = os.path.join(plot_directory, plot_name)
        plt.savefig(plot_path)
        plt.show()
        plt.close()

    # Plots computation time against the number of samples (N or M)
    def _plot_computation_times(self, x_values, computation_times, function_name):
        plot_name = (function_name.split("_")[-1]).capitalize()
        """ Plots the computation times against the number of samples """
        plt.figure()
        plt.plot(x_values, computation_times, 'o-', label='Computation Time')
        plt.xlabel(f'Number of {plot_name}')
        plt.ylabel('Computation Time (seconds)')
        plt.title(f'Computation Time vs. Number of {plot_name}')
        plt.legend(loc='best')
        plt.grid(True)

        plot_directory = os.path.join('plots', function_name)
        os.makedirs(plot_directory, exist_ok=True)

        plot_name = f'computation_time_vs_{plot_name.lower()}{self.opt_style}_{self.option_payoff}.png'
        plot_path = os.path.join(plot_directory, plot_name)
        plt.savefig(plot_path)
        plt.show()
        plt.close()

    # Plots convergence error against computation time
    def _plot_convergence_vs_computation_time(self, computation_times, errors, function_name):
        """ Plots the convergence rate of the option price estimates as a function of computation time """
        name = function_name.split("_")[-1]
        plt.figure()
        plt.xlabel('Computation Time (seconds)')
        plt.ylabel('MSE Error')
        plt.plot(computation_times, errors, label='Convergence Rate')
        plt.title(f'Convergence Rate vs. Computation Time for {name}')
        plt.grid(True)

        plot_directory = os.path.join('plots', function_name)
        os.makedirs(plot_directory, exist_ok=True)

        plot_name = f'convergence_vs_computation_time_{self.opt_style}_{self.option_payoff}.png'
        plot_path = os.path.join(plot_directory, plot_name)
        plt.savefig(plot_path)
        plt.show()
        plt.close()

    # Generates and saves tables based on the results in the YAML file
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
            'Std. Deviation Z0', 'CI Lower Bound Z0', 'CI Upper Bound Z0', 'MSE (%)', 
             'Computation Time', 'Memory Usage']
        )
        for row in rows:
            row[9] = f"{row[9]:.3e}"

        table_name = configs['plot_config'][function_name]['table_name_template'].format(
            S_0=self.S0,
            mu=self.mu,
            sigma=self.sigma,
            T=self.T,
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
    
    # Method to solve the BSDE and provide a summary of results
    def solve(self):
        """ Method called to run the program and solve the BSDE """
        start_timer = time.time()
        Y0_array, Z0_array = self._bsde_solver()  # Solve the BSDE
        finished_time = time.time() - start_timer
        est_Y0, std_Y0, CI_Y = self._confidence_interval(Y0_array)  # Calculate confidence interval for Y_0
        est_Z0, std_Z0, CI_Z = self._confidence_interval(Z0_array)  # Calculate confidence interval for Z_0
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
    # Inherits from the European class and adjusts for American options
    def __init__(self, S0, mu, sigma, correlation, K, r, div, T, N, M, confidence_level=0.025, 
                 samples=50, dims=1, option_payoff="call", H=60, delta=1, k=0): 
        super().__init__(S0, mu, sigma, correlation, K, r, div, T, N, M, confidence_level, 
                         samples, dims, option_payoff, H, delta, k)
        self._opt_style = 'american'

    # Task executed for each sample, with adjustments for early exercise (American option)
    def _sample_task(self, sample_id):
        sample_time = time.time()
        S, dw = self._generate_stock_paths()  # Simulate stock price paths
        domain_out_of_range = ((np.max(S) > (self.S0 + self.H)) or 
                               (np.min(S) < (self.S0 - self.H)))
        Y_plus = self._payoff_func(S[:, :, -1])  # Calculate the payoff at maturity
        Y = np.zeros(self.M)
        Z = np.zeros((self.M, self.dims))
        p_li_list = self._generate_hypercube_basis(S)  # Generate the basis functions for regression
        for i in range(self.N - 1, -1, -1):
            p_li = p_li_list[i] 
            A = p_li.T @ p_li
            Z = self._generate_Z(p_li, A, Y_plus, dw[:, :, i])  # Estimate Z using regression
            Y = self._generate_Y(p_li, A, Y_plus, Z)  # Estimate Y using regression
            exercise_values = self._payoff_func(S[:, :, i])  # Calculate the immediate exercise value
            continuation_value = np.exp(-self.r * self.dt) * Y  # Discount the continuation value
            Y_plus = np.maximum(exercise_values, continuation_value)  # Choose between immediate exercise or continuation

        Y0_sample = np.mean(Y_plus)  # Estimate Y_0 as the mean of the last Y_plus
        Z0_sample = np.mean(Z, axis=0)  # Estimate Z_0 as the mean of the last Z
        sample_finish_time = time.time() - sample_time
        print(f"Done with sample: {sample_id+1} in {sample_finish_time} seconds")
        return Y0_sample, Z0_sample, domain_out_of_range

    def _bsde_solver(self):
        """ Solves the backward stochastic differential equation to estimate option prices. """
        results = Parallel(n_jobs=-1)(delayed(self._sample_task)(k) for k in range(self.samples))
        
        Y0_samples = np.zeros(self.samples)
        Z0_samples = np.zeros((self.dims, self.samples))
        domain_out_of_range = False

        for k, result in enumerate(results):
            Y0_samples[k] = result[0]
            Z0_samples[:, k] = result[1]
            if result[2]:
                domain_out_of_range = True

        if domain_out_of_range:
            print("Domain possibly out of range, consider increasing H!")
            
        return Y0_samples, Z0_samples

    def __repr__(self):
        base_repr = super().__repr__()
        return base_repr[:-1] + f" opt_style='{self._opt_style}'\n)"

class BSDEOptionPricingEuropeanSpread(BSDEOptionPricingEuropean):
    # Extends the European option class for spread options
    def __init__(self, S0, mu, sigma, correlation, K, r, div, T, N, M, 
                 confidence_level, samples, dims, option_payoff, domain, delta, k, K2, R):
        super().__init__(S0, mu, sigma, correlation, K, r, div, T, N, M, 
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

    # Adjusts the payoff function for a spread option
    def _payoff_func(self, S):
        K = self.K
        K2 = self.K2
        if self.option_payoff == "call":
            payoff = (np.maximum(np.max(S, axis=0) - K, 0) - 
                    2*np.maximum(np.max(S, axis=0) - K2, 0))
        elif self.option_payoff == "put":
            payoff = (np.maximum(K - np.max(S, axis=0) , 0) - 
                    2*np.maximum(K2 - np.max(S, axis=0) , 0))

        else:
            raise ValueError(f"Unsupported option payoff type {self.option_payoff}.")
        payoff = np.expand_dims(payoff, axis=1)
        return payoff

    # Adjusts the driver for different interest rates for borrowing and lending
    def _driver(self, Y_plus, Z):
        """ Returns the driver in the BSDE for different interest rates for borrowing and lending """
        term1 = Y_plus * self.r
        term2 = Z * self.lamb
        term3 = (self.R - self.r) * np.minimum(Y_plus - Z/self.sigma, 0)

        result = - term1 - term2 + term3
        return result

    def _sample_task(self, sample_id):
        sample_time = time.time()
        S, dw = self._generate_stock_paths()  # Simulate stock price paths
        domain_out_of_range = ((np.max(S) > (self.S0 + self.H)) or 
                               (np.min(S) < (self.S0 - self.H)))
        Y_plus = self._payoff_func(S[:, :, -1])  # Calculate the payoff at maturity
        Y = np.zeros(self.M)
        Z = np.zeros((self.M, self.dims))
        p_li_list = self._generate_hypercube_basis(S)  # Generate the basis functions for regression
        for i in range(self.N - 1, -1, -1):
            p_li = p_li_list[i] 
            A = p_li.T @ p_li  # Compute the Gram matrix for least squares
            Z = self._generate_Z(p_li, A, Y_plus, dw[:, :, i])  # Estimate Z using regression
            Y = self._generate_Y(p_li, A, Y_plus, Z)  # Estimate Y using regression
            Y_plus = Y  # Update Y_plus for the next step
            del p_li, A  # Clean up memory

        Y0_sample = np.mean(Y_plus)  # Estimate Y_0 as the mean of the last Y_plus
        Z0_sample = np.mean(Z, axis=0)  # Estimate Z_0 as the mean of the last Z
        sample_finish_time = time.time() - sample_time
        print(f"Done with sample: {sample_id+1} in {sample_finish_time} seconds")
        return Y0_sample, Z0_sample, domain_out_of_range

    def _bsde_solver(self):
        """ Solves the backward stochastic differential equation to estimate option prices. """
        results = Parallel(n_jobs=-1)(delayed(self._sample_task)(k) for k in range(self.samples))
        
        Y0_samples = np.zeros(self.samples)
        Z0_samples = np.zeros((self.dims, self.samples))
        domain_out_of_range = False

        for k, result in enumerate(results):
            Y0_samples[k] = result[0]
            Z0_samples[:, k] = result[1]
            if result[2]:
                domain_out_of_range = True

        if domain_out_of_range:
            print("Domain possibly out of range, consider increasing H!")
            
        return Y0_samples, Z0_samples

    def __repr__(self):
        base_repr = super().__repr__()
        return base_repr[:-1] + f"\n  opt_style='{self._opt_style}'\n  K2='{self.K2}'\n  R='{self.R}'\n)"

class BSDEOptionPricingAmericanSpread(BSDEOptionPricingEuropeanSpread):
    # Extends the spread option class for American options
    def __init__(self, S0, mu, sigma, correlation, K, r, div, T, N, M, 
                 confidence_level, samples, dims, option_payoff, domain, delta, k, K2, R):
        super().__init__(S0, mu, sigma, correlation, K, r, div, T, N, M, 
                         confidence_level, samples, dims, option_payoff, domain, delta, k, K2, R)
        self._opt_style = 'americanspread'

    def _sample_task(self, sample_id):
        sample_time = time.time()
        S, dw = self._generate_stock_paths()  # Simulate stock price paths
        domain_out_of_range = ((np.max(S) > (self.S0 + self.H)) or 
                               (np.min(S) < (self.S0 - self.H)))
        Y_plus = self._payoff_func(S[:, :, -1])  # Calculate the payoff at maturity
        Y = np.zeros(self.M)
        Z = np.zeros((self.M, self.dims))
        p_li_list = self._generate_hypercube_basis(S)  # Generate the basis functions for regression
        for i in range(self.N - 1, -1, -1):
            p_li = p_li_list[i] 
            A = p_li.T @ p_li
            Z = self._generate_Z(p_li, A, Y_plus, dw[:, :, i])  # Estimate Z using regression
            Y = self._generate_Y(p_li, A, Y_plus, Z)  # Estimate Y using regression
            exercise_values = self._payoff_func(S[:, :, i])  # Calculate the immediate exercise value
            Y_plus = np.maximum(Y, exercise_values)  # Choose between immediate exercise or continuation
            del p_li, A  # Clean up memory

        Y0_sample = np.mean(Y_plus)  # Estimate Y_0 as the mean of the last Y_plus
        Z0_sample = np.mean(Z, axis=0)  # Estimate Z_0 as the mean of the last Z
        sample_finish_time = time.time() - sample_time
        print(f"Done with sample: {sample_id+1} in {sample_finish_time} seconds")
        return Y0_sample, Z0_sample, domain_out_of_range

    def _bsde_solver(self):
        """ Solves the backward stochastic differential equation to estimate option prices. """
        results = Parallel(n_jobs=-1)(delayed(self._sample_task)(k) for k in range(self.samples))
        
        Y0_samples = np.zeros(self.samples)
        Z0_samples = np.zeros((self.dims, self.samples))
        domain_out_of_range = False

        for k, result in enumerate(results):
            Y0_samples[k] = result[0]
            Z0_samples[:, k] = result[1]
            if result[2]:
                domain_out_of_range = True

        if domain_out_of_range:
            print("Domain possibly out of range, consider increasing H!")
            
        return Y0_samples, Z0_samples

    def __repr__(self):
        base_repr = super().__repr__()
        return base_repr[:-1] + f"\n  opt_style='{self._opt_style}'\n  K2='{self.K2}'\n  R='{self.R}'\n)"

class MemoryMonitor(Thread):
    # Monitors peak memory usage during the simulation
    def __init__(self):
        Thread.__init__(self)
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.running = True

    # Continuously checks memory usage and updates peak memory if it increases
    def run(self):
        while self.running:
            current_memory = self.process.memory_info().rss / (1024 ** 2)  
            if (current_memory > self.peak_memory):
                self.peak_memory = current_memory
            time.sleep(0.01)  

    def stop(self):
        self.running = False

