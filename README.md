BSDE Option Pricing

This Python script provides a numerical implementation for the pricing of European and American options using the Backward Stochastic Differential Equation (BSDE) method. BSDEs are a type of stochastic differential equation where the terminal condition is specified, and the solution is sought in a backward manner, i.e., from the terminal time to the initial time.

## Features

- Pricing of European and American options.
- Both Call and Put options are supported.
- Ability to adjust model parameters such as the stock price, strike price, time to maturity, interest rate, volatility, number of time steps, number of sample paths, and option type.
- Calculation of the hedge ratio (also known as the delta) for the option.

## Classes

There are two main classes in this script:

1. `BSDEOptionPricingEuropean`: This class is responsible for European option pricing using the BSDE method.
2. `BSDEOptionPricingAmerican`: This class is a subclass of `BSDEOptionPricingEuropean` and is responsible for American option pricing using the BSDE method with a penalty term.

## Usage

To use the script, create an instance of either `BSDEOptionPricingEuropean` or `BSDEOptionPricingAmerican`, and call the `run()` method. The parameters for the classes are:

- `S0`: Initial stock price (positive float or int)
- `K`: Strike price (positive float or int)
- `T`: Time to maturity (positive float or int)
- `r`: Interest rate (non-negative float or int)
- `sigma`: Volatility (positive float or int)
- `N`: Number of time steps (positive int)
- `M`: Number of sample paths (positive int)
- `opt_type`: Option type, either 'call' or 'put' (str)
- `degree`: Degree of the polynomial basis functions for regression (non-negative int, default: 3)
- `lambda_`: Penalty parameter for American option (positive float, default: 1.0)

Example:

```python
# Create an instance of the European option pricing class
european_option = BSDEOptionPricingEuropean(S0=100, K=100, T=1, r=0.05, sigma=0.2, N=100, M=10000, opt_type='call')

# Run the pricing method
european_option.run()

