# BSDE Option Pricing

This module provides implementations for option pricing based on the backward stochastic differential equation (BSDE) method for both European and American options. Additionally, it includes the traditional Black-Scholes formula for European options and the Longstaff-Schwartz method for pricing American options. It also attempts to price both American and European call spread, which is an attempt to price even a more complex problem and making the driver in the BSDE less linear.

## Features:

1. **BSDE Method**: Uses BSDE for European and American option pricing.
2. **Black-Scholes**: Provides the classic Black-Scholes formula for European option pricing.

## Requirements:

- Python (3.10 or later)

## Classes:

1. **BSDEOptionPricingEuropean**: Class to compute European option price using BSDE.
2. **BSDEOptionPricingAmerican**: Class to compute American option price using BSDE.
1. **BSDEOptionPricingEuropeanSpread**: Class to compute European call spread price using BSDE.
1. **BSDEOptionPricingAmericanSpread**: Class to compute American call spread price using BSDE.

## Usage:

Clone the repository and initalise the project by running the setup.sh (Might have to give it permission to execute first). After that, source into the python virtual enviroment and run the main file with following optional arguments

- `--S` : Initial stock price. Default is 100.0.
- `--K` : Strike price of the option. Default is 95.0.
- `--K2` : Second strike price, applicable for spread options. Default is 105.
- `--r` : Risk-free rate. Default is 0.01.
- `--R` : Second risk-free rate, applicable for spread options. Default is 0.01.
- `--mu` : Drift term of the stock. Default is 0.01.
- `--sigma` : Volatility (sigma) of the stock. Default is 0.2.
- `--T` : Time to expiration in years. Default is 0.25.
- `--N` : Number of time steps for the BSDE solver. Default is 252.
- `--M` : Number of Monte Carlo simulations. Default is 100000.
- `--L` : Lower confidence level alpha. Default is 0.025.
- `--degree` : Degree for polynomial regression in Longstaff Schwartz method. Default is 3.
- `--samples` : Number of samples of solved BSDE prices to consider. Default is 10.
- `--opt_payoff` : Type of option payoff. Choices are 'call' or 'put'. Default is 'call'.
- `--opt_style` : Style of the option. Choices are 'european', 'american', 'europeanspread', and 'americanspread'. Default is 'european'.
- `--plot` : If set, generates and displays plots. This option does not take a value.
- `--degrees` : Up to what degree to plot and generate tables for. This is applicable if `--plot` is set. Default is the value of `--degree`.


Run the script from the command line, specifying any arguments as needed. For example:

```bash
python main.py --S 100 --K 95 --sigma 0.25 --opt_style european --opt_payoff call --r 0.01 --mu 0.01 --T 1

