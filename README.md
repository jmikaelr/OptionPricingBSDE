# BSDE Option Pricing

This module provides implementations for option pricing based on the backward stochastic differential equation (BSDE) method for both European and American options. Additionally, it includes the traditional Black-Scholes formula for European options and the Longstaff-Schwartz method for pricing American options.

## Features:

1. **BSDE Method**: Uses BSDE for European and American option pricing.
2. **Black-Scholes**: Provides the classic Black-Scholes formula for European option pricing.
3. **Longstaff-Schwartz**: Uses the least squares Monte Carlo method by Longstaff and Schwartz for American option pricing.

## Requirements:

- Python (3.6 or later)
- `numpy`
- `scipy`

## Classes:

1. **BSDEOptionPricingEuropean**: Class to compute European option price using BSDE.
2. **BSDEOptionPricingAmerican**: Class to compute American option price using BSDE.
3. **BS_CALL**: Function to compute European option price using Black-Scholes formula.
4. **longstaff_schwartz**: Function to compute American option price using the Longstaff-Schwartz method.

## Usage:

Simply instantiate the appropriate class (`BSDEOptionPricingEuropean` or `BSDEOptionPricingAmerican`) with the required parameters, then run the `run()` method.

Example:
```python
S, K, r, sigma, T, N, M, opt_type = 100.0, 95.0, 0.01, 0.2, 0.25, 25, 8920, 'call'
american_opt = BSDEOptionPricingAmerican(S, K, r, sigma, T, N, M, 100, opt_type)
american_opt.run()

