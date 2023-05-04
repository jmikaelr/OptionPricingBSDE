# Option Pricing with BSDE

This repository contains the implementation of the Monte Carlo Least Squares Method for pricing European and American options using Backward Stochastic Differential Equations (BSDE). The implementation is done using Python, NumPy, and TensorFlow 1.15.0.

## Files

- `mclsq.py`: The custom module containing the main classes for option pricing.
- `main.py`: The main script that uses the classes in `mclsq.py` to run the option pricing simulations.
- `install_and_run.sh`: A bash script for creating a virtual environment, installing the required packages, and running the main script.

## Requirements

- Python 3.6 or higher
- NumPy
- TensorFlow 1.15.0
- SciPy

## Getting Started

1. Clone this repository to your local machine.
2. Open a terminal and navigate to the repository's root directory.
3. Run the `install_and_run.sh` script by executing the following command:

```bash
bash install_and_run.sh
```
## Usage

In the `main.py` file, you can adjust the option parameters to match your desired setup. The following parameters can be changed:

- `S`: Initial stock price
- `K`: Strike price
- `r`: Risk-free interest rate
- `sigma`: Volatility
- `T`: Time to maturity
- `N`: Number of time steps
- `M`: Number of simulated paths
- `opt_type`: Option type (either 'call' or 'put')

After adjusting the parameters, run the `main.py` script to see the option price and hedge ratio for both European and American options.

## Example

```python
# Create an instance of the European option pricing class
european_option = BSDEOptionPricingEuropean(S0=100, K=100, T=1, r=0.05, sigma=0.2, N=100, M=10000, opt_type='call')

# Run the pricing method
european_option.run()


