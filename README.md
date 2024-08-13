# BSDE Option Pricing

This module offers implementations for pricing options using the Backward Stochastic Differential Equation (BSDE) method. It supports both European and American options, along with variants for European and American spread options featuring complex payoff functions.

This module attempts to numerically find the pair \\((Y, Z)\\) that solves the following BSDE:

**The BSDE**  
$$-dY_t = f(t, Y_t, Z_t)dt - Z_t^*dW_t; \quad Y_T = \\xi$$  
where \\(\\xi\\) in this case represents the option's payoff at maturity. For American options, which allow early exercise for any \\(t \in [0,T]\\), \\(\\xi\\) is added as a condition at each discretized step instead. The underlying process \\(X_t\\) is the Geometric Brownian Motion under the Black-Scholes condition used to simulate the stock price paths.

## Features

- **European Options**: Implements pricing for vanilla European options with standard payoff functions:
  - Call: \\(\\phi(S_T) = \\max(S_T - K, 0)\\)
  - Put: \\(\\phi(S_T) = \\max(K - S_T, 0)\\)
- **American Options**: Includes functionality for American options, which permit early exercise, adding complexity to the pricing model.
- **Spread Options**:
  - **European Spread**: Prices options with a payoff of \\(\\phi(S_T) = \\max(S_T - K_1, 0) - 2\\max(S_T - K_2, 0)\\).
  - **American Spread**: Similar to European spread but allows early exercise.

## Requirements

- Python 3.7 or later

## Modules

1. **BSDEOptionPricingEuropean**: Computes prices for European options.
2. **BSDEOptionPricingAmerican**: Computes prices for American options.
3. **BSDEOptionPricingEuropeanSpread**: Computes prices for European spread options.
4. **BSDEOptionPricingAmericanSpread**: Computes prices for American spread options.

## Installation

1. Clone the repository:
    \`\`\`bash
    git clone https://github.com/Xehiss/OptionPricingBSDE.git
    \`\`\`
2. Navigate to the project directory:
    \`\`\`bash
    cd OptionPricingBSDE
    \`\`\`
3. Run the setup script (ensure it has execute permissions):
    \`\`\`bash 
    chmod +x setup.sh
    ./setup.sh
    \`\`\`

## Usage

After installing the module, you can use it by running the \`main.py\` script, which is configured to handle command-line arguments for flexibility in simulations. Below are the steps to activate your environment and run simulations with different configurations:

### Activating the Environment

Before running the script, make sure to activate the virtual environment if you have set one up during installation:

\`\`\`bash
source venv/bin/activate
\`\`\`

To execute the main script with default parameters, use the following command:
\`\`\`bash
python3 main.py
\`\`\`

## Command Line Arguments

- **\`S\`**: Initial stock price (default: 100.0).
- **\`K\`**: Strike price of the option (default: 95.0).
- **\`K2\`**: Second strike price for spread options (only needed for spread options).
- **\`r\`**: Risk-free interest rate (default: 0.01).
- **\`R\`**: Second interest rate for spread options (only needed for spread options).
- **\`mu\`**: Drift term of the stock's returns (default is similar to \`r\` if not specified).
- **\`sigma\`**: Volatility of the stock (default: 0.2).
- **\`T\`**: Time to expiration in years (default: 0.25).
- **\`N\`**: Number of time steps for discretization (default: 20).
- **\`M\`**: Number of Monte Carlo simulations (default: 1000).
- **\`L\`**: Confidence level alpha (default: 0.025).
- **\`delta\`**: Size of hypercubes (default: 1).
- **\`H\`**: Domain covered by the hypercubes (default: 60).
- **\`samples\`**: Number of BSDE sample paths (default: 10).
- **\`opt_payoff\`**: Option payoff type, choices are \`'call'\` or \`'put'\` (default: \`'call'\`).
- **\`opt_style\`**: Option style, choices are \`'european'\`, \`'american'\`, \`'europeanspread'\`, \`'americanspread'\` (default: \`'european'\`).
- **\`nofig\`**: If set, no plot will be shown (plots are shown by default unless this is specified).
- **\`plot_type\`**: Type of plot based on parameter variation, choices are \`'N'\`, \`'M'\`. 
- **\`plot_values\`**: Comma-separated values for the selected plot type (e.g., 10,20,30).

## Example Command

\`\`\`bash
python3 main.py --opt_style europeanspread --K 95 --K2 105 --r 0.01 --R 0.06 --mu 0.05 --S 100 --sigma 0.2 --T 0.25 --N 5 --delta 1 --M 1000 --plot_type N --plot_values 5,10,20,40
\`\`\`

This command will compute and save the option pricing results, generating plots in the \`/plots\` directory and tables in the \`/tables\` directory.

## Additional Notes

- **Extensibility**: The code is designed to be easily extended. You can add new types of options or modify existing models by extending the provided classes.
- **Performance**: The script is optimized for large-scale Monte Carlo simulations, using parallel processing where possible.

