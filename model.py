import numpy as np
from scipy.optimize import minimize

# -----------------------------
# 1. User inputs
# -----------------------------
# Example: 3 assets
expected_returns = np.array([0.0605, 0.041, 0.0707, 0.0876, 0.062, 0.0419, 0.0412, 0.037, 0.0419, 0.0587, 0.0459, 0.045, 0.042, 0.05, 0.0625, 0.0822, 0.0573, 0.0697, 0.1002, 0.0870, 0.045, 0.0381])   # expected annual returns
vols = np.array([0.2137, 0.2163, 0.1898, 0.2321, 0.1829, 0.0482, 0.0501, 0.1209, 0.0576, 0.0633, 0.0537, 0.0528, 0.0360, 0.4000, 0.1563, 0.1471, 0.1140, 0.1834, 0.2586, 0.0754, 0.1700, 0.001])              # annual volatilities (std devs)

# FIX THIS: Define asset groups for constraints
groups = {
    # Gold = 14, Bonds = 6
    # EEM = 3, IGF = 15
    # SPY = 0, GVIP = 19, IPRV = 18
    'usStocks': [0, 18, 19],
    'Emerging': [3, 15],
    'Safety': [6, 14],
}

correlation_matrix = np.array([
    [1.00, 0.86, 0.83, 0.75, 0.97, 0.19, -0.11, -0.13, 0.37, 0.79, 0.22, 0.11, 0.14, 0.32, -0.05, 0.79, 0.31, 0.69, 0.45, 0.93, 0.37, -0.01],
    [0.86, 1.00, 0.80, 0.73, 0.89, 0.20, -0.08, -0.10, 0.35, 0.76, 0.22, 0.15, 0.15, 0.32, -0.06, 0.79, 0.28, 0.73, 0.53, 0.86, 0.46, 0.00],
    [0.83, 0.80, 1.00, 0.82, 0.90, 0.24, -0.06, -0.09, 0.39, 0.75, 0.26, 0.17, 0.17, 0.32, -0.06, 0.85, 0.34, 0.65, 0.59, 0.79, 0.45, -0.02],
    [0.75, 0.73, 0.82, 1.00, 0.82, 0.19, -0.06, -0.09, 0.33, 0.67, 0.22, 0.12, 0.13, 0.33,  0.01, 0.74, 0.31, 0.58, 0.48, 0.78, 0.46, -0.02],
    [0.97, 0.89, 0.90, 0.82, 1.00, 0.21, -0.10, -0.12, 0.38, 0.81, 0.25, 0.14, 0.16, 0.33, -0.05, 0.85, 0.33, 0.72, 0.53, 0.93, 0.43, -0.01],
    [0.19, 0.20, 0.24, 0.19, 0.21, 1.00, 0.83, 0.83, 0.84, 0.43, 0.87, 0.74, 0.77, 0.01,  0.11, 0.26, 0.53, 0.22, 0.17, 0.19, 0.17, -0.03],
    [-0.11,-0.08,-0.06,-0.06,-0.10, 0.83, 1.00, 0.91, 0.66, 0.14, 0.76, 0.72, 0.71,-0.16,  0.12,-0.04, 0.37,-0.03,-0.03,-0.10, 0.02, -0.02],
    [-0.13,-0.10,-0.09,-0.09,-0.12, 0.83, 0.91, 1.00, 0.67, 0.10, 0.70, 0.71, 0.71,-0.17,  0.11,-0.08, 0.36,-0.05,-0.03,-0.10, 0.00, -0.01],
    [0.37, 0.35, 0.39, 0.33, 0.38, 0.84, 0.66, 0.67, 1.00, 0.62, 0.75, 0.65, 0.62, 0.06,  0.10, 0.40, 0.57, 0.36, 0.23, 0.35, 0.22, -0.02],
    [0.79, 0.76, 0.75, 0.67, 0.81, 0.43, 0.14, 0.10, 0.62, 1.00, 0.45, 0.29, 0.36, 0.28, -0.03, 0.76, 0.49, 0.68, 0.50, 0.75, 0.38, -0.03],
    [0.22, 0.22, 0.26, 0.22, 0.25, 0.87, 0.76, 0.70, 0.75, 0.45, 1.00, 0.67, 0.70,-0.03,  0.07, 0.30, 0.56, 0.23, 0.15, 0.18, 0.19, -0.02],
    [0.11, 0.15, 0.17, 0.12, 0.14, 0.74, 0.72, 0.71, 0.65, 0.29, 0.67, 1.00, 0.61, 0.13,  0.16, 0.20, 0.32, 0.17, 0.16, 0.12, 0.17, 0],
    [0.14, 0.15, 0.17, 0.13, 0.16, 0.77, 0.71, 0.71, 0.62, 0.36, 0.70, 0.61, 1.00,-0.06,  0.13, 0.21, 0.46, 0.23, 0.15, 0.14, 0.15, -0.05],
    [0.32, 0.32, 0.32, 0.33, 0.33, 0.01,-0.16,-0.17, 0.06, 0.28,-0.03, 0.13,-0.06, 1.00,  0.04, 0.36, 0.02, 0.27, 0.22, 0.32, 0.19, -0.01],
    [-0.05,-0.06,-0.06, 0.01,-0.05, 0.11, 0.12, 0.11, 0.10,-0.03, 0.07, 0.16, 0.13, 0.04,  1.00, 0.00, 0.05,-0.03,-0.15,-0.04, 0.00, -0.03],
    [0.79, 0.79, 0.85, 0.74, 0.85, 0.26,-0.04,-0.08, 0.40, 0.76, 0.30, 0.20, 0.21, 0.36,  0.00, 1.00, 0.41, 0.77, 0.54, 0.72, 0.45, -0.01],
    [0.31, 0.28, 0.34, 0.31, 0.33, 0.53, 0.37, 0.36, 0.57, 0.49, 0.56, 0.32, 0.46, 0.02,  0.05, 0.41, 1.00, 0.39, 0.29, 0.26, 0.27, -0.02],
    [0.69, 0.73, 0.65, 0.58, 0.72, 0.22,-0.03,-0.05, 0.36, 0.68, 0.23, 0.17, 0.23, 0.27, -0.03, 0.77, 0.39, 1.00, 0.52, 0.63, 0.42, 0],
    [0.45, 0.53, 0.59, 0.48, 0.53, 0.17,-0.03,-0.03, 0.23, 0.50, 0.15, 0.16, 0.15, 0.22, -0.15, 0.54, 0.29, 0.52, 1.00, 0.48, 0.55, -0.01],
    [0.93, 0.86, 0.79, 0.78, 0.93, 0.19,-0.10,-0.10, 0.35, 0.75, 0.18, 0.12, 0.14, 0.32, -0.04, 0.72, 0.26, 0.63, 0.48, 1.00, 0.41, -0.02],
    [0.37, 0.46, 0.45, 0.46, 0.43, 0.17, 0.02, 0.00, 0.22, 0.38, 0.19, 0.17, 0.15, 0.19,  0.00, 0.45, 0.27, 0.42, 0.55, 0.41, 1.00, 0.04],
    [-0.01, 0.00,-0.02,-0.02,-0.01,-0.03,-0.02,-0.01,-0.02,-0.03,-0.02, 0,   -0.05,-0.01, -0.03,-0.01,-0.02, 0,   -0.01,-0.02, 0.04, 1.00]
])

# -----------------------------
# 2. Build covariance matrix
# -----------------------------
cov_matrix = np.outer(vols, vols) * correlation_matrix
n_assets = len(expected_returns)

# -----------------------------
# 3. Portfolio functions
# -----------------------------
def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, vol

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free=0.0381):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - risk_free) / vol  # Maximize Sharpe ratio

# -----------------------------
# 4. Optimization
# -----------------------------
constraints = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},

    {'type': 'ineq', 'fun': lambda w: np.sum(w[groups['usStocks']]) - 0.50},  # At least 20% in Equities
    {'type': 'ineq', 'fun': lambda w: 0.90 - np.sum(w[groups['usStocks']])},  # At most 70% in Safes

    {'type': 'ineq', 'fun': lambda w: np.sum(w[groups['Emerging']]) - 0.10},  # At least 20% in Equities
    {'type': 'ineq', 'fun': lambda w: 0.45 - np.sum(w[groups['Emerging']])},  # At most 70% in Safes
    
    {'type': 'ineq', 'fun': lambda w: np.sum(w[groups['Safety']]) - 0.01},  # At least 20% in Equities
    {'type': 'ineq', 'fun': lambda w: 0.073 - np.sum(w[groups['Safety']])},  # At most 70% in Safes
]

bounds = [
    (0.407, 0.55),  # US large & med cap equities

    (0.00, 0),  # US small cap equities

    (0.00, 0),  # Europe large cap equities

    (0.024, 1),  # Emerging large cap equities

    (0.00, 0),  # Global large cap equities

    (0.00, 0),  # US aggregate bonds

    (0.0175, 1),  # US govt (all maturities)

    (0.00, 0),  # US govt (10+ years)

    (0.00, 0),  # US high credit corp bonds

    (0.00, 0),  # US high yield corp bonds

    (0.00, 0),  # US agency MBS

    (0.00, 0),  # US inflation-linked govt

    (0.00, 0),  # Global aggregate bonds

    (0.00, 0),  # Oil
    (0.00, 1),  # Gold
    (0.00, 1),  # Global infrastructure equity

    (0.00, 0),  # Infrastructure Debt

    (0.00, 0),  # Reals estate
    (0.00, 1),  # Pvt equity
    (0.00, 0.20),  # Hedge Funds
    (0.00, 0),  # Energy
    (0.00, 0)   # Cash
]

initial_guess = np.ones(n_assets) / n_assets

result = minimize(
    negative_sharpe,
    initial_guess,
    args=(expected_returns, cov_matrix, 0.0381),  # assume 2% risk-free rate
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# -----------------------------
# 5. Results
# -----------------------------
opt_weights = result.x
opt_return, opt_vol = portfolio_performance(opt_weights, expected_returns, cov_matrix)
sharpe = (opt_return - 0.0381) / opt_vol

print("Optimal weights:", np.round(opt_weights, 3))
print(f"Expected return: {opt_return:.2%}")
print(f"Volatility: {opt_vol:.2%}")
print(f"Sharpe ratio: {sharpe:.2f}")