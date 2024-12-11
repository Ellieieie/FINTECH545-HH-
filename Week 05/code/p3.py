import sys
import os
sys.path.append("/Users/ellieieie_/Desktop/FINTECH545-HH-")

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from MyLibrary.regression import fit_general_t as fit_g_t
from MyLibrary.regression import fit_normal as fit_n
from MyLibrary.returns import return_calculate as calc_returns
from MyLibrary.VaR import VaR, ESS

def pca_simulation(cov_matrix, n_sims, n_components=None):
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    sorted_indices = np.arange(len(eigvals)-1, -1, -1)
    sorted_vals = eigvals[sorted_indices]
    sorted_vecs = eigvecs[:, sorted_indices]
    total_var = np.sum(sorted_vals)
    positive_indices = np.where(sorted_vals >= 1e-8)[0]
    if n_components is not None and n_components < len(positive_indices):
        positive_indices = positive_indices[:n_components]
    selected_vals = sorted_vals[positive_indices]
    selected_vecs = sorted_vecs[:, positive_indices]
    scaling_matrix = selected_vecs @ np.diag(np.sqrt(selected_vals))
    dim = len(selected_vals)
    random_draws = np.random.randn(dim, n_sims)
    simulated_data = (scaling_matrix @ random_draws).T
    return simulated_data

def aggregate_risk(sim_data, group_cols):
    alpha = 0.05
    output = []
    grouped = sim_data.groupby(group_cols)
    for name, group_df in grouped:
        pnl_values = group_df['pnl'].values
        var_95 = VaR(pnl_values, alpha=alpha)
        es_95 = ESS(pnl_values, alpha=alpha)
        if isinstance(name, tuple):
            name = name[0]
        output.append((name, var_95, es_95))
    return pd.DataFrame(output, columns=group_cols + ['VaR95', 'ES95'])

price_data = pd.read_csv('/Users/ellieieie_/Desktop/DailyPrices.csv')
daily_returns = calc_returns(price_data, dateColumn="Date")
daily_returns = daily_returns.drop(columns=["Date"])
return_names = daily_returns.columns
current_prices = price_data.iloc[-1, :]
portfolio_data = pd.read_csv('/Users/ellieieie_/Desktop/portfolio.csv')
t_fitted_stocks = portfolio_data.loc[portfolio_data['Portfolio'].isin(["A", "B"]), 'Stock']
n_fitted_stocks = portfolio_data.loc[portfolio_data['Portfolio'].isin(["C"]), 'Stock']
all_stocks = list(t_fitted_stocks) + list(n_fitted_stocks)
for stock_name in all_stocks:
    ret_series = daily_returns[stock_name]
    daily_returns[stock_name] = ret_series - ret_series.mean()
model_fits = {}
for s in t_fitted_stocks:
    model_fits[s] = fit_g_t(daily_returns[s].values)
for s in n_fitted_stocks:
    model_fits[s] = fit_n(daily_returns[s].values)
U_data = pd.DataFrame(index=daily_returns.index)
for st in all_stocks:
    U_data[st] = model_fits[st]["u"]
spearman_corr = U_data.corr(method='spearman').values
if np.min(np.linalg.eigvals(spearman_corr)) > -1e-8:
    print("Matrix is PSD")
else:
    print("Matrix is not PSD")
num_simulations = 5000
sim_Z = pca_simulation(spearman_corr, num_simulations)
sim_U_data = pd.DataFrame(norm.cdf(sim_Z), columns=all_stocks)
sim_rets = pd.DataFrame(index=range(num_simulations), columns=all_stocks)
for st in all_stocks:
    sim_rets[st] = sim_U_data[st].apply(model_fits[st]["eval"])

def compute_portfolio_risk(sim_returns, n_sims):
    iteration_df = pd.DataFrame({'iteration': range(1, n_sims+1)})
    portfolio_data['key'] = 1
    iteration_df['key'] = 1
    merged_vals = pd.merge(portfolio_data, iteration_df, on='key').drop('key', axis=1)
    c_value, s_value, pnl_list = [], [], []
    for i, row in merged_vals.iterrows():
        this_price = current_prices[row['Stock']]
        curr_val = row['Holding'] * this_price
        sim_ret = sim_returns.loc[row['iteration']-1, row['Stock']]
        sim_val = row['Holding'] * this_price * (1.0 + sim_ret)
        c_value.append(curr_val)
        s_value.append(sim_val)
        pnl_list.append(sim_val - curr_val)
    merged_vals['currentValue'] = c_value
    merged_vals['simulatedValue'] = s_value
    merged_vals['pnl'] = pnl_list
    merged_vals['Portfolio'] = merged_vals['Portfolio'].astype(str)
    risk_report = aggregate_risk(merged_vals, ['Portfolio'])
    return risk_report

portfolio_risk = compute_portfolio_risk(sim_rets, num_simulations)
print(portfolio_risk)

def ewma_variance(series, lambda_):
    var_est = np.var(series)
    for r in series:
        var_est = lambda_ * var_est + (1 - lambda_) * (r ** 2)
    return var_est

def ewma_covariance(returns_matrix, lambda_):
    n, m = returns_matrix.shape
    weights = np.array([(1 - lambda_) * (lambda_ ** i) for i in range(n)])[::-1]
    weights /= weights.sum()
    mean_adj = returns_matrix - returns_matrix.mean(axis=0)
    weighted_cov = mean_adj.T @ (weights[:, np.newaxis] * mean_adj)
    return weighted_cov

ew_cov = ewma_covariance(daily_returns.values, 0.97)
sim_rets_cov = pd.DataFrame(pca_simulation(ew_cov, num_simulations), columns=return_names)
risk_from_cov = compute_portfolio_risk(sim_rets_cov, num_simulations)
print(risk_from_cov)
