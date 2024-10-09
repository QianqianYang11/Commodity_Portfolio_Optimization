import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch.unitroot import PhillipsPerron
import mgarch
import statsmodels.api as sm
from arch import arch_model
from scipy.optimize import fmin, minimize
from scipy.stats import t
from math import inf
from IPython.display import display
import bs4 as bs
import requests
from scipy.stats import t
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from sstudentt import SST
from scipy.optimize import minimize
import cvxpy as cp

data = pd.read_excel('D:\\2023semester\\Commodity_portfolio optimization\\Data.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data.iloc[1:]
T, N = data.shape
print(data.columns)
print(data.shape)

#covariance
qt_data = pd.read_csv('D:\\2023semester\\Commodity_portfolio optimization\\vech_t_cov.csv')
cov_matrices = []
for index, row in qt_data.iterrows():
    matrix = np.zeros((3, 3))
    var_cols = ['Var(METAL)', 'Var(ENERGY)', 'Var(AGRICULTURAL)']
    cov_cols = [('METAL', 'ENERGY'), ('METAL', 'AGRICULTURAL'), ('ENERGY', 'AGRICULTURAL')]  
    for i in range(3):
        matrix[i, i] = row[var_cols[i]]    
    for i, (col1, col2) in enumerate(cov_cols):
        cov_col = f'Cov({col1},{col2})'
        if cov_col in row:
            value = row[cov_col]
            row_idx, col_idx = (0, 1) if i == 0 else (0, 2) if i == 1 else (1, 2)
            matrix[row_idx, col_idx] = value
            matrix[col_idx, row_idx] = value
    cov_matrices.append(matrix)
Qt = np.array(cov_matrices)
Qt = np.transpose(Qt, axes=(1, 2, 0))
print("Qt:", Qt.shape) 
#correlation
rt_data = pd.read_csv('D:\\2023semester\\Commodity_portfolio optimization\\vech_t_cor.csv')
cor_matrices = []
for index, row in rt_data.iterrows():
    matrix = np.eye(3)  
    cor_cols = [('METAL', 'ENERGY'), ('METAL', 'AGRICULTURAL'), ('ENERGY', 'AGRICULTURAL')]
    for col1, col2 in cor_cols:
        cor_col = f'Cor({col1},{col2})'
        if cor_col in row:
            value = row[cor_col]
            row_idx = var_cols.index(f'Var({col1})')
            col_idx = var_cols.index(f'Var({col2})')
            matrix[row_idx, col_idx] = value
            matrix[col_idx, row_idx] = value
    cor_matrices.append(matrix)
Rt = np.array(cor_matrices)
Rt = np.transpose(Rt, axes=(1, 2, 0))
print("Rt:", Rt.shape)  # (3, 3, T)
#mean
mu_values = data.mean()
################################################mean-variance 
######################################################################
#mean-variance 
dates = data.index
T, n_assets = data.shape
gamma = 1
mu = mu_values.values
w = cp.Variable(n_assets)

returns = mu.T @ w
risk = cp.quad_form(w, Qt[:,:,0]) 
objective = cp.Maximize(returns - gamma/2 * risk)
constraints = [cp.sum(w) == 1, w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()
optimal_weights = w.value

#Time-varying
optimal_weights_time_varying = np.zeros((n_assets, T))
for t in range(T):
    Q_t = Qt[:,:,t]
    risk = cp.quad_form(w, Q_t)
    objective = cp.Maximize(returns - gamma/2 * risk)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status not in ["infeasible", "unbounded"]:
        optimal_weights_time_varying[:, t] = w.value
    else:
        print(f"Optimization issue at time {t}: {problem.status}")
        optimal_weights_time_varying[:, t] = np.nan  # Use nan to indicate failed optimization

commodity_names = ['Metal', 'Energy', 'Agricultural']
mean_variance_weight = optimal_weights_time_varying.T
mean_variance_weight = pd.DataFrame(mean_variance_weight, index=dates, columns=commodity_names)
output_excel_path = 'D:\\2023semester\\Commodity_portfolio optimization\\optimization_mean_variance.xlsx'
mean_variance_weight.to_excel(output_excel_path)
print("Successful:Mean Variance")
###########################################################################################

# ####################################################################mimimize risk
w = cp.Variable(n_assets)

risk = cp.quad_form(w, Qt[:, :, 0])
objective = cp.Minimize(risk)
constraints = [cp.sum(w) == 1, w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()
optimal_weights_min_variance = w.value

#Time-varying
optimal_weights_time_varying_min_variance = np.zeros((n_assets, T))

for t in range(T):
    Q_t = Qt[:, :, t]
    risk = cp.quad_form(w, Q_t)
    objective = cp.Minimize(risk)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status not in ["infeasible", "unbounded"]:
        optimal_weights_time_varying_min_variance[:, t] = w.value
    else:
        print(f"Optimization issue at time {t}: {problem.status}")
        optimal_weights_time_varying_min_variance[:, t] = np.nan

optimal_weights_min_variance_transposed = optimal_weights_time_varying_min_variance.T


min_variance_weight = pd.DataFrame(optimal_weights_min_variance_transposed, index=dates, columns=commodity_names)
output_excel_path = 'D:\\2023semester\\Commodity_portfolio optimization\\optimization_min_variance.xlsx'
min_variance_weight.to_excel(output_excel_path)
print("Successful: Min-Variance")
######################################################################################
######################################################################################
##Sharpe ratio
Rf = 0.0002

def objective(weights, mu, Q_t, Rf):
    portfolio_return = np.dot(weights, mu)
    portfolio_variance = np.dot(weights.T, np.dot(Q_t, weights))
    sharpe_ratio = (portfolio_return - Rf) / np.sqrt(portfolio_variance)
    return -sharpe_ratio 

constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  
bounds = tuple((0, 1) for _ in range(n_assets))
initial_weights = np.ones(n_assets) / n_assets 

optimized_weights = np.zeros((T, n_assets))

for t in range(T):
    Q_t = Qt[:, :, t] 
    result = minimize(objective, initial_weights, args=(mu, Q_t, Rf), method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        optimized_weights[t] = result.x
    else:
        print(f"Optimization issue at time {t}: {result.message}")
        optimized_weights[t] = np.nan  

Sharpe_Ratio_weight = pd.DataFrame(optimized_weights, index=dates, columns=commodity_names)
output_excel_path = 'D:\\2023semester\\Commodity_portfolio optimization\\optimization_sharpe_ratio.xlsx'
Sharpe_Ratio_weight.to_excel(output_excel_path)
print("Successful: Sharpe Ratio")

###########################################################################Sortino ratio
###########################################################################Sortino ratio
#Sortino ratio
returns_matrix = data.values

def calculate_downside_risk(weights, returns, Rf):
    downside_returns = np.minimum(returns - Rf, 0)
    downside_variance = np.dot(weights.T, np.dot(np.cov(downside_returns, rowvar=False), weights))
    return np.sqrt(downside_variance)

def objective_sortino(weights, expected_returns, returns, Rf):
    portfolio_return = np.dot(weights, expected_returns)
    downside_risk = calculate_downside_risk(weights, returns, Rf)
    sortino_ratio = (portfolio_return - Rf) / downside_risk
    return -sortino_ratio

constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0, 1) for _ in range(n_assets))
initial_weights = np.ones(n_assets) / n_assets

optimized_weights_sortino = np.zeros((T, n_assets))

for t in range(T):
    Return_t = returns_matrix[t, :]  
    result = minimize(objective_sortino, initial_weights, args=(mu, Return_t, Rf), method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        optimized_weights_sortino[t, :] = result.x
    else:
        print(f"Optimization issue at time {t}: {result.message}")
        optimized_weights_sortino[t, :] = np.nan

Sortino_Ratio_weight = pd.DataFrame(optimized_weights_sortino, index=dates, columns=commodity_names)
output_excel_path = 'D:\\2023semester\\Commodity_portfolio optimization\\optimization_sortino_ratio.xlsx'
Sortino_Ratio_weight.to_excel(output_excel_path)
print("Successful: Sortino Ratio")


############################################################################################################
######################################################################################
#minimum correlation
w = cp.Variable(n_assets)
correlation = cp.quad_form(w, Rt[:, :, 0])
objective = cp.Minimize(correlation)
constraints = [cp.sum(w) == 1, w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()
optimal_weights_min_correlation = w.value

#Time-varying
optimal_weights_time_varying_min_correlation = np.zeros((n_assets, T))

for t in range(T):
    R_t = Rt[:, :, t]
    correlation = cp.quad_form(w, R_t)
    objective = cp.Minimize(correlation)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status not in ["infeasible", "unbounded"]:
        optimal_weights_time_varying_min_correlation[:, t] = w.value
    else:
        print(f"Optimization issue at time {t}: {problem.status}")
        optimal_weights_time_varying_min_correlation[:, t] = np.nan

optimal_weights_min_correlation_transposed = optimal_weights_time_varying_min_correlation.T

min_correlation_weight = pd.DataFrame(optimal_weights_min_correlation_transposed, index=dates, columns=commodity_names)
output_excel_path = 'D:\\2023semester\\Commodity_portfolio optimization\\optimization_min_correlation.xlsx'
min_correlation_weight.to_excel(output_excel_path)
print("Successful: Min-correlation")

#############################################################################################################
#######################################################################################
#mean-CVaR
alpha = 0.95
window_size = 250
n_assets = data.shape[1]
n_days = data.shape[0]
optimal_weights_mean_CVaR = np.zeros((n_days - window_size + 1, n_assets))

for i in range(n_days - window_size + 1):
    historical_window = data[i:i + window_size].values
    sorted_returns = np.sort(historical_window, axis=0)
    var_index = int(np.ceil((1 - alpha) * sorted_returns.shape[0])) - 1
    VaR_95 = sorted_returns[var_index]
    CVaR_95 = np.mean(sorted_returns[:var_index], axis=0)
    
    w = cp.Variable(n_assets)
    portfolio_CVaR = w.T @ CVaR_95
    objective = cp.Minimize( portfolio_CVaR )
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    optimal_weights_mean_CVaR[i] = w.value

dates = data.index[window_size - 1:]
mean_CVaR_weight = pd.DataFrame(optimal_weights_mean_CVaR, index=dates, columns=data.columns)

output_excel_path = 'D:\\2023semester\\Commodity_portfolio optimization\\optimization_mean_CVaR.xlsx'
mean_CVaR_weight.to_excel(output_excel_path)
print("Successful: mean_CVaR")
#############################################################################################
#############################################################################################
print(mean_variance_weight.shape)
print(min_variance_weight.shape)
print(Sharpe_Ratio_weight.shape)
print(Sortino_Ratio_weight.shape)
print(min_correlation_weight.shape)
print(mean_CVaR_weight.shape)
#############################################################################################
#############################################################################################
# average weights
average_mean_variance = mean_variance_weight.mean(axis=0)
average_min_variance = min_variance_weight.mean(axis=0)
average_sharpe_ratio = Sharpe_Ratio_weight.mean(axis=0)
average_sortino_ratio = Sortino_Ratio_weight.mean(axis=0)
average_min_correlation = min_correlation_weight.mean(axis=0)
average_mean_CVaR = mean_CVaR_weight.mean(axis=0)

average_weights_df = pd.DataFrame({
    'Mean-Variance': average_mean_variance,
    'Min-Variance': average_min_variance,
    'Min-Correlation': average_min_correlation,
    'Sharpe Ratio': average_sharpe_ratio,
    'Sortino Ratio': average_sortino_ratio,
    'Mean-CVaR': average_mean_CVaR
})

equal_weights = pd.Series([1/3]*len(average_mean_variance), index=average_mean_variance.index)
average_weights_df.insert(0, 'Equal Weight', equal_weights)
output_excel_path = 'D:\\2023semester\\Commodity_portfolio optimization\\average_weights.xlsx'
average_weights_df.to_excel(output_excel_path, float_format='%.6f')
print("Successful: average weights")

#############################################################################################
#############################################################################################
# portfolio performance
mean_CVaR_weight = mean_CVaR_weight.reindex(data.index, fill_value=np.nan)
equal_weight_weight = pd.read_excel('D:\\2023semester\\Commodity_portfolio optimization\\optimization_equal_weight.xlsx', index_col=0)
weights = {
    "Equal Weight": equal_weight_weight,
    "Mean-Variance": mean_variance_weight,
    "Min Variance": min_variance_weight,
    "Min Correlation": min_correlation_weight,
    "Sharpe Ratio": Sharpe_Ratio_weight,
    "Sortino Ratio": Sortino_Ratio_weight,
    "Mean-CVaR": mean_CVaR_weight   }

portfolio_returns_dict = {}
for weight_name, weight_data in weights.items():
    returns = np.sum(weight_data.values * data.values, axis=1) 
    portfolio_returns_dict[weight_name] = pd.Series(returns, index=data.index)

risk_free_rate = 0.0002
significance_level = 0.05

portfolio_stats = {}
for portfolio, returns in portfolio_returns_dict.items():
    average_return = np.mean(returns)
    variance = np.var(returns)
    std_dev = np.sqrt(variance)
    sharpe_ratio = (average_return - risk_free_rate) / std_dev if std_dev != 0 else 0

    sorted_returns = np.sort(returns)
    cutoff_index = int(np.floor(significance_level * len(sorted_returns)))
    cvar = np.mean(sorted_returns[:cutoff_index])

    portfolio_stats[portfolio] = {
        'Return': average_return,
        'Variance': variance,
        'Sharpe Ratio': sharpe_ratio,
        'CVaR': cvar         }

stats_df = pd.DataFrame(portfolio_stats)
output_filepath = 'D:\\2023semester\\Commodity_portfolio optimization\\portfolio_performance.xlsx'
stats_df.to_excel(output_filepath, float_format='%.6f')
print('Successful: Performance')

print(stats_df.columns)
print(average_weights_df.columns)

############################################################################
############################################################################
############################################################################
############################################################################
#rolling performance
# mean_variance_weight = mean_variance_weight.reindex(data.index, method='ffill').fillna(0)
# min_variance_weight = min_variance_weight.reindex(data.index, method='ffill').fillna(0)
# Sharpe_Ratio_weight = Sharpe_Ratio_weight.reindex(data.index, method='ffill').fillna(0)
# Sortino_Ratio_weight = Sortino_Ratio_weight.reindex(data.index, method='ffill').fillna(0)
# min_correlation_weight = min_correlation_weight.reindex(data.index, method='ffill').fillna(0)
# mean_CVaR_weight = mean_CVaR_weight.reindex(data.index, method='ffill').fillna(0)
# equal_weight_weight = equal_weight_weight.reindex(data.index, method='ffill').fillna(0)

window_size = 250
rolling_portfolio_stats = {name: [] for name in weights.keys()}
for i in range(len(data) - window_size + 1):
    window_data = data.iloc[i:i+window_size]
    window_index = data.index[i+window_size-1]
    for name, weight_df in weights.items():
        window_weights = weight_df.iloc[i:i+window_size]
        returns = (window_data.values * window_weights.values).sum(axis=1)
        average_return = np.mean(returns)
        variance = np.var(returns)
        std_dev = np.sqrt(variance)
        sharpe_ratio = (average_return - 0.0002) / std_dev if std_dev != 0 else 0

        sorted_returns = np.sort(returns)
        VaR_index = int(len(sorted_returns) * 0.05)
        VaR = sorted_returns[VaR_index]
        cvar = np.mean(sorted_returns[:VaR_index])
        
        rolling_portfolio_stats[name].append({
            'Date': window_index,
            'Return': average_return,
            'Variance': variance,
            'Sharpe Ratio': sharpe_ratio,
            'CVaR': cvar
        })
        
all_stats_dfs = []
for name, stats in rolling_portfolio_stats.items():
    stats_df = pd.DataFrame(stats)
    stats_df.set_index('Date', inplace=True)
    stats_df['Portfolio'] = name
    all_stats_dfs.append(stats_df)

rolling_stats_df = pd.concat(all_stats_dfs)
rolling_stats_df_pivot = rolling_stats_df.pivot_table(index='Date', columns='Portfolio')
output_filepath = 'D:\\2023semester\\Commodity_portfolio optimization\\rolling_portfolio_performance.xlsx'
rolling_stats_df_pivot.to_excel(output_filepath, float_format='%.6f')
print('Successful: Rolling Performance')

############################################################################
############################################################################
############################################################################
############################################################################
#####plot
from matplotlib.ticker import FuncFormatter
portfolio_colors = ['yellow', 'green', 'orange', 'royalblue', 'darkturquoise', 'olive', 'gray']
portfolio_markers = ['o', 's', 'x', '^', 'd', 'v', '<']
line_width = 2
metrics = ['Return', 'Variance', 'CVaR']
titles = ['Equal Weight', 'Mean-Variance', 'Min Variance', 'Sharpe Ratio', 'Sortino Ratio', 'Min Correlation', 'Mean-CVaR']
def percentage_formatter(x, pos):
    return f'{x:.2f}%'
def decimal_formatter(x, pos):
    return f'{x:.2f}%'
def plot_metrics(rolling_stats_df_pivot, metrics, titles, colors, markers):
    fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(15, 8), sharex=True)
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for title, color, marker in zip(titles, colors, markers):
            if (metric, title) in rolling_stats_df_pivot.columns:
                ax.plot(rolling_stats_df_pivot.index, rolling_stats_df_pivot[(metric, title)], label=title, linewidth=line_width, color=color, marker=marker, markevery=int(len(rolling_stats_df_pivot) / 20))
        ax.set_title(f'{metric}', fontsize=18)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=10)

        if metric in ['Return', 'Variance']:
            ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
        
        if metric in [ 'CVaR']:
            ax.yaxis.set_major_formatter(FuncFormatter(decimal_formatter))

        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)  # Only create legend for the first subplot
        else:
            ax.legend().set_visible(False)  # Hide legends for the rest of the subplots

    fig.subplots_adjust(right=0.8, hspace=0.2)
    plt.show()
plot_metrics(rolling_stats_df_pivot, metrics, titles, portfolio_colors, portfolio_markers)


