"""
This implementation is based on the GitHub repository by Mark, M. (2022) 
titled RobustSharpeRatioHAC (https://github.com/majkee15/RobustSharpeRatioHAC). 
It provides routines for computing and testing Sharpe ratios following the methodologies described in:

Ledoit, O., & Wolf, M. (2008). 
“Robust performance hypothesis testing with the Sharpe ratio.” 
Journal of Empirical Finance, 15(5), 850–859.

Lo, A. W. (2002). 
“The statistics of Sharpe ratios.” 
Financial Analysts Journal, 58(4), 36–52.

In addition, we've introduced a get_returns_2 and get_returns_single functions that, given a portfolio weights DataFrame, computes and returns the corresponding returns DataFrame
"""


from typing import List
import numpy as np
import numpy.typing as npt
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
from scipy.stats import norm
from arch import bootstrap
from utilities import Portfolio
from utilities import Portfolios_Allocation
import pandas as pd
import yfinance as yf
from pandas_datareader import data as web

import pickle

NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.float64]

ret_agg = np.load("Datasets/ret_agg.npy")
ret_hedge = np.load("Datasets/ret_hedge.npy")


def relative_hac_inference(ret: npt.NDArray[np.float64], alpha: float = 0.05, rf: float = 0.0):
    """
    Computes relative Sharpe ratio statistics using asymptotic heteroscedasticity auto-correlation robust estimator
    as per
    References: 

    Ledoit, Oliver, and Michael Wolf.
    "Robust performance hypothesis testing with the Sharpe ratio.
    " Journal of Empirical Finance 15.5 (2008): 850-859.

    Lo, Andrew W.
    "The statistics of Sharpe ratios."
     Financial analysts journal 58.4 (2002): 36-52.

    Args:
        ret (np.ndarray): Return array of dim T x 2
        rf(float): Risk free rate
        alpha (float): Significance level

    Returns:
        (np.ndarray, float, tuple, float, float) : Sharpe ratios, sharpe ratio difference, confidence intervals
                p-value, standard error
    """

    # returns = np.vstack([ret1.values.flatten(), ret2.values.flatten()]).T
    ret = ret.astype(np.float64)
    sigma_hat = np.std(ret, axis=0)
    mu_hat = np.mean(ret, axis=0)
    SR_hat = mu_hat / sigma_hat
    standard_error = compute_se_parzen_relative(ret)
    ci = norm.ppf(1 - alpha / 2) * standard_error
    SR_diff = np.diff(SR_hat)[0]
    p_val = 2 * norm.cdf(-np.abs(SR_diff) / standard_error)
    return SR_hat, SR_diff, (SR_diff - ci, SR_diff + ci), p_val, standard_error


def compute_se_parzen_relative(ret: npt.NDArray[np.float64], rf: float = 0.0):
    """
    Computes the standard error of the Sharpe ratio estimator.

    References:
        Andrews, Donald WK.
        "Heteroskedasticity and autocorrelation consistent covariance matrix estimation."
        Econometrica: Journal of the Econometric Society (1991): 817-858.

    Args:
        ret (np.ndarray): Return array of dim T x 2
        rf(float): Risk free rate

    Returns:
        (float): Standard error of the Sharpe estimator
    """
    mu_hat = np.mean(ret, axis=0)
    rets_squared = np.square(ret)
    sigma_sq_hat = np.mean(rets_squared, axis=0)
    T = ret.shape[0]
    gradient = np.zeros(4)
    gradient[0] = sigma_sq_hat[0] / np.power(sigma_sq_hat[0] - mu_hat[0] ** 2, 1.5)
    gradient[1] = -sigma_sq_hat[1] / np.power(sigma_sq_hat[1] - mu_hat[1] ** 2, 1.5)
    gradient[2] = -0.5 * mu_hat[0] / np.power(sigma_sq_hat[0] - mu_hat[0] ** 2, 1.5)
    gradient[3] = -0.5 * mu_hat[1] / np.power(sigma_sq_hat[1] - mu_hat[1] ** 2, 1.5)
    v_hat = np.array([ret[:, 0] - mu_hat[0], ret[:, 1] - mu_hat[1],
                      rets_squared[:, 0] - sigma_sq_hat[0], rets_squared[:, 1] - sigma_sq_hat[1]]).T
    Psi_hat = compute_psi_hat(v_hat)  #
    standard_error = np.sqrt(gradient.T @ Psi_hat @ gradient / T)
    return standard_error


def compute_psi_hat(v_hat: npt.NDArray[np.float64]):
    """
        Computes limiting covariance matrix \hat{P\si}
    Args:
        v_hat (np.ndarray): Vector of feature parameters, e.g., (r_t - \hat{r_t}, r_t^2 - \hat{r_t)^2

    Returns:
        np.ndarray: Estimate of the limiting covariance Matrix \hat{\Psi}
    """
    T = len(v_hat)
    alpha_hat = compute_alpha_hat(v_hat)
    s_star = 2.6614 * (alpha_hat * T) ** 0.2
    s_star = np.minimum(s_star, T - 1)
    psi_hat = compute_gamma_hat(v_hat, 0)
    j = 1
    while j < s_star:
        gamma_hat = compute_gamma_hat(v_hat=v_hat, j=j)
        psi_hat = psi_hat + kernel_parzen(j / s_star) * (gamma_hat + gamma_hat.T)
        j = j + 1
    psi_hat = (T / (T - 4)) * psi_hat
    return psi_hat


def _vhat(returns: npt.NDArray[np.float64], rf: float = 0.0):
    mu_hat = np.mean(returns, axis=0)
    rets_squared = np.square(returns)
    sigma_sqr_hat = np.mean(rets_squared, axis=0)
    v_hat = np.array([returns[:, 0] - mu_hat[0], returns[:, 1] - mu_hat[1],
                      rets_squared[:, 0] - sigma_sqr_hat[0], rets_squared[:, 1] - sigma_sqr_hat[1]]).T
    return v_hat


def compute_gamma_hat(v_hat: npt.NDArray[np.float64], j: int):
    """
    Computes the gamma matrix from Ledoit & Wolf
    Args:
        v_hat (np.ndarray): Vector of feature parameters, e.g., (r_t - \hat{r_t}, r_t^2 - \hat{r_t)^2
        j (int): index of the matrix

    Returns:

    """
    T = v_hat.shape[0]
    p = v_hat.shape[1]
    gamma_hat = np.zeros((p, p))
    if j >= T:
        raise ValueError("j must be smaller than the number of observations T!")
    for i in range(j, T):
        gamma_hat = gamma_hat + np.outer(v_hat[i,].T, v_hat[i - j, :])
    gamma_hat = gamma_hat / T
    return gamma_hat


def compute_alpha_hat(v_hat):
    """
     Estimates the optimal bandwidth for the Parzen Kernel as per Andres 1991
     Args:
         v_hat (np.ndarray): Vector of feature parameters, e.g., (r_t - \hat{r_t}, r_t^2 - \hat{r_t)^2

     Returns:
         float: Alpha value for the optimal bandwidth

     """
    # Optimal bandwidth methodology
    # Andrew 1991
    # Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation
    T = v_hat.shape[0]
    p = v_hat.shape[1]
    numerator = 0.0
    denominator = 0.0
    # Model for the circular bootstrap
    # VAR(1)
    for i in range(p):
        fit = AutoReg(v_hat[:, i], lags=1, trend='c', old_names=False).fit()
        rho_hat = fit.params[1]  # select the AR 1 parameter
        sigma_hat = np.sqrt(fit.sigma2)
        numerator = numerator + 4 * (rho_hat ** 2) * (sigma_hat ** 4) / ((1 - rho_hat) ** 8)
        denominator = denominator + (sigma_hat ** 4) / ((1 - rho_hat) ** 4)
    return numerator / denominator


def kernel_parzen(x: float):
    """
    Kernel defined as per
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation
    Andrew 1991
    Parzen kernel density estimator.
    Args:
        x (float):

    Returns:

    """
    result = 0.0
    if np.abs(x) <= 0.5:
        result = 1 - 6 * (x ** 2) + 6 * (np.abs(x) ** 3)
    elif np.abs(x) <= 1:
        result = 2 * (1 - np.abs(x)) ** 3
    return result


def block_size_calibrate(returns: npt.NDArray[np.float64], b_vec: List = [1, 3, 6, 10], alpha: float = 0.05,
                         M: int = 199, K: int = 1000, b_av: int = 5, T_start: int = 50):
    """
    Bootsrap methodology for optimal block-size calibration from Ledoit & Wolf
    Args:
        returns (np.ndarray): Tx2 matrix of retruns
        b_vec (np.ndarray): Int array of possible block sizes
        alpha (float): Significance level for confidence bands
        M (int): Number of bootstrap iterations per standard error estimate
        K (int): Number of bootstrap iterations for each blocksize
        b_av (int):
        T_start (int): Offset parameter

    Returns:
        np.ndarray: Aray of reject probabilities for each blocksize
    """
    b_size = len(b_vec)
    emp_reject_prob = np.zeros(b_size)
    sigma_hat = np.std(returns, axis=0)
    mu_hat = np.mean(returns, axis=0)
    SR_hat = mu_hat / sigma_hat
    SR_diff = np.diff(SR_hat)[0]
    T = returns.shape[0]
    var_data = np.zeros((T + T_start, 2))
    var_data[0, :] = returns[0, :]
    fit1 = sm.OLS(ret_agg[1:, 0], sm.add_constant(ret_agg[:(T - 1), :])).fit()
    fit2 = sm.OLS(ret_agg[1:, 1], sm.add_constant(ret_agg[:(T - 1), :])).fit()
    coef1 = fit1.params
    coef2 = fit2.params
    resid_mat = np.vstack([fit1.resid, fit2.resid]).T
    # circumvent the number of repeats param
    resid_mat_bootstrap_cheat = np.vstack([resid_mat, resid_mat[-T_start:, :]])
    bs = bootstrap.StationaryBootstrap(b_av, resid_mat_bootstrap_cheat[1:, :])
    resid_mat_star = np.zeros_like(resid_mat_bootstrap_cheat)
    for m, _resid_mat_star in enumerate(bs.bootstrap(K)):
        resid_mat_star[1:, :] = _resid_mat_star[0][0]
        for t in range(1, T + T_start - 1):
            var_data[t, 0] = coef1[0] + coef1[1] * var_data[t - 1, 0] + coef1[2] * var_data[t - 1, 0] + resid_mat_star[
                t, 0]
            var_data[t, 1] = coef2[0] + coef2[1] * var_data[t - 1, 1] + coef2[2] * var_data[t - 1, 1] + resid_mat_star[
                t, 1]
        # no truncation here --> no t_start used
        var_data_trunc = var_data[T_start:(T_start + T)]
        for j in range(b_size):
            p_val = bootstrap_inference(var_data_trunc, block_size=b_vec[j], M=M, delta_null=SR_diff)[3]
            if p_val <= alpha:
                emp_reject_prob[j] = emp_reject_prob[j] + 1
        emp_reject_prob /= K
        closest_rejecet_prob = np.abs(emp_reject_prob - alpha)
        return closest_rejecet_prob


def sharpe_diff(returns):
    """
    Computes relative sharpe ratio
    Args:
        returns (np.ndarray): T x 2 matrix of returns

    Returns:
        float

    """
    sigma_hat = np.std(returns, axis=0)
    mu_hat = np.mean(returns, axis=0)
    SR_hat = mu_hat / sigma_hat
    SR_diff = np.diff(SR_hat)[0]
    return SR_diff


def bootstrap_inference(returns: npt.NDArray[np.float64], block_size: int, alpha: float = 0.05, M: int = 100,
                        delta_null: float = 0.0, return_resid=False):
    """
    Computes relative Sharpe ratio statistics using bootsrap methodology from Ledoit & Wolf
    Args:
        returns (np.ndarray): Return array of dim T x 2
        block_size (int): Block size for the circular block bootstrap
        alpha (float): Significance level
        M (int): Number of bootstrap draws
        delta_null (float): Risk free rate
    Returns:
        (np.ndarray, float, tuple, float, float) : Sharpe ratios, sharpe ratio difference, confidence intervals
                p-value, standard error

    """
    returns = returns.astype(np.float64)
    T = returns.shape[0]
    l = int(np.floor(T / block_size))
    sigma_hat = np.std(returns, axis=0)
    mu_hat = np.mean(returns, axis=0)
    SR_hat = mu_hat / sigma_hat
    SR_diff = np.diff(SR_hat)[0]
    hac_se = compute_se_parzen_relative(returns)
    d = np.abs(SR_diff - delta_null) / hac_se
    p_value = 1.0
    se = 0.0
    bs = bootstrap.CircularBlockBootstrap(block_size, returns)
    d_star_arr = np.zeros(M)
    d_star_arr_non_abs = np.zeros(M)
    for m, ret_star_boot in enumerate(bs.bootstrap(M)):
        ret_star = ret_star_boot[0][0]
        sigma_hat = np.std(ret_star, axis=0)
        mu_hat_star = np.mean(ret_star, axis=0)
        SR_hat_star = mu_hat_star / sigma_hat
        SR_diff_star = np.diff(SR_hat_star)[0]
        ret_star_squared = np.square(ret_star)
        sigma_sq_hat_star = np.mean(ret_star_squared, axis=0)
        gradient = np.zeros(4)
        gradient[0] = sigma_sq_hat_star[0] / np.power(sigma_sq_hat_star[0] - mu_hat_star[0] ** 2, 1.5)
        gradient[1] = -sigma_sq_hat_star[1] / np.power(sigma_sq_hat_star[1] - mu_hat_star[1] ** 2, 1.5)
        gradient[2] = -0.5 * mu_hat_star[0] / np.power(sigma_sq_hat_star[0] - mu_hat_star[0] ** 2, 1.5)
        gradient[3] = -0.5 * mu_hat_star[1] / np.power(sigma_sq_hat_star[1] - mu_hat_star[1] ** 2, 1.5)
        y_star = np.array([ret_star[:, 0] - mu_hat_star[0], ret_star[:, 1] - mu_hat_star[1],
                           ret_star_squared[:, 0] - sigma_sq_hat_star[0],
                           ret_star_squared[:, 1] - sigma_sq_hat_star[1]]).T
        psi_hat_star = np.zeros((4, 4), dtype='float64')
        for j in range(1, l):
            zeta_star = (block_size ** 0.5) * np.mean(y_star[((j - 1) * block_size):(j * block_size), :], axis=0)
            psi_hat_star = psi_hat_star + np.outer(zeta_star.T, zeta_star)
        psi_hat_star = psi_hat_star / l
        psi_hat_star = (T / (T - 4)) * psi_hat_star
        se_star = np.sqrt(gradient.T @ psi_hat_star @ gradient / T)
        d_star = np.abs(SR_diff_star - SR_diff) / se_star
        d_star_arr[m] = d_star
        d_star_arr_non_abs[m] = (SR_diff_star - SR_diff) / se_star
        se = se + se_star
        if d_star >= d:
            p_value = p_value + 1

    p_value = p_value / (M + 1)
    se = se / (M + 1)
    ci = np.quantile(d_star_arr, 1 - alpha) * hac_se
    if return_resid:
        return SR_hat, SR_diff, (SR_diff - ci, SR_diff + ci), p_value, se, d_star_arr_non_abs
    else:
        return SR_hat, SR_diff, (SR_diff - ci, SR_diff + ci), p_value, se


def single_hac_inference(returns: npt.NDArray[np.float64], rf: float = 0.0, alpha: float = 0.05):
    """
    Computes an absolute (for a single returns series) Sharpe ratio statistics using asymptotic heteroscedasticity auto-correlation robust estimator
    as per
    References:

    Ledoit, Oliver, and Michael Wolf.
    "Robust performance hypothesis testing with the Sharpe ratio.
    " Journal of Empirical Finance 15.5 (2008): 850-859.

    Lo, Andrew W.
    "The statistics of Sharpe ratios."
     Financial analysts journal 58.4 (2002): 36-52.

    Args:
        ret (np.ndarray): Return array of dim T x 2
        rf(float): Risk free rate
        alpha (float): Significance level

    Returns:
        (np.ndarray, float, tuple, float, float) : Sharpe ratios, sharpe ratio difference, confidence intervals
                p-value, standard error
    """
    returns = returns.astype(np.float64)
    returns = returns - rf
    sigma_hat = np.std(returns, axis=0)
    mu_hat = np.mean(returns, axis=0)
    SR_hat = mu_hat / sigma_hat
    standard_error = compute_se_parzen_single(returns)
    ci = norm.ppf(1 - alpha / 2) * standard_error
    p_val = 2 * norm.cdf(-np.abs(SR_hat) / standard_error)
    return SR_hat, (SR_hat - ci, SR_hat + ci), p_val, standard_error


def compute_se_parzen_single(returns: npt.NDArray[np.float64], rf: float = 0.0):
    mu_hat = np.mean(returns, axis=0)
    rets_squared = np.square(returns)
    sigma_hat = np.mean(rets_squared, axis=0)
    gradient = np.zeros(2)
    T = len(returns)
    # note that we are working with uncentered moments therefore for the delta method the function describing
    # sharpe ratio is given as
    # g(a, b) = a / sqrt(b - a^2)
    # gradient[0] = 1 / sigma_hat
    gradient[0] = (mu_hat - 2 * sigma_hat ** 2) / (2 * np.power(mu_hat - sigma_hat ** 2, 1.5))
    gradient[1] = (mu_hat * sigma_hat) / np.power(mu_hat - sigma_hat ** 2, 1.5)
    v_hat = np.array([returns - mu_hat, rets_squared - sigma_hat]).T
    Psi_hat = compute_psi_hat(v_hat)
    standard_error = np.sqrt(gradient @ Psi_hat @ gradient.T / T)
    return standard_error


def simple_sharpe_single(rets: npt.NDArray[np.float64], alpha=0.05, rf=0.0):
    """
    This method assumes that returns are independent and identically distributed draws from a normal distribution.
    Args:
        rets: ret (np.ndarray): Return array of dim T x 1
        rf(float): Risk free rate
        alpha (float): Significance level

    Returns:

    """
    T = rets.shape[0]
    sr = (np.mean(rets) - rf) / np.std(rets)
    se = np.sqrt((1 / T) * (1 + (sr * sr) / 2))
    ci = norm.ppf(1 - alpha / 2) * se
    p_val = 2 * norm.cdf(-np.abs(sr) / se)
    return sr, (sr - ci, sr + ci),  p_val, se

def get_returns_2(Weights, Prices, daily_rf_rate, transac_rate = 0.):
    Weights[0] = list(zip(Weights['SP500'], Weights['Bonds'], Weights['Commodities'], Weights['Credits'], Weights['HY']))

    lst = Weights[[0]].loc['2002':'2024']
    lst = pd.DataFrame(lst).rename(columns={0:'Weights'})
    lst['label'] = pd.factorize(lst['Weights'].apply(tuple))[0]
    lab_encountered = []
    portfolio_list = []
    new_prices = Prices[['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']]
    new_prices = new_prices[new_prices.index.isin(lst.index)]
    assets = ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']

    for t in lst.index:
        if lst['label'].loc[t] not in lab_encountered:
            weights = {}
            for i, asset in enumerate(assets):
                weights[asset] = lst['Weights'].loc[t][i]

            
            new_portfolio = Portfolio(f"Portfolio {lst['label'].loc[t]}", weights
                                    , assets, new_prices)
            portfolio_list.append(new_portfolio)
            lab_encountered.append(lst['label'].loc[t])

    lst['label'] = "Portfolio " + lst['label'].astype(str)
    prices_used = new_prices

    portfolios = portfolio_list
    allocation = pd.DataFrame(lst['label'])

    PA = Portfolios_Allocation(prices_used, portfolios, allocation, assets)

    _,_,Returns,_,_,_ = PA.calculate_PnL_transactions(transaction_cost_rate=transac_rate)

    Returns.dropna(inplace = True)

    prices_used_sp = Prices[['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']].loc[Weights.index]
    new_prices = prices_used_sp[prices_used_sp.index.isin(Weights.index)]

    Port_SP = Portfolio('Portfolio SP', {'SP500' : 1/5, 'Bonds' : 1/5, 'Commodities':1/5, 'Credits':1/5, 'HY' : 1/5}, ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY'], new_prices)
    portfolios_sp = [Port_SP]
    allocation_sp = pd.DataFrame({'Allocation': 'Portfolio SP'}, index = Weights.index)
    assets = ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']
    PA_base = Portfolios_Allocation(new_prices, portfolios_sp, allocation_sp, assets)
    _,_,Returns_base,_,_,_ = PA_base.calculate_PnL_transactions(transaction_cost_rate=transac_rate)
    Returns_base.dropna(inplace = True)

    RF_Rate = pd.DataFrame((1 + daily_rf_rate['rf']).resample('MS').prod() - 1)
    RF_Rate = RF_Rate.loc[Returns.index[0]:]

    Returns['Combined Port Return'] = Returns['Combined Port Return'] - RF_Rate['rf']
    Returns = Returns.rename(columns = {'Combined Port Return' : 'Excess-Return'})

    Returns_base['Combined Port Return'] = Returns_base['Combined Port Return'] - RF_Rate['rf']
    Returns_base = Returns_base.rename(columns = {'Combined Port Return' : 'Excess-Return'})

    Ret_comb = pd.concat([Returns, Returns_base], axis = 1).dropna()

    return Ret_comb.values

def get_returns_single(Weights, Prices, daily_rf_rate, transac_rate = 0.):
    Weights[0] = list(zip(Weights['SP500'], Weights['Bonds'], Weights['Commodities'], Weights['Credits'], Weights['HY']))

    lst = Weights[[0]].loc['2002':'2024']
    lst = pd.DataFrame(lst).rename(columns={0:'Weights'})
    lst['label'] = pd.factorize(lst['Weights'].apply(tuple))[0]
    lab_encountered = []
    portfolio_list = []
    new_prices = Prices[['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']]
    new_prices = new_prices[new_prices.index.isin(lst.index)]
    assets = ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']

    for t in lst.index:
        if lst['label'].loc[t] not in lab_encountered:
            weights = {}
            for i, asset in enumerate(assets):
                weights[asset] = lst['Weights'].loc[t][i]

            
            new_portfolio = Portfolio(f"Portfolio {lst['label'].loc[t]}", weights
                                    , assets, new_prices)
            portfolio_list.append(new_portfolio)
            lab_encountered.append(lst['label'].loc[t])

    lst['label'] = "Portfolio " + lst['label'].astype(str)
    prices_used = new_prices

    portfolios = portfolio_list
    allocation = pd.DataFrame(lst['label'])

    PA = Portfolios_Allocation(prices_used, portfolios, allocation, assets)

    _,_,Returns,_,_,_ = PA.calculate_PnL_transactions(transaction_cost_rate=transac_rate)

    Returns.dropna(inplace = True)

    RF_Rate = pd.DataFrame((1 + daily_rf_rate['rf']).resample('MS').prod() - 1)
    RF_Rate = RF_Rate.loc[Returns.index[0]:]

    Returns['Combined Port Return'] = Returns['Combined Port Return'] - RF_Rate['rf']

    Returns = Returns.rename(columns = {'Combined Port Return' : 'Excess-Return'})
    return Returns.values

def bootstrap_sharpe_ci(returns, block_size, alpha=0.05, M=1000, rf=0.0):
    excess = returns - rf
    T = len(excess)
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    SR_hat = mu / sigma

    bs = bootstrap.CircularBlockBootstrap(block_size, excess)
    SR_boot = np.empty(M)
    for m, data in enumerate(bs.bootstrap(M)):
        sample = data[0][0]
        SR_boot[m] = (sample.mean()) / sample.std(ddof=1)

    lower = np.quantile(SR_boot, alpha/2)
    upper = np.quantile(SR_boot, 1 - alpha/2)
    se_bs = SR_boot.std(ddof=1)
    p_bs = np.mean(np.abs(SR_boot) >= abs(SR_hat))
    return SR_hat, se_bs, (lower, upper), p_bs


    

    



    






