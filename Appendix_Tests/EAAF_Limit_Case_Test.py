import numpy as np
import pandas as pd

import cvxpy as cp

from sklearn.covariance import LedoitWolf
from pandas_datareader import data as web

from dateutil.relativedelta import relativedelta
import pickle
import time

import os
import sys

THIS_DIR = os.path.dirname(__file__)                          
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))  


sys.path.insert(1, PROJECT_ROOT)
from EAAF import macro_lagged, EAAF_parallelized, compute_maxreturn_capped_vol, EAAForest
from utilities_plots import compare_weights_score_with_baseline


def compute_maxreturn_no_capped_vol(
    data,
    Prices,
    daily_rf_rate,
    previous_weights=None,
    prev_day=None,
    lambda_s=0.,
    lambda_tc=0.
):
    """
    One‐step convex optimization: maximize portfolio expected return subject to
    1) w ≥ 0
    2) sum(w) = 1

    Inputs:
    - data:             A pandas DataFrame whose index aligns with reindexed Returns.
    - Prices:           A pandas DataFrame of historical asset prices (indexed by date).
    - daily_rf_rate:    (Unused here, but kept for API consistency.)
    - previous_weights: (optional) last period’s weights, used only if you want to
                        include a turnover‐penalty. If None, turnover penalty is ignored.
    - prev_day:         (optional) a date label corresponding to the row in Returns for realized returns.
    - lambda_s:            shrinkage‐penalty coefficient (only used if shrinkage_alloc is provided).
    - lambda_tc:        turnover penalty coefficient (only used if previous_weights is provided).
    - shrinkage_alloc:  (optional) function that returns (shrinkage_sigma, w_shrink) 
                        if you want an additional “shrinkage‐target” penalty. 
                        If None, no shrinkage term is added.

    Returns
    -------
    tuple
        - float: optimal portfolio expected return (μᵀwₒₚₜ) for use as split score.
        - np.array : a vector containing the optimizal weights.
    """

    # 1) Compute monthly returns aligned with `data` index
    Returns = (
        Prices
        .resample('MS')      # month‐start frequency
        .ffill()
        .pct_change()
        .shift(-1)           # so that return at month t corresponds to period [t, t+1)
        .dropna()
    )

    # 2) If we want realized returns for turnover adjustment:
    if prev_day is not None and previous_weights is not None:
        realized = Returns.loc[prev_day].values

    # 3) Restrict to the same dates as `data.index`
    Returns = Returns.reindex(data.index).dropna().sort_index()
    assets = Returns.columns
    n = len(assets)

    # 4) Compute historical expected returns (sample mean)
    T, p = Returns.shape
    mu_vec = Returns.mean().values

    # 5) Compute a shrinkage covariance estimate
    lw = LedoitWolf().fit(Returns)
    Sigma = lw.covariance_  # monthly covariance matrix

    EW_weights = np.ones(n)/n 
    var_ew_monthly = EW_weights @ Sigma @ EW_weights

    # 7) Set up previous‐weight drift (if applying turnover penalty)
    if previous_weights is None:
        w_prev_drift = np.ones(n) / n
    else:
        w_prev = previous_weights.copy()
        if prev_day is not None:
            w_prev_drift = w_prev * (1 + realized)
            w_prev_drift = w_prev_drift / w_prev_drift.sum()
        else:
            w_prev_drift = w_prev.copy()

    # 8) Build CVXPY variables
    w = cp.Variable(n, nonneg=True)     # portfolio weights
    # Epigraph variables for turnover penalty
    p = cp.Variable(n, nonneg=True)
    q = cp.Variable(n, nonneg=True)
    # Epigraph variables for shrinkage penalty (if used)
    if lambda_s > 0:
        u = cp.Variable(n, nonneg=True)
        v = cp.Variable(n, nonneg=True)

    # 9) Constraints:
    constraints = []
    # (a) Fully invested
    constraints.append(cp.sum(w) == 1)

    # (c) Turnover‐penalty epigraph (only if lambda_tc > 0)
    if lambda_tc > 0:
        constraints.append(p >= w - w_prev_drift)
        constraints.append(q >= w_prev_drift - w)

    # (d) Shrinkage‐penalty epigraph (only if gamma > 0 and shrinkage_alloc is provided)
    if lambda_s > 0:
        w_shrink = EW_weights.copy()
        constraints.append(u >= w - w_shrink)
        constraints.append(v >= w_shrink - w)

    # 10) Objective: maximize expected return − λ_tc * turnover_penalty − γ * shrinkage_penalty
    exp_ret = mu_vec @ w
    if lambda_tc > 0:
        turnover_term = lambda_tc * cp.sum(p + q)
    else:
        turnover_term = 0

    if lambda_s > 0:
        shrinkage_term = lambda_s * cp.sum(u + v)
    else:
        shrinkage_term = 0

    objective = cp.Maximize(exp_ret - turnover_term - shrinkage_term)

    # 11) Solve the QP
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(
            solver=cp.ECOS,
            abstol=1e-6,
            reltol=1e-6,
            feastol=1e-8,
            max_iters=5000
        )
    except cp.SolverError as e:
        print(f"SolverError occurred: {e}. Using equal‐weighted weights instead.")
        w_opt = EW_weights.copy()
    else:
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Solver did not succeed. Status = {problem.status}. Using equal‐weighted weights instead.")
            w_opt = EW_weights.copy()
        else:
            w_opt = np.maximum(w.value, 0)
            w_opt = w_opt / np.sum(w_opt)

    # 12) Compute realized annualized volatility and annualized expected return
    port_var_monthly = w_opt @ Sigma @ w_opt
    port_vol_annual = np.sqrt(port_var_monthly * 12.0)
    sharpe = mu_vec @ w_opt  / port_var_monthly
    sharpe_annual = sharpe * 12.0

    return mu_vec @ w_opt, w_opt

def expanding_window_training_limit_cases(MacroEconomic_indicators, Prices, daily_rf_rate):

    features = list(MacroEconomic_indicators.columns)

    test_periods = []
    test_start   = pd.Timestamp('2002-06-01')
    last_date    = pd.Timestamp('2024-06-01') #MacroEconomic_indicators.index.max()

    first_last = test_start  - relativedelta(months=1)


    while test_start < last_date:
        test_end = test_start + relativedelta(months=6) - relativedelta(days=1)
        if test_end > last_date:
            test_end = last_date
        test_periods.append((test_start, test_end))
        test_start = test_start + relativedelta(months=6)

    # number of independent repetitions
    Limit_Cases = ['High_Shrinkage_Penalty', 'High_Transaction_Cost_Penalty']

    # container for all runs
    Weights_test = []


    for j, case in enumerate(Limit_Cases):
        print(f"\n=== TEST RUN {j+1}/{case} ===")
        Weights= []

        weight_last = None
        last_day = first_last

        for start, end in test_periods:
            train_end = start - relativedelta(days=1)
            print(f" Train up to {train_end.date()} → Test {start.date()}–{end.date()}")

            # slice train/test
            train_macro = MacroEconomic_indicators.loc[:train_end]
            train_rf    = daily_rf_rate.loc[:train_end]
            test_macro  = MacroEconomic_indicators.loc[start:end]

            if case == "High_Shrinkage_Penalty":
                    
                Forest = EAAF_parallelized(
                    Prices,
                    ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY'],
                    train_macro,
                    train_rf,
                    compute_maxreturn_capped_vol,
                    max_depth=4,
                    min_samples_split=72, 
                    n_trees=360,
                    bootstrap_frac=1,
                    lambda_s = 1000, 		
                    lambda_tc = 0.,     
                    boot = False, 
                    previous_weights=weight_last,
                    prev_day=last_day,
                    random_state=42
                )
            
            elif case == "High_Transaction_Cost_Penalty":

                Forest = EAAF_parallelized(
                    Prices,
                    ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY'],
                    train_macro,
                    train_rf,
                    compute_maxreturn_no_capped_vol,
                    max_depth=4,
                    min_samples_split=72, 
                    n_trees=360,
                    bootstrap_frac=1,
                    lambda_s = 0., 		
                    lambda_tc = 1000,     
                    boot = False, 
                    previous_weights=weight_last,
                    prev_day=last_day,
                    random_state=42
                )


            Forest.fit(features)

            # one allocation call over the full 6-month block
            PA_6m = Forest.predict_sequence(test_macro, weight_last)
            _, _, _, weight_6m, _, _ = PA_6m.calculate_PnL_transactions()
            Weights.append(weight_6m)

            last_day = Weights[-1].iloc[-1].name
            weight_last = []

            for key in PA_6m.portfolios[-1].weights.keys():
                weight_last.append(PA_6m.portfolios[-1].weights[key])

        Weights_df = pd.concat(Weights, axis=0)

        Weights_df['Name'] = f"experiment_{j}, case = {case}"

        # collect this run’s list of DataFrames
        Weights_test.append(Weights_df)

    return Weights_test

if __name__ == '__main__':

    start_time = time.time()

    MacroEconomic_indicators, Prices, daily_rf_rate = macro_lagged()

    WT = expanding_window_training_limit_cases(MacroEconomic_indicators, Prices, daily_rf_rate)

    with open("EAAF_TEST_LIMIT_CASES.pkl", "wb") as f:
       pickle.dump(WT, f)

    compare_weights_score_with_baseline(WT, Prices, MacroEconomic_indicators, daily_rf_rate)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total elapsed time: {elapsed:.2f} seconds")

