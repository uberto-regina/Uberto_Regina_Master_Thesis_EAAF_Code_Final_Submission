import numpy as np
import pandas as pd
import cvxpy as cp

from sklearn.covariance import LedoitWolf

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

"""
Equivalence test from Appendix A, where we test if maximising expected returns under a volatility cap (Eq. 3.3) is dual 
to minimizing variance under a target return (Eq.3.4) to beat in the sense specified in Section 3.
"""

def compute_minvar_target_return(
    data,
    Prices,
    daily_rf_rate,
    previous_weights=None,
    prev_day=None,
    lambda_s = 0,
    lambda_tc = 0
):
    """
    One‑step convex optimization: minimize portfolio variance subject to
    1) w ≥ 0
    2) sum(w) = 1
    3) expected return ≥ return of the vol-capped max‑return portfolio

    Returns
    -------
    tuple
        - float: optimal portfolio expected return (μᵀwₒₚₜ) for use as split score.
        - np.array : a vector containing the optimizal weights.
    """
    # 1) Compute the vol‑capped max‑return portfolio (no L1 penalties)
    mu_cap, w_cap = compute_maxreturn_capped_vol(
        data, Prices, daily_rf_rate,
        previous_weights=None,
        prev_day=None,
        lambda_s=0,
        lambda_tc=0
    )
    R_target = mu_cap  # monthly expected return threshold obtained through the first optimization

    # 2) Compute monthly returns aligned with `data` index
    Returns = (
        Prices
        .resample('MS')
        .ffill()
        .pct_change()
        .shift(-1)
        .dropna()
    )
    Returns = Returns.reindex(data.index).dropna().sort_index()
    assets = Returns.columns
    n = len(assets)

    # 3) Sample means and shrinkage covariance
    mu_vec = Returns.mean().values
    Sigma = LedoitWolf().fit(Returns).covariance_

    # 4) Build CVXPY variable
    w = cp.Variable(n, nonneg=True)

    # 5) Constraints
    constraints = [
        cp.sum(w) == 1,
        mu_vec @ w >= R_target
    ]

    # 6) Objective: minimize variance
    objective = cp.Minimize(cp.quad_form(w, Sigma))

    # 7) Solve
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
        w_opt =  np.ones(n)/n
    else:
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Solver did not succeed. Status = {problem.status}. Using equal‐weighted weights instead.")
            w_opt = np.ones(n)/n
        else:
            w_opt = np.maximum(w.value, 0)
            w_opt = w_opt / np.sum(w_opt)

    # 8) Extract solution
    w_opt = np.maximum(w.value, 0)
    w_opt /= w_opt.sum()
    port_var_monthly = w_opt @ Sigma @ w_opt
    port_vol_annual = np.sqrt(port_var_monthly * 12)

    w_opt_series = pd.Series(w_opt, index=assets)
    return mu_vec @ w_opt, w_opt


def expanding_window_training_equivalence(MacroEconomic_indicators, Prices, daily_rf_rate):

    features = list(MacroEconomic_indicators.columns)

    test_periods = []
    test_start   = pd.Timestamp('2002-06-01')
    last_date    = pd.Timestamp('2024-06-01') 

    first_last = test_start  - relativedelta(months=1)


    while test_start < last_date:
        test_end = test_start + relativedelta(months=6) - relativedelta(days=1)
        if test_end > last_date:
            test_end = last_date
        test_periods.append((test_start, test_end))
        test_start = test_start + relativedelta(months=6)


    names = ['Max_Return', 'Min_Var']

    # container for all runs
    Weights_test = []


    for j, name in enumerate(names):
        print(f"\n=== TEST RUN {j+1}/{name} ===")
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


            if name == "Max_Return":
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
                    lambda_s =0., 		   
                    lambda_tc = 0.,     
                    boot = False, 
                    previous_weights=weight_last,
                    prev_day=last_day,
                    random_state=42
                )
            elif name == "Min_Var":
                Forest = EAAF_parallelized(
                    Prices,
                    ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY'],
                    train_macro,
                    train_rf,
                    compute_minvar_target_return,
                    max_depth=4,
                    min_samples_split=72,
                    n_trees=360,
                    bootstrap_frac=1,
                    lambda_s =0., 		   	
                    lambda_tc = 0.,     
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

        Weights_df['Name'] = f"experiment_{j}, {names[j]}"

        # collect this run’s list of DataFrames
        Weights_test.append(Weights_df)

    return Weights_test


if __name__ == '__main__':

    start_time = time.time()

    MacroEconomic_indicators, Prices, daily_rf_rate = macro_lagged()

    WT = expanding_window_training_equivalence(MacroEconomic_indicators, Prices, daily_rf_rate)
    
    with open("EAAF_TEST_EQUIVALENCE.pkl", "wb") as f:
       pickle.dump(WT, f)

    compare_weights_score_with_baseline(WT, Prices, MacroEconomic_indicators, daily_rf_rate)

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Total elapsed time: {elapsed:.2f} seconds")
