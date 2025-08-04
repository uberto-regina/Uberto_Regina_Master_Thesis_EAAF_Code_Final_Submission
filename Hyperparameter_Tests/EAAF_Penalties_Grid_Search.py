import numpy as np
import pandas as pd

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

"""
Grid search over the two L1 penalties, as described in Section 5.2.
"""

def expanding_window_training_penalties_search(MacroEconomic_indicators, Prices, daily_rf_rate):

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

    

    # Create logspace grid for shrinkage penalty parameter
    n_total = 8

    # Number of strictly positive, log‑spaced points
    n_nonzero = n_total - 1  # = 6

    # Upper bound
    max_val = 0.004

    # Choose a small epsilon for smallest nonzero point
    epsilon = 0.5e-3

    # Generate 6 log‑spaced values from ε up to 0.002
    nonzeros = np.logspace(
        np.log10(epsilon),
        np.log10(max_val),
        num=n_nonzero
    )

    # Prepend exact zero
    grid = np.concatenate(([0.0], nonzeros))

    shrinkages_penaltys = grid 

    transaction_penaltys = [0,0.001,0.002,0.003,0.004]


    # container for all runs
    Weights_test = []

    for transac_penalty in transaction_penaltys:
        for shrinkage_penalty in shrinkages_penaltys:
            print(f"\n=== TEST RUN Transac Penalty {transac_penalty}/ Shrinkage Penalty {shrinkage_penalty} ===")
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

                # fit on all history so far  
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
                    lsmbda_s = shrinkage_penalty, 		 
                    lambda_tc = transac_penalty,     
                    boot = False, 
                    previous_weights=weight_last,
                    prev_day=last_day,
                    random_state=42
                )

                Forest.fit(features)

                # one allocation call over the full 6-month block
                PA_6m = Forest.predict_sequence(test_macro, weight_last)
                _, _, _, weight_6m, _,_ = PA_6m.calculate_PnL_transactions()
                Weights.append(weight_6m)

                last_day = Weights[-1].iloc[-1].name
                weight_last = []

                for key in PA_6m.portfolios[-1].weights.keys():
                    weight_last.append(PA_6m.portfolios[-1].weights[key])

            Weights_df = pd.concat(Weights, axis=0)

            Weights_df['Name'] = f"Transaction_penal_{transac_penalty}, Shrinkage_penal_{shrinkage_penalty}"

            # collect this run’s list of DataFrames
            Weights_test.append(Weights_df)

    return Weights_test

if __name__ == '__main__':

    start_time = time.time()

    MacroEconomic_indicators, Prices, daily_rf_rate = macro_lagged()

    WT = expanding_window_training_penalties_search(MacroEconomic_indicators, Prices, daily_rf_rate)
    
    with open("EAAF_TEST_PENALTIES_GRID_SEARCH.pkl", "wb") as f:
        pickle.dump(WT, f)

    compare_weights_score_with_baseline(WT, Prices, MacroEconomic_indicators, daily_rf_rate)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total elapsed time: {elapsed:.2f} seconds")

