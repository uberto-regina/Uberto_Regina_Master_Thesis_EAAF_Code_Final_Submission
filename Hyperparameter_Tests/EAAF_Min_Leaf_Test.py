import numpy as np
import pandas as pd

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
Implements the procedure from Section 5.3.2 to identify the optimal minimum number of observations per leaf.
"""


def expanding_window_training_min_leaf(MacroEconomic_indicators, Prices, daily_rf_rate):

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

    min_leafs = [2, 20, 24, 28, 32, 36, 40, 44, 48, 52] 

    # container for all runs
    Weights_test = []


    for j, leaf in enumerate(min_leafs):
        print(f"\n=== TEST RUN {j+1}/{len(min_leafs)} ===")
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

            min_split = 2*leaf
            
            Forest = EAAF_parallelized(
                Prices,
                ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY'],
                train_macro,
                train_rf,
                compute_maxreturn_capped_vol,
                max_depth=4,
                min_samples_split=min_split, 
                n_trees= 360,
                bootstrap_frac=1,
                lambda_s = 0, 			
                lambda_tc = 0.,     
                boot = False, 
                previous_weights=weight_last,
                prev_day=last_day,
                random_state=42
            )

            Forest.fit(features)

            # one allocation call over the full 6-month block
            PA_6m = Forest.predict_sequence(test_macro, weight_last)
            _, _, _, weight_6m, _ , _ = PA_6m.calculate_PnL_transactions()
            Weights.append(weight_6m)

            last_day = Weights[-1].iloc[-1].name
            weight_last = []

            for key in PA_6m.portfolios[-1].weights.keys():
                weight_last.append(PA_6m.portfolios[-1].weights[key])

        Weights_df = pd.concat(Weights, axis=0)

        Weights_df['Name'] = f"experiment_{j}, leaf_size = {leaf}"

        # collect this run’s list of DataFrames
        Weights_test.append(Weights_df)

    return Weights_test


if __name__ == '__main__':

    start_time = time.time()

    MacroEconomic_indicators, Prices, daily_rf_rate = macro_lagged()

    WT = expanding_window_training_min_leaf(MacroEconomic_indicators, Prices, daily_rf_rate)

    with open("EAAF_TEST_MIN_LEAF.pkl", "wb") as f:
       pickle.dump(WT, f)

    compare_weights_score_with_baseline(WT, Prices, MacroEconomic_indicators, daily_rf_rate)

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Total elapsed time: {elapsed:.2f} seconds")
