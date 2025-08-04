import numpy as np
import pandas as pd

from pandas_datareader import data as web

import matplotlib.pyplot as plt

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
Test from Section 5.3.1 to determine how many trees are needed for output stability.
"""

def expanding_window_training_number_trees_test(MacroEconomic_indicators, Prices, daily_rf_rate):

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

    # number of independent repetitions
    n_test = 4

    number_trees = [45,90,135,180,225,270,315,360,405,450,495]

    # container for all runs

    Weights_final = []

    for i, nb_tree in enumerate(number_trees):
        Weights_test = []

        for j in range(n_test):
            print(f"\n=== TEST RUN {j+1}/{n_test} ===")
            print("number_trees", nb_tree)
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
                    n_trees=nb_tree,
                    bootstrap_frac=1,
                    previous_weights=weight_last,
                    prev_day=last_day,
                    lambda_s = 0., 		   
                    lambda_tc = 0.     
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

            Weights_df['Name'] = f"experiment_{j}, {nb_tree}"

            # collect this run’s list of DataFrames
            Weights_test.append(Weights_df)

        # collect this run’s list of DataFrames
        Weights_final.append(Weights_test)

    return Weights_final


def compare_weights_variability(list_weights, names = None):
    """
    Assess and visualize how portfolio weights vary across multiple model runs.

    For each asset weight column in the provided list of DataFrames, this function computes, at each date:
      - Standard deviation of weights across runs
      - Range (max minus min)
      - Mean weight
      - Coefficient of variation (std_dev / mean, with zero-mean handled safely)

    It then plots, in separate subplots (see Appendix D, Figure D.1), the weight trajectories from all runs for each asset.

    Parameters
    ----------
    list_weights : list of pd.DataFrame
        A sequence of DataFrames (one per test run), each indexed by date and containing the same set of weight columns.
    names : list of str, optional
        Labels for each run to use in the plot legends; if omitted, runs are labeled "Test 0", "Test 1", etc.

    Returns
    -------
    variability_all : pd.DataFrame
        A concatenated DataFrame with a two-level column index: first level is asset/weight name, second level is
        one of 'std_dev', 'range', or 'cv', containing the corresponding variability metric at each date.
    """
    
    n_weights = 5  # We have 4 weight columns in each test DataFrame.
    variability_stats = {}
    weight_run_data = {}

    # Loop over each weight column
    for weight_idx in range(n_weights):
        weight_name = f"Weight {weight_idx}"
        
        # Create DataFrame to hold values of the current weight across test runs
        weight_df = pd.DataFrame(index=list_weights[0].index)
        for i, test_df in enumerate(list_weights):
            weight_df[f'Test {i}'] = test_df.iloc[:, weight_idx]
        
        # Save the raw weight data for later plotting
        weight_run_data[weight_name] = weight_df.copy()
        
        # Compute variability metrics row-by-row (each time point)
        weight_df['std_dev'] = weight_df.std(axis=1)
        weight_df['range']   = weight_df.max(axis=1) - weight_df.min(axis=1)
        weight_df['mean']    = weight_df.mean(axis=1)
        weight_df['cv']      = weight_df['std_dev'] / weight_df['mean'].replace(0, np.nan)
        
        # Save only the variability metrics for this weight
        variability_stats[weight_name] = weight_df[['std_dev', 'range', 'cv']]
    
    # Combine the variability stats for all weights into one DataFrame with hierarchical columns.
    variability_all = pd.concat(variability_stats, axis=1)

    from matplotlib.gridspec import GridSpec
    import matplotlib.dates as mdates

    # ensure DateTimeIndex
    for df in weight_run_data.values():
        df.index = pd.to_datetime(df.index)

    fig = plt.figure(figsize=(12, 16))
    gs  = GridSpec(3, 2, figure=fig, height_ratios=[1,1,1])
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    axes.append(fig.add_subplot(gs[2, :]))

    locator   = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    for ax, (weight_name, df) in zip(axes, weight_run_data.items()):
        # plot
        for i, col in enumerate(df.columns):
            if col.startswith("Test"):
                label = names[i] if names else col
                ax.plot(df.index, df[col], marker='o', label=label)

        ax.set_title(f'{weight_name} Across Tests')
        ax.set_ylabel("Weight")
        ax.grid(True)
        ax.legend(loc='upper left', fontsize='small')

        # **apply date locator/formatter**
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # **rotate every x‑tick label**
        #plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.show()
    
    return variability_all

def EAAF_stability_CV_MAD_testing(list_weights_across_depth):
    """
    Assess CV and MAD stability across different EAAF model depths.

    For each model depth (outer list), this function expects four weight‐series DataFrames (inner list corresponding to four model run).
    It uses `weights_variability_test` to compute, for each asset weight:
      - Fraction of dates where CV ≤ 0.10
      - Fraction of dates where MAD ≤ 0.025

    It then plots two line charts side by side:
      - CV stability vs. number of trees
      - MAD stability vs. number of trees

    Parameters
    ----------
    list_weights_across_depth : list of list of pd.DataFrame
        Nested lists where each top-level element corresponds to a specific tree depth
        and contains four DataFrames of weight allocations from independent runs.

    Returns
    -------
    None
        Generates Matplotlib figures showing CV and MAD stability across tree counts as in Figure 5.4.
    """

    def weights_variability_test(list_weights,
                                cv_threshold=0.10,
                                mad_threshold=0.02,
                                cv_floor=0.):
        """
        Evaluate and summarize weight variability across multiple model runs.

        For each asset weight column in `list_weights`, this function computes at each date:
        - Standard deviation of weights across runs
        - Range (max minus min)
        - Mean weight
        - Coefficient of variation (std_dev / max(mean, cv_floor))
        - Mean absolute deviation (MAD) from the mean

        It then aggregates these metrics into:
        - A detailed time‐indexed DataFrame of variability statistics per weight
        - A summary table per weight indicating:
            * Fraction of dates where CV ≤ cv_threshold
            * Overall mean MAD
            * (If mad_threshold is set) Fraction of dates where MAD ≤ mad_threshold

        Parameters
        ----------
        list_weights : list of pd.DataFrame
            Sequence of weight DataFrames (one per test run), each indexed by date with identical columns.
        cv_threshold : float, optional
            CV cutoff for pass/fail classification (default 0.10).
        mad_threshold : float, optional
            MAD cutoff for pass/fail classification (default 0.02). If None, MAD thresholds are not evaluated.
        cv_floor : float, optional
            Minimum mean used in CV denominator to prevent division by small values (default 0.0).

        Returns
        -------
        variability_all : pd.DataFrame
            Multi‐level DataFrame of 'std_dev', 'range', 'mean', 'cv', and 'mad' for each weight, indexed by date.
        summary : pd.DataFrame
            One‐row per weight with columns:
            - pct_cv_within: fraction of dates with CV ≤ cv_threshold
            - mean_mad: average MAD over all dates
            - pct_mad_within (if mad_threshold set): fraction of dates with MAD ≤ mad_threshold
        """
        # Number of weight columns
        n_weights = list_weights[0].shape[1] -1
        variability_stats = {}

        # Compute stats for each weight
        for idx in range(n_weights):
            name = f"Weight {idx}"
            # Assemble runs DataFrame
            runs = pd.DataFrame({f"Test {i}": df.iloc[:, idx]
                                for i, df in enumerate(list_weights)},
                                index=list_weights[0].index)
            stats = pd.DataFrame(index=runs.index)
            stats['std_dev'] = runs.std(axis=1)
            stats['range']   = runs.max(axis=1) - runs.min(axis=1)
            stats['mean']    = runs.mean(axis=1)
            # Determine denominator for CV
            if cv_floor is not None:
                denom = stats['mean'].clip(lower=cv_floor)
            else:
                denom = stats['mean'].replace(0, np.nan)
            stats['cv']      = stats['std_dev'] / denom
            stats['mad']     = runs.sub(stats['mean'], axis=0).abs().mean(axis=1)

            variability_stats[name] = stats

        # Combine into one DataFrame
        variability_all = pd.concat(variability_stats, axis=1)

        # Build summary
        cols = ['pct_cv_within', 'mean_mad']
        if mad_threshold is not None:
            cols += ['pct_mad_within']

        # no dtype=… here
        summary = pd.DataFrame(index=variability_stats.keys(), columns=cols)

        for weight, df in variability_stats.items():
            pct_cv = (df['cv'] <= cv_threshold).mean()
            summary.loc[weight, 'pct_cv_within']        = pct_cv
            summary.loc[weight, 'mean_mad']            = df['mad'].mean()

            if mad_threshold is not None:
                pct_mad = (df['mad'] <= mad_threshold).mean()
                summary.loc[weight, 'pct_mad_within']    = pct_mad     

        return summary


    w0, w1, w2, w3, w4 = [], [], [], [], []
    mad0, mad1, mad2, mad3, mad4 = [], [], [], [], []
    for WT in list_weights_across_depth:
        w_df = weights_variability_test(WT, 0.1, 0.025, 0.)
        w0.append(w_df['pct_cv_within'].iloc[0])
        w1.append(w_df['pct_cv_within'].iloc[1])
        w2.append(w_df['pct_cv_within'].iloc[2])
        w3.append(w_df['pct_cv_within'].iloc[3])
        w4.append(w_df['pct_cv_within'].iloc[4])

        w_df = weights_variability_test(WT, 0.1, 0.025, 0.)
        mad0.append(w_df['pct_mad_within'].iloc[0])
        mad1.append(w_df['pct_mad_within'].iloc[1])
        mad2.append(w_df['pct_mad_within'].iloc[2])
        mad3.append(w_df['pct_mad_within'].iloc[3])
        mad4.append(w_df['pct_mad_within'].iloc[4])

    depths = [45, 90, 135, 180, 225, 270, 315, 360, 405, 450, 495]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    lists1 = [w0, w1, w2, w3, w4]
    labels1 = ['Weight 0', 'Weight 1', 'Weight 2', 'Weight 3', 'Weight 4']


    # First plot: stability with critical line
    ax = axes[1]
    for y, label in zip(lists1, labels1):
        ax.plot(depths, y, label=label)

    ax.set_xlabel('Number of Trees')
    ax.set_ylabel('Percentage of points with CV within 0.1')
    ax.set_title('CV Testing')
    ax.legend()
    ax.grid(True)

    lists2 = [mad0, mad1, mad2, mad3, mad4]
    labels2 = ['Weight 0', 'Weight 1', 'Weight 2', 'Weight 3', 'Weight 4']

    # Second plot: additional series
    ax2 = axes[0]
    for y, label in zip(lists2, labels2):
        ax2.plot(depths, y, label=label)

    ax2.set_xlabel('Number of Trees')
    ax2.set_ylabel('Percentage of points with MAD within 0.025')
    ax2.set_title('MAD Testing')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    start_time = time.time()

    MacroEconomic_indicators, Prices, daily_rf_rate = macro_lagged()

    WT = expanding_window_training_number_trees_test(MacroEconomic_indicators, Prices, daily_rf_rate)

    with open("EAAF_TEST_NUMBER_TREES.pkl", "wb") as f:
       pickle.dump(WT, f)

    EAAF_stability_CV_MAD_testing(WT)

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Total elapsed time: {elapsed:.2f} seconds")
