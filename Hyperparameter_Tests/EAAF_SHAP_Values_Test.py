import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import time
import shap
import concurrent.futures

from shap.plots import beeswarm
from shap import Explanation

import os
import sys

THIS_DIR = os.path.dirname(__file__)                          
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))  

sys.path.insert(1, PROJECT_ROOT)
from EAAF import macro_lagged, EAAF_parallelized, compute_maxreturn_capped_vol, EAAForest
from utilities_plots import compare_weights_score_with_baseline

"""
SHAP analysis scripts used in Section 6.
"""

_explainer = None
_features  = None
_max_evals  = None

def init_shap_worker(forest, features, background, max_evals):
    """
    Run once _in each child process_ to set up:
      - _explainer: a shap.PermutationExplainer bound to your fitted forest
      - _features, _max_evals for use in shap_worker
    """
    global _explainer, _features, _max_evals
    _features  = features
    _max_evals  = max_evals

    def predict_fn(X):
        # shap sends you either a DataFrame or ndarray
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=_features)
        raw = np.vstack(forest.predict_sequence(X, None, True).values)
        return raw

    _explainer = shap.PermutationExplainer(predict_fn, background)


def shap_worker(df_chunk):
    """
    Once init_shap_worker has run, this uses the global _explainer
    to compute SHAP values for a chunk.
    """
    return _explainer(df_chunk, max_evals=_max_evals).values


def EAAF_training_and_compute_SHAP_values(
    macroeco: pd.DataFrame,
    Prices,
    daily_rf_rate: pd.Series,
    n_samples: int = 100,
    background_size: int = 100,
    max_evals: int = 200,
    n_workers: int = None,
):
    # 1) Fit once in the parent
    features     = macroeco.columns.tolist()
    weight_names = ['SP500','Bonds','Commodities','Credits','HY']
    Forest = EAAF_parallelized(
        Prices, weight_names, macroeco, daily_rf_rate, compute_maxreturn_capped_vol,
        max_depth=4, min_samples_split=72, n_trees=360, bootstrap_frac=1,
        lambda_s=0., lambda_tc=0., previous_weights=None, prev_day=None, random_state=42
    )
    Forest.fit(features)

    # 2) Draw explain & background samples
    X_explain = macroeco.sample(n=min(n_samples, len(macroeco)), random_state=0)
    background = macroeco.sample(n=min(background_size, len(macroeco)), random_state=1)

    if n_workers is None:
        # ProcessPoolExecutor._max_workers defaults to number of CPUs
        n_workers = concurrent.futures.ProcessPoolExecutor()._max_workers


    n_workers = min(n_workers, len(X_explain))
    chunks = np.array_split(X_explain, n_workers)


    # 3) Launch pool with our top‐level initializer & worker
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_shap_worker,
            initargs=(Forest, features, background, max_evals)
        ) as exe:
        # submit each chunk for SHAP
        futures = [exe.submit(shap_worker, chunk) for chunk in chunks]
        # collect results in the same submit order
        list_of_arrays = [f.result() for f in futures]

    # 4) Stitch back into one array: (n_explain, n_features, n_weights)
    shap_arr = np.vstack(list_of_arrays)


    return shap_arr, X_explain

def plot_shap_results(
    shap_arr: np.ndarray,
    X_explain: pd.DataFrame,
    feature_names: list[str],
    weight_names: list[str],
    plot_bar: bool = True,
    plot_beeswarm: bool = True,
    figsize: tuple[int,int] = (15, 10)
):
    """
    Plot SHAP summaries for each output weight, with each plot in its own figure.

    Parameters
    ----------
    shap_arr : np.ndarray
        Array of shape (n_explain, n_features, n_weights) containing SHAP values.
    X_explain : pd.DataFrame
        The DataFrame of samples you explained (n_explain x n_features).
    feature_names : list of str
        Column names for the features (length = n_features).
    weight_names : list of str
        Names of each model output / weight (length = n_weights).
    plot_bar : bool
        If True, draw a mean-|SHAP| bar chart for each weight.
    plot_beeswarm : bool
        If True, draw a SHAP beeswarm (dot) plot for each weight.
    figsize : (width, height)
        Figure size passed through to shap summary/plots.
    """

    n_weights = len(weight_names)

    for i, wname in enumerate(weight_names):
        # extract SHAP values for this output
        sv = shap_arr[:, :, i]                    # (n_explain, n_features)
        expl = Explanation(values=sv,
                           data=X_explain.values,
                           feature_names=feature_names)

        # new figure per weight
        plt.figure(figsize=figsize)
        plt.title(f"{wname} — (n={sv.shape[0]})", fontsize=14)

        if plot_bar:
            # compute mean(|SHAP|) and plot bar chart
            bar_exp = expl.abs.mean(0)
            shap.plots.bar(bar_exp, show=False)
        elif plot_beeswarm:
            # beeswarm plot
            shap.plots.beeswarm(expl, show=False)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    start_time = time.time()

    MacroEconomic_indicators, Prices, daily_rf_rate = macro_lagged()

    shap_arr, X_explain = EAAF_training_and_compute_SHAP_values(MacroEconomic_indicators, Prices, daily_rf_rate)
    
    with open("shap_test_final.pkl", "wb") as f:
       pickle.dump(shap_arr, f)

    with open("X_explain_shap_test_final.pkl", "wb") as f:
       pickle.dump(X_explain, f)

    Features = ['Credit Spread', 'Term Spread', 'NFC Index', 'Ind. Production', 'Inflation', 'VIX', 'GDP Growth', 'PCEPI', 'Momentum']

    Assets = ['SP500', 'Gov. Bonds', 'Commodities', 'Credits', 'High Yield']
    
    plot_shap_results(shap_arr, X_explain, Features, Assets, plot_bar = False)

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Total elapsed time: {elapsed:.2f} seconds")
