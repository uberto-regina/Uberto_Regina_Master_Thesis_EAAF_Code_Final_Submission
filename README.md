# Enhancing the Asset Allocation Forest: A Robust Extension for Dynamic Portfolio Optimization
**Uberto Regina**, M.Sc. in Applied Mathematics  <br>
ETH Zürich, August 2025  <br>
*Supervisor:* Prof. Dr. Patrick Cheridito <br>
*Co-Supervisors:* Dr. Urban Ulrych and Antonello Cirulli, OLZ AG

This thesis introduces the **Enhanced Asset Allocation Forest (EAAF)**, an extension of the base Asset Allocation Forest (AAF) model presented in [Bettencourt et al., 2024][1]. The EAAF model refines and builds upon the AAF architecture to improve portfolio allocation performance.

In the following section, figure, and equation numbers refer to those in the thesis.

This repository includes:

- **EAAF model implementation and utilities**
- **Testing and evaluation scripts**
- **Hyperparameter search tests**

---

## To train and run an EAAF model

1. **Install dependencies.**  <be>
2. **Load your datasets in the `Datasets` folder and modify the `macro_lagged` function in `EAAF.py` to import them correctly**
2. **Execute the `EAAF.py`script:** <br>
   - **With parallel computing:**  <br>
     - Keep `EAAF.py` as-is (you can adjust the number of workers).  <br>
     - Run the script to train the full EAAF model and compare it to baselines using a 6-month expanding-window evaluation for out-of-sample training and testing. <br>
   - **Without parallel computing:**  <br>
     - In the training function, replace `EAAF_parallelized` with `EAAForest`. <br>
     - Run the script; all other steps remain the same. <br>

By default, `six_month_expanding_window_training` implements the scheme from Section 3.2.2: the model is trained up to each cutoff date, then evaluated strictly out-of-sample on subsequent data in sequence. To use a rolling-window approach or change the evaluation period, call the appropriate function or adjust the window length.

## Core Python Modules

1. **`EAAF.py`** <br>
   - Contains the model’s *core classes and functions*: <br>
     - `compute_maxreturn_capped_vol`: implements the main optimization of the model as described in Eq. 3.5. <br>
     - `Node` and `EAATree`, the fundamental building blocks of the ensemble. <br>
     - `EAAForest`, which assembles multiple `EAATree` instances into the final EAAF model according to the thesis specifications. <br>
   - Implements **`EAAF_parallelized`**, a parallelized version of the EAAF training routine. By distributing tree training across multiple CPU cores (default: 45, configurable), this function dramatically reduces computation time. <br>
   - Once trained, the EAAF model can generate out-of-sample asset weights for any given data point or `DataFrame` (indexed by date), given that you provide a starting weight that corresponds to the most recent allocation; if none, the model will assume an EW weight. The prediction routine returns a `Portfolios_Allocation`, which is a custom class explained below. <br>
   - Contains also the model's *core training and evaluation* functions: <br>
     - `six_month_expanding_window_training`: Implements the expanding window evaluation with 6-month out-of-sample periods as described in Section 4.2 <br>
     - `six_month_fixed_rolling_window_training`: Implements the fixed rolling-window evaluation as described in Section 5.1 with a customizable rolling window length. <br>
     - To customize the evaluation period (e.g., a different window length), adjust the relevant parameters in these functions.

2. **`utilities.py`** <br>
   - Contains some helper functions <br>
   - Defines the **`Portfolio`** class, representing a portfolio at a specific date, including asset names, corresponding weights, and optional share conversions. <br>
   - Introduces **`Portfolios_Allocation`**, a container for multiple `Portfolio` instances. Supply: <br>
     1. A list of `Portfolio` objects (each used once). <br>
     2. A `DataFrame` mapping each date to its portfolio name. <br>
   - Implements allocation-level metrics such as: <br>
     - `calculate_Sharpe_Ratio_rebalancing` (with or without transaction costs), which not only computes the Arithmetic Sharpe Ratio but also the Sortino Ratio, the Geometric Sharpe Ratio, the Volatility, the Maximum Drawdown, the Ulcer Index, and the total Turnover. <br>
     - `calculate_PnL_transactions` which computes the PnL (with or without transaction costs)

3. **`utilities_plot.py`** <br>
    - Contains the main function for plotting purposes <br>
    - `get_and_plot_weights`: it takes a DataFrame indexed by time where each column is an asset and its corresponding weight and transforms it into a `Portfolios_Allocation` for computing metrics. It can also plot the weights with respect to time, as shown in Figure 5.1. <br>
    - `compare_weights_score_with_baseline`: it takes as input a list of DataFrames indexed by time, where each column is an asset and its corresponding weight, which each corresponds to a model run, and compares all the wealth performance of the model with the baselines as in Figure 5.2. Note that to confront it with the AAF baseline instead of retraining everything, we have already pre-saved `AAF_TEST_BASE.pkl`, where a model's typical allocation is obtained. Use the script `AAF_Base_test.py` to generate your AAF and compare it. <br>
    - `max_return_capped_vol_rolling_sharpe`: function used inside  `compare_weights_score_with_baseline` to compute the rolling Sharpe baseline described in Section 4.2. <br>
    - `statistics_comparison_EW`: same input as `compare_weights_score_with_baseline`, and compares each model run to EW and AAF baselines with respect to the comparative metrics used throughout the paper, and gives the table seen for example in Table 5.1.

4. **`LedoitWolfEst.py`** <br>
   - Implements robust Sharpe ratio tests using HAC estimators, based on the “RobustSharpeRatioHAC” repository by Mark M. (2022) ([GitHub link](https://github.com/majkee15/RobustSharpeRatioHAC)).  <br>
   - Follows methodologies from:  <br>
     - Ledoit, O. & Wolf, M. (2008). _Robust performance hypothesis testing with the Sharpe ratio._ Journal of Empirical Finance, 15(5), 850–859.  <br>
     - Lo, A. W. (2002). _The statistics of Sharpe ratios._ Financial Analysts Journal, 58(4), 36–52.  <br>
   - Adds two functions, `get_returns_2` and `get_returns_single`, which convert a portfolio weights DataFrame into the corresponding returns series.  

## Folders

There are three main folders: `Datasets`, `Hyperparameter_Tests`, and `Appendix_Tests`

### Datasets

This folder should contain the main datasets used for training and evaluating the model as described in Section 4.1. Please note that because we do not own the datasets, they have not been added to the Datasets folder, so the code will not run as is. Before executing any script, load your data in this `Datasets` folder and update the `macro_lagged` function in `EAAF.py` to point to your data. <br>

The only files present are the two needed for running `LedoitWolfEst.py`.

### Hyperparameter_Tests

This folder includes all tests from Section 5 to determine the best hyperparameters, as well as the SHAP analysis from Section 6. Specifically:

- **`AAF_Base_Test.py`** <br>
 Functions to replicate the base AAF model of [Bettencourt et al., 2024][1].

- **`EAAF_Bootstrap.py`**  <br>
 Bootstrapping test from Section 5.3.4 to evaluate the benefit of adding bootstrapping.

- **`EAAF_JS_Mean_Shrinkage_Test.py`**  <br>
 JS mean shrinkage test from Section 5.3.3 to assess the value of JS mean shrinkage.

- **`EAAF_LW_Covariance_Shrinkage_Test.py`**  <br>
 Ledoit–Wolf covariance shrinkage test from Section 5.3.3 to assess the value of covariance shrinkage.

- **`EAAF_Min_Leaf_Test.py`** <br>
 Implements the procedure from Section 5.3.2 to identify the optimal minimum number of observations per leaf.
 
- **`EAAF_Numb_Trees_Test.py`**  <br>
 Test from Section 5.3.1 to determine how many trees are needed for output stability.

- **`EAAF_Penalties_Grid.py`**  <br>
 Grid search over the two L1 penalties, as described in Section 5.2.

- **`EAAF_SHAP_Values.py`**  <br>
 SHAP analysis scripts used in Section 6.

### Appendix_Tests

This folder includes all tests from the Appendix. Specifically:

- **`EAAF_Equivalence_Test`** <br>
 Equivalence test from Appendix A, where we test if maximising expected returns under a volatility cap (Eq. 3.3) is dual to minimizing variance under a target return (Eq.3.4) to beat in the sense specified in Section 3.

 - **`EAAF_Limit_Case_Test.py`** <br>
 The two limit test cases from Appendix C, where we test the model under high shrinkage penalty toward the EW benchmark (Figure C.1) and the model under high transaction cost penalty (without volatility constraint), as in Table C.2.

 - **`EAAF_Scoring_Function_Test.py`** <br>
 The Scoring Function test, where we test whether the scoring function (Eq. 3.8) is better using expected returns or the Sharpe Ratio.

...


[1]: Bettencourt, L. O., Tetereva, A., & Petukhina, A. (2024). _Advancing Markowitz: Asset Allocation Forest_. SSRN. https://doi.org/10.2139/ssrn.4781685
