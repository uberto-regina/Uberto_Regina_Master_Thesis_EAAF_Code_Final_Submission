import numpy as np
import pandas as pd
from datetime import datetime
import cvxpy as cp

from utilities import Portfolio
from sklearn.covariance import LedoitWolf

from utilities import Portfolios_Allocation

import matplotlib.pyplot as plt

import pickle
from LedoitWolfEst import single_hac_inference, relative_hac_inference, bootstrap_inference, get_returns_2

def get_and_plot_weights(Weights, Prices, daily_rf_rate, weights_plot = True, verbose = True):
    """
    Convert a time-indexed weights DataFrame into a `Portfolios_Allocation` object for metric calculation 
    and optionally plot asset weights over time as in Figure 5.1.

    Parameters
    ----------
    Weights : pd.DataFrame
        DataFrame indexed by date with each column representing an asset's portfolio weight.
    Prices : pd.DataFrame
        Historical price series for each asset.
    daily_rf_rate : float or pd.Series
        Constant or time-indexed risk-free rate used in performance metrics.
    weights_plot : bool, optional
        If True, generate a time-series plot of asset weights (see Figure 5.1). Defaults to True.
    verbose : bool, optional
        If True, calculate and print allocation metrics such as the Sharpe ratio (default is True).

    Returns
    -------
    Portfolios_Allocation
        An object encapsulating the full portfolio allocation for metrics computation.
    """

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

    if verbose == True:
        PA.calculate_Sharpe_Ratio_rebalancing(daily_rf_rate=daily_rf_rate.loc['2002':'2024'], freq = 'M')

        PnL,_,_,_,_, _ = PA.calculate_PnL_rebalancing()

        print("The PnL is ", PnL)

        print("-----adding transactions cost-----")

        PA.calculate_Sharpe_Ratio_rebalancing(daily_rf_rate=daily_rf_rate.loc['2002':'2024'], freq = 'M', ret = False, verbose = True, transactions= True)

        PnL_t,_,_,_,_ ,_ = PA.calculate_PnL_transactions()

        print("The PnL is ", PnL_t)
    
    if weights_plot:

        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Left subplot: Stacked Area Chart
        ax1.stackplot(Weights.index, Weights['SP500'], Weights['Bonds'], Weights['Commodities'], Weights['Credits'], Weights['HY'],
                    labels=['SP500', 'Gov. Bonds', 'Commodities', 'Credits', 'High Yield'])
        highlight_cov = datetime.strptime('2020-02-20', "%Y-%m-%d")

        highlight_cred_sui = datetime.strptime('2023-03-15', "%Y-%m-%d")

        highlight_lehman_brot = datetime.strptime('2008-09-15', "%Y-%m-%d")


        # Highlight by drawing a vertical line
        ax1.axvline(x=highlight_cov, color='red', linestyle='--', linewidth=2, label='Covid-19')
        ax1.axvline(x=highlight_cred_sui, color='red', linestyle='--', linewidth=2, label='Credit-Suisse Crisis')
        ax1.axvline(x=highlight_lehman_brot, color='red', linestyle='--', linewidth=2, label='Lehmann Brothers Collapse')
        ax1.legend(loc='upper left')
        ax1.set_title('Weights')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Weight')

        # Right subplot: Another graph (example: line plot of one column)
        ax2.plot(Prices.loc['2002':'2024'].index,  Prices.loc['2002':'2024']['SP500'] / Prices.loc['2002':'2024']['SP500'].iloc[0], label='SP500')
        ax2.plot(Prices.loc['2002':'2024'].index, Prices.loc['2002':'2024']['Bonds'] / Prices.loc['2002':'2024']['Bonds'].iloc[0], label='Gov. Bonds')
        ax2.plot(Prices.loc['2002':'2024'].index, Prices.loc['2002':'2024']['Commodities'] / Prices.loc['2002':'2024']['Commodities'].iloc[0], label='Commodities')
        ax2.plot(Prices.loc['2002':'2024'].index, Prices.loc['2002':'2024']['Credits'] / Prices.loc['2002':'2024']['Credits'].iloc[0], label='Credits')
        ax2.plot(Prices.loc['2002':'2024'].index, Prices.loc['2002':'2024']['HY'] / Prices.loc['2002':'2024']['HY'].iloc[0], label='High Yield')
        ax2.axvline(x=highlight_cov, color='red', linestyle='--', linewidth=2, label='Covid-19')
        ax2.axvline(x=highlight_cred_sui, color='red', linestyle='--', linewidth=2, label='Credit-Suisse Crisis')
        ax2.axvline(x=highlight_lehman_brot, color='red', linestyle='--', linewidth=2, label='Lehmann Brothers Collapse')
        ax2.set_title('Prices')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Weight')
        ax2.legend(loc='upper left')

        plt.tight_layout()
        plt.show()
    
    return PA

def max_return_capped_vol_rolling_sharpe(data, Prices, allocation_index):
    """
    Function to compute the Rolling Sharpe Strategy of Section 4.2.
    """
    def compute_max_return_capped(R):
        # expected returns
        mu = R.mean().values
        n  = len(mu)

        # shrinkage covariance estimate
        Sigma = LedoitWolf().fit(R).covariance_

        # equal-weight vol target
        w_ew    = np.ones(n) / n
        vol_ew2 = float(w_ew @ Sigma @ w_ew)

        # CVXPY variable
        w = cp.Variable(n, nonneg=True)
        # objective: max μᵀ w
        obj = cp.Maximize(mu @ w)
        # constraints: full invest, non-negative, cap vol
        cons = [
            cp.sum(w) == 1,
            cp.quad_form(w, Sigma) <= vol_ew2
        ]

        prob = cp.Problem(obj, cons)
        prob.solve()   

        w_opt = np.clip(w.value, 0, None)
        w_opt /= w_opt.sum()
        return w_opt

    assets     = ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']
    start_date = allocation_index[0]
    end_date   = allocation_index[-1]
    valid_dates = Prices.index[(Prices.index >= start_date) & (Prices.index <= end_date)]
    Weights   = pd.DataFrame(index=Prices.index, columns=['w'])

    for date in valid_dates:
        # 8-year window
        t0 = date - pd.DateOffset(years=8)
        Psn = Prices[assets].loc[t0:date]
        # monthly returns
        R   = Psn.resample('MS').ffill().pct_change().dropna()

        R   = R.reindex(data.index).dropna()

        w_opt = compute_max_return_capped(R)
        Weights.at[date, 'w'] = w_opt

    # Build Portfolio objects as before
    lst = Weights.dropna().rename(columns={'w':'Weights'})
    lst['label'] = pd.factorize(lst['Weights'].apply(tuple))[0]
    lab_seen, portfolios = [], []
    newP = Prices[assets].loc[lst.index]

    for t in lst.index:
        lbl = lst.at[t, 'label']
        if lbl not in lab_seen:
            wt = lst.at[t, 'Weights']
            weights_dict = {a: wt[i] for i,a in enumerate(assets)}
            p = Portfolio(f"Portfolio {lbl}", weights_dict, assets, newP)
            portfolios.append(p)
            lab_seen.append(lbl)

    lst['label'] = "Portfolio " + lst['label'].astype(str)
    allocation = pd.DataFrame(lst['label'])
    PA = Portfolios_Allocation(newP, portfolios, allocation, assets)
    return PA

def compare_weights_score_with_baseline(
    list_weights, Prices, MacroEconomic_indicators,
    daily_rf_rate, verbose=False, names=None, transa=0.002
):
    """
    Compare EAAF model-generated portfolio performance against benchmark allocations.

    This function takes a list of DataFrames (one per model run), each indexed by date with asset columns representing portfolio weights. 
    It calculates the resulting wealth trajectories for each set of weights and benchmarks them against predefined baselines. 
    To include the AAF baseline without retraining, it loads the allocation stored in `AAF_TEST_BASE.pkl`; you can create your own AAF allocation file using the `AAF_Base_test.py` script.

    Parameters
    ----------
    list_weights : list of pd.DataFrame
        Each DataFrame maps dates to asset weights for a single model run.
    Prices : pd.DataFrame
        Historical price data for each asset.
    MacroEconomic_indicators : pd.DataFrame
        Time series of macroeconomic variables used in the models.
    daily_rf_rate : float or pd.Series
        The daily risk-free rate, either as a constant or a time-indexed series.
    verbose : bool, optional
        If True, print detailed progress and diagnostic information (default is False).
    names : list of str, optional
        Labels for each model run, used in plotting and output (default is None).
    transa : float, optional
        Transaction cost rate applied to portfolio rebalancing (default is 0.002).

   Returns
    -------
    None
        Generates and displays a comparison plot of cumulative wealth trajectories for the EAAF model portfolios against benchmark allocations (see Figure 5.2).
    """

    Sortinos, Geom_Sharpes, Arithm_Sharpes = [], [], []
    Volas, Max_Draws, Ulcers = [], [], []
    Pnls, Port_vals = [], []
    allocations_indexs, Returns, Turnovers = [], [], []

    # Evaluate each weight vector
    for i, w in enumerate(list_weights):
        Pa_w = get_and_plot_weights(
            w.loc[w.index.intersection(daily_rf_rate.index)],
            Prices, daily_rf_rate, verbose=verbose, weights_plot=verbose,
        )
        allocations_indexs.append(Pa_w.allocation.index)

        # Performance & risk metrics with rebalancing
        (Sortino_w, Geom_Sharpe_w, Arith_Sharpe_w,
         Vola_w, Max_Draw_w, Ulcer_w, Turnover_w) = \
            Pa_w.calculate_Sharpe_Ratio_rebalancing(
                daily_rf_rate=daily_rf_rate,
                freq="M", ret=True, verbose=verbose,
                transactions=True, transactions_cost=transa
            )

        # P&L and returns with transaction costs
        PnL_w, _, Returns_w, _, Port_val_w, _ = \
            Pa_w.calculate_PnL_transactions(
                transaction_cost_rate=transa
            )

        Sortinos.append(Sortino_w)
        Geom_Sharpes.append(Geom_Sharpe_w)
        Arithm_Sharpes.append(Arith_Sharpe_w)
        Volas.append(Vola_w * np.sqrt(12))
        Max_Draws.append(Max_Draw_w)
        Ulcers.append(Ulcer_w)
        Pnls.append(PnL_w)
        Port_vals.append(Port_val_w)
        Returns.append(Returns_w)
        Turnovers.append(Turnover_w)

    # Check all allocation dates align
    if not all(idx.equals(allocations_indexs[0]) for idx in allocations_indexs):
        print("Error on the indices")

    # Baseline AAF 

    try:
        with open("AAF_TEST_BASE.pkl", "rb") as f:
            wAAT = pickle.load(f)
    except Exception as e:
        print(f"Couldn't load AAF_TEST_BASE.pkl, use AAF_Base_Test.py to train at least one the base AAF model")


    
    PA_AAT = get_and_plot_weights(
            wAAT[0].loc[wAAT[0].index.intersection(daily_rf_rate.index)],
            Prices, daily_rf_rate, verbose=verbose, weights_plot=verbose,
        )
    
    PnL_wAAT, _, Returns_wAAT, _, Port_val_wAAT, _ = \
            PA_AAT.calculate_PnL_transactions(
                transaction_cost_rate=transa
            )
    
    Port_vals.append(Port_val_wAAT)

    # Baseline Equal-Weight SP500, Bonds, Commodities, Credits, HY
    assets_sp = ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']
    new_prices = Prices[assets_sp].loc[allocations_indexs[0]]
    Port_SP = Portfolio(
        'Portfolio SP', {a: 1/5 for a in assets_sp}, assets_sp, new_prices
    )
    PA_base = Portfolios_Allocation(
        new_prices, [Port_SP],
        pd.DataFrame({'Allocation': 'Portfolio SP'}, index=allocations_indexs[0]),
        assets_sp
    )
    PA_base.calculate_Sharpe_Ratio_rebalancing(
        daily_rf_rate=daily_rf_rate, freq='M', ret=True,
        transactions=True, verbose=verbose, transactions_cost=transa
    )
    PnL_base, _, Returns_base, _, Port_val_base, _ = PA_base.calculate_PnL_transactions(
        transaction_cost_rate=transa
    )


    # Baseline 60-40
    assets_sp2 = ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']
    new_prices = Prices[assets_sp].loc[allocations_indexs[0]]
    Port_SP2 = Portfolio(
        'Portfolio SP2', {'SP500' : 0.6, 'Bonds' : 0.4, 'Commodities' : 0, 'Credits' : 0, 'HY' : 0}, assets_sp2, new_prices
    )
    PA_base2 = Portfolios_Allocation(
        new_prices, [Port_SP2],
        pd.DataFrame({'Allocation': 'Portfolio SP2'}, index=allocations_indexs[0]),
        assets_sp2
    )
    PA_base2.calculate_Sharpe_Ratio_rebalancing(
        daily_rf_rate=daily_rf_rate, freq='M', ret=True,
        transactions=True, verbose = verbose, transactions_cost=transa
    )
    PnL_base2, _, Returns_base2, _, Port_val_base2, _ = PA_base2.calculate_PnL_transactions(
        transaction_cost_rate=transa
    )
    Port_vals.append(Port_val_base2)


    # Baseline Rolling Sharpe
    PA_sharpe = max_return_capped_vol_rolling_sharpe(
        MacroEconomic_indicators, Prices, allocations_indexs[0]
    )
    PA_sharpe.calculate_Sharpe_Ratio_rebalancing(
        daily_rf_rate=daily_rf_rate, freq='M', ret=True,
        transactions=True, verbose = verbose, transactions_cost=transa
    )
    PnL_sh, _, Return_sh, _, Port_val_sh, _ = PA_sharpe.calculate_PnL_transactions(
        transaction_cost_rate=transa
    )
    Port_vals.append(Port_val_sh)

    Port_vals.append(Port_val_base)

    # Combine wealth trajectories
    Combined_portos = pd.concat(Port_vals, axis=1)

    # Set up names
    if names is not None:
        names1 = names.copy()
    else:
        names1 = [f'Test {i}' for i in range(len(list_weights))]
    names1 += ['Baseline AAF', 'Baseline 60-40', 'Baseline Rolling Sharpe', 'Baseline EW']
    Combined_portos.columns = names1

    # Split names into model vs. baseline
    model_names    = names1[:-4]
    baseline_names = names1[-4:]

    outperformances = []
    for ret in Returns:
        out_perf = pd.DataFrame((ret - Returns_base) / (1 + Returns_base))
        out_perf = out_perf.rename(columns={ret.name if hasattr(ret, 'name') else out_perf.columns[0]: 'Return'})
        out_perf['Return'] = (1 + out_perf['Return']).cumprod() - 1
        outperformances.append(out_perf)
    out_perf = pd.DataFrame((Returns_wAAT - Returns_base) / (1 + Returns_base))
    out_perf = out_perf.rename(columns={ret.name if hasattr(ret, 'name') else out_perf.columns[0]: 'Return'})
    out_perf['Return'] = (1 + out_perf['Return']).cumprod() - 1
    outperformances.append(out_perf)

    out_perf = pd.DataFrame((Returns_base2 - Returns_base) / (1 + Returns_base))
    out_perf = out_perf.rename(columns={ret.name if hasattr(ret, 'name') else out_perf.columns[0]: 'Return'})
    out_perf['Return'] = (1 + out_perf['Return']).cumprod() - 1
    outperformances.append(out_perf)

    out_perf = pd.DataFrame((Return_sh - Returns_base) / (1 + Returns_base))
    out_perf = out_perf.rename(columns={ret.name if hasattr(ret, 'name') else out_perf.columns[0]: 'Return'})
    out_perf['Return'] = (1 + out_perf['Return']).cumprod() - 1
    outperformances.append(out_perf)

    combined_outperf = pd.concat(outperformances, axis=1)
    combined_outperf.columns = names1[:-1]


    def get_distinct_colors(n, cmap_name="tab10"):
        """
        Pull the first n colors from a qualitative colormap.
        If n exceeds the map’s size, cycles through it again.
        """
        cmap = plt.get_cmap(cmap_name)
        base_colors = cmap.colors  # tuple of RGBA
        # pick n of them
        return [base_colors[i % len(base_colors)] for i in range(n)]

    model_colors = get_distinct_colors(len(model_names), cmap_name="tab10")

    baseline_colors = plt.get_cmap("Set2").colors[:4]

    # plotting

    all_colors = model_colors + list(baseline_colors)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Wealth trajectories
    Combined_portos[model_names].plot(
    ax=axes[0],
    color=model_colors,
    linewidth=1.5,
    linestyle='-',
    legend=False
)

    Combined_portos[baseline_names].plot(
        ax=axes[0],
        color=baseline_colors,
        linewidth=1.5,
        linestyle='--'
    )

    axes[0].set_title('Model vs. Baseline Wealth')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Portfolio Value')
    axes[0].legend(model_names + baseline_names, ncol=2, fontsize='small')


    combined_outperf[model_names].plot(
    ax=axes[1],
    color=model_colors,
    linewidth=1.5,
    linestyle='-',
    legend=False
)

    combined_outperf[baseline_names[:-1]].plot(
        ax=axes[1],
        color=list(baseline_colors)[:-1],
        linewidth=1.5,
        linestyle='--'
    )

    # Labels, title, legend
    axes[1].set_title('Outperformance Plot')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Cumulative Outperformance')
    axes[1].legend(model_names + baseline_names, title='Series', ncol=2, fontsize='small')

    plt.tight_layout()
    plt.show()

def statistics_comparison_EW(list_weights, Prices, MacroEconomic_indicators, daily_rf_rate,
                          verbose=False, names=None, transac = 0.002):
    """
    Compare EAAF model-generated portfolios against equal-weight and AAF benchmarks.

    This function accepts the same inputs as `compare_weights_score_with_baseline`: 
    a list of weight DataFrames (one per model run), each indexed by date with asse-‐column weights. 
    It computes performance metrics (e.g., cumulative return, Sharpe ratio) for each model allocation,
    then benchmarks them against both an equal-weight (EW) portfolio and the pre-saved AAF allocation.
    The results are returned as a summary table analogous to Table 5.1 in the paper.

    Parameters
    ----------
    list_weights : list of pd.DataFrame
        Each DataFrame maps dates to asset weights for a single model run.
    Prices : pd.DataFrame
        Historical price data for each asset.
    MacroEconomic_indicators : pd.DataFrame
        Time series of macroeconomic variables used in the models.
    daily_rf_rate : float or pd.Series
        Daily risk-free rate, as a constant or date-indexed series.
    verbose : bool, optional
        If True, print detailed performance metrics for each run during computation (default is False).
    names : list of str, optional
        Labels for each model run, used in table headings (default is None).
    transac : float, optional
        Transaction cost rate applied to portfolio rebalancing (default is 0.002).

    Returns
    -------
    pd.DataFrame
        Summary table of comparative statistics for each model, the EW benchmark,
        and the AAF allocation (e.g., cumulative return, volatility, Sharpe ratio).
    """

    Sortinos = []
    Volas, Max_Draws, Ulcers      = [], [], []
    Pnls              = []
    Returns, Turnovers            = [], []
    SR_hats, CIS                  = [], []

    for i, w in enumerate(list_weights):
        Pa_w = get_and_plot_weights(w.loc[w.index.intersection(daily_rf_rate.index)], Prices, daily_rf_rate,weights_plot = False, verbose = verbose)
        Sortino_w, _ , _ , Vola_w, Max_Draw_w, Ulcer_w, Turnover_w = Pa_w.calculate_Sharpe_Ratio_rebalancing(daily_rf_rate=daily_rf_rate, freq = "M", ret = True, verbose = verbose, transactions= True, transactions_cost=transac)
        PnL_w,_,Returns_w,_,Port_val_w, _ = Pa_w.calculate_PnL_transactions(transaction_cost_rate=transac) #Pa_w.calculate_PnL_rebalancing()
        ret_agg = np.load("Datasets/ret_agg.npy")
        ret_hedge = np.load("Datasets/ret_hedge.npy")

        Returns_combined = get_returns_2(w.copy(), Prices, daily_rf_rate, transac)
        SR_hat,  ci_raw, _, _ = single_hac_inference(Returns_combined[:,0])
        # Append into lists:
        Sortinos.append(Sortino_w)
        Volas.append(Vola_w * np.sqrt(12))           # annualized
        Max_Draws.append(Max_Draw_w)
        Ulcers.append(Ulcer_w)
        Pnls.append(PnL_w)
        Returns.append(Returns_w)
        Turnovers.append(Turnover_w)
        SR_hats.append(SR_hat * np.sqrt(12))          # annualized
        # adjust CI to be (lower, upper) around the ann. Sharpe
        ci = [x * np.sqrt(12) for x in ci_raw]
        ci = [SR_hats[-1] - ci[0], ci[1] - SR_hats[-1]]
        CIS.append(ci)

    ### baseline EW

    prices_used_sp = Prices[['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']].loc[list_weights[0].index]
    new_prices = prices_used_sp[prices_used_sp.index.isin(list_weights[0].index)]  #[0.4,0.3,0.1,0.05,0.15]
    Port_SP = Portfolio('Portfolio SP', {'SP500' : 1/5, 'Bonds' : 1/5, 'Commodities': 1/5, 'Credits':  1/5, 'HY' : 1/5}, ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY'], new_prices)
    portfolios_sp = [Port_SP]
    allocation_sp = pd.DataFrame({'Allocation': 'Portfolio SP'}, index = list_weights[0].index)
    assets = ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']
    PA_base = Portfolios_Allocation(new_prices, portfolios_sp, allocation_sp, assets)
    Sortino_baseline, Geom_Sharpe_baseline, Arith_Sharpe_baseline, Vola_baseline, Max_Draw_baseline, Ulcer_baseline, Turnover_baseline = PA_base.calculate_Sharpe_Ratio_rebalancing(daily_rf_rate=daily_rf_rate, freq = 'M', ret = True, verbose = False, transactions = True, transactions_cost=transac)
    PnL_base,_,Returns_base,_,Port_val_base, _ = PA_base.calculate_PnL_transactions(transaction_cost_rate=transac) #PA_base.calculate_PnL_rebalancing()

    Sortinos.append(Sortino_baseline)
    Volas.append(Vola_baseline * np.sqrt(12))           # annualized
    Max_Draws.append(Max_Draw_baseline)
    Ulcers.append(Ulcer_baseline)
    Pnls.append(PnL_base)
    Returns.append(Returns_base)
    Turnovers.append(Turnover_baseline)

    SR_hat_base,  ci_raw_b, _, _ = single_hac_inference(Returns_combined[:,1])
    SR_hats.append(SR_hat_base* np.sqrt(12)) 

    #### baseline AAF

    try:
        with open("AAF_TEST_BASE.pkl", "rb") as f:
            wAAT = pickle.load(f)
    except Exception as e:
        print(f"Couldn't load AAF_TEST_BASE.pkl, use AAF_Base_Test.py to train at least one the base AAF model")

    
    PA_AAT = get_and_plot_weights(
            wAAT[0].loc[wAAT[0].index.intersection(daily_rf_rate.index)],
            Prices, daily_rf_rate, verbose=verbose
        )
    
    Sortino_AAT, Geom_Sharpe_AAT, Arith_Sharpe_AAT, Vola_AAT, Max_Draw_AAT, Ulcer_AAT, Turnover_AAT = \
    PA_AAT.calculate_Sharpe_Ratio_rebalancing(
        daily_rf_rate=daily_rf_rate, freq = 'M', ret = True, verbose = False, transactions = True, transactions_cost=transac
        )
    
    PnL_wAAT, _, Returns_wAAT, _, Port_val_wAAT, _ = \
            PA_AAT.calculate_PnL_transactions(
                transaction_cost_rate=transac
            )
    
    Returns_combined_AAT = get_returns_2(wAAT[0].copy(), Prices, daily_rf_rate, transac)
    SR_hat_AAT,  ci_raw_AAT, _, _ = single_hac_inference(Returns_combined_AAT[:,0])

    Sortinos.append(Sortino_AAT)
    Volas.append(Vola_AAT * np.sqrt(12))           # annualized
    Max_Draws.append(Max_Draw_AAT)
    Ulcers.append(Ulcer_AAT)
    Pnls.append(PnL_wAAT)
    Returns.append(Returns_wAAT)
    Turnovers.append(Turnover_AAT)

    SR_hats.append(SR_hat_AAT* np.sqrt(12)) 


    # Build strategy names
    if names is not None:
        strategies = names.copy()
    else:
        strategies = [f"Test {i}" for i in range(len(list_weights))]

    strategies.append("EW")
    strategies.append("AAF")
    # Unpack CIs
    ci_lower = [c[0] for c in CIS]
    ci_upper = [c[1] for c in CIS]

    # Mean returns and turnover
    mean_returns  = [np.mean(r) * 12 for r in Returns]      # annualized
    mean_turnover = [np.mean(t) * 12 for t in Turnovers]

    # Create DataFrame
    stats_df = pd.DataFrame({
        "Sortino":            Sortinos,
        "Sharpe Ratio":            SR_hats,
        "Volatility (ann.)":  Volas,
        "Max Drawdown":       Max_Draws,
        "Ulcer Index":        Ulcers,
        "Mean Return (ann.)": mean_returns,
        "Mean Turnover":      mean_turnover,
        "Total PnL":          Pnls,
    }, index=strategies)

    
    print(stats_df.T)


    return stats_df.T, stats_df.T.to_latex()
