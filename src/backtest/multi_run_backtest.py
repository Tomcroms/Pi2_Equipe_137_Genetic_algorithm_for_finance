import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model.data_loader import DataLoader
from src.model.stock import Stock
from src.model.genetic_algorithm import GeneticAlgorithm

# -------------------- Paths --------------------
PROJ_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJ_ROOT / "results"
FIG_DIR = PROJ_ROOT / "paper" / "figures"
FILE_PATH = PROJ_ROOT / "data" / "Cac40_Prices_2000_to_Today.xlsx"

# -------------------- Config defaults --------------------
WINDOW_MONTHS = 36
BUDGET = 1_000_000
IS_SHORT_AVAILABLE = False
SEED = 137
POP = 100
MAX_GENS = 250
RISK_AVERSION = 6

# -------------------- Helpers --------------------
def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def monthly_prices_from_loader():
    loader = DataLoader(str(FILE_PATH))
    loader.load_data()
    loader.preprocess_data()
    daily_prices = loader.raw_data
    monthly_prices = daily_prices.resample("M").last()
    monthly_prices = monthly_prices.dropna(axis=1, how="all")
    monthly_prices = monthly_prices.ffill().dropna(how="any")
    return monthly_prices

def to_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    return monthly_prices.pct_change().dropna(how="any")

def ewma_cov(window_rets: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    R = window_rets.values  # W x N
    W = R.shape[0]
    w = np.array([(1.0 - lam) * (lam ** k) for k in range(W-1, -1, -1)], dtype=float)
    w = w / w.sum()
    mu_w = (w[:, None] * R).sum(axis=0, keepdims=True)
    X = R - mu_w
    Sigma = np.einsum('t,ti,tj->ij', w, X, X)
    return Sigma * 12.0  # annualize monthly -> yearly

def build_window_stocks(prices_t: pd.Series, window_rets: pd.DataFrame,
                        cov_method: str, lam: float) -> tuple[list, np.ndarray]:
    mu_ann = window_rets.mean() * 12.0
    std_ann = window_rets.std() * np.sqrt(12.0)
    if cov_method == "sample":
        cov_ann = window_rets.cov().values * 12.0
    elif cov_method == "ewma":
        cov_ann = ewma_cov(window_rets, lam=lam)
    else:
        raise ValueError(f"Unknown cov_method {cov_method}")

    stocks = []
    for name in window_rets.columns:
        stocks.append(Stock(
            name=name,
            expected_return=float(mu_ann[name]),
            std_dev=float(std_ann[name]),
            price=float(prices_t[name])
        ))
    return stocks, cov_ann

def drift_weights(prev_w: np.ndarray, asset_rets: np.ndarray) -> np.ndarray:
    rp = float(np.dot(prev_w, asset_rets))
    denom = 1.0 + rp
    if np.isclose(denom, 0.0):
        return prev_w
    return prev_w * (1.0 + asset_rets) / denom

def turnover_and_cost(prev_w: np.ndarray, target_w: np.ndarray, asset_rets_t: np.ndarray, cost_bps_per_side: float):
    w_tilde = drift_weights(prev_w, asset_rets_t)
    delta = np.abs(target_w - w_tilde).sum()
    tcost = cost_bps_per_side / 10_000.0
    cost = tcost * delta
    return float(delta), float(cost)

def shares_to_weights(stocks: list, shares: np.ndarray, is_short: bool) -> np.ndarray:
    prices = np.array([s.price for s in stocks])
    positions = prices * shares
    if not is_short:
        long_pos = np.maximum(positions, 0.0)
        total = long_pos.sum()
        if total <= 0:
            return np.ones_like(positions) / len(positions)
        return long_pos / total
    else:
        total = np.sum(np.abs(positions))
        if total <= 0:
            u = np.random.randn(len(positions))
            return u / np.sum(np.abs(u))
        return positions / total

# -------------------- Metrics & Figures --------------------
def equity_curve(ret_net: pd.Series) -> pd.Series:
    return (1.0 + ret_net).cumprod()

def max_drawdown(eq: pd.Series) -> float:
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    return float(dd.min())

def annualized_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    rf_ann = 0.0
    for (strat, covm, costbps), g in df.groupby(["strategy","cov_method","cost_bps"]):
        g = g.sort_values("date")
        r = g["ret_net"]; n = len(r)
        if n == 0: 
            continue
        ret_geom = (1.0 + r).prod() ** (12.0 / n) - 1.0
        vol_ann = r.std(ddof=1) * np.sqrt(12.0)
        sharpe = (ret_geom - rf_ann) / vol_ann if vol_ann > 1e-12 else np.nan
        downside = r[r < 0.0]
        sortino = np.nan if len(downside)==0 else (ret_geom - rf_ann) / (downside.std(ddof=1)*np.sqrt(12.0) + 1e-12)
        eq = equity_curve(r)
        mdd = max_drawdown(eq)
        turnover = g["turnover"].mean()
        cost_drag = g["cost"].sum()
        out.append([strat, covm, costbps, ret_geom, vol_ann, sharpe, sortino, mdd, turnover, cost_drag])
    return pd.DataFrame(out, columns=["Strategy","CovMethod","CostBps","Ann.Return","Ann.Vol","Sharpe","Sortino","MaxDD","Avg.Turnover","Cost.Sum"])

def plot_equity(df: pd.DataFrame, outpath: Path):
    plt.figure(figsize=(9,4.5))
    for strat, g in df.groupby("strategy"):
        g = g.sort_values("date")
        plt.plot(g["date"], equity_curve(g["ret_net"]), label=strat)
    plt.xlabel("Date"); plt.ylabel("Cumulative net value"); plt.legend(); plt.tight_layout()
    plt.savefig(outpath); plt.close()

def plot_boxplot(df: pd.DataFrame, outpath: Path):
    data = [g.sort_values("date")["ret_net"].values for _, g in df.groupby("strategy")]
    labels = [s for s, _ in df.groupby("strategy")]
    plt.figure(figsize=(6.5,4.5))
    # Matplotlib >=3.9
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.ylabel("Monthly net returns"); plt.tight_layout()
    plt.savefig(outpath); plt.close()

def plot_rolling_sharpe(df: pd.DataFrame, outpath: Path, window=12):
    plt.figure(figsize=(9,4.5))
    for strat, g in df.groupby("strategy"):
        g = g.sort_values("date").set_index("date")
        r = g["ret_net"]
        rs = (r.rolling(window).mean() / (r.rolling(window).std() + 1e-12)) * np.sqrt(12.0)
        plt.plot(rs.index, rs.values, label=strat)
    plt.xlabel("Date"); plt.ylabel(f"Rolling Sharpe ({window}m)"); plt.legend(); plt.tight_layout()
    plt.savefig(outpath); plt.close()

# -------------------- Single run --------------------
def run_one(cov_method: str, lam: float, cost_bps: int, seed: int) -> pd.DataFrame:
    np.random.seed(seed)
    monthly_prices = monthly_prices_from_loader()
    monthly_rets = to_monthly_returns(monthly_prices)
    monthly_rets = monthly_rets.dropna(axis=1, how="any")
    monthly_prices = monthly_prices[monthly_rets.columns]
    dates = monthly_rets.index
    rows = []

    prev_w_1N = None
    prev_w_GA = None

    for t in range(WINDOW_MONTHS, len(dates) - 1):
        window = monthly_rets.iloc[t-WINDOW_MONTHS:t].dropna(axis=1, how="any")
        if window.shape[1] < 5:
            continue
        date_t = dates[t]
        date_next = dates[t+1]
        prices_t = monthly_prices.loc[date_t, window.columns]
        next_rets = monthly_rets.loc[date_next, window.columns].values

        stocks, cov_ann = build_window_stocks(prices_t, window, cov_method=cov_method, lam=lam)

        # 1/N
        N = len(window.columns)
        w_1N = np.ones(N) / N
        if prev_w_1N is None:
            to_1N, cost_1N = 0.0, 0.0
        else:
            to_1N, cost_1N = turnover_and_cost(prev_w_1N, w_1N, window.iloc[-1].values, cost_bps)
        ret_net_1N = float(np.dot(w_1N, next_rets)) - cost_1N
        rows.append({"date": date_next, "strategy": "1/N", "ret_net": ret_net_1N,
                     "turnover": to_1N, "cost": cost_1N, "n_assets": N,
                     "cov_method": cov_method, "cost_bps": cost_bps, "seed": seed})
        prev_w_1N = w_1N

        # GA — visuals OFF, quiet
        ga = GeneticAlgorithm(
            stocks, cov_ann,
            population_size=POP,
            is_short_available=IS_SHORT_AVAILABLE,
            fitness_function="quadratic utility",
            crossover_function="simulated binary crossover",
            mutation_function="gaussian mutation",
            selection_method="tournament selection",
            risk_aversion=RISK_AVERSION,
            budget=BUDGET,
            max_generations=MAX_GENS,
            enable_visuals=False,   # <- no plots
            verbose=False           # <- no per-gen prints
        )
        best = ga.evolve(fitness_threshold=np.inf)
        w_ga = shares_to_weights(stocks, best.shares, IS_SHORT_AVAILABLE)

        if prev_w_GA is None:
            to_ga, cost_ga = 0.0, 0.0
        else:
            to_ga, cost_ga = turnover_and_cost(prev_w_GA, w_ga, window.iloc[-1].values, cost_bps)
        ret_net_ga = float(np.dot(w_ga, next_rets)) - cost_ga
        rows.append({"date": date_next, "strategy": "GA-UQ", "ret_net": ret_net_ga,
                     "turnover": to_ga, "cost": cost_ga, "n_assets": N,
                     "cov_method": cov_method, "cost_bps": cost_bps, "seed": seed})
        prev_w_GA = w_ga

    df = pd.DataFrame(rows).sort_values(["date","strategy"])
    suffix = f"_{cov_method}" + (f"_l{lam:.2f}".replace(".","") if cov_method=="ewma" else "")
    out_path = RESULTS_DIR / f"oos_results{suffix}_c{cost_bps}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(df)} rows)")
    return df

# -------------------- Main: batch runs --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cov", nargs="+", default=["sample","ewma"], choices=["sample","ewma"],
                   help="Which covariance estimators to run.")
    p.add_argument("--lam", type=float, default=0.94, help="EWMA lambda (used if cov includes ewma).")
    p.add_argument("--costs", nargs="+", type=int, default=[10], help="Costs in bps per side, e.g. 5 10 25.")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()

def main():
    ensure_dirs()
    args = parse_args()
    all_dfs = []
    for covm in args.cov:
        for c in args.costs:
            df = run_one(cov_method=covm, lam=args.lam, cost_bps=c, seed=args.seed)
            all_dfs.append(df)
    big = pd.concat(all_dfs, ignore_index=True)
    combined_path = RESULTS_DIR / "oos_results_all.csv"
    big.to_csv(combined_path, index=False)
    print(f"Saved combined: {combined_path}  ({len(big)} rows)")

    # Metrics
    table = annualized_metrics(big)
    table_path = RESULTS_DIR / "summary_metrics.csv"
    table.to_csv(table_path, index=False)
    print(f"Saved metrics: {table_path}")

    # Figures (pour lisibilité: on trace celles de 'sample' au coût minimal)
    df_sample = big[big["cov_method"]=="sample"]
    if not df_sample.empty:
        plot_equity(df_sample, FIG_DIR / "fig_equity_curves.pdf")
        plot_boxplot(df_sample, FIG_DIR / "fig_boxplot_monthly_returns.pdf")
        plot_rolling_sharpe(df_sample, FIG_DIR / "fig_rolling_sharpe.pdf")
        print(f"Saved figures to {FIG_DIR}")

if __name__ == "__main__":
    main()
