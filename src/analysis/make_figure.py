import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_CSV = "results/oos_results.csv"
FIG_DIR = "paper/figures"

def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)

def equity_curve(ret_net: pd.Series) -> pd.Series:
    return (1.0 + ret_net).cumprod()

def max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    dd = series / roll_max - 1.0
    return float(dd.min())

def annualized_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: colonnes ['date','strategy','ret_net','turnover','cost']
    Retourne un tableau par stratégie: Return_ann, Vol_ann, Sharpe, Sortino, MaxDD, Turnover
    """
    out = []
    rf_ann = 0.0  # ajuste si tu utilises un Rf != 0
    for strat, g in df.groupby("strategy"):
        g = g.sort_values("date")
        r = g["ret_net"]
        n = len(r)
        if n == 0:
            continue
        # annualisation
        ret_geom = (1.0 + r).prod() ** (12.0 / n) - 1.0
        vol_ann = r.std(ddof=1) * np.sqrt(12.0)
        sharpe = (ret_geom - rf_ann) / vol_ann if vol_ann > 1e-12 else np.nan
        downside = r[r < 0.0]
        if len(downside) == 0:
            sortino = np.nan
        else:
            down_dev = downside.std(ddof=1) * np.sqrt(12.0)
            sortino = (ret_geom - rf_ann) / down_dev if down_dev > 1e-12 else np.nan
        # max DD sur la courbe nette
        eq = equity_curve(r)
        mdd = max_drawdown(eq)
        turnover = g["turnover"].mean()
        cost_drag = g["cost"].sum()
        out.append([strat, ret_geom, vol_ann, sharpe, sortino, mdd, turnover, cost_drag])
    return pd.DataFrame(out, columns=["Strategy","Ann.Return","Ann.Vol","Sharpe","Sortino","MaxDD","Avg.Turnover","Cost.Sum"])

def plot_equity(df: pd.DataFrame, path: str):
    plt.figure(figsize=(9,4.5))
    for strat, g in df.groupby("strategy"):
        g = g.sort_values("date")
        eq = equity_curve(g["ret_net"])
        plt.plot(g["date"], eq, label=strat)
    plt.xlabel("Date")
    plt.ylabel("Cumulative net value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_boxplot(df: pd.DataFrame, path: str):
    data = [g.sort_values("date")["ret_net"].values for _, g in df.groupby("strategy")]
    labels = [s for s, _ in df.groupby("strategy")]
    plt.figure(figsize=(6.5,4.5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Monthly net returns")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_rolling_sharpe(df: pd.DataFrame, path: str, window=12):
    plt.figure(figsize=(9,4.5))
    for strat, g in df.groupby("strategy"):
        g = g.sort_values("date").set_index("date")
        r = g["ret_net"]
        roll_mean = r.rolling(window).mean()
        roll_std = r.rolling(window).std()
        rs = (roll_mean / (roll_std + 1e-12)) * np.sqrt(12.0)
        plt.plot(rs.index, rs.values, label=strat)
    plt.xlabel("Date")
    plt.ylabel(f"Rolling Sharpe ({window}m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    ensure_dirs()
    df = pd.read_csv(RESULTS_CSV, parse_dates=["date"])
    # Table
    table = annualized_metrics(df)
    table_path = "results/summary_metrics.csv"
    table.to_csv(table_path, index=False)
    print(f"Saved: {table_path}")
    # Figures
    plot_equity(df, os.path.join(FIG_DIR, "fig_equity_curves.pdf"))
    plot_boxplot(df, os.path.join(FIG_DIR, "fig_boxplot_monthly_returns.pdf"))
    plot_rolling_sharpe(df, os.path.join(FIG_DIR, "fig_rolling_sharpe.pdf"))
    print(f"Saved figures to {FIG_DIR}")

if __name__ == "__main__":
    main()
