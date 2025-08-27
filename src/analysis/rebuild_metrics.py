import pandas as pd, numpy as np, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

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

def safe_to_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)   # atomic replace sur Windows
        print(f"Saved metrics: {path}")
    except PermissionError:
        alt = path.with_name(path.stem + f"_{int(time.time())}.csv")
        os.replace(tmp, alt)
        print(f"WARNING: Can't overwrite {path} (locked). Wrote {alt} instead.")

def main():
    all_path = RES / "oos_results_all.csv"
    if not all_path.exists():
        raise FileNotFoundError(all_path)
    df = pd.read_csv(all_path, parse_dates=["date"])
    table = annualized_metrics(df)
    safe_to_csv(table, RES / "summary_metrics.csv")

if __name__ == "__main__":
    main()
