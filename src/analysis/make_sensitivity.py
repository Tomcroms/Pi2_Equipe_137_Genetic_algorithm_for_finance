import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES  = ROOT / "results"
FIG  = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def equity_curve(r): return (1+r).cumprod()
def max_dd(eq): 
    roll = eq.cummax()
    return float((eq/roll - 1).min())

def annualized_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out=[]
    rf=0.0
    for (strat,covm,cost), g in df.groupby(["strategy","cov_method","cost_bps"]):
        g=g.sort_values("date"); r=g["ret_net"]; n=len(r)
        if n==0: continue
        ret_geom = (1+r).prod()**(12.0/n)-1.0
        vol_ann  = r.std(ddof=1)*np.sqrt(12.0)
        sharpe   = (ret_geom-rf)/vol_ann if vol_ann>1e-12 else np.nan
        sortino  = (ret_geom-rf)/(r[r<0].std(ddof=1)*np.sqrt(12.0)+1e-12) if (r<0).any() else np.nan
        eq       = equity_curve(r); mdd=max_dd(eq)
        out.append([strat,covm,int(cost),ret_geom,vol_ann,sharpe,sortino,mdd,g["turnover"].mean(),g["cost"].sum()])
    return pd.DataFrame(out, columns=["Strategy","CovMethod","CostBps","Ann.Return","Ann.Vol","Sharpe","Sortino","MaxDD","Avg.Turnover","Cost.Sum"])

def bar_sensitivity(table: pd.DataFrame, metric: str, outpath: Path):
    # barres par coût, couleurs = cov, groupes = stratégies
    order_cost = sorted(table["CostBps"].unique())
    covs = ["sample","ewma"]
    strategies = ["1/N","GA-UQ"]
    width=0.35

    plt.figure(figsize=(8.5,4.5))
    x = np.arange(len(order_cost))
    for i, strat in enumerate(strategies):
        for j, cov in enumerate(covs):
            subset = table[(table["Strategy"]==strat)&(table["CovMethod"]==cov)]
            ys = [subset[subset["CostBps"]==c][metric].values[0] if (subset["CostBps"]==c).any() else np.nan for c in order_cost]
            offset = (i*len(covs)+j - (len(strategies)*len(covs)-1)/2)* (width/len(covs))
            plt.bar(x+offset, ys, width/len(covs), label=f"{strat}-{cov}")

    plt.xticks(x, [f"{c} bps" for c in order_cost])
    plt.ylabel(metric); plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(outpath); plt.close()

def main():
    df = pd.read_csv(RES/"oos_results_all.csv", parse_dates=["date"])
    table = annualized_metrics(df)
    # Sauvegarde tableau détaillé
    table.sort_values(["Strategy","CovMethod","CostBps"]).to_csv(RES/"summary_metrics.csv", index=False)
    # Figure sensibilité (Sharpe par défaut)
    bar_sensitivity(table, "Sharpe", FIG/"fig_sensitivity_sharpe.pdf")
    print(f"Saved: {RES/'summary_metrics.csv'} and {FIG/'fig_sensitivity_sharpe.pdf'}")

if __name__=="__main__":
    main()
