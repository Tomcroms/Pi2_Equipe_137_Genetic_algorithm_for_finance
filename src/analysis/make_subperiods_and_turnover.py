import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
RES  = ROOT/"results"
FIG  = ROOT/"paper"/"figures"
FIG.mkdir(parents=True, exist_ok=True)

PERIODS = [
    ("2004-01-01","2009-12-31"),
    ("2010-01-01","2015-12-31"),
    ("2016-01-01","2020-12-31"),
    ("2021-01-01","2024-12-31"),
]

def ann_metrics(df):
    out=[]
    rf=0.0
    for (p,strat), g in df.groupby(["Period","strategy"]):
        g=g.sort_values("date"); r=g["ret_net"]; n=len(r)
        if n==0: continue
        ret = (1+r).prod()**(12.0/n)-1.0
        vol = r.std(ddof=1)*np.sqrt(12.0)
        sharpe = (ret-rf)/vol if vol>1e-12 else np.nan
        out.append([p,strat,ret,vol,sharpe])
    return pd.DataFrame(out, columns=["Period","Strategy","Ann.Return","Ann.Vol","Sharpe"])

def main():
    df = pd.read_csv(RES/"oos_results_all.csv", parse_dates=["date"])
    # On garde par ex. cov_method=='sample' et cost_bps==10 pour lisibilité
    base = df[(df["cov_method"]=="sample") & (df["cost_bps"]==10)].copy()
    # Period tag
    def tag(d):
        for i,(a,b) in enumerate(PERIODS,1):
            if pd.Timestamp(a)<=d<=pd.Timestamp(b): return f"P{i} ({a[:4]}–{b[:4]})"
        return "Other"
    base["Period"]=base["date"].apply(tag)
    table = ann_metrics(base)
    table.to_csv(RES/"subperiod_metrics.csv", index=False)

    # Turnover time-series (GA vs 1/N)
    plt.figure(figsize=(9,4.5))
    for strat, g in base.groupby("strategy"):
        g=g.sort_values("date")
        plt.plot(g["date"], g["turnover"].rolling(6).mean(), label=strat)  # lissage 6m
    plt.xlabel("Date"); plt.ylabel("Turnover (6m MA)"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG/"fig_turnover_ma6.pdf"); plt.close()
    print("Saved:", RES/"subperiod_metrics.csv", "and", FIG/"fig_turnover_ma6.pdf")

if __name__=="__main__":
    main()
