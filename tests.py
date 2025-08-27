import pandas as pd
from scipy import stats

df = pd.read_csv("results/oos_results.csv", parse_dates=["date"])
pivot = df.pivot(index="date", columns="strategy", values="ret_net").dropna()

x = pivot["GA-UQ"].values
y = pivot["1/N"].values

# t-test apparié
t_stat, p_t = stats.ttest_rel(x, y)

# Wilcoxon (non-paramétrique)
# (attention: nécessite des différences non nulles)
w_stat, p_w = stats.wilcoxon(x, y, zero_method="wilcox", correction=True, alternative="two-sided")

print(f"Paired t-test: t={t_stat:.3f}, p={p_t:.4f}")
print(f"Wilcoxon: W={w_stat:.3f}, p={p_w:.4f}")
