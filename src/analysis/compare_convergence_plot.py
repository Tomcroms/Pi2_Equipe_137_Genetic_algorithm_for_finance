# src/analysis/compare_convergence_plot.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.model.data_loader import DataLoader
from src.model.stock import Stock
from src.model.genetic_algorithm import GeneticAlgorithm

# -------------------- Config --------------------
FILE_PATH = "data/Cac40_Prices_2000_to_Today.xlsx"
WINDOW_MONTHS = 36
RISK_AVERSION = 6
SEED = 137

PROJ_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJ_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Data utils --------------------
def monthly_prices_from_loader():
    loader = DataLoader(FILE_PATH)
    loader.load_data()
    loader.preprocess_data()
    daily_prices = loader.raw_data
    m_prices = daily_prices.resample("M").last()
    m_prices = m_prices.dropna(axis=1, how="all").ffill().dropna(how="any")
    return m_prices

def to_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    return monthly_prices.pct_change().dropna(how="any")

def build_window_stocks(prices_t: pd.Series, window_rets: pd.DataFrame):
    mu_ann = window_rets.mean() * 12.0
    std_ann = window_rets.std() * np.sqrt(12.0)   # conservé pour cohérence
    cov_ann = window_rets.cov().values * 12.0
    stocks = []
    for name in window_rets.columns:
        stocks.append(Stock(
            name=name,
            expected_return=float(mu_ann[name]),
            std_dev=float(std_ann[name]),
            price=float(prices_t[name])
        ))
    return stocks, cov_ann, mu_ann.values

def shares_to_weights_long_only(stocks, shares):
    prices = np.array([s.price for s in stocks])
    pos = np.maximum(prices * shares, 0.0)
    tot = pos.sum()
    if tot <= 0:
        return np.ones_like(pos) / len(pos)
    return pos / tot

def utility_quadratic(w, mu, Sigma, lam):
    # U(w) = mu^T w - (lam/2) w^T Sigma w
    return float(mu @ w - 0.5 * lam * (w @ Sigma @ w))

# -------------------- Projection sur simplexe --------------------
def project_simplex(v):
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.where(u > (cssv - 1) / (np.arange(n) + 1))[0]
    if len(rho_idx) == 0:
        theta = 0.0
    else:
        rho = rho_idx[-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w

# -------------------- PGD avec historique --------------------
def pgd_markowitz_with_history(mu, Sigma, lam, w0=None, max_iter=2000, tol=1e-9):
    n = mu.size
    if w0 is None:
        w = np.ones(n) / n
    else:
        w = project_simplex(w0)

    L = lam * max(1e-12, np.linalg.eigvalsh(Sigma).max())  # Lipschitz du gradient
    step = 1.0 / L

    u_prev = utility_quadratic(w, mu, Sigma, lam)
    util_hist = [u_prev]
    time_hist = [0.0]
    t0 = time.perf_counter()

    for k in range(1, max_iter + 1):
        grad = mu - lam * (Sigma @ w)       # gradient de U
        w = project_simplex(w + step * grad)
        u = utility_quadratic(w, mu, Sigma, lam)
        util_hist.append(u)
        time_hist.append(time.perf_counter() - t0)
        if abs(u - u_prev) < tol:
            return w, u, k, (time.perf_counter() - t0), np.array(util_hist), np.array(time_hist)
        u_prev = u

    return w, u, max_iter, (time.perf_counter() - t0), np.array(util_hist), np.array(time_hist)

# -------------------- Main --------------------
def main():
    np.random.seed(SEED)

    m_prices = monthly_prices_from_loader()
    m_rets = to_monthly_returns(m_prices).dropna(axis=1, how="any")
    m_prices = m_prices[m_rets.columns]
    dates = m_rets.index
    if len(dates) <= WINDOW_MONTHS:
        raise RuntimeError("Pas assez d'historique.")

    t = len(dates) - 1
    window = m_rets.iloc[t - WINDOW_MONTHS : t].dropna(axis=1, how="any")
    prices_t = m_prices.loc[dates[t], window.columns]

    stocks, cov_ann, mu_ann = build_window_stocks(prices_t, window)

    # ---- PGD ----
    w_pgd, u_pgd, it_pgd, dt_pgd, util_hist_pgd, time_hist_pgd = pgd_markowitz_with_history(
        mu_ann, cov_ann, RISK_AVERSION, w0=None, max_iter=2000, tol=1e-9
    )
    print("[PGD] iters=%d  time=%.4fs  U=%.6f" % (it_pgd, dt_pgd, u_pgd))

    # ---- GA ----
    ga = GeneticAlgorithm(
        stocks, cov_ann,
        population_size=100,
        is_short_available=False,
        fitness_function="quadratic utility",
        crossover_function="simulated binary crossover",
        mutation_function="gaussian mutation",
        selection_method="tournament selection",
        risk_aversion=RISK_AVERSION,
        budget=1_000_000,
        max_generations=250,
        enable_visuals=False,
        verbose=False
    )
    t0 = time.perf_counter()
    best = ga.evolve(fitness_threshold=np.inf)
    dt_ga = time.perf_counter() - t0
    util_hist_ga = np.array(ga.best_fitness_history, dtype=float)
    # Approximation du temps par génération (linéaire)
    time_hist_ga = np.linspace(0.0, dt_ga, num=len(util_hist_ga))

    w_ga = shares_to_weights_long_only(stocks, best.shares)
    u_ga = utility_quadratic(w_ga, mu_ann, cov_ann, RISK_AVERSION)
    print("[GA ] gens=%d  time=%.4fs  U=%.6f" % (ga.max_generations, dt_ga, u_ga))
    print("ΔU (PGD - GA) = %.6f" % (u_pgd - u_ga))

    # ---- Distance de poids (optionnel à commenter/afficher) ----
    # L1 et L2 entre w_pgd et w_ga
    l1 = float(np.sum(np.abs(w_pgd - w_ga)))
    l2 = float(np.linalg.norm(w_pgd - w_ga))
    print("||w_PGD - w_GA||_1 = %.6f   ||w_PGD - w_GA||_2 = %.6f" % (l1, l2))

    # ---- Figure 1: utilité vs itération/génération ----
    plt.figure(figsize=(8.5, 4.6))
    plt.plot(util_hist_pgd, label="PGD (itérations)")
    plt.plot(util_hist_ga, label="GA (générations)")
    plt.xlabel("Itération / Génération")
    plt.ylabel("Utilité quadratique")
    plt.legend()
    plt.tight_layout()
    f1 = FIG_DIR / "fig_convergence_iters.pdf"
    plt.savefig(f1)
    plt.savefig(str(f1).replace(".pdf", ".png"), dpi=180)
    plt.close()
    print(f"Saved: {f1}")

    # ---- Figure 2: utilité vs temps ----
    plt.figure(figsize=(8.5, 4.6))
    plt.plot(time_hist_pgd, util_hist_pgd, label="PGD")
    plt.plot(time_hist_ga, util_hist_ga, label="GA (approx. temps par génération)")
    plt.xlabel("Temps (secondes)")
    plt.ylabel("Utilité quadratique")
    plt.legend()
    plt.tight_layout()
    f2 = FIG_DIR / "fig_convergence_time.pdf"
    plt.savefig(f2)
    plt.savefig(str(f2).replace(".pdf", ".png"), dpi=180)
    plt.close()
    print(f"Saved: {f2}")

if __name__ == "__main__":
    main()
