# src/analysis/compare_convergence.py
import time
import numpy as np
import pandas as pd

from src.model.data_loader import DataLoader
from src.model.stock import Stock
from src.model.genetic_algorithm import GeneticAlgorithm

# -------------------- Configs (cohérentes avec ton backtest) --------------------
FILE_PATH = "data/Cac40_Prices_2000_to_Today.xlsx"
WINDOW_MONTHS = 36
RISK_AVERSION = 6
SEED = 137

# -------------------- Données mensuelles (identique en esprit à multi_run_backtest) --------------------
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
    std_ann = window_rets.std() * np.sqrt(12.0)   # (pas utilisé ici, mais conservé)
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

# -------------------- Outils : poids GA, utilité quadratique --------------------
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

# -------------------- Projection sur le simplexe (w>=0, somme=1) --------------------
def project_simplex(v):
    # Proj. Euclidienne sur { w>=0, sum w = 1 } (algorithme de tri + seuillage)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u > (cssv - 1) / (np.arange(n) + 1))[0]
    if len(rho) == 0:
        theta = 0.0
    else:
        rho = rho[-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w

# -------------------- Descente de gradient projetée (PGD) --------------------
def pgd_markowitz(mu, Sigma, lam, w0=None, max_iter=1000, tol=1e-9):
    n = mu.size
    if w0 is None:
        w = np.ones(n) / n
    else:
        w = project_simplex(w0)

    # Pas ~ 1/L où L est la plus grande valeur propre de lam*Sigma (Lipschitz du gradient)
    # Ici on calcule L exactement (n<=40 => OK)
    L = lam * np.linalg.eigvalsh(Sigma).max()
    step = 1.0 / (L + 1e-12)

    u_prev = utility_quadratic(w, mu, Sigma, lam)
    t0 = time.perf_counter()
    for k in range(1, max_iter + 1):
        grad = mu - lam * (Sigma @ w)              # gradient de U
        w = project_simplex(w + step * grad)       # pas de gradient + projection
        u = utility_quadratic(w, mu, Sigma, lam)
        if abs(u - u_prev) < tol:
            return w, u, k, (time.perf_counter() - t0)
        u_prev = u
    return w, u, max_iter, (time.perf_counter() - t0)

# -------------------- Main : comparaison GA vs PGD sur la DERNIÈRE fenêtre --------------------
def main():
    np.random.seed(SEED)

    m_prices = monthly_prices_from_loader()
    m_rets = to_monthly_returns(m_prices).dropna(axis=1, how="any")
    m_prices = m_prices[m_rets.columns]
    dates = m_rets.index

    if len(dates) <= WINDOW_MONTHS:
        raise RuntimeError("Pas assez d'historique pour une fenêtre de 36 mois.")

    # Dernière fenêtre
    t = len(dates) - 1
    window = m_rets.iloc[t - WINDOW_MONTHS : t].dropna(axis=1, how="any")
    prices_t = m_prices.loc[dates[t], window.columns]

    stocks, cov_ann, mu_ann = build_window_stocks(prices_t, window)

    # --------- PGD (convexe, rapide) ---------
    w_pgd, u_pgd, it_pgd, dt_pgd = pgd_markowitz(mu_ann, cov_ann, RISK_AVERSION, w0=None, max_iter=2000)
    print("\n[PGD] iters=%d  time=%.4fs  U=%.6f" % (it_pgd, dt_pgd, u_pgd))

    # --------- GA (heuristique) ---------
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
    best = ga.evolve(fitness_threshold=np.inf)   # s'arrête à max_generations
    dt_ga = time.perf_counter() - t0

    w_ga = shares_to_weights_long_only(stocks, best.shares)
    u_ga = utility_quadratic(w_ga, mu_ann, cov_ann, RISK_AVERSION)
    print("[GA ] gens=%d  time=%.4fs  U=%.6f" % (ga.max_generations, dt_ga, u_ga))

    # --------- Résumé lisible ---------
    print("\n=== Résumé convergence (dernière fenêtre) ===")
    print("PGD :  U=%.6f  | iters=%d  | %.3fs" % (u_pgd, it_pgd, dt_pgd))
    print("GA  :  U=%.6f  | gens=%d   | %.3fs" % (u_ga, ga.max_generations, dt_ga))
    print("ΔU (PGD - GA) = %.6f" % (u_pgd - u_ga))

if __name__ == "__main__":
    main()
