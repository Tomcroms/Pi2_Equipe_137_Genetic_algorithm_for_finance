import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

# IMPORTANT: use non-interactive backend to avoid GUI popups during GA evolution
import matplotlib
matplotlib.use("Agg")

from model.data_loader import DataLoader
from model.stock import Stock
from model.genetic_algorithm import GeneticAlgorithm

# ---------------- Config par défaut ----------------
FILE_PATH = "data/Cac40_Prices_2000_to_Today.xlsx"
WINDOW_MONTHS = 36          # fenêtre d'estimation (mois)
BUDGET = 1_000_000          # budget notionnel (peu importe l'unité car on travaille en poids)
COST_BPS_PER_SIDE = 10      # 10 bps par côté => 0.001 par unité de turnover
REBALANCE_FREQ = "M"        # mensuel
IS_SHORT_AVAILABLE = False  # long-only par défaut
UPPER_BOUND = 0.10          # borne supérieure par actif pour les poids (optionnel, côté GA)
SEED = 137

# ---------------- Utils ----------------
def ensure_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("paper/figures", exist_ok=True)

def monthly_prices_from_loader(loader: DataLoader) -> pd.DataFrame:
    """
    Utilise le DataLoader pour charger les prix quotidiens,
    puis resample en fin de mois (dernier cours).
    """
    loader.load_data()
    loader.preprocess_data()
    daily_prices = loader.raw_data  # index = Dates, colonnes = tickers
    monthly_prices = daily_prices.resample("M").last()
    # nettoyage basique: enlève colonnes quasi vides
    monthly_prices = monthly_prices.dropna(axis=1, how="all")
    # forward-fill occasionnel si trous (rare), puis drop lignes avec NaN résiduels
    monthly_prices = monthly_prices.ffill().dropna(how="any")
    return monthly_prices

def to_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    rets = monthly_prices.pct_change().dropna(how="any")
    return rets

def build_window_stocks(prices_t: pd.Series, window_rets: pd.DataFrame) -> tuple[list, np.ndarray]:
    """
    Construit la liste de Stock (mu, std, prix) et la matrice de covariance annualisées
    à partir des rendements mensuels de la fenêtre d'estimation.
    """
    mu_ann = window_rets.mean() * 12.0
    std_ann = window_rets.std() * np.sqrt(12.0)
    cov_ann = window_rets.cov().values * 12.0

    stocks = []
    for name in window_rets.columns:
        price = float(prices_t[name])
        s = Stock(name=name, expected_return=float(mu_ann[name]), std_dev=float(std_ann[name]), price=price)
        stocks.append(s)
    return stocks, cov_ann

def shares_to_weights(stocks: list, shares: np.ndarray, is_short_available: bool) -> np.ndarray:
    """
    Convertit des 'shares' en poids cohérents avec ta classe Portfolio:
    - long-only: poids >=0, somme = 1 sur la valeur des longs
    - long/short: poids signés, somme des valeurs absolues = 1 (exposition brute)
    """
    prices = np.array([s.price for s in stocks])
    positions = prices * shares
    if not is_short_available:
        long_pos = np.maximum(positions, 0.0)
        total = long_pos.sum()
        if total <= 0:
            # fallback égal-pondéré
            w = np.ones_like(positions) / len(positions)
        else:
            w = long_pos / total
    else:
        total = np.sum(np.abs(positions))
        if total <= 0:
            u = np.random.randn(len(positions))
            w = u / np.sum(np.abs(u))
        else:
            w = positions / total
    return w

def drift_weights(prev_w: np.ndarray, asset_rets: np.ndarray) -> np.ndarray:
    """
    Drift des poids avant rééquilibrage:
    w_tilde_i = w_{i,t-1} * (1 + r_i) / (1 + r_p)
    """
    rp = float(np.dot(prev_w, asset_rets))
    denom = 1.0 + rp
    if np.isclose(denom, 0.0):
        return prev_w  # pathologie
    w_tilde = prev_w * (1.0 + asset_rets) / denom
    return w_tilde

def turnover_and_cost(prev_w: np.ndarray, target_w: np.ndarray, asset_rets_t: np.ndarray, cost_bps_per_side: float) -> tuple[float, float]:
    """
    Calcule turnover et coût proportionnel (simple).
    """
    w_tilde = drift_weights(prev_w, asset_rets_t)
    delta = np.abs(target_w - w_tilde).sum()
    tcost = (cost_bps_per_side / 10_000.0)  # 10 bps -> 0.001
    cost = tcost * delta
    return float(delta), float(cost)

def run_backtest():
    np.random.seed(SEED)
    ensure_dirs()

    # 1) Données mensuelles
    loader = DataLoader(FILE_PATH)
    monthly_prices = monthly_prices_from_loader(loader)
    monthly_rets = to_monthly_returns(monthly_prices)

    # pour éviter des colonnes à trous: on garde les colonnes complètes sur toute la période
    monthly_rets = monthly_rets.dropna(axis=1, how="any")
    monthly_prices = monthly_prices[monthly_rets.columns]

    dates = monthly_rets.index
    n_months = len(dates)
    if n_months <= WINDOW_MONTHS + 1:
        raise RuntimeError("Pas assez d'historique pour une fenêtre de 36 mois + OOS.")

    # 2) Journal de résultats
    rows = []
    prev_w_1N = None
    prev_w_GA = None

    for t in range(WINDOW_MONTHS, n_months - 1):
        window = monthly_rets.iloc[t-WINDOW_MONTHS:t]
        date_t = dates[t]          # date d'estimation / rééquilibrage
        date_next = dates[t+1]     # mois OOS

        # garder uniquement colonnes sans NaN dans la fenêtre courante
        window = window.dropna(axis=1, how="any")
        if window.shape[1] < 5:   # trop peu d'actifs, saute
            continue

        # aligner prix et rets
        prices_t = monthly_prices.loc[date_t, window.columns]
        next_rets = monthly_rets.loc[date_next, window.columns].values

        # 2.1 Stocks et cov annualisés sur la fenêtre
        stocks, cov_ann = build_window_stocks(prices_t, window)

        # 2.2 Stratégie 1/N
        N = len(window.columns)
        w_1N = np.ones(N) / N
        if prev_w_1N is None:
            to_1N, cost_1N = 0.0, 0.0
        else:
            to_1N, cost_1N = turnover_and_cost(prev_w_1N, w_1N, window.iloc[-1].values, COST_BPS_PER_SIDE)
        ret_gross_1N = float(np.dot(w_1N, next_rets))
        ret_net_1N = ret_gross_1N - cost_1N
        rows.append({
            "date": date_next, "strategy": "1/N",
            "ret_gross": ret_gross_1N, "ret_net": ret_net_1N,
            "turnover": to_1N, "cost": cost_1N,
            "n_assets": N
        })
        prev_w_1N = w_1N

        # 2.3 Stratégie GA (utilité quadratique par défaut)
        ga = GeneticAlgorithm(
            stocks, cov_ann,
            population_size=100,
            is_short_available=IS_SHORT_AVAILABLE,
            fitness_function="quadratic utility",
            crossover_function="simulated binary crossover",
            mutation_function="gaussian mutation",
            selection_method="tournament selection",
            risk_aversion=6,
            budget=BUDGET,
            max_generations=250
        )
        best = ga.evolve(fitness_threshold=np.inf)  # on s'arrête à max_generations
        w_ga = shares_to_weights(stocks, best.shares, IS_SHORT_AVAILABLE)

        if prev_w_GA is None:
            to_ga, cost_ga = 0.0, 0.0
        else:
            to_ga, cost_ga = turnover_and_cost(prev_w_GA, w_ga, window.iloc[-1].values, COST_BPS_PER_SIDE)
        ret_gross_ga = float(np.dot(w_ga, next_rets))
        ret_net_ga = ret_gross_ga - cost_ga

        rows.append({
            "date": date_next, "strategy": "GA-UQ",
            "ret_gross": ret_gross_ga, "ret_net": ret_net_ga,
            "turnover": to_ga, "cost": cost_ga,
            "n_assets": N
        })
        prev_w_GA = w_ga

        print(f"{date_next.date()}  1/N net={ret_net_1N:.4%}  |  GA net={ret_net_ga:.4%}")

    df = pd.DataFrame(rows).sort_values(["date", "strategy"])
    out_path = "results/oos_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(df)} rows)")

if __name__ == "__main__":
    run_backtest()
