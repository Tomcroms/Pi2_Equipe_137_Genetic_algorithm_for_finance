from stock import Stock
from genetic_algorithm import GeneticAlgorithm
import numpy as np
from data_loader import DataLoader
import time


def main():

    file_path = 'data/Cac40_Prices_2000_to_Today.xlsx'
    data_loader = DataLoader(file_path)
    
    try:
        stocks, cov_matrix = data_loader.get_data()
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    try:
        ga = GeneticAlgorithm(stocks, cov_matrix, population_size=50, risk_aversion=4, budget=1_000_000_000, max_generations=10000)
        best_portfolio = ga.evolve(fitness_threshold=0.02)

        print("Best portfolio:")
        for stock, shares in zip(best_portfolio.stocks, best_portfolio.shares):
            print(f"{stock.name}: {shares} shares")
        print(f"Total Investment: ${best_portfolio.total_investment:.2f}")
        print(f"Expected Return: ${best_portfolio.expected_return:.2f}")
        print(f"Variance: {best_portfolio.variance:.2f}")

    except Exception as e:
        print(f"Erreur inconnue:\n{e}")


if __name__ == "__main__":
    main()







################################################################
                        #Donnees test

# stocks = [
#     Stock('Stock A', expected_return=0.1, std_dev=0.2, price=50),
#     Stock('Stock B', expected_return=0.15, std_dev=0.25, price=100),
#     Stock('Stock C', expected_return=0.07, std_dev=0.15, price=30),
# ]

# # Covariance matrix (example)
# cov_matrix = np.array([
#     [0.04, 0.006, 0.002],
#     [0.006, 0.0625, 0.0015],
#     [0.002, 0.0015, 0.0225],
# ])

################################################################