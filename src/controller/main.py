import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)


from model.genetic_algorithm import GeneticAlgorithm
from model.data_loader import DataLoader
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
        ga = GeneticAlgorithm(stocks, cov_matrix, population_size=100, selection_method=None, risk_aversion=6, budget=1_000_000_000, max_generations=1000)
        best_portfolio = ga.evolve(fitness_threshold=0.030)

        print("Best portfolio:")
        for stock, shares in zip(best_portfolio.stocks, best_portfolio.shares):
            print(f"{stock.name}: {shares} shares")
        print(f"Total Investment: ${best_portfolio.total_investment:.2f}")
        print(f"Expected Return: ${best_portfolio.expected_return:.2f} ({best_portfolio.get_expected_return_percentage():.2f}%)")
        print(f"Variance: {best_portfolio.variance:.2f}")
        print(f"Standard Deviation of Returns: {best_portfolio.get_standard_deviation_percentage():.2f}%")

        time.sleep(1000)

    except Exception as e:
        print(f"Erreur inconnue:\n{e}")


if __name__ == "__main__":
    main()