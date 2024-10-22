from stock import Stock
from genetic_algorithm import GeneticAlgorithm
import numpy as np


################################################################

                        #A MODIFIER !!!
stocks = [
    Stock('Action A', 0.12, 0.20),
    Stock('Action B', 0.10, 0.15),
    Stock('Action C', 0.08, 0.10)
]

# Matrice de covariance fictive
cov_matrix_values = np.array([
    [0.04, 0.006, 0.004],
    [0.006, 0.0225, 0.003],     
    [0.004, 0.003, 0.01]
])

population_size = 50
risk_aversion = 4                   
fitness_threshold = 0.02 

################################################################


def main():

    try:
        ga = GeneticAlgorithm(stocks, cov_matrix_values, population_size, risk_aversion=risk_aversion)
        best_portfolio = ga.evolve(fitness_threshold)

        print("Meilleur portefeuille trouvé:")
        for i, stock in enumerate(stocks):
            print(f"{stock.name}: {best_portfolio.weights[i]:.4f}")
        print(f"Rendement attendu: {best_portfolio.expected_return:.4f}")
        print(f"Variance: {best_portfolio.variance:.6f}")
        print(f"Fitness: {best_portfolio.fitness:.6f}")

    except Exception as e:
        print(f"Erreur inconnue:\n{e}")


if __name__ == "__main__":
    main()