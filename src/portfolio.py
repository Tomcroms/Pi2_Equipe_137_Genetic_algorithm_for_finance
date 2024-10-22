import numpy as np

class Portfolio:
    def __init__(self, weights, stocks, cov_matrix):
        self.weights = np.array(weights)  # Poids du portefeuille
        self.stocks = stocks  # Liste des objets Stock
        self.cov_matrix = cov_matrix  # Matrice de covariance
        self.expected_return = self.calculate_expected_return()
        self.variance = self.calculate_variance()
        self.fitness = None  # À calculer avec la fonction d'utilité

    def calculate_expected_return(self):
        expected_returns = np.array([stock.expected_return for stock in self.stocks])
        return np.dot(self.weights, expected_returns)

    def calculate_variance(self):
        return np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights))

    def calculate_fitness(self, risk_aversion):
        # Fonction d'utilité quadratique
        self.fitness = self.expected_return - (risk_aversion / 2) * self.variance
