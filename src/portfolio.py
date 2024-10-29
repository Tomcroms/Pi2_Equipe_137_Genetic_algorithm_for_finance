import numpy as np

class Portfolio:
    def __init__(self, shares, stocks, cov_matrix, budget, risk_aversion):
        self.shares = np.array(shares)
        self.stocks = stocks  # List of Stock objects
        self.cov_matrix = cov_matrix  # Covariance matrix
        self.budget = budget
        self.risk_aversion = risk_aversion
        self.total_investment = self.calculate_total_investment()
        self.expected_return = self.calculate_expected_return()
        self.variance = self.calculate_variance()
        self.fitness = None  # To be calculated with the utility function
        self.calculate_fitness()  # Calculate fitness during initialization

    def calculate_total_investment(self):
        prices = np.array([stock.price for stock in self.stocks])
        return np.dot(self.shares, prices)

    def calculate_expected_return(self):
        expected_returns = np.array([stock.expected_return for stock in self.stocks])
        prices = np.array([stock.price for stock in self.stocks])
        investment = prices * self.shares
        return np.dot(investment, expected_returns)

    def calculate_variance(self):
        prices = np.array([stock.price for stock in self.stocks])
        investment = prices * self.shares
        weights = investment / self.total_investment
        return np.dot(weights.T, np.dot(self.cov_matrix, weights)) * self.total_investment**2

    def calculate_fitness(self, penalty=0):
        # Quadratic utility function with optional investment penalty
        self.fitness = self.expected_return - (self.risk_aversion / 2) * self.variance - penalty * self.total_investment

    def __add__(self, other):
        # Simulated Binary Crossover (SBX)
        eta = 2  # Crossover distribution index
        child_shares = []
        for i in range(len(self.shares)):
            u = np.random.rand()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            child_share = 0.5 * ((1 + beta) * self.shares[i] + (1 - beta) * other.shares[i])
            child_shares.append(child_share)
        # Ensure non-negative and within budget
        child_shares = np.maximum(child_shares, 0)
        child_portfolio = Portfolio(child_shares, self.stocks, self.cov_matrix, self.budget, self.risk_aversion)
        if child_portfolio.total_investment > self.budget:
            scale_factor = self.budget / child_portfolio.total_investment
            child_portfolio.shares *= scale_factor
            child_portfolio.total_investment = self.budget
        child_portfolio.calculate_fitness()
        return child_portfolio

    def __invert__(self):
        # Gaussian mutation
        mutation_rate = 0.1  # Mutation rate
        sigma = 0.1  # Standard deviation for mutation
        mutated_shares = self.shares.copy()
        for i in range(len(mutated_shares)):
            if np.random.rand() < mutation_rate:
                mutated_shares[i] += np.random.normal(0, sigma * mutated_shares[i])
        # Ensure non-negative and within budget
        mutated_shares = np.maximum(mutated_shares, 0)
        mutated_portfolio = Portfolio(mutated_shares, self.stocks, self.cov_matrix, self.budget, self.risk_aversion)
        if mutated_portfolio.total_investment > self.budget:
            scale_factor = self.budget / mutated_portfolio.total_investment
            mutated_portfolio.shares *= scale_factor
            mutated_portfolio.total_investment = self.budget
        mutated_portfolio.calculate_fitness()
        return mutated_portfolio
