import numpy as np

class Portfolio:

    eta = 2  # Default SBX crossover distribution index
    mutation_rate = 0.1  # Default mutation rate
    sigma = 0.1  # Default mutation standard deviation

    def __init__(self, shares, stocks, cov_matrix, is_short_available, fitness_function, crossover_function, mutation_function, budget, risk_aversion):
        self.shares = np.array(shares)
        self.stocks = stocks  # List of Stock objects
        self.cov_matrix = cov_matrix  # Covariance matrix
        self.is_short_available = is_short_available
        self.fitness_function = fitness_function
        #self.selection_method = selection_method
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        self.budget = budget
        self.risk_aversion = risk_aversion
        self.total_investment = self.calculate_total_investment()
        self.expected_return = self.calculate_expected_return()
        self.variance = self.calculate_variance()
        self.fitness = None  # To be calculated with the utility function
        self.calculate_fitness()  # Calculate fitness during initialization

    def calculate_total_investment(self):
        prices = np.array([stock.price for stock in self.stocks])
        shares = np.abs(self.shares)
        return np.dot(shares, prices)
    
    def calculate_expected_return(self):
        expected_returns = np.array([stock.expected_return for stock in self.stocks])
        prices = np.array([stock.price for stock in self.stocks])
        total_investment = np.dot(prices, self.shares)
        weights = (prices * self.shares) / total_investment
        return np.dot(weights, expected_returns)

    def calculate_variance(self):
        prices = np.array([stock.price for stock in self.stocks])
        investment = prices * self.shares
        weights = investment / self.total_investment
        return np.dot(weights.T, np.dot(self.cov_matrix, weights)) * self.total_investment**2

    def calculate_fitness(self):
        # Avoid division by zero
        if self.total_investment == 0:
            self.fitness = -np.inf  # Assign a very low fitness if there's no investment
            return
        
        # Calculate per-unit return and variance
        return_per_unit = self.expected_return
        variance_per_unit = self.variance / (self.total_investment ** 2)

        if(not self.fitness_function or self.fitness_function == "quadratic utility"):
            self.fitness_with_quadratic_utility_function(return_per_unit, variance_per_unit)
            return
        elif(self.fitness_function=="sharpe ratio"):
            self.fitness_with_sharpe_ratio(return_per_unit, variance_per_unit)
            return 

    def fitness_with_quadratic_utility_function(self, return_per_unit, variance_per_unit):
        self.fitness = return_per_unit - (self.risk_aversion / 2) * variance_per_unit

    def fitness_with_sharpe_ratio(self, return_per_unit, variance_per_unit):
        self.fitness = (np.sqrt(return_per_unit) / variance_per_unit)

    def adjust_shares_to_budget(self):
        total_value = self.calculate_total_investment()
        if total_value == 0:
            return
        scaling_factor = self.budget / total_value
        self.shares *= scaling_factor
        self.total_investment = self.calculate_total_investment()

    def __add__(self, other):
        if(not self.crossover_function or self.crossover_function == "simulated binary crossover"):
            crossed_child = self.simulated_binary_crossover(other)
        
        elif(self.crossover_function == "blend crossover"):
            crossed_child = self.blx_alpha_crossover(other)

        elif(self.crossover_function == "arithmetic crossover"):
            crossed_child = self.arithmetic_crossover(other)

        return crossed_child

    def __invert__(self):
        if(not self.mutation_function or self.mutation_function == "gaussian mutation"):
            try:
                mutated_child = self.gaussian_mutation()
            except Exception as e:
                print(f"Erreur gaussian mutation {e}")
        
        elif(self.mutation_function == "polynomial mutation"):
            mutated_child = self.polynomial_mutation()

        elif(self.mutation_function == "uniform mutation"):
            mutated_child = self.uniform_mutation()

        return mutated_child


    #Crossover methods
    def simulated_binary_crossover(self, other):
        # Simulated Binary Crossover (SBX)
        child_shares = []
        for i in range(len(self.shares)):
            u = np.random.rand()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (Portfolio.eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (Portfolio.eta + 1))
            child_share = 0.5 * ((1 + beta) * self.shares[i] + (1 - beta) * other.shares[i])
            child_shares.append(child_share)
        # Ensure non-negative shares
        if(not self.is_short_available): 
            child_shares = np.maximum(child_shares, 0)
        # Re-scale shares to match the budget
        child_portfolio = Portfolio(child_shares, self.stocks, self.cov_matrix, self.is_short_available, self.fitness_function, self.crossover_function, self.mutation_function, self.budget, self.risk_aversion)
        child_portfolio.adjust_shares_to_budget()
        child_portfolio.calculate_fitness()
        return child_portfolio

    def blx_alpha_crossover(self, other, alpha=0.5):
        # Croisement BLX-α (Blend Crossover)
        child_shares = []
        for x1, x2 in zip(self.shares, other.shares):
            L = min(x1, x2)
            U = max(x1, x2)
            I = U - L
            # On choisit un point dans [L - alpha*I, U + alpha*I]
            val = np.random.uniform(L - alpha * I, U + alpha * I)
            child_shares.append(val)
        if(not self.is_short_available): 
            child_shares = np.maximum(child_shares, 0)
        child_portfolio = Portfolio(child_shares, self.stocks, self.cov_matrix, self.is_short_available, self.fitness_function, self.crossover_function, self.mutation_function, self.budget, self.risk_aversion)
        child_portfolio.adjust_shares_to_budget()
        child_portfolio.calculate_fitness()
        return child_portfolio
        
    def arithmetic_crossover(self, other):
        r = np.random.rand()
        child_shares = r * self.shares + (1 - r) * other.shares
        if(not self.is_short_available): 
            child_shares = np.maximum(child_shares, 0)
        child_portfolio = Portfolio(child_shares, self.stocks, self.cov_matrix, self.fitness_function, self.crossover_function, self.mutation_function, self.budget, self.risk_aversion)
        child_portfolio.adjust_shares_to_budget()
        child_portfolio.calculate_fitness()
        return child_portfolio


    #Mutation methods
    def gaussian_mutation(self):
        # Gaussian mutation
        mutated_shares = self.shares.copy()
        for i in range(len(mutated_shares)):
            if np.random.rand() < Portfolio.mutation_rate:
                mutated_shares[i] += np.random.normal(0, Portfolio.sigma * abs(mutated_shares[i]))
        if(not self.is_short_available): 
            mutated_shares = np.maximum(mutated_shares, 0)
        
        mutated_portfolio = Portfolio(mutated_shares, self.stocks, self.cov_matrix, self.is_short_available, self.fitness_function, self.crossover_function, self.mutation_function, self.budget, self.risk_aversion)
        try:
            mutated_portfolio.adjust_shares_to_budget()
        except Exception as e:
            print(f"Erreur adjust shares {e}")
        try:    
            mutated_portfolio.calculate_fitness()
        except Exception as e:
            print(f"Erreur adjust shares {e}")
        return mutated_portfolio
    
    def polynomial_mutation(self):
        eta_m = Portfolio.eta*2
        mutated_shares = self.shares.copy()
        for i in range(len(mutated_shares)):
            if np.random.rand() < Portfolio.mutation_rate:
                u = np.random.rand()
                delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1 if u <= 0.5 else 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                mutated_shares[i] = mutated_shares[i] * (1 + delta)
                # Non negative shares
                if(not self.is_short_available and mutated_shares[i] < 0):
                    mutated_shares[i] = 0
        mutated_portfolio = Portfolio(mutated_shares, self.stocks, self.cov_matrix, self.is_short_available, self.fitness_function, self.crossover_function, self.mutation_function, self.budget, self.risk_aversion)
        mutated_portfolio.adjust_shares_to_budget()
        mutated_portfolio.calculate_fitness()
        return mutated_portfolio
    
    def uniform_mutation(self):
        mutated_shares = self.shares.copy()
        for i in range(len(mutated_shares)):
            if np.random.rand() < Portfolio.mutation_rate:
                mutated_shares[i] = np.random.uniform(self.shares[i] - np.sqrt(self.shares[i]), self.shares[i] + np.sqrt(self.shares[i]))
        if(not self.is_short_available): 
            mutated_shares = np.maximum(mutated_shares, 0)
        mutated_portfolio = Portfolio(mutated_shares, self.stocks, self.cov_matrix, self.is_short_available, self.fitness_function, self.crossover_function, self.mutation_function, self.budget, self.risk_aversion)
        mutated_portfolio.adjust_shares_to_budget()
        mutated_portfolio.calculate_fitness()
        return mutated_portfolio


    #Utils methods
    def get_expected_return_percentage(self):
        if self.total_investment == 0:
            return 0
        return self.expected_return

    def get_volatility_percentage(self):
        if self.total_investment == 0:
            return 0
        variance_of_returns = self.variance / (self.total_investment ** 2)*100
        return variance_of_returns


    #Static methods
    @staticmethod
    def tournament_selection(population, k=3):
        selected = []
        for _ in range(2):  # Sélectionner deux parents
            individuals = np.random.choice(population, k)
            best = max(individuals, key=lambda ind: ind.fitness)
            selected.append(best)
        return selected
    
    @staticmethod
    def selection(population, selection_method):
        if(not selection_method or selection_method=="tournament selection"):
            return Portfolio.tournament_selection(population)
        
        elif(selection_method=="autre_methode"):
            return "autre methode"
        
        else:
            print("Unknown selection method...")
            raise Exception

