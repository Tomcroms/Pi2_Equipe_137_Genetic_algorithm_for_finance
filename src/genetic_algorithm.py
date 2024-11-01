# genetic_algorithm.py
import numpy as np
from stock import Stock
from portfolio import Portfolio

class GeneticAlgorithm:
    def __init__(self, stocks, cov_matrix, population_size, risk_aversion=4, budget=100000, max_generations=None):
        self.stocks = stocks
        self.cov_matrix = cov_matrix
        self.population_size = population_size  #number of portfolio generated at each step
        self.risk_aversion = risk_aversion
        self.budget = budget
        self.population = self.initialize_population()
        self.max_generations = max_generations

    def initialize_population(self):
        population = []
        prices = np.array([stock.price for stock in self.stocks])
        for _ in range(self.population_size):
            # Generate random number of shares within the budget
            shares = np.random.rand(len(self.stocks)) * (self.budget / prices)
            shares = np.floor(shares)  # Use whole shares
            portfolio = Portfolio(shares, self.stocks, self.cov_matrix, self.budget, self.risk_aversion)
            population.append(portfolio)
        return population

    def tournament_selection(self, k=3):
        # Tournament selection
        selected = []
        for _ in range(2):  # Select two parents
            individuals = np.random.choice(self.population, k)
            best = max(individuals, key=lambda ind: ind.fitness)
            selected.append(best)
        return selected

    def evolve(self, fitness_threshold):
        generation = 0
        best_fitness = -np.inf
        while best_fitness < fitness_threshold:
            generation += 1
            new_population = []
            # Elitism: retain the top individuals
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            elite = self.population[:int(0.1 * self.population_size)]           #selection du top 10% des Portfolio de la population 
            new_population.extend(elite)
            while len(new_population) < self.population_size:
                # Selection
                parent1, parent2 = self.tournament_selection()
                # Crossover using overloaded '+' operator
                child = parent1 + parent2
                # Mutation using overloaded '~' operator
                child = ~child
                new_population.append(child)
            self.population = new_population
            # Update best fitness
            best_portfolio = max(self.population, key=lambda ind: ind.fitness)
            best_fitness = best_portfolio.fitness
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
            # Check for maximum generations
            if self.max_generations and generation >= self.max_generations:
                print("Maximum number of generations reached.")
                break
        print(f"Stopped at generation {generation} with a fitness of {best_fitness:.6f}")
        return best_portfolio