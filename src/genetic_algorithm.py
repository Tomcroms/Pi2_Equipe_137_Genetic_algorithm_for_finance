# genetic_algorithm.py
import numpy as np
from stock import Stock
from portfolio import Portfolio
from portfolio_visualizer import PortfolioVisualizer
import time

class GeneticAlgorithm:
    def __init__(self, stocks, cov_matrix, population_size, risk_aversion=4, budget=10000000, max_generations=None):
        self.stocks = stocks
        self.cov_matrix = cov_matrix
        self.population_size = population_size  #number of portfolio generated at each step
        self.risk_aversion = risk_aversion
        self.budget = budget
        self.population = self.initialize_population()
        self.max_generations = max_generations
        self.stagnation_counter = 0
        self.best_fitness_history = []

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
    
    def adjust_parameters(self):
        Portfolio.eta = max(0.5, Portfolio.eta * 0.9)                                       # Decrease eta, but not below 0.5
        Portfolio.mutation_rate = min(1, Portfolio.mutation_rate * 1.1)                     # Increase up to 100%
        Portfolio.sigma = min(0.5, Portfolio.sigma * 1.1)                                             # Increase sigma
        print(f"New parameters: eta={Portfolio.eta}, mutation_rate={Portfolio.mutation_rate}, sigma={Portfolio.sigma}")

    def fitness_stagnation(self, generation, stagnation_limit=3):
        stagnation = False
        if generation > 2:
            fitness_change = self.best_fitness_history[-1] - self.best_fitness_history[-2]
            min_improvement = 1e-6                                                          # Threshold for minimal improvement
            if abs(fitness_change) < min_improvement:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            if self.stagnation_counter >= stagnation_limit:
                stagnation=True
        
        return stagnation

    def evolve(self, fitness_threshold):
        generation = 0
        best_fitness = -np.inf

        visualizer = PortfolioVisualizer()


        while best_fitness < fitness_threshold:
            generation += 1
            new_population = []
                                                                                            # Elitism: retain the top 10% individuals
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            elite = self.population[:int(0.1 * self.population_size)]
            new_population.extend(elite)

            while len(new_population) < self.population_size:
                # Selection
                parent1, parent2 = Portfolio.tournament_selection(self.population)
                # Crossover using overloaded '+' operator
                child = parent1 + parent2
                # Mutation using overloaded '~' operator
                child = ~child
                new_population.append(child)

            self.population = new_population

            # Update best fitness
            best_portfolio = max(self.population, key=lambda ind: ind.fitness)
            best_fitness = best_portfolio.fitness
            self.best_fitness_history.append(best_fitness)
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")

            visualizer.update(self.population)

            #Check for stagnation
            if(self.fitness_stagnation(generation)):
                self.adjust_parameters()

            # Check for maximum generations
            if self.max_generations and generation >= self.max_generations:
                print("Maximum number of generations reached.")
                break

        print(f"Stopped at generation {generation} with a fitness of {best_fitness:.6f}")

        return best_portfolio