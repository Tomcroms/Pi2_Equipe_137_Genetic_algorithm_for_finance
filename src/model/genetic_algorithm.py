import numpy as np
from model.stock import Stock
from model.portfolio import Portfolio
from view.portfolio_visualizer import PortfolioVisualizer
from view.fitness_visualization import FitnessVisualizer
import time

class GeneticAlgorithm:
    def __init__(self, stocks, cov_matrix, population_size=100, is_short_available=False, fitness_function=None, crossover_function=None, mutation_function=None, selection_method=None, risk_aversion=4, budget=10000000, max_generations=None):
        self.stocks = stocks
        self.cov_matrix = cov_matrix
        self.population_size = population_size  #number of portfolio generated at each step
        self.is_short_available = is_short_available
        self.fitness_function = fitness_function
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        self.risk_aversion = risk_aversion
        self.budget = budget
        self.population = self.initialize_population()
        self.selection_method = selection_method
        self.max_generations = max_generations
        self.stagnation_counter = 0
        self.best_fitness_history = []
        self.portfolio_visualizer = PortfolioVisualizer()
        self.fitness_visualizer = FitnessVisualizer()

    def initialize_population(self):
        population = []
        prices = np.array([stock.price for stock in self.stocks])
        for _ in range(self.population_size):
            # Generate random number of shares within the budget
            shares = np.random.rand(len(self.stocks)) * (self.budget / prices)
            shares = np.floor(shares)  # Use whole shares
            portfolio = Portfolio(shares, self.stocks, self.cov_matrix, self.is_short_available, self.fitness_function, self.crossover_function, self.mutation_function, self.budget, self.risk_aversion)
            population.append(portfolio)
        return population
    
    def adjust_parameters(self):
        Portfolio.eta = max(0.5, Portfolio.eta * 0.9)                                       # Decrease eta, but not below 0.5
        Portfolio.mutation_rate = min(1, Portfolio.mutation_rate * 1.1)                     # Increase up to 100%
        Portfolio.sigma = min(0.5, Portfolio.sigma * 1.1)                                   # Increase sigma
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

    def fitness_stagnation2(self, generation, threshold=0.001):
        """
        Check if the fitness function shows stagnation based on the derivative of the moving average.
        """
        window_size = 5
        if len(self.best_fitness_history) < window_size:
            return False

        # Compute the moving average
        moving_avg = np.convolve(
            self.best_fitness_history, np.ones(window_size) / window_size, mode='valid'
        )

        # Compute the derivative of the moving average
        if len(moving_avg) > 1:
            derivative = np.diff(moving_avg)
            print(f"Moving average derivative (last): {derivative[-1]:.6f}")

            # Check if the last derivative is below the threshold
            if derivative[-1] < threshold:
                print("Fitness is stagnating based on moving average.")
                return True

        return False

    def compute_global_progression(self):
        """
        Compute and print global progression metrics: cumulative improvement and mean rate.
        """
        if len(self.best_fitness_history) < 2:
            print("Not enough data for global progression metrics.")
            return

        # Total progression
        cumulative_progression = sum(
            abs(self.best_fitness_history[i] - self.best_fitness_history[i - 1])
            for i in range(1, len(self.best_fitness_history))
        )

        # Mean rate of progression
        mean_progression_rate = cumulative_progression / len(self.best_fitness_history)

        print("\n=== Global Progression Metrics ===")
        print(f"Cumulative progression: {cumulative_progression:.6f}")
        print(f"Mean progression rate: {mean_progression_rate:.6f}")
        print("==================================")

    def evolve(self, fitness_threshold):
        generation = 0
        best_fitness = -np.inf

        while best_fitness < fitness_threshold:
            generation += 1
            new_population = []
                                                                                            # Elitism: retain the top 10% individuals
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            elite = self.population[:int(0.1 * self.population_size)]
            new_population.extend(elite)

            while len(new_population) < self.population_size:
                # Selection                
                parent1, parent2 = Portfolio.selection(self.population, self.selection_method)
                
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

            self.portfolio_visualizer.update(self.population, best_portfolio)
            self.fitness_visualizer.update(best_portfolio, generation)

            #Check for stagnation
            status = self.detect_stagnation()
            if status == 'stagnation':
                self.adjust_parameters('stagnation')
            elif status == 'recovered':
                self.adjust_parameters('recovered')

            # Check for maximum generations
            if self.max_generations and generation >= self.max_generations:
                print("Maximum number of generations reached.")
                break

        print(f"Stopped at generation {generation} with a fitness of {best_fitness:.6f}")

        return best_portfolio



