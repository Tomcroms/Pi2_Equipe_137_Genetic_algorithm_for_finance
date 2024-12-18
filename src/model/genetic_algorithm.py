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
        self.in_stagnation_phase = False
        self.stagnation_counter = 0
        self.recovery_counter = 0
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

    def fitness_progress_metric(self, window_size=5):
        # Compute the moving average of best fitness
        if len(self.best_fitness_history) < window_size:
            return None, None
        moving_avg = np.convolve(self.best_fitness_history, np.ones(window_size) / window_size, mode='valid')
        
        # Compute the derivative of the moving average
        if len(moving_avg) > 1:
            derivative = np.diff(moving_avg)
            return moving_avg, derivative
        return moving_avg, None
    
    def detect_stagnation(self, window_size=5, stagnation_threshold=2e-5, recovery_threshold=2e-5, consecutive=3):
        """
        Uses hysteresis:
        - If derivative < stagnation_threshold for `consecutive` times, declare stagnation.
        - If derivative > recovery_threshold for `consecutive` times, declare recovery.
        """
        moving_avg, derivative = self.fitness_progress_metric(window_size)
        if derivative is None:
            return 'none'
        
        last_values = derivative[-consecutive:] if len(derivative) >= consecutive else derivative
        avg_last_derivative = np.mean(last_values)
        
        if self.in_stagnation_phase:
            # If already in stagnation, check for recovery
            if avg_last_derivative > recovery_threshold:
                self.recovery_counter += 1
                if self.recovery_counter >= consecutive:
                    # Declare recovery
                    self.in_stagnation_phase = False
                    self.recovery_counter = 0
                    return 'recovered'
            else:
                self.recovery_counter = 0
        else:
            # Not in stagnation, check for stagnation
            if avg_last_derivative < stagnation_threshold:
                self.stagnation_counter += 1
                if self.stagnation_counter >= consecutive:
                    # Declare stagnation
                    self.in_stagnation_phase = True
                    self.stagnation_counter = 0
                    return 'stagnation'
            else:
                self.stagnation_counter = 0
        
        return 'none'

    def detect_stagnation2(self, window_size=5, stagnation_threshold=2e-5, recovery_threshold=2e-5, consecutive=3):
        """
        Uses hysteresis:
        - If derivative < stagnation_threshold for `consecutive` times, declare stagnation.
        - If derivative > recovery_threshold for `consecutive` times, declare recovery.
        """
        moving_avg, derivative = self.fitness_progress_metric(window_size)
        if derivative is None:
            return 'none'
        
        last_values = derivative[-consecutive:] if len(derivative) >= consecutive else derivative
        avg_last_derivative = np.mean(last_values)
        
        if self.in_stagnation_phase:
            # Still check derivative
            if avg_last_derivative < stagnation_threshold:
                self.stagnation_extension_counter += 1
                if self.stagnation_extension_counter >= consecutive:
                    # Already in stagnation but no improvement? Increase parameters again
                    self.adjust_parameters('stagnation') 
                    self.stagnation_extension_counter = 0
            else:
                self.stagnation_extension_counter = 0

            # Check if we can recover
            if avg_last_derivative > recovery_threshold:
                self.recovery_counter += 1
                if self.recovery_counter >= consecutive:
                    self.in_stagnation_phase = False
                    self.recovery_counter = 0
                    self.stagnation_extension_counter = 0
                    return 'recovered'
            else:
                self.recovery_counter = 0
        else:
            # Normal check for stagnation
            if avg_last_derivative < stagnation_threshold:
                self.stagnation_counter += 1
                if self.stagnation_counter >= consecutive:
                    self.in_stagnation_phase = True
                    self.stagnation_counter = 0
                    self.stagnation_extension_counter = 0
                    return 'stagnation'
            else:
                self.stagnation_counter = 0
        
        return 'none'

    def adjust_parameters(self, mode):
        if mode == 'stagnation':
            # Increase exploration
            Portfolio.mutation_rate = min(1, Portfolio.mutation_rate * 1.1)
            Portfolio.sigma = min(0.5, Portfolio.sigma * 1.1)
            Portfolio.eta = max(0.5, Portfolio.eta * 0.9)
            print(f"Stagnation: Increased mutation. eta={Portfolio.eta}, mutation_rate={Portfolio.mutation_rate}, sigma={Portfolio.sigma}")
        elif mode == 'recovered':
            # Decrease exploration slowly back to baseline
            Portfolio.mutation_rate = max(0.01, Portfolio.mutation_rate * 0.90)
            Portfolio.sigma = max(0.01, Portfolio.sigma * 0.90)
            Portfolio.eta = min(1.0, Portfolio.eta * 1.1)
            print(f"Recovery: Reduced mutation. eta={Portfolio.eta}, mutation_rate={Portfolio.mutation_rate}, sigma={Portfolio.sigma}")

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
            status = self.detect_stagnation2()
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



