import numpy as np
from src.model.stock import Stock
from src.model.portfolio import Portfolio
from src.view.portfolio_visualizer import PortfolioVisualizer
from src.view.fitness_visualization import FitnessVisualizer
import time

class GeneticAlgorithm:
    def __init__(self, stocks, cov_matrix, population_size=100, is_short_available=False, fitness_function=None, crossover_function=None, mutation_function=None, selection_method=None, risk_aversion=4, budget=10000000, max_generations=None, enable_visuals=True, verbose=None, log_every=10):
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
        self.stagnation_extension_counter = 0
        self.best_fitness_history = []
        self.enable_visuals = enable_visuals
        if self.enable_visuals:
            self.portfolio_visualizer = PortfolioVisualizer()
            self.fitness_visualizer = FitnessVisualizer()
        else:
            self.portfolio_visualizer = None
            self.fitness_visualizer = None

    def initialize_population(self):
        """
        Create the initial population of portfolios.

        Design choices:
        - Long-only: sample weights from a Dirichlet so they are >= 0 and sum to 1,
        then convert to shares that exactly match the cash budget.
        - Long/short: build a signed weight vector where the gross exposure
        (sum of absolute weights) equals 1, then scale so gross exposure equals
        the cash budget. This keeps the notion of "budget" consistent for both modes.
        - After building shares, we always call `adjust_shares_to_budget()` as a safety
        net to guarantee the final exposure is exactly equal to the budget model.
        """
        population = []
        prices = np.array([stock.price for stock in self.stocks])
        n = len(self.stocks)

        for _ in range(self.population_size):
            if self.is_short_available:
                # ---- Long/short initialization ----
                # We want signed weights whose L1 norm (sum of abs weights) is 1.
                # A simple way: draw a long mix and a short mix from Dirichlet,
                # then combine them with a chosen gross split (e.g., 50% long / 50% short).
                long_mix = np.random.dirichlet(np.ones(n))   # non-negative, sums to 1
                short_mix = np.random.dirichlet(np.ones(n))  # non-negative, sums to 1

                gross_long_share = 0.5  # 50% gross long / 50% gross short to start
                signed_w = gross_long_share * long_mix - (1.0 - gross_long_share) * short_mix

                # Normalize so that sum(|w|) = 1 (pure gross exposure of 1 unit).
                l1 = np.sum(np.abs(signed_w))
                # Edge case: if numerically tiny (extremely unlikely), fall back to equal long-only
                if l1 <= 0:
                    signed_w = np.ones(n) / n
                    l1 = 1.0
                signed_w = signed_w / l1

                # Turn weights into dollar allocations. With long/short, "budget" means
                # target gross exposure (|long| + |short|) equals the cash budget.
                dollar_alloc = signed_w * self.budget

            else:
                # ---- Long-only initialization ----
                # Draw a random allocation on the simplex (>=0, sums to 1)
                w = np.random.dirichlet(np.ones(n))

                # Convert weights into dollar allocations so total invested equals budget.
                dollar_alloc = w * self.budget

            # Convert dollars to number of shares (real-valued; we keep it continuous here).
            # If later need integer shares, floor/round BEFORE calling adjust_shares_to_budget(),
            # but be aware the rescale will make them continuous again.
            shares = dollar_alloc / prices

            # Build the Portfolio object
            portfolio = Portfolio(
                shares, self.stocks, self.cov_matrix,
                self.is_short_available, self.fitness_function,
                self.crossover_function, self.mutation_function,
                self.budget, self.risk_aversion
            )

            # Safety: force the exposure to match the budget model exactly
            # (long-only => sum of long dollars = budget ; long/short => gross = budget).
            portfolio.adjust_shares_to_budget()

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
            Portfolio.eta = max(2, Portfolio.eta * 0.9)
            print(f"Stagnation: Increased mutation. eta={Portfolio.eta}, mutation_rate={Portfolio.mutation_rate}, sigma={Portfolio.sigma}")
        elif mode == 'recovered':
            # Decrease exploration slowly back to baseline
            Portfolio.mutation_rate = max(0.01, Portfolio.mutation_rate * 0.90)
            Portfolio.sigma = max(0.01, Portfolio.sigma * 0.90)
            Portfolio.eta = min(25, Portfolio.eta * 1.1)
            print(f"Recovery: Reduced mutation. eta={Portfolio.eta}, mutation_rate={Portfolio.mutation_rate}, sigma={Portfolio.sigma}")

    def evolve(self, fitness_threshold):
        generation = 0
        best_fitness = -np.inf

        while best_fitness < fitness_threshold:
            if generation > 50 and len(self.best_fitness_history) >= 20:
                recent = self.best_fitness_history[-20:]
                if max(recent) - min(recent) < 1e-6:
                    print("Early stopping due to stagnation.")
                    break
                
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

            if self.portfolio_visualizer:
                self.portfolio_visualizer.update(self.population, best_portfolio)
            if self.fitness_visualizer:
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



