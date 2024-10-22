import numpy as np
from .stock import Stock
from .portfolio import Portfolio


class GeneticAlgorithm:
    def __init__(self, stocks, cov_matrix, population_size, risk_aversion=4, max_generations=None):
        self.stocks = stocks  # Liste des objets Stock
        self.cov_matrix = cov_matrix  # Matrice de covariance
        self.population_size = population_size
        self.risk_aversion = risk_aversion
        self.population = self.initialize_population()
        if(max_generations):
            self.max_generations = max_generations
        else:
            self.max_generations = max_generations

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            # Générer des poids aléatoires et normaliser
            weights = np.random.rand(len(self.stocks))
            weights = weights / np.sum(weights)
            portfolio = Portfolio(weights, self.stocks, self.cov_matrix)
            portfolio.calculate_fitness(self.risk_aversion)
            population.append(portfolio)
        return population

    def tournament_selection(self, k=3):
        # Sélection par tournoi
        selected = []
        for _ in range(2):  # Sélectionner deux parents
            individuals = np.random.choice(self.population, k)
            best = max(individuals, key=lambda ind: ind.fitness)
            selected.append(best)
        return selected

    def sbx_crossover(self, parent1, parent2, eta=2):
        # Simulated Binary Crossover (SBX)
        child_weights = []
        for i in range(len(parent1.weights)):
            u = np.random.rand()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            child_weight = 0.5 * ((1 + beta) * parent1.weights[i] + (1 - beta) * parent2.weights[i])
            child_weights.append(child_weight)
        # Normaliser les poids
        child_weights = np.array(child_weights)
        child_weights[child_weights < 0] = 0
        child_weights = child_weights / np.sum(child_weights)
        child = Portfolio(child_weights, self.stocks, self.cov_matrix)
        child.calculate_fitness(self.risk_aversion)
        return child

    def gaussian_mutation(self, portfolio: Portfolio, mutation_rate=0.1, sigma=0.1):
        # Mutation gaussienne
        mutated_weights = []
        portfolio.__class__.__annotations__
        for weight in portfolio.weights:
            if np.random.rand() < mutation_rate:
                weight += np.random.normal(0, sigma)
            mutated_weights.append(weight)
        # Normaliser les poids
        mutated_weights = np.array(mutated_weights)
        mutated_weights[mutated_weights < 0] = 0
        mutated_weights = mutated_weights / np.sum(mutated_weights)
        mutated_portfolio = Portfolio(mutated_weights, self.stocks, self.cov_matrix)
        mutated_portfolio.calculate_fitness(self.risk_aversion)
        return mutated_portfolio

    def evolve(self, fitness_threshold):
        generation = 0
        best_fitness = -np.inf  # Initialisation à une valeur très basse
        while best_fitness < fitness_threshold:
            generation += 1
            new_population = []
            # Élitisme : conserver les meilleurs individus
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            elite = self.population[:int(0.1 * self.population_size)]
            new_population.extend(elite)
            while len(new_population) < self.population_size:
                # Sélection
                parent1, parent2 = self.tournament_selection()
                # Croisement
                child = self.sbx_crossover(parent1, parent2)
                # Mutation
                child = self.gaussian_mutation(child)
                new_population.append(child)
            self.population = new_population
            # Mise à jour du meilleur fitness
            best_portfolio = max(self.population, key=lambda ind: ind.fitness)
            best_fitness = best_portfolio.fitness
            print(f"Génération {generation}: Meilleur fitness = {best_fitness:.6f}")
            # Ici si on a définie un max_generations, on vérifie si on l'a dépassé (pour éviter boucle infinie par ex)
            if self.max_generations and generation >= self.max_generations:
                print("Nombre maximal de générations atteint.")
                break
        print(f"Arrêt à la génération {generation} avec un fitness de {best_fitness:.6f}")
        return best_portfolio



