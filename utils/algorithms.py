# utils/algorithms.py - ALGORITHMES TSP AVANCÉS (VERSION COMPLÈTE)
import numpy as np
import random
import time
from typing import Dict, List, Tuple
import math

class TSPSolver:
    def __init__(self):
        self.distance_matrix_cache = {}
        
    def calculate_distance(self, city1: Tuple[float, float], city2: Tuple[float, float]) -> float:
        """Calcule la distance euclidienne entre deux villes"""
        return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
    
    def create_distance_matrix(self, cities: List[Tuple[float, float]]) -> np.ndarray:
        """Crée la matrice des distances avec cache pour optimisation"""
        cities_tuple = tuple(map(tuple, cities))
        if cities_tuple in self.distance_matrix_cache:
            return self.distance_matrix_cache[cities_tuple]
            
        n = len(cities)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i][j] = self.calculate_distance(cities[i], cities[j])
        
        self.distance_matrix_cache[cities_tuple] = matrix
        return matrix
    
    def nearest_neighbor(self, cities: List[Tuple[float, float]]) -> Tuple[float, List[int]]:
        """Algorithme du plus proche voisin optimisé"""
        n = len(cities)
        if n == 0:
            return 0, []
            
        distance_matrix = self.create_distance_matrix(cities)
        unvisited = set(range(1, n))
        path = [0]  # Commencer à la première ville
        current = 0
        total_distance = 0
        
        while unvisited:
            # Trouver la ville la plus proche
            next_city = min(unvisited, key=lambda city: distance_matrix[current][city])
            total_distance += distance_matrix[current][next_city]
            path.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        
        # Retour au point de départ
        total_distance += distance_matrix[current][0]
        path.append(0)
        
        return total_distance, path
    
    def two_opt_swap(self, path: List[int], i: int, k: int) -> List[int]:
        """Effectue un swap 2-opt"""
        return path[:i] + path[i:k+1][::-1] + path[k+1:]
    
    def two_opt(self, cities: List[Tuple[float, float]], max_iterations: int = 70) -> Tuple[float, List[int]]:
        """Algorithme 2-opt avec limite d'itérations"""
        n = len(cities)
        if n == 0:
            return 0, []
            
        distance_matrix = self.create_distance_matrix(cities)
        
        # Solution initiale avec plus proche voisin
        current_distance, current_path = self.nearest_neighbor(cities)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            for i in range(1, n-1):
                for k in range(i+1, n):
                    if k - i == 1:
                        continue
                    
                    new_path = self.two_opt_swap(current_path, i, k)
                    new_distance = self.calculate_path_distance(new_path, distance_matrix)
                    
                    if new_distance < current_distance:
                        current_path = new_path
                        current_distance = new_distance
                        improved = True
                        break
                if improved:
                    break
            iterations += 1
        
        return current_distance, current_path
    
    def calculate_path_distance(self, path: List[int], distance_matrix: np.ndarray) -> float:
        """Calcule la distance totale d'un chemin"""
        total = 0
        for i in range(len(path) - 1):
            total += distance_matrix[path[i]][path[i+1]]
        return total
    
    def genetic_algorithm(self, cities: List[Tuple[float, float]], 
                         population_size: int = 100, 
                         generations: int = 500,
                         mutation_rate: float = 0.01) -> Tuple[float, List[int]]:
        """Algorithme génétique pour le TSP"""
        n = len(cities)
        if n == 0:
            return 0, []
            
        distance_matrix = self.create_distance_matrix(cities)
        
        def create_individual():
            """Crée un individu aléatoire"""
            individual = list(range(1, n))
            random.shuffle(individual)
            return [0] + individual + [0]
        
        def fitness(individual):
            """Fonction de fitness (inverse de la distance)"""
            return 1.0 / self.calculate_path_distance(individual, distance_matrix)
        
        def crossover(parent1, parent2):
            """Croisement par ordre"""
            start, end = sorted(random.sample(range(1, n), 2))
            child = [None] * (n + 1)
            child[0] = child[-1] = 0
            
            # Copier le segment du parent1
            child[start:end] = parent1[start:end]
            
            # Remplir le reste avec parent2
            pointer = 1
            for gene in parent2[1:-1]:
                if gene not in child:
                    while child[pointer] is not None:
                        pointer += 1
                    child[pointer] = gene
            
            return child
        
        def mutate(individual):
            """Mutation par échange"""
            if random.random() < mutation_rate:
                i, j = random.sample(range(1, n), 2)
                individual[i], individual[j] = individual[j], individual[i]
            return individual
        
        # Initialiser la population
        population = [create_individual() for _ in range(population_size)]
        
        for generation in range(generations):
            # Évaluation
            fitnesses = [fitness(ind) for ind in population]
            
            # Sélection
            selected = random.choices(
                population, 
                weights=fitnesses, 
                k=population_size
            )
            
            # Croisement et mutation
            new_population = []
            for i in range(0, population_size, 2):
                parent1, parent2 = selected[i], selected[i+1]
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                child1 = mutate(child1)
                child2 = mutate(child2)
                new_population.extend([child1, child2])
            
            population = new_population
        
        # Meilleure solution
        best_individual = max(population, key=fitness)
        best_distance = self.calculate_path_distance(best_individual, distance_matrix)
        
        return best_distance, best_individual
    
    def simulated_annealing(self, cities: List[Tuple[float, float]],
                          initial_temperature: float = 1000,
                          cooling_rate: float = 0.995,
                          max_iterations: int = 10000) -> Tuple[float, List[int]]:
        """Recuit simulé pour le TSP"""
        n = len(cities)
        if n == 0:
            return 0, []
            
        distance_matrix = self.create_distance_matrix(cities)
        
        # Solution initiale
        current_solution = list(range(n)) + [0]
        current_distance = self.calculate_path_distance(current_solution, distance_matrix)
        
        best_solution = current_solution[:]
        best_distance = current_distance
        
        temperature = initial_temperature
        
        for iteration in range(max_iterations):
            # Générer un voisin
            i, j = random.sample(range(1, n), 2)
            new_solution = current_solution[:]
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            new_distance = self.calculate_path_distance(new_solution, distance_matrix)
            
            # Critère d'acceptation
            if new_distance < current_distance:
                current_solution = new_solution
                current_distance = new_distance
                if new_distance < best_distance:
                    best_solution = new_solution[:]
                    best_distance = new_distance
            else:
                # Acceptation probabiliste
                probability = math.exp((current_distance - new_distance) / temperature)
                if random.random() < probability:
                    current_solution = new_solution
                    current_distance = new_distance
            
            # Refroidissement
            temperature *= cooling_rate
            
            # Arrêt précoce si la température est trop basse
            if temperature < 1e-10:
                break
        
        return best_distance, best_solution

    # =========================================================================
    # MÉTHODES OPTIMISÉES AJOUTÉES
    # =========================================================================

    def nearest_neighbor_optimized(self, cities: List[Tuple[float, float]]) -> Tuple[float, List[int]]:
        """Plus proche voisin optimisé avec early stopping"""
        n = len(cities)
        if n <= 1:
            return 0, [0] if n == 1 else []
        
        distance_matrix = self.create_distance_matrix(cities)
        
        # Essayer différents points de départ
        best_distance = float('inf')
        best_path = []
        
        for start_city in range(min(5, n)):  # Tester les 5 premières villes comme départ
            unvisited = set(range(n))
            unvisited.remove(start_city)
            path = [start_city]
            current = start_city
            total_distance = 0
            
            while unvisited:
                # Trouver les k plus proches voisins
                k_neighbors = min(10, len(unvisited))
                candidates = sorted(unvisited, key=lambda city: distance_matrix[current][city])[:k_neighbors]
                
                next_city = candidates[0]
                total_distance += distance_matrix[current][next_city]
                path.append(next_city)
                unvisited.remove(next_city)
                current = next_city
            
            # Retour au point de départ
            total_distance += distance_matrix[current][start_city]
            
            if total_distance < best_distance:
                best_distance = total_distance
                best_path = path + [start_city]
        
        return best_distance, best_path

    def hybrid_algorithm(self, cities: List[Tuple[float, float]]) -> Tuple[float, List[int]]:
        """Algorithme hybride : Génétique + 2-opt"""
        n = len(cities)
        if n <= 1:
            return 0, [0] if n == 1 else []
        
        # Étape 1: Solution initiale avec algorithme génétique
        genetic_distance, genetic_path = self.genetic_algorithm(
            cities, 
            population_size=50,  # Plus petit pour la rapidité
            generations=200
        )
        
        # Étape 2: Amélioration avec 2-opt
        improved_distance, improved_path = self.two_opt(cities)
        
        # Choisir la meilleure solution
        if genetic_distance < improved_distance:
            return genetic_distance, genetic_path
        else:
            return improved_distance, improved_path

    def parallel_genetic_algorithm(self, cities: List[Tuple[float, float]], 
                                 population_size: int = 100,
                                 generations: int = 500,
                                 num_islands: int = 4) -> Tuple[float, List[int]]:
        """Algorithme génétique parallèle (îles)"""
        # Implémentation simplifiée du modèle d'îles
        n = len(cities)
        if n == 0:
            return 0, []
        
        distance_matrix = self.create_distance_matrix(cities)
        
        def create_island_population(pop_size):
            """Crée une population pour une île"""
            return [self._create_individual(n) for _ in range(pop_size)]
        
        # Créer les îles
        island_populations = [create_island_population(population_size // num_islands) 
                             for _ in range(num_islands)]
        
        best_global = None
        best_global_fitness = float('inf')
        
        for generation in range(generations):
            # Évolution dans chaque île
            for i, population in enumerate(island_populations):
                # Sélection et reproduction dans l'île
                fitnesses = [1/self.calculate_path_distance(ind, distance_matrix) for ind in population]
                selected = random.choices(population, weights=fitnesses, k=len(population))
                
                new_population = []
                for j in range(0, len(selected), 2):
                    parent1, parent2 = selected[j], selected[j+1]
                    child1 = self._crossover(parent1, parent2, n)
                    child2 = self._crossover(parent2, parent1, n)
                    child1 = self._mutate(child1, n)
                    child2 = self._mutate(child2, n)
                    new_population.extend([child1, child2])
                
                island_populations[i] = new_population
                
                # Mettre à jour la meilleure solution globale
                island_best = min(population, key=lambda ind: self.calculate_path_distance(ind, distance_matrix))
                island_fitness = self.calculate_path_distance(island_best, distance_matrix)
                
                if island_fitness < best_global_fitness:
                    best_global = island_best
                    best_global_fitness = island_fitness
            
            # Migration entre îles (toutes les 20 générations)
            if generation % 20 == 0 and generation > 0:
                # Échanger les meilleurs individus entre îles
                migrants = []
                for population in island_populations:
                    best_individual = min(population, key=lambda ind: self.calculate_path_distance(ind, distance_matrix))
                    migrants.append(best_individual)
                
                # Mélanger les migrants
                random.shuffle(migrants)
                for i, population in enumerate(island_populations):
                    # Remplacer le pire individu par un migrant
                    worst_index = max(range(len(population)), 
                                    key=lambda idx: self.calculate_path_distance(population[idx], distance_matrix))
                    population[worst_index] = migrants[i]
        
        return best_global_fitness, best_global

    def _create_individual(self, n: int) -> List[int]:
        """Crée un individu pour l'algorithme génétique"""
        individual = list(range(1, n))
        random.shuffle(individual)
        return [0] + individual + [0]

    def _crossover(self, parent1: List[int], parent2: List[int], n: int) -> List[int]:
        """Croisement pour l'algorithme génétique"""
        start, end = sorted(random.sample(range(1, n), 2))
        child = [None] * (n + 1)
        child[0] = child[-1] = 0
        child[start:end] = parent1[start:end]
        
        pointer = 1
        for gene in parent2[1:-1]:
            if gene not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = gene
        
        return child

    def _mutate(self, individual: List[int], n: int) -> List[int]:
        """Mutation pour l'algorithme génétique"""
        if random.random() < 0.1:  # 10% de chance de mutation
            i, j = random.sample(range(1, n), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def compare_algorithms(self, cities: List[Tuple[float, float]]) -> Dict:
        """Compare tous les algorithmes et retourne les résultats"""
        algorithms = {
            "Plus Proche Voisin": self.nearest_neighbor,
            "Plus Proche Voisin Optimisé": self.nearest_neighbor_optimized,
            "2-Opt": self.two_opt,
            "Génétique": self.genetic_algorithm,
            "Génétique Parallèle": self.parallel_genetic_algorithm,
            "Recuit Simulé": self.simulated_annealing,
            "Hybride": self.hybrid_algorithm
        }
        
        results = {}
        
        for name, algorithm in algorithms.items():
            start_time = time.time()
            try:
                distance, path = algorithm(cities)
                execution_time = time.time() - start_time
                results[name] = {
                    'distance': distance,
                    'path': path,
                    'time': execution_time,
                    'success': True
                }
            except Exception as e:
                results[name] = {
                    'distance': float('inf'),
                    'path': [],
                    'time': 0,
                    'success': False,
                    'error': str(e)
                }
        
        return results

# Test des algorithmes
if __name__ == "__main__":
    solver = TSPSolver()
    test_cities = [(0,0), (1,2), (3,1), (2,3), (4,2)]
    print("Test des algorithmes TSP avancés:")
    print(f"Villes: {test_cities}")
    
    results = solver.compare_algorithms(test_cities)
    for algo, result in results.items():
        if result['success']:
            print(f"{algo}: Distance={result['distance']:.2f}, Temps={result['time']:.4f}s")
        else:
            print(f"{algo}: Erreur - {result['error']}")