# utils/algorithms.py - ALGORITHMES TSP AVANCÉS (VERSION COMPLÈTE)
import numpy as np
import random
import time
from typing import Dict, List, Tuple
import math
from concurrent.futures import ProcessPoolExecutor
def solve_from_city(args):
        self, cities, start_city = args
        
        # Créer une permutation de villes où start_city est en première position
        reordered = [cities[start_city]] + cities[:start_city] + cities[start_city+1:]
        
        
        improved_distance, improved_path = self.two_opt_improve(reordered)
        
        return improved_distance, improved_path
class TSPSolver:
    def __init__(self):
        self.distance_matrix_cache = {}
        self.nearest_cache = {}

        
    def calculate_distance(self, city1: Tuple[float, float], city2: Tuple[float, float]) -> float:
        #Calcule la distance euclidienne entre deux villes
        return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
    
    def create_distance_matrix(self, cities: List[Tuple[float, float]]) -> np.ndarray:
        #Crée la matrice des distances 
        cities_tuple = tuple(map(tuple, cities))
        #Si elle existe dans la cache
        if cities_tuple in self.distance_matrix_cache:
            return self.distance_matrix_cache[cities_tuple]
        n = len(cities)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i][j] = self.calculate_distance(cities[i], cities[j])
        
        self.distance_matrix_cache[cities_tuple] = matrix
        return matrix
        #Algorithme du plus proche voisin
    def nearest_neighbor(self, cities: List[Tuple[float, float]]) -> Tuple[float, List[int]]:
         
        n = len(cities)
        if n == 0:
            return 0, []
        cities_tuple = tuple(map(tuple, cities))    
        if cities_tuple in self.nearest_cache:
            return self.nearest_cache[cities_tuple]
        distance_matrix = self.create_distance_matrix(cities)
        unvisited = set(range(1, n))
        path = [0]  
        current = 0
        total_distance = 0
        
        while unvisited:
            next_city = min(unvisited, key=lambda city: distance_matrix[current][city])
            total_distance += distance_matrix[current][next_city]
            path.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        
        # Retour au point de départ
        total_distance += distance_matrix[current][0]
        path.append(0)
        self.nearest_cache[cities_tuple] = (total_distance, path)
        return total_distance, path
        # two_opt_delta
    def two_opt_delta(self, path: List[int], distance_matrix: List[List[float]], 
                      i: int, k: int) -> float:
       
        n = len(path)
        
        # Les 4 arêtes 
        a = path[i - 1]
        b = path[i]
        c = path[k]
        d = path[(k + 1) % n]
        if a == c or b == d:
            return 0
        
        current_distance = distance_matrix[a][b] + distance_matrix[c][d]
        
        # Distance des nouvelles arêtes après le swap
        new_distance = distance_matrix[a][c] + distance_matrix[b][d]
        if new_distance >= current_distance:
            return 0
        return new_distance - current_distance
    

    
    def two_opt_swap(self, path: List[int], i: int, k: int) -> List[int]:
        """Effectue un swap 2-opt"""
        path[i:k+1] = path[i:k+1][::-1]
        return path
    
    
    def multi_start_nn_2opt(self, cities: List[Tuple[float, float]], 
                        num_starts: int = 15) -> Tuple[float, List[int]]:
        """
        Multi-start Nearest Neighbor + 2-opt
        Essaie plusieurs points de départ et garde le meilleur
        """
        n = len(cities)
        if n == 0:
            return 0, []
        
        distance_matrix = self.create_distance_matrix(cities)
        best_distance = float('inf')
        best_path = []
        
        print(f"\n Multi-start NN + 2-opt: {num_starts} départs")
        
        for start_idx in range(min(num_starts, n)):
            
            unvisited = set(range(n))
            unvisited.remove(start_idx)
            path = [start_idx]
            current = start_idx
            
            while unvisited:
                next_city = min(unvisited, key=lambda city: distance_matrix[current][city])
                path.append(next_city)
                unvisited.remove(next_city)
                current = next_city
            
            # Convertir en ordre de villes pour two_opt_improve
            cities_order = [cities[i] for i in path]
            
            # Améliorer avec 2-opt
            try:
                improved_dist, improved_cities = self.two_opt_improve2(cities_order)
                
                if improved_dist < best_distance:
                    best_distance = improved_dist
                    best_path = improved_cities
                    print(f"  Départ {start_idx+1}: Nouveau meilleur = {best_distance:.2f}")
            except Exception as e:
                print(f"  Départ {start_idx+1}: Erreur - {str(e)}")
        
        print(f" Meilleure distance: {best_distance:.2f}\n")
        return best_distance, best_path
        
    def two_opt_improve(self, path, cities, max_iterations=30):

        n = len(path)
        current_path = path.copy()
        distance_matrix = self.create_distance_matrix(cities)

        current_distance = sum(distance_matrix[current_path[i]][current_path[i+1]]
                            for i in range(len(current_path)-1))

        improved = True
        iterations = 0

        while improved and iterations < max_iterations:
            improved = False
            best_delta = 0
            best_i = best_k = -1

            for i in range(1, n - 2):
                for k in range(i + 1, n - 1):
                    delta = self.two_opt_delta(current_path, distance_matrix, i, k)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_k = k

            if best_delta < -1e-6:
                current_path = self.two_opt_swap(current_path, best_i, best_k)
                current_distance += best_delta
                improved = True

            iterations += 1

        return current_distance, current_path
    
    
   
    def two_opt_improve2(self, cities: List[Tuple[float, float]]) -> Tuple[float, List[int]]:
        
        max_iterations = 90
        n = len(cities)
        initial_distance, initial_path = self.nearest_neighbor(cities)

        current_path = initial_path.copy()
        current_distance = initial_distance

        distance_matrix = self.create_distance_matrix(cities)
        
        
        
        improved = True
        iterations = 0
        
        
        while improved and iterations < max_iterations:
            improved = False
            best_delta = 0
            best_i = -1
            best_k = -1
            
            # Tester toutes les paires d'arêtes possibles
            for i in range(1, n - 1):
                for k in range(i + 1, n):
                    if k - i == 1:  
                        continue
                    
                    
                    delta = self.two_opt_delta(current_path, distance_matrix, i, k)
                    
                    # Garder la meilleure amélioration
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_k = k
            
            # Appliquer la meilleure amélioration si elle existe
            if best_delta < -0.000001:  # Seuil d'amélioration minimum
                current_path = self.two_opt_swap(current_path, best_i, best_k)
                current_distance += best_delta
                improved = True
            
            iterations += 1
        
        return current_distance, current_path
    
    
    
    
    def _select_strategic_starts(self, cities: List[Tuple[float, float]], 
                                num_starts: int) -> List[int]:
        """Sélectionne des points de départ stratégiques"""
        n = len(cities)
        cities_array = np.array(cities)
        
        selected = []
        
        # 1. Coins extrêmes (4 coins)
        x_coords = cities_array[:, 0]
        y_coords = cities_array[:, 1]
        
        corners = [
            np.argmin(x_coords + y_coords),  # Coin bas-gauche
            np.argmax(x_coords + y_coords),  # Coin haut-droit
            np.argmin(x_coords - y_coords),  # Coin bas-droit
            np.argmax(x_coords - y_coords),  # Coin haut-gauche
        ]
        selected.extend(corners)
        
        # 2. Centres
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        distances_to_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        center_city = np.argmin(distances_to_center)
        selected.append(center_city)
        
        # 3. Points répartis uniformément
        remaining = num_starts - len(selected)
        if remaining > 0:
            step = n // remaining
            for i in range(remaining):
                idx = (i * step) % n
                if idx not in selected:
                    selected.append(idx)
        
        return selected[:num_starts]
    

    def calculate_path_distance(self, path: List[int], distance_matrix: np.ndarray) -> float:
        """Calcule la distance totale d'un chemin"""
        total = 0
        for i in range(len(path) - 1):
            total += distance_matrix[path[i]][path[i+1]]
        return total
    

    

    def ga_tsp(self, cities: List[Tuple[float, float]]):

        POP = 40
        GEN = 120
        ELITE = 6
        MUT_RATE = 0.25

        n = len(cities)
        D = self.create_distance_matrix(cities)

        def random_tour():
            p = list(range(n))
            random.shuffle(p)
            
            return p

        # ---- initial population (light 2-opt)
        population = []
        for _ in range(POP):
            tour = random_tour()
            # On passe le tour à 2-opt
            d, optimized_tour = self.two_opt_improve(tour, cities, max_iterations=3)
            population.append((d, optimized_tour))

        # ---- evolution
        for g in range(GEN):
            population.sort(key=lambda x: x[0])
            new_pop = population[:ELITE]

            while len(new_pop) < POP:
                p1 = random.choice(population)[1]
                p2 = random.choice(population)[1]

                # ---- OX crossover
                a, b = sorted(random.sample(range(1, n), 2))
                child = [None]*n
                p1_set = set(child[a:b])
                child[a:b] = p1[a:b]
                fill = [x for x in p2 if x not in p1_set]
                j = 0
                for i in range(n):
                    if child[i] is None:
                        child[i] = fill[j]
                        j += 1
                

                # ---- mutation
                if random.random() < MUT_RATE:
                    i, k = sorted(random.sample(range(n), 2))
                    child[i:k] = reversed(child[i:k])

                # ---- local improvement
                d, child = self.two_opt_improve(child, cities, max_iterations=6)
                new_pop.append((d, child))

            population = new_pop
            print(f"Gen {g+1}: best = {population[0][0]:.2f}")
         
        best_distance, best_path = population[0]

        return best_distance, best_path
    


    

    def genetic_algorithm(self, cities: List[Tuple[float, float]], 
                     population_size: int = 500, 
                     generations: int = 1000,
                     elite_size: int = 20,
                     mutation_rate: float = 0.02) -> Tuple[float, List[int]]:
        """
        Algorithme génétique simple pour TSP
        """
        n = len(cities)
        if n <= 1:
            return 0, [0] if n == 1 else []
        
        distance_matrix = self.create_distance_matrix(cities)
        
        def create_route():
            return random.sample(range(n), n)
        
        def calculate_fitness(route):
            total = sum(distance_matrix[route[i]][route[(i+1) % n]] for i in range(n))
            return 1 / total if total > 0 else 0
        
        def ordered_crossover(parent1, parent2):
            child = [-1] * n
            start, end = sorted(random.sample(range(n), 2))
            child[start:end] = parent1[start:end]
            parent2_filtered = [item for item in parent2 if item not in child]
            j = 0
            for i in range(n):
                if child[i] == -1:
                    child[i] = parent2_filtered[j]
                    j += 1
            return child
        
        def mutate(route):
            for i in range(n):
                if random.random() < mutation_rate:
                    j = random.randint(0, n - 1)
                    route[i], route[j] = route[j], route[i]
            return route
        
        # Population initiale
        population = [create_route() for _ in range(population_size)]
        
        for gen in range(generations):
            # Évaluation et tri
            fitness_scores = [(i, calculate_fitness(population[i])) for i in range(population_size)]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Sélection élite
            next_gen = [population[fitness_scores[i][0]] for i in range(elite_size)]
            
            # Reproduction
            while len(next_gen) < population_size:
                parent1 = population[fitness_scores[random.randint(0, elite_size-1)][0]]
                parent2 = population[fitness_scores[random.randint(0, elite_size-1)][0]]
                child = ordered_crossover(parent1, parent2)
                child = mutate(child)
                next_gen.append(child)
            
            population = next_gen
        
        # Meilleur individu
        best_route = min(population, key=lambda r: sum(distance_matrix[r[i]][r[(i+1)%n]] for i in range(n)))
        best_distance = sum(distance_matrix[best_route[i]][best_route[(i+1)%n]] for i in range(n))
        
        return best_distance, best_route + [best_route[0]]

    def compare_algorithms(self, cities: List[Tuple[float, float]]) -> Dict:
        """Compare tous les algorithmes et retourne les résultats"""
        algorithms = {
            "Plus Proche Voisin": self.nearest_neighbor,
            
            "two_opt_improve": self.two_opt_improve2,
            "genetic": self.genetic_algorithm,
            "multi_start_nn_2opt": self.multi_start_nn_2opt
            
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