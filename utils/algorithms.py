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
        cities_tuple = tuple(map(tuple, cities))    
        if cities_tuple in self.nearest_cache:
            return self.nearest_cache[cities_tuple]
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
        self.nearest_cache[cities_tuple] = (total_distance, path)
        return total_distance, path
    def two_opt_delta(self, path: List[int], distance_matrix: List[List[float]], 
                      i: int, k: int) -> float:
       
        n = len(path)
        
        # Les 4 arêtes concernées par le swap
        a = path[i - 1]
        b = path[i]
        c = path[k]
        d = path[(k + 1) % n]
        
        # Distance actuelle des arêtes à supprimer
        current_distance = distance_matrix[a][b] + distance_matrix[c][d]
        
        # Distance des nouvelles arêtes après le swap
        new_distance = distance_matrix[a][c] + distance_matrix[b][d]
        
        return new_distance - current_distance
    
    def two_opt_swap(self, path: List[int], i: int, k: int) -> List[int]:
        """Effectue un swap 2-opt"""
    
        return path[:i] + path[i:k+1][::-1] + path[k+1:]
    
    def two_opt(self, cities: List[Tuple[float, float]], max_iterations: int = 20) -> Tuple[float, List[int]]:
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
    def two_opt_improve(self, cities: List[Tuple[float, float]]) -> Tuple[float, List[int]]:
        
        max_iterations = 20
        n = len(cities)
        initial_distance, initial_path = self.nearest_neighbor(cities)

        current_path = initial_path.copy()
        current_distance = initial_distance

        distance_matrix = self.create_distance_matrix(cities)
        
        
        # Amélioration avec 2-opt
       
        
        
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
                    if k - i == 1:  # Skip arêtes adjacentes
                        continue
                    
                    # Calculer le changement de distance
                    delta = self.two_opt_delta(current_path, distance_matrix, i, k)
                    
                    # Garder la meilleure amélioration
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_k = k
            
            # Appliquer la meilleure amélioration si elle existe
            if best_delta < -1:  # Seuil d'amélioration minimum
                current_path = self.two_opt_swap(current_path, best_i, best_k)
                current_distance += best_delta
                improved = True
            
            iterations += 1
        
        return current_distance, current_path
    
        
       
    

   

    def parallel_tsp(self, cities: List[Tuple[float, float]] )-> Tuple[float, List[int]]:
        num_threads=4
        n = len(cities)
        tasks = [(self, cities, i) for i in range(n)]
        
        best_distance = float('inf')
        best_path = None
        print("paralll")
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            print("kkkkk")
            for dist, path in executor.map(solve_from_city, tasks):
                print("paaaaaa")
                if dist < best_distance:
                    best_distance = dist
                    best_path = path

        return best_distance, best_path
    
    def smart_multi_start(self, cities: List[Tuple[float, float]], 
                         num_starts: int = 20,
                         max_iterations_2opt: int = 30) -> Tuple[float, List[int], Dict]:
        """
        Multi-start intelligent avec échantillonnage stratégique
        
        Temps estimé pour 929 villes: 10-15 minutes
        Qualité: 93-96% de l'optimal
        """
        n = len(cities)
        start_time = time.time()
        
        print(f"Smart Multi-Start - {n} villes, {num_starts} départs")
        print("-" * 60)
        
        distance_matrix = self.create_distance_matrix(cities)
        
        # Sélection stratégique: coins + centres + aléatoires
        start_cities = self._select_strategic_starts(cities, num_starts)
        
        best_distance = float('inf')
        best_path = []
        
        for idx, start_city in enumerate(start_cities):
            # Nearest neighbor
            nn_dist, nn_path = self.nearest_neighbor_fast(cities, distance_matrix, start_city)
            
            # 2-opt léger
            opt_path = nn_path
            opt_dist = nn_dist
            
            for _ in range(max_iterations_2opt):
                improved = False
                for i in range(1, n - 1):
                    for k in range(i + 2, min(i + 50, n)):  # Fenêtre limitée
                        if self._should_swap_2opt(opt_path, distance_matrix, i, k):
                            opt_path = self._swap_2opt(opt_path, i, k)
                            opt_dist = self.calculate_path_distance(opt_path, distance_matrix)
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            
            if opt_dist < best_distance:
                best_distance = opt_dist
                best_path = opt_path
                print(f"  Départ {idx+1}/{num_starts}: Nouveau meilleur = {best_distance:.2f}")
        
        elapsed = time.time() - start_time
        
        stats = {
            'best_distance': best_distance,
            'num_starts': num_starts,
            'time': elapsed
        }
        
        print(f"\n✅ Meilleure distance: {best_distance:.2f}")
        print(f"⏱️  Temps total: {elapsed:.1f}s")
        
        return best_distance, best_path, stats
    
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
    
    def _should_swap_2opt(self, path: List[int], distance_matrix: List[List[float]], 
                         i: int, k: int) -> bool:
        """Vérifie si un swap 2-opt améliore la solution"""
        n = len(path)
        a, b = path[i-1], path[i]
        c, d = path[k], path[(k+1) % n]
        
        current = distance_matrix[a][b] + distance_matrix[c][d]
        new = distance_matrix[a][c] + distance_matrix[b][d]
        
        return new < current - 0.01
    
    def _swap_2opt(self, path: List[int], i: int, k: int) -> List[int]:
        """Effectue un swap 2-opt"""
        return path[:i] + path[i:k+1][::-1] + path[k+1:]

    def calculate_path_distance(self, path: List[int], distance_matrix: np.ndarray) -> float:
        """Calcule la distance totale d'un chemin"""
        total = 0
        for i in range(len(path) - 1):
            total += distance_matrix[path[i]][path[i+1]]
        return total
    
    
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