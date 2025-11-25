# utils/analysis.py - ANALYSE STATISTIQUE AVANCÉE
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

class TSPAnalyzer:
    def __init__(self):
        self.results_history = []
    
    def add_execution_result(self, cities_count: int, results: Dict, parameters: Dict = None):
        """Ajoute un résultat d'exécution à l'historique"""
        execution_data = {
            'timestamp': pd.Timestamp.now(),
            'cities_count': cities_count,
            'results': results,
            'parameters': parameters or {}
        }
        self.results_history.append(execution_data)
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Retourne un résumé des performances"""
        summary_data = []
        
        for execution in self.results_history:
            cities_count = execution['cities_count']
            
            for algo_name, result in execution['results'].items():
                if result['success']:
                    summary_data.append({
                        'Algorithm': algo_name,
                        'Cities': cities_count,
                        'Distance': result['distance'],
                        'Time': result['time'],
                        'Efficiency': result['distance'] / result['time'] if result['time'] > 0 else 0,
                        'Timestamp': execution['timestamp']
                    })
        
        return pd.DataFrame(summary_data)
    
    def compare_algorithms(self) -> Dict:
        """Compare les algorithmes sur toutes les exécutions"""
        df = self.get_performance_summary()
        if df.empty:
            return {}
        
        comparison = {}
        
        for algo in df['Algorithm'].unique():
            algo_data = df[df['Algorithm'] == algo]
            comparison[algo] = {
                'mean_distance': algo_data['Distance'].mean(),
                'std_distance': algo_data['Distance'].std(),
                'mean_time': algo_data['Time'].mean(),
                'std_time': algo_data['Time'].std(),
                'mean_efficiency': algo_data['Efficiency'].mean(),
                'best_distance': algo_data['Distance'].min(),
                'worst_distance': algo_data['Distance'].max(),
                'execution_count': len(algo_data),
                'success_rate': len(algo_data) / len(self.results_history)
            }
        
        return comparison
    
    def scalability_analysis(self) -> Dict:
        """Analyse l'évolutivité des algorithmes"""
        df = self.get_performance_summary()
        if df.empty:
            return {}
        
        scalability = {}
        city_sizes = sorted(df['Cities'].unique())
        
        for algo in df['Algorithm'].unique():
            algo_data = df[df['Algorithm'] == algo]
            scalability[algo] = {
                'city_sizes': city_sizes,
                'mean_distances': [],
                'mean_times': [],
                'time_complexity': self.estimate_time_complexity(algo_data)
            }
            
            for city_size in city_sizes:
                size_data = algo_data[algo_data['Cities'] == city_size]
                if not size_data.empty:
                    scalability[algo]['mean_distances'].append(size_data['Distance'].mean())
                    scalability[algo]['mean_times'].append(size_data['Time'].mean())
        
        return scalability
    
    def estimate_time_complexity(self, algo_data: pd.DataFrame) -> str:
        """Estime la complexité temporelle de l'algorithme"""
        if len(algo_data) < 3:
            return "Insufficient data"
        
        # Régression pour estimer la complexité
        X = algo_data['Cities'].values
        y = algo_data['Time'].values
        
        # Test différentes complexités
        complexities = {
            'O(n)': X,
            'O(n log n)': X * np.log(X),
            'O(n²)': X**2,
            'O(n³)': X**3
        }
        
        best_fit = None
        best_r2 = -np.inf
        
        for name, X_transformed in complexities.items():
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(X_transformed, y)
                r2 = r_value**2
                if r2 > best_r2:
                    best_r2 = r2
                    best_fit = name
            except:
                continue
        
        return f"{best_fit} (R²={best_r2:.3f})" if best_fit else "Unknown"
    
    def generate_performance_report(self) -> str:
        """Génère un rapport de performance détaillé"""
        comparison = self.compare_algorithms()
        scalability = self.scalability_analysis()
        
        report = []
        report.append("=" * 60)
        report.append("📊 RAPPORT DE PERFORMANCE TSP")
        report.append("=" * 60)
        
        # Résumé comparatif
        report.append("\n🎯 COMPARAISON DES ALGORITHMES")
        report.append("-" * 40)
        
        for algo, stats in comparison.items():
            report.append(f"\n{algo}:")
            report.append(f"  • Distance moyenne: {stats['mean_distance']:.2f} ± {stats['std_distance']:.2f}")
            report.append(f"  • Temps moyen: {stats['mean_time']:.4f}s ± {stats['std_time']:.4f}s")
            report.append(f"  • Efficacité: {stats['mean_efficiency']:.2f}")
            report.append(f"  • Meilleure distance: {stats['best_distance']:.2f}")
            report.append(f"  • Taux de succès: {stats['success_rate']:.1%}")
        
        # Analyse d'évolutivité
        report.append("\n📈 ANALYSE D'ÉVOLUTIVITÉ")
        report.append("-" * 40)
        
        for algo, data in scalability.items():
            report.append(f"\n{algo}:")
            report.append(f"  • Complexité estimée: {data['time_complexity']}")
            if data['mean_times']:
                growth_factor = data['mean_times'][-1] / data['mean_times'][0] if data['mean_times'][0] > 0 else 0
                report.append(f"  • Facteur de croissance: {growth_factor:.2f}x")
        
        # Recommandations
        report.append("\n💡 RECOMMANDATIONS")
        report.append("-" * 40)
        
        if comparison:
            best_algo = min(comparison.items(), key=lambda x: x[1]['mean_distance'])[0]
            fastest_algo = min(comparison.items(), key=lambda x: x[1]['mean_time'])[0]
            most_efficient = max(comparison.items(), key=lambda x: x[1]['mean_efficiency'])[0]
            
            report.append(f"• Meilleure qualité: {best_algo}")
            report.append(f"• Plus rapide: {fastest_algo}")
            report.append(f"• Plus efficace: {most_efficient}")
            report.append(f"• Exécutions analysées: {len(self.results_history)}")
        
        return "\n".join(report)
    
    def create_scalability_plot(self):
        """Crée un graphique d'évolutivité"""
        df = self.get_performance_summary()
        if df.empty:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Graphique des distances
        for algo in df['Algorithm'].unique():
            algo_data = df[df['Algorithm'] == algo]
            ax1.plot(algo_data['Cities'], algo_data['Distance'], 'o-', label=algo, markersize=6)
        
        ax1.set_xlabel('Nombre de Villes')
        ax1.set_ylabel('Distance Moyenne')
        ax1.set_title('Évolutivité - Distance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique des temps
        for algo in df['Algorithm'].unique():
            algo_data = df[df['Algorithm'] == algo]
            ax2.plot(algo_data['Cities'], algo_data['Time'], 'o-', label=algo, markersize=6)
        
        ax2.set_xlabel('Nombre de Villes')
        ax2.set_ylabel('Temps Moyen (s)')
        ax2.set_title('Évolutivité - Temps d\'Exécution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_performance_radar(self):
        """Crée un graphique radar des performances"""
        comparison = self.compare_algorithms()
        if not comparison:
            return None
        
        algorithms = list(comparison.keys())
        metrics = ['mean_distance', 'mean_time', 'mean_efficiency', 'success_rate']
        
        # Normalisation des données
        normalized_data = {}
        for algo in algorithms:
            normalized_data[algo] = []
            for metric in metrics:
                value = comparison[algo][metric]
                if metric == 'mean_distance':
                    # Plus c'est bas, mieux c'est
                    min_val = min(comparison[a][metric] for a in algorithms)
                    max_val = max(comparison[a][metric] for a in algorithms)
                    normalized = 1 - (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                elif metric == 'mean_time':
                    # Plus c'est bas, mieux c'est
                    min_val = min(comparison[a][metric] for a in algorithms)
                    max_val = max(comparison[a][metric] for a in algorithms)
                    normalized = 1 - (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                else:
                    # Plus c'est haut, mieux c'est
                    min_val = min(comparison[a][metric] for a in algorithms)
                    max_val = max(comparison[a][metric] for a in algorithms)
                    normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                
                normalized_data[algo].append(max(0, min(1, normalized)))
        
        # Création du graphique radar
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Fermer le cercle
        
        metric_labels = ['Distance', 'Temps', 'Efficacité', 'Taux Succès']
        
        for algo in algorithms:
            values = normalized_data[algo] + [normalized_data[algo][0]]  # Fermer le cercle
            ax.plot(angles, values, 'o-', linewidth=2, label=algo, markersize=6)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Comparaison des Performances (Radar)', size=14, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        return fig

# Test de l'analyseur
if __name__ == "__main__":
    analyzer = TSPAnalyzer()
    
    # Données de test
    test_results = {
        'Plus Proche Voisin': {'distance': 100, 'time': 0.1, 'success': True},
        '2-Opt': {'distance': 90, 'time': 1.0, 'success': True},
        'Génétique': {'distance': 85, 'time': 10.0, 'success': True}
    }
    
    analyzer.add_execution_result(10, test_results)
    analyzer.add_execution_result(20, test_results)
    
    print(analyzer.generate_performance_report())