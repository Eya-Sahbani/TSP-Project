# AJOUTER CES IMPORTATIONS EN HAUT DU FICHIER
# app.py - APPLICATION TSP COMPLÈTE ET FONCTIONNELLE
import streamlit as st
import numpy as np
import pandas as pd
import json
import csv
from io import StringIO
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.algorithms import TSPSolver
from utils.analysis import TSPAnalyzer
import time

from typing import Dict, List, Tuple
def load_tsp_file(file_content: str):
    """
    Parse un fichier TSPLIB .tsp et retourne une liste de villes (x,y).
    Supporte les formats EUC_2D classiques.
    """
    lines = file_content.splitlines()
    coords = []
    reading_nodes = False
    
    for line in lines:
        line = line.strip()
        
        # Début des coordonnées
        if line.startswith("NODE_COORD_SECTION"):
            reading_nodes = True
            continue
        
        # Fin
        if line.startswith("EOF"):
            break
        
        # Lecture des points
        if reading_nodes:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    # TSPLIB format: index X Y
                    x = float(parts[1])
                    y = float(parts[2])
                    coords.append((x, y))
                except:
                    pass
    
    return coords




       
    
    

def add_comparison_download_button(solutions: Dict, cities: List[Tuple[float, float]]):
    """Ajoute un bouton pour télécharger le rapport de comparaison"""
    report_data = create_comparison_report(solutions, cities)
    st.download_button(
        label=" Télécharger le Rapport de Comparaison",
        data=report_data,
        file_name="tsp_comparison_report.txt",
        mime="text/plain",
        help="Télécharger le rapport complet de comparaison",
        use_container_width=True
    )


# Configuration de la page
st.set_page_config(
    page_title="Solveur TSP Académique",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .algorithm-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .best-solution {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

class TSPApp:
    def __init__(self):
        self.solver = TSPSolver()
        
        self.analyzer = TSPAnalyzer()
        self.n_cities = 15
        self.population_size = 100
        self.generations = 500
        self.max_iterations = 10000
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialise l'état de la session Streamlit"""
        if 'cities' not in st.session_state:
            st.session_state.cities = []
        if 'solutions' not in st.session_state:
            st.session_state.solutions = {}
        if 'city_names' not in st.session_state:
            st.session_state.city_names = []
        if 'best_algorithm' not in st.session_state:
            st.session_state.best_algorithm = None
            
    def run(self):
        """Lance l'application web"""
        self.render_header()
        self.render_sidebar()
        self.render_main_content()
        
    def render_header(self):
        """Affiche l'en-tête de l'application"""
        st.markdown('<h1 class="main-header"> Solveur TSP </h1>', unsafe_allow_html=True)
        st.markdown(" Problème du Voyageur de Commerce - Interface Interactive")
        
    def render_sidebar(self):
        """Affiche la barre latérale avec les contrôles"""
        with st.sidebar:
            st.header("Menu")
            
            # Génération de villes
            st.subheader("Génération de Villes")
            uploaded_file = st.file_uploader(" Importer un fichier TSPLIB (.tsp)", type=["tsp"])

            if uploaded_file is not None:
                content = uploaded_file.read().decode("utf-8")
                cities = load_tsp_file(content)

                if len(cities) > 0:
                    st.session_state.cities = cities
                    st.session_state.city_names = [f"Ville_{i}" for i in range(len(cities))]
                    st.session_state.solutions = {}
                    st.session_state.best_algorithm = None
        
                    st.success(f" Fichier chargé : {len(cities)} villes importées !")
                else:
                    st.error(" Erreur : aucune ville détectée dans ce fichier.")


            
            
            if st.button(" Aléatoires", use_container_width=True):
                self.generate_random_cities()
           
            
            self.n_cities = st.slider("Nombre de villes", 5, 50, 15)
            
            # Algorithmes
            st.subheader(" Algorithmes")
            self.selected_algorithms = st.multiselect(
                "Choisir les algorithmes à comparer:",
                ["multi_start_nn_2opt","two_opt_improve", "Plus Proche Voisin", "genetic",],
                default=[ "two_opt_improve"]
            )
            
            # Paramètres avancés
            with st.expander(" Paramètres Avancés"):
                self.max_iterations = st.slider("Itérations max", 20, 50, 100)
            
            # Bouton de résolution
            if st.button(" Lancer la Comparaison", type="primary", use_container_width=True):
                self.compare_algorithms()
            
            # Affichage des résultats rapides
            if st.session_state.solutions:
                self.render_quick_results()
                
    def render_quick_results(self):
        """Affiche les résultats rapides dans la sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.subheader(" Résultats Rapides")
        
        if st.session_state.best_algorithm:
            best_result = st.session_state.solutions[st.session_state.best_algorithm]
            st.sidebar.metric("Meilleur algorithme", st.session_state.best_algorithm)
            st.sidebar.metric("Distance", f"{best_result['distance']:.2f}")
            st.sidebar.metric("Temps", f"{best_result['time']:.4f}s")
    
    def render_main_content(self):
        """Affiche le contenu principal"""
        if not st.session_state.cities:
            self.render_welcome_screen()
        else:
            tab1, tab2, tab3 = st.tabs(["🎯 Visualisation", "📈 Comparaison", "📊 Analyse"])
            
            with tab1:
                self.render_visualization_tab()
            with tab2:
                self.render_comparison_tab()
            with tab3:
                self.render_analysis_tab()
    
    
    
    def render_visualization_tab(self):
        """Onglet de visualisation"""
        st.subheader(" Visualisation des Solutions")
        
        if st.session_state.solutions:
            # Sélecteur d'algorithme
            selected_algo = st.selectbox(
                "Choisir l'algorithme à visualiser:",
                list(st.session_state.solutions.keys())
            )
            fig = self.create_solution_plot(selected_algo)
        else:
            fig = self.create_cities_plot()
            
        st.plotly_chart(fig, use_container_width=True)
        
        # Détails de la solution
        if st.session_state.solutions:
            self.render_solution_details()
    
    def render_comparison_tab(self):
        """Onglet de comparaison"""
        st.subheader(" Comparaison des Algorithmes")
        
        if not st.session_state.solutions:
            st.warning(" Lancez d'abord une comparaison pour voir les résultats !")
            return
        
        # Métriques comparatives
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            best_algo = st.session_state.best_algorithm
            st.metric("Meilleur", best_algo if best_algo else "N/A")
        
        with col2:
            successful = [r for r in st.session_state.solutions.values() if r['success']]
            if successful:
                best_dist = min(r['distance'] for r in successful)
                st.metric("Meilleure distance", f"{best_dist:.2f}")
        
        with col3:
            if successful:
                fastest = min(st.session_state.solutions.items(), 
                            key=lambda x: x[1]['time'] if x[1]['success'] else float('inf'))[0]
                st.metric("Plus rapide", fastest)
        
        with col4:
            total_time = sum(r['time'] for r in successful)
            st.metric("Temps total", f"{total_time:.2f}s")
        
        # Graphique de comparaison
        fig = self.create_comparison_chart()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Tableau détaillé
        self.render_comparison_table()
    
    def render_analysis_tab(self):
        """Onglet d'analyse"""
        st.subheader(" Analyse des Performances")
        
        if len(self.analyzer.results_history) == 0:
            st.info("""
            ##  Analyse Statistique
            
            **Effectuez plusieurs comparaisons pour débloquer :**
            - 📊 **Analyses statistiques** avancées
            - 📈 **Graphiques d'évolutivité**
            - 🎯 **Recommandations** intelligentes
            
            *Lancez au moins 2-3 comparaisons avec différentes tailles de problèmes*
            """)
            return
        
        # Rapport de performance
        report = self.analyzer.generate_performance_report()
        st.text_area(" Rapport Complet", report, height=300)
        
        # Graphiques d'analyse
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = self.analyzer.create_scalability_plot()
            if fig1:
                st.pyplot(fig1)
        
        with col2:
            fig2 = self.analyzer.create_performance_radar()
            if fig2:
                st.pyplot(fig2)
    
    def generate_random_cities(self):
        """Génère des villes aléatoires"""
        st.session_state.cities = [
            (np.random.uniform(0, 100), np.random.uniform(0, 100))
            for _ in range(self.n_cities)
        ]
        st.session_state.city_names = [f"Ville_{i}" for i in range(self.n_cities)]
        st.session_state.solutions = {}
        st.session_state.best_algorithm = None
        st.success(f" {self.n_cities} villes aléatoires générées !")
        
    
    
    def compare_algorithms(self):
        """Compare les algorithmes sélectionnés"""
        
        if not st.session_state.cities:
            st.error("❌ Veuillez d'abord générer des villes !")
            return
        
        if not self.selected_algorithms:
            st.error("❌ Veuillez sélectionner au moins un algorithme !")
            return
        
        with st.spinner("🔍 Comparaison des algorithmes en cours..."):
            results = {}
            parameters = {
                
                'max_iterations': self.max_iterations
            }
            
            for algo_name in self.selected_algorithms:
                start_time = time.time()
                
                try:
                    if algo_name == "Plus Proche Voisin":
                        distance, path = self.solver.nearest_neighbor(st.session_state.cities)
                    elif algo_name == "multi_start_nn_2opt":
                        distance, path = self.solver.multi_start_nn_2opt(st.session_state.cities)
                    
                    
                    elif algo_name == "two_opt_improve": 
                        distance, path = self.solver.two_opt_improve2(st.session_state.cities)
                    
                    elif algo_name =="genetic":
                        distance, path = self.solver.genetic_algorithm(st.session_state.cities)
                    
                    
                    execution_time = time.time() - start_time
                    
                    results[algo_name] = {
                        'distance': distance,
                        'path': path,
                        'time': execution_time,
                        'success': True
                    }
                    
                except Exception as e:
                    results[algo_name] = {
                        'distance': float('inf'),
                        'path': [],
                        'time': 0,
                        'success': False,
                        'error': str(e)
                    }
            
            # Mettre à jour l'état
            st.session_state.solutions = results
            
            # Enregistrer dans l'analyseur
            self.analyzer.add_execution_result(
                len(st.session_state.cities), 
                results, 
                parameters
            )
            
            # Trouver le meilleur algorithme
            successful = {k: v for k, v in results.items() if v['success']}
            if successful:
                st.session_state.best_algorithm = min(successful.items(), 
                                                    key=lambda x: x[1]['distance'])[0]
        
        successful_count = len([r for r in results.values() if r['success']])
        st.success(f"✅ Comparaison terminée ! {successful_count}/{len(results)} algorithmes réussis")
    
    def create_demo_plot(self):
        """Crée un graphique de démonstration"""
        fig = go.Figure()
        fig.update_layout(
            title="Exemple de Solution TSP",
            xaxis_title="Coordonnée X",
            yaxis_title="Coordonnée Y",
            height=400,
            annotations=[dict(
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                text="Générez des villes pour commencer l'analyse",
                showarrow=False,
                font=dict(size=16, color="gray")
            )]
        )
        return fig
    
    def create_cities_plot(self):
        """Crée un graphique des villes seulement"""
        cities = st.session_state.cities
        fig = go.Figure()
        
        x_coords = [city[0] for city in cities]
        y_coords = [city[1] for city in cities]
        
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode='markers+text',
            marker=dict(size=15, color='red'),
            text=st.session_state.city_names,
            textposition="top center",
            name="Villes"
        ))
        
        fig.update_layout(
            title=f"Problème TSP - {len(cities)} villes",
            xaxis_title="Coordonnée X",
            yaxis_title="Coordonnée Y",
            height=800
        )
        
        return fig
    
    def create_solution_plot(self, algorithm_name):
        """Crée un graphique de solution"""
        cities = st.session_state.cities
        result = st.session_state.solutions[algorithm_name]
        
        fig = go.Figure()
        
        # Villes
        x_coords = [city[0] for city in cities]
        y_coords = [city[1] for city in cities]
        
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode='markers+text',
            marker=dict(size=15, color='red'),
            text=st.session_state.city_names,
            textposition="top center",
            name="Villes"
        ))
        
        # Chemin solution
        if result['success']:
            path = result['path']
            path_x = [cities[i][0] for i in path]
            path_y = [cities[i][1] for i in path]
            
            fig.add_trace(go.Scatter(
                x=path_x, y=path_y,
                mode='lines+markers',
                line=dict(color='blue', width=3),
                marker=dict(size=8, color='blue'),
                name=f"Chemin {algorithm_name}"
            ))
            
            # Point de départ
            fig.add_trace(go.Scatter(
                x=[cities[path[0]][0]], y=[cities[path[0]][1]],
                mode='markers',
                marker=dict(size=20, color='green', symbol='star'),
                name="Départ/Arrivée"
            ))
        
        fig.update_layout(
            title=f"Solution {algorithm_name} - Distance: {result['distance']:.2f}",
            xaxis_title="Coordonnée X",
            yaxis_title="Coordonnée Y",
            height=500
        )
        
        return fig
    
    def create_comparison_chart(self):
        """Crée un graphique de comparaison"""
        successful = {k: v for k, v in st.session_state.solutions.items() if v['success']}
        if not successful:
            return None
        
        algorithms = list(successful.keys())
        distances = [result['distance'] for result in successful.values()]
        times = [result['time'] for result in successful.values()]
        
        fig = go.Figure()
        
        # Barres des distances
        fig.add_trace(go.Bar(
            name='Distance',
            x=algorithms,
            y=distances,
            marker_color='lightblue'
        ))
        
        # Ligne des temps
        fig.add_trace(go.Scatter(
            name='Temps (s)',
            x=algorithms,
            y=times,
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Comparaison des Algorithmes",
            xaxis_title="Algorithmes",
            yaxis_title="Distance",
            yaxis2=dict(
                title="Temps (secondes)",
                overlaying='y',
                side='right'
            ),
            showlegend=True
        )
        
        return fig
    
    def render_comparison_table(self):
        """Affiche le tableau comparatif"""
        st.subheader("📋 Tableau Comparatif Détaillé")
        
        data = []
        for algo, result in st.session_state.solutions.items():
            if result['success']:
                data.append({
                    'Algorithme': algo,
                    'Distance': f"{result['distance']:.2f}",
                    'Temps (s)': f"{result['time']:.4f}",
                    'Efficacité': f"{result['distance']/result['time']:.2f}" if result['time'] > 0 else "N/A"
                })
            else:
                data.append({
                    'Algorithme': algo,
                    'Distance': "Échec",
                    'Temps (s)': "Échec", 
                    'Efficacité': "Échec"
                })
        
        st.dataframe(pd.DataFrame(data), use_container_width=True)
    
    def render_solution_details(self):
        st.subheader("📝 Détails des Solutions")
        
        for algo, result in st.session_state.solutions.items():
            if result['success']:
                with st.expander(f"🔍 {algo} - Distance: {result['distance']:.2f}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Temps d'exécution", f"{result['time']:.4f}s")
                    with col2:
                        st.metric("Nombre d'étapes", len(result['path']))
                    
                    st.write("**Chemin complet :**", result['path'])

                    # --- Bouton de téléchargement ---
                    solution_text = (
                        f"Algorithme : {algo}\n"
                        f"Distance : {result['distance']:.2f}\n"
                        f"Temps : {result['time']:.4f}s\n"
                        "Chemin :\n" +
                        " -> ".join(str(p) for p in result['path'])
                    )

                    st.download_button(
                        label="📥 Télécharger la solution",
                        data=solution_text,
                        file_name=f"solution_{algo.replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
    def render_welcome_screen(self):
        """Affiche l'écran d'accueil"""
        st.markdown("""
        ##  Bienvenue dans le Solveur TSP Académique

        Cette application vous permet de :
        - Générer des villes (aléatoires ou françaises)
        - Importer des fichiers TSPLIB (.tsp)
        - Exécuter plusieurs algorithmes (2-Opt, Génétique, Recuit simulé)
        - Comparer les performances
        - Visualiser les chemins optimaux
        - Analyser l'efficacité selon la taille du problème

        
        """)
def main():
    app = TSPApp()
    app.run()

if __name__ == "__main__":
    main()