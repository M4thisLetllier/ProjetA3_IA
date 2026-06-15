=====================================================================
PROJET FISE3 - PARTIE IA
Livrable Besoin Client 1 : Cartographie et Clustering Spatial
=====================================================================

1. PRÉREQUIS ET ENVIRONNEMENT
Ce module utilise des bibliothèques externes pour l'analyse de données et la cartographie web. 
Assurez-vous que votre environnement Python dispose des paquets suivants :
> pip install pandas folium matplotlib

2. CONTENU DU DOSSIER
Ce répertoire comprend 4 éléments fondamentaux :
- IRVE_clean_FINAL.csv : La base de données nettoyée issue du module Big Data.
- besoin1_exploration.ipynb : Le Notebook d'exploration. Il contient notre démarche, le nettoyage, les analyses statistiques et la justification de nos choix d'algorithmes (MarkerCluster).
- script_besoin1.py : Le script de production exécutable en ligne de commande.
- Readme.txt : La présente documentation.

3. EXÉCUTION DU SCRIPT
Pour générer les cartographies, ouvrez un terminal dans ce répertoire et exécutez :
> python script_besoin1.py

4. RÉSULTATS ATTENDUS
L'exécution du script prend environ 10 à 15 secondes. Il génèrera deux fichiers HTML directement dans le dossier :
- carte_implantations.html : Visualisation dynamique de l'ensemble du réseau avec gestion algorithmique de la densité (MarkerCluster) selon la charte graphique des implantations.
- carte_chaleur_rapide.html : Analyse de densité (Heatmap) sur fond sombre ciblant spécifiquement la répartition des bornes de recharge rapide à haute puissance.

Ces deux fichiers peuvent être ouverts avec n'importe quel navigateur Web moderne.
=====================================================================