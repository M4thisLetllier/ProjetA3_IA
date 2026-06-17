import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score,calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import  KMeans
import folium
import os
import joblib
from configuration import COULEURS

def creation_modele_Kmeans(chemin_modele, coords, k, df):
    if os.path.exists(chemin_modele):
        # Le modèle existe déjà ! On le charge
        print(f"Chargement du modèle existant pour k={k}...")
        kmeans = joblib.load(chemin_modele)
        df['cluster_kmeans'] = kmeans.predict(coords)
    else:
        # Le modèle n'existe pas. On l'entraîne et on le sauvegarde.
        print(f"Entraînement d'un nouveau modèle pour k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")

        # On entraîne et on récupère les labels
        df['cluster_kmeans'] = kmeans.fit_predict(coords)

        # On sauvegarde le modèle sur le disque
        joblib.dump(kmeans, chemin_modele)
        print("Modèle sauvegardé avec succès.")
    return df

def creation_carte_kmeans(df_kmeans, k):
    print("Creation de la carte ...")
    carte_france = folium.Map(location=[46.5, 2.5], zoom_start=6, tiles='OpenStreetMap')
    # Sécurité
    data_to_plot = df_kmeans[['nom_station', 'lon', 'lat', 'cluster_kmeans']].dropna()
    print("Ajout des marqueurs ...")
    # 4. Ajout des points (marqueurs) sur la carte
    for index, row in data_to_plot.iterrows():
        # On récupère le numéro du cluster
        num_cluster = int(row['cluster_kmeans'])

        # On choisit la couleur associée dans la palette
        couleur_point = COULEURS[num_cluster % len(COULEURS)]

        # On crée le marqueur
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=2,
            popup=f"Station: {row['nom_station']}<br>Cluster K-Means: {num_cluster}",
            color=couleur_point,
            fill=True,
            fill_color=couleur_point,
            fill_opacity=0.7
        ).add_to(carte_france)

    # 5. Sauvegarde de la carte dans un fichier HTML
    carte_france.save(f"carte/carte_kmeans_k{k}.html")

    print("La carte interactive a été générée : carte_kmeans_k.html")


def silhouette(coords):
    meilleur_score = -1
    meilleur_k = 2
    # On limite la recherche (par exemple à 10, ou au nombre max de points - 1)
    k_max_test = min(10, len(coords) - 1)

    for k in range(2, k_max_test + 1):
        print(f'teste n {k}')
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        score = silhouette_score(coords, labels, sample_size=2000, random_state=42)

        if score > meilleur_score:
            meilleur_score = score
            meilleur_k = k
    return meilleur_k,meilleur_score


def calinski_plushaut(coords) :
    meilleur_score = -1
    meilleur_k = 2
    k_max_test = 15


    for k in range(2, k_max_test + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(coords)

        # Le calcul magique et ultra-rapide
        score = calinski_harabasz_score(coords, labels)

        if score > meilleur_score:
            meilleur_score = score
            meilleur_k = k

    return meilleur_k,meilleur_score


def calinski_saut(coords):
    score = -1
    meilleur_score = -1
    meilleur_k = 2
    k_max_test =15
    saut= 0
    meilleur_saut=0

    for k in range(2, k_max_test + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(coords)

        # Le calcul magique et ultra-rapide
        temp = calinski_harabasz_score(coords, labels)
        saut =  temp - score
        score = temp


        if saut > meilleur_saut:
            meilleur_saut = saut
            meilleur_score = score
            meilleur_k = k

    return meilleur_k, meilleur_saut

def davies(coords):
    meilleur_score = float('inf')
    meilleur_k = 2
    k_max_test = 15

    for k in range(2, k_max_test + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(coords)

        # Calcul du score de Davies-Bouldin
        score = davies_bouldin_score(coords, labels)

        # Si le score est PLUS PETIT, c'est qu'il est meilleur !
        if score < meilleur_score:
            meilleur_score = score
            meilleur_k = k
    return meilleur_k,meilleur_score


def KmeansCluster(df,nb_clusters :int):
    """Permet la creation d'un modele et d'une carte de celui ci"""
    assert 0 < nb_clusters < 187,"Number of cluster must be between 0 and 22"
    #1. Création du modèle et prédiction
    print("Chargement des clusters ...")
    coords = df[['lon', 'lat']].dropna()
    kmeans = KMeans(n_clusters=nb_clusters, random_state=42, n_init=10)
    df['cluster_kmeans'] = kmeans.fit_predict(X = coords)
    data_to_plot = df[['nom_station', 'lon', 'lat', 'cluster_kmeans']].dropna()

    # 2. Définition de la palette de couleurs
    grandes_couleurs = [
        'red', 'blue', 'green', 'purple', 'orange',
        'darkred', 'darkblue', 'darkgreen', 'cadetblue',
         'pink', 'lightblue', 'lightgreen', 'gray',
        'black', 'lightgray', 'gold', 'magenta', 'cyan',
        'brown', 'lime', 'teal', 'navy', 'olive'
    ]
    # 3. Création de la carte de base
    # On centre la carte sur la France
    print("Creation de la carte ...")
    carte_france = folium.Map(location=[46.5, 2.5], zoom_start=6, tiles='OpenStreetMap')
    print("Ajout des marqueurs ...")
    # 4. Ajout des points (marqueurs) sur la carte
    for index, row in data_to_plot.iterrows():
        # On récupère le numéro du cluster
        num_cluster = int(row['cluster_kmeans'])

        # On choisit la couleur associée dans la palette
        couleur_point = grandes_couleurs[num_cluster % len(grandes_couleurs)]

        # On crée le marqueur
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=2,
            popup=f"Station: {row['nom_station']}<br>Cluster K-Means: {num_cluster}",
            color=couleur_point,
            fill=True,
            fill_color=couleur_point,
            fill_opacity=0.7
        ).add_to(carte_france)

    # 5. Sauvegarde de la carte dans un fichier HTML
    carte_france.save(f'carte/carte_clusters{nb_clusters}.html')

    print("La carte interactive a été générée : carte_clusters.html")


def analyse_globale_kmeans(coords,  dossier_sauvegarde,k_max=15):
    """
    Exécute les 4 métriques de K-Means, calcule un score global,
    et affiche une planche de 4 graphiques.
    """
    print(f"Lancement de l'analyse globale de k=2 à k={k_max}...")

    valeurs_k = list(range(2, k_max + 1))
    inerties = []
    silhouettes = []
    calinskis = []
    davies_vals = []

    # 1. Calcul de toutes les métriques pour chaque K
    taille_echantillon = min(2000, len(coords))  # Optimisation pour la Silhouette

    for k in valeurs_k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(coords)
        labels = kmeans.labels_

        inerties.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(coords, labels, sample_size=taille_echantillon, random_state=42))
        calinskis.append(calinski_harabasz_score(coords, labels))
        davies_vals.append(davies_bouldin_score(coords, labels))

    # 2. Identification des meilleurs scores individuels (pour les points rouges)
    meilleur_k_sil = valeurs_k[np.argmax(silhouettes)]
    meilleur_k_cal = valeurs_k[np.argmax(calinskis)]
    meilleur_k_dav = valeurs_k[np.argmin(davies_vals)]

    # 3. CALCUL DU SCORE COMPOSITE GLOBAL (Le "Juge de Paix")
    # Fonction locale pour ramener une liste de valeurs entre 0 (pire) et 1 (meilleur)
    def normaliser(tableau, inverser=False):
        t_min, t_max = np.min(tableau), np.max(tableau)
        if t_max == t_min: return np.zeros(len(tableau))
        norm = (tableau - t_min) / (t_max - t_min)
        return 1 - norm if inverser else norm

    # On normalise les 3 scores (l'inertie n'est pas incluse car elle descend toujours mathématiquement)
    norm_sil = normaliser(silhouettes, inverser=False)
    norm_cal = normaliser(calinskis, inverser=False)
    norm_dav = normaliser(davies_vals, inverser=True)  # True car Davies cherche le minimum

    # Le score final est la somme des 3 notes (sur 3 points maximum)
    scores_globaux = norm_sil + norm_cal + norm_dav
    meilleur_k_global = valeurs_k[np.argmax(scores_globaux)]

    print(f"-> Analyse terminée. Le meilleur nombre de clusters global calculé est : k={meilleur_k_global}")

    # 4. CRÉATION DES 4 GRAPHIQUES
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Analyse K-Means - Meilleur K Global estimé : {meilleur_k_global}", fontsize=16, fontweight='bold',
                 color='#2c3e50')

    # Graphique 1 : Méthode du Coude (Inertie)
    axes[0, 0].plot(valeurs_k, inerties, marker='o', linestyle='-', color='#3498db')
    axes[0, 0].set_title("Méthode du Coude (Inertie - Plus bas = meilleur)")
    axes[0, 0].set_xlabel("Nombre de clusters (k)")
    # On met un marqueur spécifique pour montrer où le score global a tranché sur l'inertie
    idx_global = valeurs_k.index(meilleur_k_global)
    axes[0, 0].plot(meilleur_k_global, inerties[idx_global], marker='*', color='black', markersize=12,
                    label="Choix Global")
    axes[0, 0].legend()

    # Graphique 2 : Silhouette
    axes[0, 1].plot(valeurs_k, silhouettes, marker='o', linestyle='-', color='#2ecc71')
    axes[0, 1].plot(meilleur_k_sil, np.max(silhouettes), 'ro', markersize=10, label=f"Max local (k={meilleur_k_sil})")
    axes[0, 1].set_title("Score de Silhouette (Plus haut = meilleur)")
    axes[0, 1].legend()

    # Graphique 3 : Calinski-Harabasz
    axes[1, 0].plot(valeurs_k, calinskis, marker='o', linestyle='-', color='#f39c12')
    axes[1, 0].plot(meilleur_k_cal, np.max(calinskis), 'ro', markersize=10, label=f"Max local (k={meilleur_k_cal})")
    axes[1, 0].set_title("Indice de Calinski-Harabasz (Plus haut = meilleur)")
    axes[1, 0].legend()

    # Graphique 4 : Davies-Bouldin
    axes[1, 1].plot(valeurs_k, davies_vals, marker='o', linestyle='-', color='#9b59b6')
    axes[1, 1].plot(meilleur_k_dav, np.min(davies_vals), 'ro', markersize=10, label=f"Min local (k={meilleur_k_dav})")
    axes[1, 1].set_title("Indice de Davies-Bouldin (Plus bas = meilleur)")
    axes[1, 1].legend()

    # Finitions visuelles pour tous les graphiques
    for ax in axes.flatten():
        ax.set_xticks(valeurs_k)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()

    # Sauvegarde optionnelle
    os.makedirs(dossier_sauvegarde, exist_ok=True)
    chemin_sauvegarde = os.path.join(dossier_sauvegarde, "analyse_multicriteres_kmeans.png")
    plt.savefig(chemin_sauvegarde, dpi=150, bbox_inches="tight")
    plt.close()  # On ferme pour éviter l'affichage direct dans la console

    # On retourne le meilleur K calculé et le chemin de l'image pour l'afficher dans PyQt
    return meilleur_k_global, chemin_sauvegarde