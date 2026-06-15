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


def calinski(coords) :
    meilleur_score = -1
    meilleur_k = 2
    k_max_test = 187


    for k in range(2, k_max_test + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(coords)

        # Le calcul magique et ultra-rapide
        score = calinski_harabasz_score(coords, labels)

        if score > meilleur_score:
            meilleur_score = score
            meilleur_k = k

    return meilleur_k,meilleur_score


def calinskisaut(coords):
    score = -1
    meilleur_score = -1
    meilleur_k = 2
    k_max_test = 200
    saut= 0
    meilleur_saut=0

    for k in range(2, k_max_test + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(coords)

        # Le calcul magique et ultra-rapide
        temp = calinski_harabasz_score(coords, labels)
        saut = score - temp
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
