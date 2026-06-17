import os
import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
import warnings

# Désactivation des messages d'avertissement inutiles dans le terminal
warnings.filterwarnings('ignore')

def generation_carte():
    print("=====================================================")
    print("  Démarrage du script - Besoin Client 1 (Cartographie)")
    print("=====================================================")

    file_path = 'IRVE_clean_FINAL.csv'
    
    # Sécurité : Vérification de la présence de la base de données
    if not os.path.exists(file_path):
        print(f"Erreur fatale : Le fichier '{file_path}' est introuvable.")
        print("Veuillez le placer dans le même dossier que ce script.")
        return

    # 1. Chargement et Nettoyage
    print("\n 1/4 - Chargement et nettoyage des données géographiques...")
    colonnes_interet = ['nom_station', 'implantation_station', 'lat', 'lon']
    df = pd.read_csv(file_path, usecols=colonnes_interet)

    # Nettoyage strict : suppression des NaN et des points hors France métropolitaine
    df = df.dropna(subset=['lat', 'lon'])
    df = df[(df['lat'] >= 41) & (df['lat'] <= 51.5) & (df['lon'] >= -5) & (df['lon'] <= 10)]
    print(f"Données prêtes : {df.shape[0]} bornes validées.")

    # Attribution de la charte graphique
    COULEURS = {
        'Voirie': '#2196F3',
        'Parking privé à usage public': '#4CAF50',
        'Parking public': '#FF9800',
        'Station dédiée à la recharge rapide': '#F44336',
        'Parking privé réservé à la clientèle': '#9C27B0',
    }
    df['couleur_hex'] = df['implantation_station'].map(COULEURS).fillna('#757575')
    
    # S'assurer que le dossier de destination existe
    os.makedirs('carte', exist_ok=True)

    # 2. Carte des Clusters
    print("\n2/4 - Génération de la carte des implantations (MarkerCluster)...")
    carte_clustering = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles='CartoDB Positron')
    cluster_moteur = MarkerCluster(disableClusteringAtZoom=15).add_to(carte_clustering)

    for lat, lon, imp, hex_c in zip(df['lat'], df['lon'], df['implantation_station'], df['couleur_hex']):
        folium.CircleMarker(
            location=[lat, lon], radius=4, color=hex_c, fill=True, fill_color=hex_c, fill_opacity=0.7,
            popup=folium.Popup(f"<b>Type :</b> {imp}", max_width=300)
        ).add_to(cluster_moteur)

    carte_clustering.save('carte/carte_implantations.html')
    print("Fichier 'carte_implantations.html' sauvegardé.")

    # 3. Carte de Chaleur (Recharge Rapide)
    print("\n3/4 - Génération de la carte de chaleur (Recharge Rapide)...")
    df_fast = df[df['implantation_station'] == 'Station dédiée à la recharge rapide']
    
    carte_thermique_rapide = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles='CartoDB Dark_Matter')
    points_rapides = df_fast[['lat', 'lon']].values.tolist()

    HeatMap(
        points_rapides, radius=9, blur=14, max_zoom=11,
        gradient={0.2: 'navy', 0.5: 'cyan', 0.8: 'yellow', 1.0: 'crimson'}
    ).add_to(carte_thermique_rapide)

    carte_thermique_rapide.save('carte/carte_chaleur_rapide.html')
    print("Fichier 'carte_chaleur_rapide.html' sauvegardé.")

    # 4. Carte de Chaleur (Globale)
    print("\n 4/4 - Génération de la carte de chaleur (Réseau Global)...")
    carte_thermique_globale = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles='CartoDB Dark_Matter')
    points_globaux = df[['lat', 'lon']].values.tolist()

    # On utilise un rayon un peu plus petit et plus flou car il y a énormément de points (70 000+)
    HeatMap(
        points_globaux, radius=7, blur=10, max_zoom=11,
        gradient={0.2: 'blue', 0.4: 'lime', 0.7: 'yellow', 1.0: 'red'}
    ).add_to(carte_thermique_globale)

    carte_thermique_globale.save('carte/carte_chaleur_globale.html')
    print("Fichier 'carte_chaleur_globale.html' sauvegardé.")

    print("\n=====================================================")
    print("  OPÉRATION TERMINÉE ! Les 3 livrables HTML sont prêts.")
    print("=====================================================")

if __name__ == "__main__":
    generation_carte()