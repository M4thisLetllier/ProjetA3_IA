# geo_utils.py
import os
import json
from shapely.geometry import shape, Point
from shapely.ops import unary_union

# --- CONFIGURATION INTERNE ---
# On détermine le chemin du GeoJSON par rapport à ce fichier
CHEMIN_GEOJSON = os.path.join(os.path.dirname(__file__), "france.geojson")
FRANCE_GEOMETRY = None

def initialiser_geometrie():
    """Charge et fusionne le GeoJSON de la France une bonne fois pour toutes."""
    global FRANCE_GEOMETRY
    if FRANCE_GEOMETRY is not None:
        return  # Déjà chargé

    if not os.path.exists(CHEMIN_GEOJSON):
        print(f"Attention : Le fichier {CHEMIN_GEOJSON} est introuvable. "
              f"La vérification précise sera désactivée.")
        return

    try:
        with open(CHEMIN_GEOJSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        polygones = [shape(feature["geometry"]) for feature in data["features"]]
        FRANCE_GEOMETRY = unary_union(polygones)
    except Exception as e:
        print(f"Erreur lors de l'initialisation géométrique : {e}")
        FRANCE_GEOMETRY = None

# On force le chargement dès que le module est importé
initialiser_geometrie()


def est_en_france(lon, lat):
    """
    Vérifie si un point (longitude, latitude) est situé en France métropolitaine.
    Renvoie True si oui, False sinon.
    """
    # 1. Pré-filtrage rapide par boîte englobante (évite les calculs Shapely inutiles)
    if not (-5.0 <= lon <= 9.5) or not (41.3 <= lat <= 51.1):
        return False

    # 2. Vérification précise avec Shapely si le GeoJSON est disponible
    if FRANCE_GEOMETRY is not None:
        point = Point(lon, lat)
        return point.within(FRANCE_GEOMETRY)

    # Si le GeoJSON est manquant, on se contente de la boîte englobante
    return True