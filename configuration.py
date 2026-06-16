import os

# --- CONFIGURATION GLOBALE ---
DOSSIER_CARTES = "carte"
DOSSIER_MODELES = "modeles"

DOSSIER_GRAPHE = "graphe"
DOSSIER_GRAPHE_B3 = "graphe/besoin3"
DOSSIER_GRAPHE_B4 = "graphe/besoin4"
COULEURS = [
        'red', 'blue', 'green', 'purple', 'orange',
        'darkred', 'darkblue', 'darkgreen', 'cadetblue',
         'pink', 'lightblue', 'lightgreen', 'gray',
         'lightgray', 'gold', 'magenta', 'cyan',
        'brown', 'lime', 'teal', 'navy', 'olive'
    ]
def creation_dossier():
    os.makedirs(DOSSIER_CARTES, exist_ok=True)
    os.makedirs(DOSSIER_MODELES, exist_ok=True)
    os.makedirs(DOSSIER_GRAPHE, exist_ok=True)
    os.makedirs(DOSSIER_GRAPHE_B3, exist_ok=True)
    os.makedirs(DOSSIER_GRAPHE_B4, exist_ok=True)
