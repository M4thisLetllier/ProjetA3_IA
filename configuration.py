import os
import sys

# --- DÉTECTION DYNAMIQUE DE LA RACINE ---
if getattr(sys, 'frozen', False):
    # L'application est exécutée depuis le .exe compilé par PyInstaller
    RACINE_PROJET = os.path.dirname(sys.executable)
    Hawke = False
else:
    # L'application est exécutée en mode développement (Python classique)
    RACINE_PROJET = os.path.dirname(os.path.abspath(__file__))

# --- CONFIGURATION GLOBALE VIA CHEMINS ABSOLUS ---
DOSSIER_CARTES = os.path.join(RACINE_PROJET, "carte")
DOSSIER_MODELES = os.path.join(RACINE_PROJET, "modeles")
DOSSIER_GRAPHE = os.path.join(RACINE_PROJET, "graphe")
DOSSIER_GRAPHE_B3 = os.path.join(RACINE_PROJET, "graphe", "besoin3")
DOSSIER_GRAPHE_B4 = os.path.join(RACINE_PROJET, "graphe", "besoin4")
DATA_BORNE = os.path.join(RACINE_PROJET, "IRVE_clean_FINAL.csv")
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
