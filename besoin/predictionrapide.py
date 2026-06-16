"""
==============================================================================
SCRIPT D'UTILISATION : PRÉDICTION RAPIDE
Projet IRVE - Besoin Client 3
==============================================================================
À exécuter dans le terminal.
"""

import pandas as pd
import joblib

def predire_implantation(caracteristiques_borne):
    """
    Charge le modèle pré-entraîné et retourne la prédiction pour une borne donnée.
    """
    try:
        modele = joblib.load("modele_prediction_implantation.pkl")
    except FileNotFoundError:
        return "Erreur : Le fichier 'modele_prediction_implantation.pkl' est introuvable. Lancez l'entraînement d'abord."

    # Conversion du dictionnaire d'entrée en DataFrame (attendu par le pipeline)
    df_input = pd.DataFrame([caracteristiques_borne])
    
    # Prédiction
    prediction = modele.predict(df_input)
    return prediction[0]

# ==============================================================================
# TEST DU SCRIPT
# ==============================================================================
if __name__ == "__main__":
    # Exemple d'une borne type (ex: borne ultra-rapide sur autoroute)
    borne_test = {
        "prise_type_ef": 0,
        "prise_type_2": 1,
        "prise_type_combo_ccs": 1,
        "prise_type_chademo": 0,
        "prise_type_autre": 0,
        "gratuit": 0,
        "paiement_acte": 1,
        "paiement_cb": 1,
        "paiement_autre": 0,
        "reservation": 0,
        "station_deux_roues": 0,
        "cable_t2_attache": 1,
        "condition_acces": "Accès libre",
        "accessibilite_pmr": "Accessible non réservé",
        "puissance_nominale": 150.0,
        "nbre_pdc": 4,
        "lon": 4.8320,  # Ex: Coordonnées vers Lyon
        "lat": 45.7640
    }

    print("=" * 50)
    print("TEST DE PRÉDICTION DU TYPE D'IMPLANTATION")
    print("=" * 50)
    resultat = predire_implantation(borne_test)
    print(f"-> L'IA prédit que cette borne se situe en : ** {resultat} **")