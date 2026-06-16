"""
==============================================================================
APPRENTISSAGE SUPERVISÉ : PRÉDICTION DE LA PUISSANCE NOMINALE
Projet IRVE - Besoin Client 4
==============================================================================
Ce script réalise :
  1. Le chargement des données via un chemin dynamique sécurisé
  2. La préparation des variables validées par l'analyse statistique
  3. L'optimisation des hyperparamètres d'un RandomForestRegressor (GridSearchCV)
  4. L'évaluation du modèle via des métriques de régression (MAE, RMSE, R²)
  5. La visualisation des performances (Prédit vs Réel)
  6. La sauvegarde du modèle final au format .pkl
==============================================================================
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 1. CHARGEMENT ET GESTION SÉCURISÉE DES CHEMINS
# ==============================================================================
print("Chargement des données pour le Besoin 4...")

# Détection dynamique des dossiers pour éviter le piège des slashes Windows
base_dir = os.path.dirname(os.path.abspath(__file__))  # Dossier 'besoin4'
projet_dir = os.path.dirname(base_dir)                # Dossier racine 'ProjetA3_IA'
csv_path = os.path.join(projet_dir, "Besoin3,4", "IRVE_clean_FINAL.csv")

df = pd.read_csv(csv_path, low_memory=False)

# Définition de la cible (Variable numérique continue)
TARGET = "puissance_nominale"

# Nettoyage des lignes sans cible ou sans coordonnées (essentiel pour les modèles)
df = df.dropna(subset=[TARGET, "lon", "lat"])

# ==============================================================================
# 2. PRÉPARATION DES ARRAYS (FEATURES / TARGET)
# ==============================================================================

# Listes exactes validées par ton script de justification
BOOL_COLS = [
    "prise_type_ef", "prise_type_2", "prise_type_combo_ccs", 
    "prise_type_chademo", "prise_type_autre", "gratuit", "paiement_acte", 
    "paiement_cb", "paiement_autre", "reservation", "station_deux_roues", 
    "cable_t2_attache"
]

CAT_COLS = ["condition_acces", "accessibilite_pmr", "implantation_station"]

NUM_COLS = ["nbre_pdc", "lon", "lat"]

X = df[BOOL_COLS + CAT_COLS + NUM_COLS]
y = df[TARGET]

# Séparation Entraînement / Test (80% / 20%)
# Note : Pas de paramètre 'stratify' ici car la cible est continue (régression)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================================================================
# 3. PIPELINE DE PRÉTRAITEMENT (PREPROCESSING)
# ==============================================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUM_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS),
        ('bool', 'passthrough', BOOL_COLS)
    ]
)

# ==============================================================================
# 4. CONFIGURATION DU MODÈLE ET GRID SEARCH CV
# ==============================================================================
# Utilisation de RandomForestRegressor pour la régression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
])

# Grille d'hyperparamètres (similaire à ton besoin 3 mais adaptée à la régression)
param_grid = {
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': [10, 20, None]
}

print("Recherche des meilleurs hyperparamètres (GridSearchCV)...")
# On utilise le R² (scoring='r2') pour évaluer la qualité de la régression
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\nMeilleurs hyperparamètres : {grid_search.best_params_}")

# ==============================================================================
# 5. ÉVALUATION ET MÉTRIQUES DE RÉGRESSION
# ==============================================================================
print("\nÉvaluation sur le jeu de test...")
y_pred = best_model.predict(X_test)

# Calcul des indicateurs clés
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("RAPPORT DE PERFORMANCE DE LA RÉGRESSION")
print("="*50)
print(f"Erreur Absolue Moyenne (MAE) : {mae:.2f} kW")
print(f"Racine de l'Erreur Quadratique Moyenne (RMSE) : {rmse:.2f} kW")
print(f"Coefficient de Détermination (R²) : {r2:.4f}")
print("="*50)

# ==============================================================================
# 6. VISUALISATIONS DIAGNOSTIQUE
# ==============================================================================
plt.figure(figsize=(9, 7))
# Tracé des points Réel vs Prédit
plt.scatter(y_test, y_pred, alpha=0.3, color='#185FA5', s=12)

# Ajout de la ligne idéale Y = X (Prédiction parfaite)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, color='red', linestyle='--', lw=2, label="Prédiction Parfaite (Y = X)")

plt.title('Validation du Modèle B4 : Puissance Prédite vs Puissance Réelle', fontsize=12, fontweight='bold')
plt.xlabel('Valeurs Réelles (kW)', fontsize=10)
plt.ylabel('Valeurs Prédites (kW)', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()

# Sauvegarde du graphique dans le dossier 'besoin4'
plt.savefig("graphe_regression_prediction_vs_realite.png", dpi=150)
print("✓ Graphique 'Prédiction vs Réalité' sauvegardé sous 'graphe_regression_prediction_vs_realite.png'")

# ==============================================================================
# 7. SAUVEGARDE DU MODÈLE FINAL
# ==============================================================================
joblib.dump(best_model, "modele_prediction_puissance.pkl")
print("✓ Modèle de régression sauvegardé sous 'modele_prediction_puissance.pkl'")