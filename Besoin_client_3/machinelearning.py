"""
==============================================================================
APPRENTISSAGE SUPERVISÉ : PRÉDICTION DU TYPE D'IMPLANTATION
Projet IRVE - Besoin Client 3
==============================================================================
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==============================================================================
print("Chargement des données...")
df = pd.read_csv("../IRVE_clean_FINAL.csv", low_memory=False)

# Définition de la cible et des variables (issues de ton analyse statistique)
TARGET = "implantation_station"

# Remplacement des valeurs manquantes critiques pour éviter les plantages
df = df.dropna(subset=[TARGET, "lon", "lat"])

# Séparation des features
BOOL_COLS = [
    "prise_type_ef", "prise_type_2", "prise_type_combo_ccs",
    "prise_type_chademo", "prise_type_autre", "gratuit", "paiement_acte", 
    "paiement_cb", "paiement_autre", "reservation", "station_deux_roues", 
    "cable_t2_attache"
]
CAT_COLS = ["condition_acces", "accessibilite_pmr"]
NUM_COLS = ["puissance_nominale", "nbre_pdc", "lon", "lat"]

X = df[BOOL_COLS + CAT_COLS + NUM_COLS]
y = df[TARGET]

# Split Train / Test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==============================================================================
# 2. PIPELINE DE PRÉTRAITEMENT (Preprocessing)
# ==============================================================================
# Normalisation des numériques (Z-score) et Encodage One-Hot des textuelles
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUM_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS),
        ('bool', 'passthrough', BOOL_COLS) # Les booléens (0/1) passent directement
    ]
)

# ==============================================================================
# 3. MODÈLE ET GRID SEARCH CV
# ==============================================================================
# Choix du modèle : Forêt Aléatoire (Random Forest)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Grille d'hyperparamètres (simplifiée pour des temps de calcul raisonnables)
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [10, 20, None]
}

print("Recherche des meilleurs hyperparamètres (GridSearchCV)...")
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\nMeilleurs hyperparamètres : {grid_search.best_params_}")

# ==============================================================================
# 4. ÉVALUATION ET MÉTRIQUES
# ==============================================================================
print("\nÉvaluation sur le jeu de test...")
y_pred = best_model.predict(X_test)

# Rapport de classification
print("\n" + "="*50)
print("RAPPORT DE CLASSIFICATION")
print("="*50)
print(classification_report(y_test, y_pred))

# Matrice de confusion graphique
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title('Matrice de Confusion : Prédiction des Implantations')
plt.ylabel('Réalité')
plt.xlabel('Prédiction')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("matrice_confusion_implantation.png")
print("✓ Matrice de confusion sauvegardée sous 'matrice_confusion_implantation.png'")

# ==============================================================================
# 5. SAUVEGARDE DU MODÈLE POUR LE SCRIPT FINAL / WEB
# ==============================================================================
joblib.dump(best_model, "modele_prediction_implantation.pkl")
print("\n✓ Modèle (incluant le prétraitement) sauvegardé sous 'modele_prediction_implantation.pkl'")