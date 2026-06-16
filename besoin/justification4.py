"""
==============================================================================
JUSTIFICATION STATISTIQUE ET GRAPHIQUE DES VARIABLES
Projet IRVE - Besoin Client 4 : Prédiction de la puissance nominale
==============================================================================
Ce script réalise :
  1. Sélection et exclusion des variables
  2. Tests de Kruskal-Wallis pour les variables catégorielles/booléennes (vs cible continue)
  3. Tests de corrélation de Spearman pour les variables numériques (vs cible continue)
  4. Visualisations : Boxplots par type de prise/implantation + Scatter plot géo
  5. Tableau récapitulatif des p-values
==============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, spearmanr
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CHARGEMENT ET PRÉPARATION
# ==============================================================================



# Dossier actuel du script (.../ProjetA3_IA/besoin4)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Dossier parent du projet (.../ProjetA3_IA)
projet_dir = os.path.dirname(base_dir)

# On construit le chemin proprement sans aucun antislash manuel
csv_path = os.path.join(projet_dir, "Besoin3,4", "IRVE_clean_FINAL.csv")

print("Chargement des données...")
df = pd.read_csv(csv_path, low_memory=False)

# Cible du Besoin 4
TARGET = "puissance_nominale"

# Nettoyage de la cible
df = df.dropna(subset=[TARGET])

# ==============================================================================
# 1. VARIABLES EXCLUES ET JUSTIFICATION
# ==============================================================================

print("=" * 70)
print("VARIABLES EXCLUES DE L'ANALYSE (BESOIN 4)")
print("=" * 70)

exclusions = {
    "nom_amenageur"        : "Identifiant texte libre, trop de modalités → bruit",
    "contact_amenageur"    : "Contact administratif, non lié à la physique de la borne",
    "nom_operateur"        : "Trop de modalités (>200), risque de sur-apprentissage",
    "contact_operateur"    : "Coordonnée administrative, non prédictive",
    "id_station_itinerance": "Identifiant unique, aucune info prédictive",
    "id_pdc_itinerance"    : "Identifiant unique",
    "nom_station"          : "Texte libre, très variable",
    "adresse_station"      : "Trop spécifique",
    "code_insee_commune"   : "Remplacé par lon/lat (plus précis géographiquement)",
    "horaires"             : "Format texte OSM, nécessiterait un parsing complexe",
    "raccordement"         : "52% de valeurs manquantes",
    "tarif_kwh_clean"      : "76% de valeurs manquantes",
}

for var, raison in exclusions.items():
    print(f"  ✗  {var:<35} → {raison}")

# ==============================================================================
# 2. VARIABLES RETENUES
# ==============================================================================

print("\n" + "=" * 70)
print("VARIABLES RETENUES")
print("=" * 70)

BOOL_COLS = [
    "prise_type_ef", "prise_type_2", "prise_type_combo_ccs",
    "prise_type_chademo", "prise_type_autre",
    "gratuit", "paiement_acte", "paiement_cb", "paiement_autre",
    "reservation", "station_deux_roues", "cable_t2_attache",
]

# Note : 'implantation_station' glisse ici comme variable explicative
CAT_COLS = ["condition_acces", "accessibilite_pmr", "implantation_station"]
NUM_COLS = ["nbre_pdc"]
GEO_COLS = ["lon", "lat"]

print("\n  Variables booléennes (0/1) :", BOOL_COLS)
print("\n  Variables catégorielles    :", CAT_COLS)
print("\n  Variables numériques       :", NUM_COLS)
print("\n  Variables géographiques    :", GEO_COLS)

# ==============================================================================
# 3. TESTS STATISTIQUES - KRUSKAL-WALLIS (Qualitatif vs Continu)
# ==============================================================================

print("\n" + "=" * 70)
print("TEST DE KRUSKAL-WALLIS (H0 : Distributions de puissance identiques)")
print("=" * 70)

resultats_qual = []

for col in BOOL_COLS + CAT_COLS:
    df_sub = df[[col, TARGET]].dropna()
    groupes = [df_sub[df_sub[col] == val][TARGET].values for val in df_sub[col].unique()]
    groupes = [g for g in groupes if len(g) > 0]
    
    if len(groupes) > 1:
        stat, p = kruskal(*groupes)
        significatif = "✓ SIGNIFICATIF" if p < 0.05 else "✗ non significatif"
    else:
        stat, p, significatif = 0, 1.0, "✗ non significatif"
        
    resultats_qual.append({
        "Variable"   : col,
        "Stat"       : round(stat, 2),
        "p-value"    : f"{p:.2e}",
        "p < 0.05"   : p < 0.05,
        "Conclusion" : significatif,
        "Test"       : "Kruskal-Wallis"
    })
    print(f"  {col:<35} KW={stat:>10.1f}  p={p:.2e}  {significatif}")

# ==============================================================================
# 4. TESTS STATISTIQUES - CORRÉLATION DE SPEARMAN (Continu vs Continu)
# ==============================================================================

print("\n" + "=" * 70)
print("TEST DE SPEARMAN (H0 : Pas de dépendance monotone avec la puissance)")
print("=" * 70)

resultats_quant = []

for col in NUM_COLS + GEO_COLS:
    df_sub = df[[col, TARGET]].dropna()
    stat, p = spearmanr(df_sub[col], df_sub[TARGET])
    significatif = "✓ SIGNIFICATIF" if p < 0.05 else "✗ non significatif"
    
    resultats_quant.append({
        "Variable"   : col,
        "Stat"       : round(stat, 4),
        "p-value"    : f"{p:.2e}",
        "p < 0.05"   : p < 0.05,
        "Conclusion" : significatif,
        "Test"       : "Spearman Corr"
    })
    print(f"  {col:<35} Rho={stat:>10.4f}  p={p:.2e}  {significatif}")

# ==============================================================================
# 5. TABLEAU RÉCAPITULATIF COMPLET
# ==============================================================================

recap = pd.concat([pd.DataFrame(resultats_qual), pd.DataFrame(resultats_quant)], ignore_index=True)
print("\n\nTABLEAU RÉCAPITULATIF COMPLET - BESOIN 4")
print(recap[["Test","Variable","Stat","p-value","Conclusion"]].to_string(index=False))

# ==============================================================================
# 6. VISUALISATIONS
# ==============================================================================

# ── Figure 1 : Boxplots de la puissance selon le type de prise (Clés) ───────
bool_interessants = ["prise_type_combo_ccs", "prise_type_chademo", "prise_type_2", "prise_type_ef"]
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Impact des types de connecteurs sur la Puissance Nominale (kW)", fontsize=13, fontweight="bold")

for ax, col in zip(axes, bool_interessants):
    sns.boxplot(data=df, x=col, y=TARGET, ax=ax, palette="Set2", showfliers=False)
    ax.set_title(col.replace("_", " "), fontsize=10, fontweight="bold")
    ax.set_xlabel("Présence (0=Non, 1=Oui)")
    ax.set_ylabel("Puissance Nominale (kW)" if ax == axes[0] else "")
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("b4_graphe_boxplots_prises.png", dpi=150)
plt.close()

# ── Figure 2 : Boxplot selon l'implantation de la station ──────────────────
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="implantation_station", y=TARGET, palette="viridis", showfliers=False)
plt.title("Distribution de la Puissance Nominale par Type d'Implantation", fontsize=12, fontweight="bold")
plt.xticks(rotation=25, ha="right", fontsize=9)
plt.xlabel("Implantation")
plt.ylabel("Puissance Nominale (kW)")
plt.gca().spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("b4_graphe_boxplot_implantation.png", dpi=150)
plt.close()

# ── Figure 3 : Distribution géographique de la puissance (Carte) ────────────
plt.figure(figsize=(10, 8))
sc = plt.scatter(df["lon"], df["lat"], c=df[TARGET], cmap="turbo", alpha=0.6, s=1, vmin=0, vmax=150)
plt.colorbar(sc, label="Puissance Nominale (kW) - Max tronqué à 150 pour lisibilité")
plt.title("Cartographie de l'intensité de la Puissance des Bornes en France", fontsize=12, fontweight="bold")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.xlim(-5.5, 10)
plt.ylim(41, 51.5)
plt.gca().spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("b4_graphe_geo_puissance.png", dpi=150)
plt.close()

print("\n✓ Graphiques du Besoin 4 sauvegardés avec succès.")