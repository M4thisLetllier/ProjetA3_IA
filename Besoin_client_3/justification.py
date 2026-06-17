"""
==============================================================================
JUSTIFICATION STATISTIQUE ET GRAPHIQUE DES VARIABLES
Projet IRVE - Besoin Client 3 : Prédiction du type d'implantation
==============================================================================
Ce script réalise :
  1. Sélection et exclusion des variables
  2. Tests Chi-deux pour les variables catégorielles/booléennes
  3. Tests de Kruskal-Wallis pour les variables numériques
  4. Visualisations : barres empilées 100% + boxplots
  5. Tableau récapitulatif des p-values
==============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import chi2_contingency, kruskal
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CHARGEMENT ET PRÉPARATION
# ==============================================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "../IRVE_clean_FINAL.csv")
df = pd.read_csv(csv_path, low_memory=False)

# Cible
TARGET = "implantation_station"

# On simplifie les labels longs pour les graphiques
label_map = {
    "Voirie"                                 : "Voirie",
    "Parking privé à usage public"           : "Parking public",
    "Parking public"                         : "Parking publ.",
    "Station dédiée à la recharge rapide"    : "Station rapide",
    "Parking privé réservé à la clientèle"   : "Parking privé",
}
df["implantation_court"] = df[TARGET].map(label_map).fillna(df[TARGET])

# Ordre d'affichage
ORDER = ["Voirie", "Parking publ.", "Parking public", "Station rapide", "Parking privé"]

# ==============================================================================
# 1. VARIABLES EXCLUES ET JUSTIFICATION
# ==============================================================================

print("=" * 70)
print("VARIABLES EXCLUES DE L'ANALYSE")
print("=" * 70)

exclusions = {
    "nom_amenageur"        : "Identifiant texte libre, trop de modalités → bruit",
    "contact_amenageur"    : "Contact administratif, non lié à la physique de la borne",
    "nom_operateur"        : "Trop de modalités (>200), risque de sur-apprentissage",
    "contact_operateur"    : "Coordonnée administrative, non prédictive",
    "telephone_operateur"  : "Idem",
    "nom_enseigne"         : "Doublon de nom_operateur",
    "id_station_itinerance": "Identifiant unique, aucune info prédictive",
    "id_pdc_itinerance"    : "Identifiant unique",
    "nom_station"          : "Texte libre, très variable",
    "adresse_station"      : "Trop spécifique, risque fuite de données",
    "code_insee_commune"   : "Remplacé par lon/lat (plus précis géographiquement)",
    "consolidated_commune" : "Trop de modalités, remplacé par lon/lat",
    "consolidated_code_postal": "Idem",
    "horaires"             : "Format texte OSM, nécessiterait un parsing complexe",
    "raccordement"         : "52% de valeurs manquantes",
    "date_mise_en_service" : "Variable temporelle à traiter séparément si besoin",
    "date_maj"             : "Métadonnée de mise à jour, non prédictive",
    "tarif_kwh_clean"      : "76% de valeurs manquantes",
    "restriction_gabarit"  : "Très faible lien avec le type d'implantation",
}

for var, raison in exclusions.items():
    print(f"  ✗  {var:<35} → {raison}")

# ==============================================================================
# 2. VARIABLES RETENUES
# ==============================================================================

print("\n" + "=" * 70)
print("VARIABLES RETENUES")
print("=" * 70)

# Variables booléennes (0/1)
BOOL_COLS = [
    "prise_type_ef", "prise_type_2", "prise_type_combo_ccs",
    "prise_type_chademo", "prise_type_autre",
    "gratuit", "paiement_acte", "paiement_cb", "paiement_autre",
    "reservation", "station_deux_roues", "cable_t2_attache",
]

# Variables catégorielles texte
CAT_COLS = ["condition_acces", "accessibilite_pmr"]

# Variables numériques
NUM_COLS = ["puissance_nominale", "nbre_pdc"]

# Variables géographiques
GEO_COLS = ["lon", "lat"]

print("\n  Variables booléennes (0/1) :", BOOL_COLS)
print("\n  Variables catégorielles    :", CAT_COLS)
print("\n  Variables numériques       :", NUM_COLS)
print("\n  Variables géographiques    :", GEO_COLS)

# ==============================================================================
# 3. TESTS STATISTIQUES - CHI-DEUX (variables catégorielles + booléennes)
# ==============================================================================

print("\n" + "=" * 70)
print("TEST CHI-DEUX (H0 : indépendance avec implantation_station)")
print("=" * 70)

resultats_chi2 = []

for col in BOOL_COLS + CAT_COLS:
    contingence = pd.crosstab(df[col], df[TARGET])
    chi2, p, dof, _ = chi2_contingency(contingence)
    significatif = "✓ SIGNIFICATIF" if p < 0.05 else "✗ non significatif"
    resultats_chi2.append({
        "Variable"      : col,
        "Chi2"          : round(chi2, 2),
        "p-value"       : f"{p:.2e}",
        "p < 0.05"      : p < 0.05,
        "Conclusion"    : significatif,
    })
    print(f"  {col:<35} Chi²={chi2:>10.1f}  p={p:.2e}  {significatif}")

# ==============================================================================
# 4. TESTS STATISTIQUES - KRUSKAL-WALLIS (variables numériques)
# ==============================================================================

print("\n" + "=" * 70)
print("TEST DE KRUSKAL-WALLIS (H0 : distributions identiques entre groupes)")
print("=" * 70)

resultats_kw = []

for col in NUM_COLS + GEO_COLS:
    groupes = [df[df[TARGET] == impl][col].dropna().values
               for impl in df[TARGET].unique()]
    groupes = [g for g in groupes if len(g) > 0]
    stat, p = kruskal(*groupes)
    significatif = "✓ SIGNIFICATIF" if p < 0.05 else "✗ non significatif"
    resultats_kw.append({
        "Variable"   : col,
        "Stat KW"    : round(stat, 2),
        "p-value"    : f"{p:.2e}",
        "p < 0.05"   : p < 0.05,
        "Conclusion" : significatif,
    })
    print(f"  {col:<35} KW={stat:>10.1f}  p={p:.2e}  {significatif}")

# ==============================================================================
# 5. TABLEAU RÉCAPITULATIF
# ==============================================================================

df_chi2 = pd.DataFrame(resultats_chi2)[["Variable","Chi2","p-value","Conclusion"]]
df_chi2["Test"] = "Chi-deux"
df_kw   = pd.DataFrame(resultats_kw)[["Variable","Stat KW","p-value","Conclusion"]]
df_kw["Test"] = "Kruskal-Wallis"
df_kw = df_kw.rename(columns={"Stat KW": "Chi2"})

recap = pd.concat([df_chi2, df_kw], ignore_index=True)
print("\n\nTABLEAU RÉCAPITULATIF COMPLET")
print(recap[["Test","Variable","Chi2","p-value","Conclusion"]].to_string(index=False))

# ==============================================================================
# 6. VISUALISATIONS
# ==============================================================================

palette = {
    "Voirie"         : "#E8593C",
    "Parking publ."  : "#3B6D11",
    "Parking public" : "#185FA5",
    "Station rapide" : "#F5A623",
    "Parking privé"  : "#7B4FBF",
}

# ── Figure 1 : Barres empilées 100% pour les booléens clés ──────────────────

bool_interessants = [
    "prise_type_combo_ccs", "prise_type_chademo", "prise_type_2",
    "cable_t2_attache", "gratuit", "paiement_cb", "reservation", "station_deux_roues"
]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle(
    "Répartition du type d'implantation selon les variables booléennes\n(barres empilées à 100%)",
    fontsize=14, fontweight="bold", y=1.01
)

for ax, col in zip(axes.flatten(), bool_interessants):
    ct = pd.crosstab(df[col], df["implantation_court"], normalize="index") * 100
    ct = ct.reindex(columns=[c for c in ORDER if c in ct.columns])
    colors = [palette.get(c, "#AAAAAA") for c in ct.columns]
    ct.plot(kind="bar", stacked=True, ax=ax, color=colors, legend=False, width=0.6)
    ax.set_title(col.replace("_", " "), fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticklabels(["Non (0)", "Oui (1)"], rotation=0, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, 100)
    ax.spines[["top","right"]].set_visible(False)

# Légende commune
handles = [plt.Rectangle((0,0),1,1, color=palette.get(l,"#AAA")) for l in ORDER if l in palette]
labels  = [l for l in ORDER if l in palette]
fig.legend(handles, labels, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.04), fontsize=9, title="Type d'implantation")

plt.tight_layout()
plt.savefig("graphe_barres_empilees.png", dpi=150, bbox_inches="tight")
print("\n✓  Graphique sauvegardé : graphe_barres_empilees.png")
plt.close()

# ── Figure 2 : Boxplots pour puissance_nominale et nbre_pdc ─────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Distribution des variables numériques par type d'implantation\n(boxplots)",
    fontsize=13, fontweight="bold"
)

for ax, col, titre in zip(axes, ["puissance_nominale", "nbre_pdc"],
                           ["Puissance nominale (kW)", "Nombre de points de charge"]):
    data_plot = df[["implantation_court", col]].dropna()
    order_box = [o for o in ORDER if o in data_plot["implantation_court"].unique()]
    colors_box = [palette.get(o, "#AAAAAA") for o in order_box]

    bp = ax.boxplot(
    [data_plot[data_plot["implantation_court"] == o][col].values for o in order_box],
    tick_labels=order_box,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    flierprops=dict(marker="o", markersize=2, alpha=0.3),
    widths=0.5
)
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title(titre, fontsize=11, fontweight="bold")
    ax.set_xlabel("Type d'implantation", fontsize=9)
    ax.set_ylabel(titre, fontsize=9)
    ax.set_xticklabels(order_box, rotation=20, ha="right", fontsize=8)
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("graphe_boxplots.png", dpi=150, bbox_inches="tight")
print("✓  Graphique sauvegardé : graphe_boxplots.png")
plt.close()

# ── Figure 3 : Barres empilées 100% pour les variables catégorielles ─────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Répartition du type d'implantation selon les variables catégorielles",
    fontsize=13, fontweight="bold"
)

for ax, col in zip(axes, CAT_COLS):
    ct = pd.crosstab(df[col], df["implantation_court"], normalize="index") * 100
    ct = ct.reindex(columns=[c for c in ORDER if c in ct.columns])
    colors = [palette.get(c, "#AAAAAA") for c in ct.columns]
    ct.plot(kind="bar", stacked=True, ax=ax, color=colors, legend=False, width=0.6)
    ax.set_title(col.replace("_", " "), fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, 100)
    ax.spines[["top","right"]].set_visible(False)

handles = [plt.Rectangle((0,0),1,1, color=palette.get(l,"#AAA")) for l in ORDER if l in palette]
labels  = [l for l in ORDER if l in palette]
fig.legend(handles, labels, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.08), fontsize=9, title="Type d'implantation")

plt.tight_layout()
plt.savefig("graphe_categoriel.png", dpi=150, bbox_inches="tight")
print("✓  Graphique sauvegardé : graphe_categoriel.png")
plt.close()

# ── Figure 4 : Carte de densité GPS par type d'implantation ─────────────────

fig, axes = plt.subplots(1, len(label_map), figsize=(20, 4))
fig.suptitle(
    "Distribution géographique (lon/lat) par type d'implantation\n→ justifie l'intérêt des variables lon et lat",
    fontsize=12, fontweight="bold"
)

for ax, (impl_long, impl_court) in zip(axes, label_map.items()):
    subset = df[df[TARGET] == impl_long]
    color  = palette.get(impl_court, "#AAAAAA")
    ax.scatter(subset["lon"], subset["lat"],
               alpha=0.05, s=0.5, color=color)
    ax.set_title(impl_court, fontsize=9, fontweight="bold", color=color)
    ax.set_xlim(-5.5, 10)
    ax.set_ylim(41, 51.5)
    ax.set_xlabel("Longitude", fontsize=7)
    ax.set_ylabel("Latitude", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("graphe_geo.png", dpi=150, bbox_inches="tight")
print("✓  Graphique sauvegardé : graphe_geo.png")
plt.close()

# ==============================================================================
# 7. RÉSUMÉ FINAL
# ==============================================================================

print("\n" + "=" * 70)
print("RÉSUMÉ - VARIABLES RETENUES POUR LE MODÈLE")
print("=" * 70)

tous_tests = resultats_chi2 + resultats_kw
variables_retenues = [r["Variable"] for r in tous_tests if r["p < 0.05"]]
variables_rejetees = [r["Variable"] for r in tous_tests if not r["p < 0.05"]]

print(f"\n  ✓  Variables statistiquement significatives ({len(variables_retenues)}) :")
for v in variables_retenues:
    print(f"       - {v}")

if variables_rejetees:
    print(f"\n  ✗  Variables non significatives ({len(variables_rejetees)}) :")
    for v in variables_rejetees:
        print(f"       - {v}")

print(f"\n  +  Variables géographiques (lon, lat) : retenues sur base du graphe de")
print(f"     distribution géographique différenciée par type d'implantation")

print("\n  Variables finales recommandées pour le modèle :")
features_finales = variables_retenues + ["lon", "lat"]
print(f"     {features_finales}")

print("\n✓  Script terminé. 4 graphiques exportés dans le répertoire courant.")
