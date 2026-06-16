import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QMessageBox, QFrame, QDialog, QTextEdit)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

# =============================================================================
# DICTIONNAIRE DES EXPLICATIONS MÉTIER
# =============================================================================
# Ce dictionnaire relie chaque nom de fichier à son titre et son explication.
EXPLICATIONS = {
    "graphe_barres_empilees.png": {
        "titre": "Impact des Équipements et Services (Booléens)",
        "texte": (
            "<b>Que montre ce graphique ?</b><br>"
            "Il présente la répartition à 100% du type d'implantation en fonction de la présence (1) ou l'absence (0) "
            "de certaines caractéristiques (gratuité, type de prise, etc.).<br><br>"
            "<b>Analyse :</b><br>"
            "On remarque que certaines caractéristiques influencent fortement le lieu d'implantation. "
            "Par exemple, la gratuité est très rare pour les stations de recharge rapide, tandis que "
            "la présence d'un câble attaché (T2) est un fort indicateur d'un type d'implantation spécifique. "
            "<i>Ces variables sont donc d'excellents prédicteurs pour notre modèle d'Intelligence Artificielle.</i>"
        )
    },
    "graphe_boxplots.png": {
        "titre": "Distribution des Puissances et Points de Charge",
        "texte": (
            "<b>Que montre ce graphique ?</b><br>"
            "Les boîtes à moustaches (boxplots) montrent la répartition des valeurs numériques "
            "(Puissance en kW et Nombre de prises) selon le type de station.<br><br>"
            "<b>Analyse :</b><br>"
            "On observe une fracture très nette : les 'Stations dédiées à la recharge rapide' ont une "
            "puissance nominale médiane drastiquement supérieure aux bornes en voirie ou sur les parkings publics. "
            "Le test statistique de Kruskal-Wallis a confirmé que ces différences sont hautement significatives"
            "(p < 0.05)."
        )
    },
    "graphe_categoriel.png": {
        "titre": "Influence des Variables Catégorielles",
        "texte": (
            "<b>Que montre ce graphique ?</b><br>"
            "Il illustre comment l'accessibilité PMR et les conditions d'accès sont réparties selon le lieu.<br><br>"
            "<b>Analyse :</b><br>"
            "Les règles d'accessibilité varient fortement d'un domaine à l'autre. Par exemple, la voirie a des "
            "obligations légales différentes des parkings privés. Le test du Chi-Deux a validé la pertinence "
            "de conserver ces variables pour l'entraînement du modèle final."
        )
    },
    "graphe_geo.png": {
        "titre": "Empreinte Géographique des Implantations",
        "texte": (
            "<b>Que montre ce graphique ?</b><br>"
            "Il s'agit de mini-cartes de France (nuages de points basés sur la Longitude et Latitude) "
            "filtrées par type d'implantation.<br><br>"
            "<b>Analyse :</b><br>"
            "La voirie couvre l'ensemble du territoire de manière assez uniforme, tandis que les "
            "stations de recharge rapide se concentrent sur les grands axes (autoroutes) et les parkings "
            "privés près des grands bassins de population. <i>Ceci justifie l'intégration des coordonnées "
            "GPS comme variables majeures dans le Machine Learning.</i>"
        )
    }
}


# =============================================================================
# FENÊTRE VISIONNEUSE (Affiche l'image + le texte)
# =============================================================================
class ViewerGraphe(QDialog):
    def __init__(self, chemin_image, nom_fichier):
        super().__init__()
        self.chemin_image = chemin_image
        self.infos = EXPLICATIONS.get(nom_fichier, {"titre": "Analyse", "texte": "Pas de description disponible."})
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.infos["titre"])
        self.resize(1000, 650)  # Fenêtre de belle taille

        layout_principal = QHBoxLayout()  # Disposition horizontale : Image à gauche, Texte à droite

        # --- 1. Zone Image (Gauche) ---
        self.lbl_image = QLabel("Chargement de l'image...")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: white; border: 1px solid #ccc;")

        # Chargement et redimensionnement propre de l'image
        pixmap = QPixmap(self.chemin_image)
        # On redimensionne l'image pour qu'elle tienne dans environ 700x600 pixels tout en gardant les proportions
        pixmap_scale = pixmap.scaled(700, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_image.setPixmap(pixmap_scale)

        layout_principal.addWidget(self.lbl_image, stretch=7)  # L'image prend 70% de la largeur

        # --- 2. Zone Texte Explicatif (Droite) ---
        layout_texte = QVBoxLayout()

        lbl_titre = QLabel(self.infos["titre"])
        lbl_titre.setWordWrap(True)
        lbl_titre.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout_texte.addWidget(lbl_titre)

        zone_explication = QTextEdit()
        zone_explication.setHtml(self.infos["texte"])
        zone_explication.setReadOnly(True)  # Empêche l'utilisateur d'écrire dedans
        zone_explication.setStyleSheet("font-size: 13px; background-color: #f8f9fa; padding: 10px; border: none;")
        layout_texte.addWidget(zone_explication)

        btn_fermer = QPushButton("Fermer l'analyse")
        btn_fermer.setStyleSheet("padding: 10px; background-color: #e74c3c; color: white; font-weight: bold;")
        btn_fermer.clicked.connect(self.close)
        layout_texte.addWidget(btn_fermer)

        layout_principal.addLayout(layout_texte, stretch=3)  # Le texte prend 30% de la largeur

        self.setLayout(layout_principal)


# =============================================================================
# HUB PRINCIPAL
# =============================================================================
class HubJustification(QWidget):
    def __init__(self, df):
        super().__init__()
        self.dossier_graphes = "./graphe/besoin3"
        self.fichier_csv = "../IRVE_clean_FINAL.csv"
        os.makedirs(self.dossier_graphes, exist_ok=True)
        self.df = df.copy()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Hub - Justification des Variables (Besoin 3)")
        self.setFixedSize(450, 400)
        layout = QVBoxLayout()

        titre = QLabel("Analyses Statistiques et Justifications")
        titre.setStyleSheet("font-size: 16px; font-weight: bold;")
        titre.setAlignment(Qt.AlignCenter)
        layout.addWidget(titre)

        info = QLabel("Ces graphiques justifient le choix des variables pour le Machine Learning.")
        info.setStyleSheet("color: gray; font-style: italic;")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)
        layout.addWidget(QFrame(frameShape=QFrame.HLine))

        # --- Boutons d'accès aux graphiques ---
        self.btn_barres = QPushButton("Impact des Variables Booléennes")
        self.btn_barres.setStyleSheet("padding: 12px; background-color: #3498db; color: white;")
        self.btn_barres.clicked.connect(lambda: self.ouvrir_ou_generer("graphe_barres_empilees.png"))
        layout.addWidget(self.btn_barres)

        self.btn_box = QPushButton("Boxplots : Puissances et Prises")
        self.btn_box.setStyleSheet("padding: 12px; background-color: #9b59b6; color: white;")
        self.btn_box.clicked.connect(lambda: self.ouvrir_ou_generer("graphe_boxplots.png"))
        layout.addWidget(self.btn_box)

        self.btn_cat = QPushButton("Impact des Variables Catégorielles")
        self.btn_cat.setStyleSheet("padding: 12px; background-color: #f1c40f; color: black;")
        self.btn_cat.clicked.connect(lambda: self.ouvrir_ou_generer("graphe_categoriel.png"))
        layout.addWidget(self.btn_cat)

        self.btn_geo = QPushButton("Empreinte Géographique (GPS)")
        self.btn_geo.setStyleSheet("padding: 12px; background-color: #2ecc71; color: white;")
        self.btn_geo.clicked.connect(lambda: self.ouvrir_ou_generer("graphe_geo.png"))
        layout.addWidget(self.btn_geo)

        self.setLayout(layout)

    # =========================================================================
    # LOGIQUE DE GESTION (Vérification -> Génération -> Ouverture)
    # =========================================================================
    def ouvrir_ou_generer(self, nom_fichier):
        chemin_absolu = os.path.abspath(os.path.join(self.dossier_graphes, nom_fichier))

        # Si le fichier n'existe pas, on lance la génération de TOUS les graphiques d'un coup
        if not os.path.exists(chemin_absolu):
            reponse = QMessageBox.question(
                self, "Génération requise",
                "Les graphiques n'ont pas encore été générés. Cela peut prendre quelques secondes.\nVoulez-vous les générer maintenant ?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reponse == QMessageBox.Yes:
                # Changement du texte des boutons pour faire patienter l'utilisateur
                titre_original = self.windowTitle()
                self.setWindowTitle("Génération en cours... Veuillez patienter.")
                QApplication.processEvents()

                succes = self.generer_tous_les_graphes()
                self.setWindowTitle(titre_original)

                if not succes:
                    return  # Arrêt si la génération a échoué
            else:
                return  # L'utilisateur a annulé

        # Ouverture de la fenêtre PyQt personnalisée
        self.viewer = ViewerGraphe(chemin_absolu, nom_fichier)
        self.viewer.exec_()  # Utilisation de exec_() car c'est un QDialog (fenêtre modale)

    # =========================================================================
    # LOGIQUE DE GÉNÉRATION
    # =========================================================================
    def generer_tous_les_graphes(self):
        """Génère les 4 graphiques et les sauvegarde dans le dossier spécifié."""

        try:
            # 1. Préparation des données
            TARGET = "implantation_station"
            label_map = {
                "Voirie": "Voirie",
                "Parking privé à usage public": "Parking public",
                "Parking public": "Parking publ.",
                "Station dédiée à la recharge rapide": "Station rapide",
                "Parking privé réservé à la clientèle": "Parking privé",
            }
            self.df["implantation_court"] = self.df[TARGET].map(label_map).fillna(self.df[TARGET])
            ORDER = ["Voirie", "Parking publ.", "Parking public", "Station rapide", "Parking privé"]
            palette = {"Voirie": "#E8593C", "Parking publ.": "#3B6D11", "Parking public": "#185FA5",
                       "Station rapide": "#F5A623", "Parking privé": "#7B4FBF"}

            # 2. Graphique 1 : Barres empilées (Booléens)
            bool_cols = ["prise_type_combo_ccs", "prise_type_chademo", "prise_type_2",
                         "cable_t2_attache", "gratuit", "paiement_cb", "reservation", "station_deux_roues"]
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            for ax, col in zip(axes.flatten(), bool_cols):
                if col in self.df.columns:
                    ct = pd.crosstab(self.df[col], self.df["implantation_court"], normalize="index") * 100
                    ct = ct.reindex(columns=[c for c in ORDER if c in ct.columns])
                    colors = [palette.get(c, "#AAAAAA") for c in ct.columns]
                    ct.plot(kind="bar", stacked=True, ax=ax, color=colors, legend=False, width=0.6)
                    ax.set_title(col.replace("_", " "), fontsize=10, fontweight="bold")
                    ax.set_xticklabels(["Non (0)", "Oui (1)"], rotation=0)
                    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
                    ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.dossier_graphes, "graphe_barres_empilees.png"), dpi=100)
            plt.close()

            # 3. Graphique 2 : Boxplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, col, titre in zip(axes, ["puissance_nominale", "nbre_pdc"], ["Puissance (kW)", "Points de charge"]):
                if col in self.df.columns:
                    data_plot = self.df[["implantation_court", col]].dropna()
                    order_box = [o for o in ORDER if o in data_plot["implantation_court"].unique()]
                    bp = ax.boxplot([data_plot[data_plot["implantation_court"] == o][col].values for o in order_box],
                                    tick_labels=order_box, patch_artist=True,
                                    flierprops=dict(marker="o", markersize=2, alpha=0.3))
                    for patch, color in zip(bp["boxes"], [palette.get(o, "#AAA") for o in order_box]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    ax.set_title(titre, fontweight="bold")
                    ax.set_xticklabels(order_box, rotation=15, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(self.dossier_graphes, "graphe_boxplots.png"), dpi=100)
            plt.close()

            # 4. Graphique 3 : Variables Catégorielles
            cat_cols = ["condition_acces", "accessibilite_pmr"]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, col in zip(axes, cat_cols):
                if col in self.df.columns:
                    ct = pd.crosstab(self.df[col], self.df["implantation_court"], normalize="index") * 100
                    ct = ct.reindex(columns=[c for c in ORDER if c in ct.columns])
                    colors = [palette.get(c, "#AAAAAA") for c in ct.columns]
                    ct.plot(kind="bar", stacked=True, ax=ax, color=colors, legend=False, width=0.6)
                    ax.set_title(col.replace("_", " "), fontweight="bold")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
                    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
            plt.tight_layout()
            plt.savefig(os.path.join(self.dossier_graphes, "graphe_categoriel.png"), dpi=100)
            plt.close()

            # 5. Graphique 4 : Carte Géo (GPS)
            # On crée une grille 2x3 et on augmente la hauteur (figsize de 4 à 8) pour que les 2 lignes respirent
            fig, axes = plt.subplots(2, 3, figsize=(18, 8))
            # On "aplatit" la grille en une simple liste de 6 axes pour faciliter la boucle
            axes_flat = axes.flatten()
            for i, (impl_long, impl_court) in enumerate(label_map.items()):
                ax = axes_flat[i]  # On récupère l'axe correspondant à l'index
                subset = self.df[self.df[TARGET] == impl_long]
                color = palette.get(impl_court, "#AAAAAA")
                ax.scatter(subset["lon"], subset["lat"], alpha=0.05, s=0.5, color=color)
                ax.set_title(impl_court, fontsize=11, fontweight="bold",
                             color=color)
                ax.set_xlim(-5.5, 10)
                ax.set_ylim(41, 51.5)
                ax.axis('off')
            axes_flat[5].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(self.dossier_graphes, "graphe_geo.png"), dpi=100)
            plt.close()

            return True

        except Exception as e:
            QMessageBox.critical(self, "Erreur de génération",
                                 f"Une erreur s'est produite lors de la création des graphiques :\n{str(e)}")
            return False
