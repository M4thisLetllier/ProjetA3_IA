import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QMessageBox, QFrame, QDialog, QTextEdit)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

from configuration import DOSSIER_GRAPHE_B4

# =============================================================================
# DICTIONNAIRE DES EXPLICATIONS MÉTIER (BESOIN 4)
# =============================================================================
EXPLICATIONS = {
    "graphe_bool_puissance.png": {
        "titre": "Impact des Equipements sur la Puissance Moyenne",
        "texte": (
            "<b>Que montre ce graphique ?</b><br>"
            "Il presente la puissance nominale moyenne (en kW) selon la presence (1) ou l'absence (0) "
            "des differents types de prises, connecteurs et services de la station.<br><br>"
            "<b>Analyse :</b><br>"
            "On constate que la presence de prises de type Combo CCS ou CHAdeMO est directement correlee "
            "a des niveaux de puissance moyenne extremement eleves (caracteristiques de la recharge rapide). "
            "A l'inverse, la gratuite ou l'amenagement pour deux roues concernent majoritairement des puissances "
            "plus faibles. Ces variables binaires apportent une separation franche indispensable pour notre "
            "modele de regression."
        )
    },
    "graphe_num_pdc.png": {
        "titre": "Relation entre Points de Charge et Puissance",
        "texte": (
            "<b>Que montre ce graphique ?</b><br>"
            "Il illustre l'evolution de la puissance nominale moyenne en fonction du nombre total de points "
            "de charge (pdc) installes au sein d'une meme station.<br><br>"
            "<b>Analyse :</b><br>"
            "Le nombre de points de charge montre une tendance specifique : les stations possedant un nombre modere "
            "mais cible de connecteurs delivrent regulierement de fortes puissances (stations autoroutieres), tandis "
            "que les tres grands parcs de stationnement urbains affichent des puissances individuelles plus divisees. "
            "Cette feature numerique guide precisement l'algorithme de Foret Aleatoire."
        )
    },
    "graphe_cat_puissance.png": {
        "titre": "Distribution de la Puissance par Categorie",
        "texte": (
            "<b>Que montre ce graphique ?</b><br>"
            "Ces boites a moustaches (boxplots) analysent les ecarts et la dispersion de la puissance nominale "
            "selon les conditions d'acces, l'accessibilite PMR et le type d'implantation de la borne.<br><br>"
            "<b>Analyse :</b><br>"
            "On observe des ecarts majeurs selon les contextes. Les stations implantees dans des zones dediees "
            "a la recharge rapide affichent des medianes de puissance nettement superieures aux installations en voirie. "
            "L'analyse de variance statistique confirme que ces facteurs categoriels sont hautement significatifs "
            "pour expliquer la variabilite de la puissance sur le territoire."
        )
    },
    "graphe_geo_puissance.png": {
        "titre": "Repartition Geographique des Puissances",
        "texte": (
            "<b>Que montre ce graphique ?</b><br>"
            "Ce nuage de points cartographique separe le territoire national en trois grandes categories "
            "de puissance : standard, rapide et haute puissance.<br><br>"
            "<b>Analyse :</b><br>"
            "Les infrastructures de puissance standard (inferieure ou egale a 22 kW) sont distribuees de facon "
            "homogene dans toutes les communes de France. En revanche, les bornes de haute puissance (superieure "
            "a 50 kW) suivent lineairement les corridors routiers majeurs et les autoroutes. Les coordonnees GPS "
            "(Longitude et Latitude) sont donc cruciales pour predire la puissance locale maximale."
        )
    }
}


# =============================================================================
# FENÊTRE VISIONNEUSE
# =============================================================================
class ViewerGraphe(QDialog):
    def __init__(self, chemin_image, nom_fichier):
        super().__init__()
        self.chemin_image = chemin_image
        self.infos = EXPLICATIONS.get(nom_fichier, {"titre": "Analyse", "texte": "Pas de description disponible."})
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.infos["titre"])
        self.resize(1000, 650)

        layout_principal = QHBoxLayout()

        # Zone Image (Gauche)
        self.lbl_image = QLabel("Chargement de l'image...")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: white; border: 1px solid #ccc;")

        pixmap = QPixmap(self.chemin_image)
        pixmap_scale = pixmap.scaled(700, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_image.setPixmap(pixmap_scale)
        layout_principal.addWidget(self.lbl_image, stretch=7)

        # Zone Texte Explicatif (Droite)
        layout_texte = QVBoxLayout()

        lbl_titre = QLabel(self.infos["titre"])
        lbl_titre.setWordWrap(True)
        lbl_titre.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout_texte.addWidget(lbl_titre)

        zone_explication = QTextEdit()
        zone_explication.setHtml(self.infos["texte"])
        zone_explication.setReadOnly(True)
        zone_explication.setStyleSheet("font-size: 13px; background-color: #f8f9fa; padding: 10px; border: none;")
        layout_texte.addWidget(zone_explication)

        btn_fermer = QPushButton("Fermer l'analyse")
        btn_fermer.setStyleSheet("padding: 10px; background-color: #e74c3c; color: white; font-weight: bold;")
        btn_fermer.clicked.connect(self.close)
        layout_texte.addWidget(btn_fermer)

        layout_principal.addLayout(layout_texte, stretch=3)
        self.setLayout(layout_principal)


# =============================================================================
# HUB PRINCIPAL (BESOIN 4)
# =============================================================================
class HubJustificationB4(QWidget):
    def __init__(self, df):
        super().__init__()
        self.dossier_graphes = DOSSIER_GRAPHE_B4
        self.df = df.copy()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Hub - Justification des Variables (Besoin 4)")
        self.setFixedSize(450, 400)
        layout = QVBoxLayout()

        titre = QLabel("Analyses Statistiques et Justifications (Regresseur)")
        titre.setStyleSheet("font-size: 16px; font-weight: bold;")
        titre.setAlignment(Qt.AlignCenter)
        layout.addWidget(titre)

        info = QLabel("Ces graphiques justifient le choix des variables pour l'estimation de la puissance.")
        info.setStyleSheet("color: gray; font-style: italic;")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)
        layout.addWidget(QFrame(frameShape=QFrame.HLine))

        # Boutons d'accès aux graphiques
        self.btn_barres = QPushButton("Impact des Variables Booleennes sur la Puissance")
        self.btn_barres.setStyleSheet("padding: 12px; background-color: #3498db; color: white;")
        self.btn_barres.clicked.connect(lambda: self.ouvrir_ou_generer("graphe_bool_puissance.png"))
        layout.addWidget(self.btn_barres)

        self.btn_num = QPushButton("Relation Points de Charge et Puissance")
        self.btn_num.setStyleSheet("padding: 12px; background-color: #9b59b6; color: white;")
        self.btn_num.clicked.connect(lambda: self.ouvrir_ou_generer("graphe_num_pdc.png"))
        layout.addWidget(self.btn_num)

        self.btn_cat = QPushButton("Boxplots : Variables Categorielles vs Puissance")
        self.btn_cat.setStyleSheet("padding: 12px; background-color: #f1c40f; color: black;")
        self.btn_cat.clicked.connect(lambda: self.ouvrir_ou_generer("graphe_cat_puissance.png"))
        layout.addWidget(self.btn_cat)

        self.btn_geo = QPushButton("Cartographie Geographique de la Puissance")
        self.btn_geo.setStyleSheet("padding: 12px; background-color: #2ecc71; color: white;")
        self.btn_geo.clicked.connect(lambda: self.ouvrir_ou_generer("graphe_geo_puissance.png"))
        layout.addWidget(self.btn_geo)

        self.setLayout(layout)

    def ouvrir_ou_generer(self, nom_fichier):
        chemin_absolu = os.path.abspath(os.path.join(self.dossier_graphes, nom_fichier))

        if not os.path.exists(chemin_absolu):
            reponse = QMessageBox.question(
                self, "Generation requise",
                "Les graphiques analytiques n'ont pas encore ete generes. Voulez-vous les creer maintenant ?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reponse == QMessageBox.Yes:
                titre_original = self.windowTitle()
                self.setWindowTitle("Generation en cours... Veuillez patienter.")
                QApplication.processEvents()

                succes = self.generer_tous_les_graphes_b4()
                self.setWindowTitle(titre_original)

                if not succes:
                    return
            else:
                return

        self.viewer = ViewerGraphe(chemin_absolu, nom_fichier)
        self.viewer.exec_()

    # =========================================================================
    # LOGIQUE DE GÉNÉRATION DES GRAPHES POUR LA RÉGRESSION (BESOIN 4)
    # =========================================================================
    def generer_tous_les_graphes_b4(self):
        try:
            TARGET = "puissance_nominale"
            self.df = self.df.dropna(subset=[TARGET, "lon", "lat"])

            # 1. Graphique 1 : Barres d'impact des Booleens (Moyenne de Puissance)
            bool_cols = ["prise_type_combo_ccs", "prise_type_chademo", "prise_type_2",
                         "cable_t2_attache", "gratuit", "paiement_cb", "reservation", "station_deux_roues"]
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            for ax, col in zip(axes.flatten(), bool_cols):
                if col in self.df.columns:
                    means = self.df.groupby(col)[TARGET].mean()
                    means.plot(kind="bar", ax=ax, color=["#185FA5", "#E8593C"], width=0.5)
                    ax.set_title(col.replace("_", " "), fontsize=10, fontweight="bold")
                    ax.set_xticklabels(["Non (0)", "Oui (1)"], rotation=0)
                    ax.set_ylabel("Puissance Moyenne (kW)")
                    ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.dossier_graphes, "graphe_bool_puissance.png"), dpi=100)
            plt.close()

            # 2. Graphique 2 : Impact de la variable numerique discrete nbre_pdc
            fig, ax = plt.subplots(figsize=(10, 5))
            if "nbre_pdc" in self.df.columns:
                means_pdc = self.df.groupby("nbre_pdc")[TARGET].mean().sort_index().head(15)
                means_pdc.plot(kind="bar", ax=ax, color="#7B4FBF", width=0.6)
                ax.set_title("Puissance Moyenne (kW) selon le Nombre de Points de Charge", fontweight="bold")
                ax.set_xlabel("Nombre de Points de Charge (pdc)")
                ax.set_ylabel("Puissance Moyenne (kW)")
                ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.dossier_graphes, "graphe_num_pdc.png"), dpi=100)
            plt.close()

            # 3. Graphique 3 : Boxplots pour les Variables Categorielles Features
            cat_cols = ["condition_acces", "accessibilite_pmr", "implantation_station"]
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            for ax, col in zip(axes, cat_cols):
                if col in self.df.columns:
                    data_sub = self.df[[col, TARGET]].dropna()
                    means_cat = data_sub.groupby(col)[TARGET].mean().sort_values()
                    categories_sorted = means_cat.index
                    
                    bp = ax.boxplot([data_sub[data_sub[col] == c][TARGET].values for c in categories_sorted],
                                    tick_labels=[str(c)[:15] for c in categories_sorted], patch_artist=True,
                                    flierprops=dict(marker="o", markersize=2, alpha=0.3))
                    for patch in bp["boxes"]:
                        patch.set_facecolor("#3498db")
                        patch.set_alpha(0.7)
                    ax.set_title(col.replace("_", " "), fontweight="bold")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
                    ax.set_ylabel("Puissance Nominale (kW)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.dossier_graphes, "graphe_cat_puissance.png"), dpi=100)
            plt.close()

            # 4. Graphique 4 : Cartographie par classes de puissance (GPS)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            conditions = [
                (self.df[TARGET] <= 22),
                (self.df[TARGET] > 22) & (self.df[TARGET] <= 50),
                (self.df[TARGET] > 50)
            ]
            cat_names = ["Puissance Standard (<= 22 kW)", "Recharge Rapide (22 a 50 kW)", "Haute Puissance (> 50 kW)"]
            colors_geo = ["#185FA5", "#F5A623", "#E8593C"]
            
            for i, ax in enumerate(axes):
                subset = self.df[conditions[i]]
                ax.scatter(subset["lon"], subset["lat"], alpha=0.1, s=1, color=colors_geo[i])
                ax.set_title(cat_names[i], fontsize=11, fontweight="bold", color=colors_geo[i])
                ax.set_xlim(-5.5, 10)
                ax.set_ylim(41, 51.5)
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.dossier_graphes, "graphe_geo_puissance.png"), dpi=100)
            plt.close()

            return True

        except Exception as e:
            QMessageBox.critical(self, "Erreur de generation",
                                 f"Une erreur est survenue lors de la creation des graphiques du Besoin 4 :\n{str(e)}")
            return False