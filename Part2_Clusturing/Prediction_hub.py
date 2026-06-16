import os
import joblib
import folium
import glob
import numpy as np
import webbrowser
from sklearn.cluster import KMeans
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,QComboBox,
                             QLabel, QPushButton, QMessageBox, QDoubleSpinBox, QFrame, QApplication)
from PyQt5.QtCore import Qt
from configuration import DOSSIER_CARTES, DOSSIER_MODELES, COULEURS


class PredictionPage(QWidget):
    def __init__(self, df_coords):
        super().__init__()
        self.coords = df_coords
        self.k_cible = 6  # Le modèle conseillé par défaut
        self.model = None
        self.init_ui()
        self.lister_et_charger_modeles()

    def init_ui(self):
        self.setWindowTitle("Prédiction : Nouvelle Borne")
        self.setFixedSize(450, 480)  # Légèrement agrandie pour le menu déroulant
        layout = QVBoxLayout()

        titre = QLabel("Testez l'emplacement d'une nouvelle borne")
        titre.setStyleSheet("font-size: 16px; font-weight: bold;")
        titre.setAlignment(Qt.AlignCenter)
        layout.addWidget(titre)

        # --- NOUVEAU : Menu déroulant pour le choix du modèle ---
        layout.addWidget(QLabel("Choisissez le modèle de prédiction (Sectorisation) :"))
        self.combo_modeles = QComboBox()
        self.combo_modeles.setStyleSheet("padding: 5px; font-size: 14px;")
        self.combo_modeles.currentIndexChanged.connect(self.changer_modele)
        layout.addWidget(self.combo_modeles)

        line_top = QFrame();
        line_top.setFrameShape(QFrame.HLine);
        layout.addWidget(line_top)

        # --- Champs de saisie (Longitude / Latitude) ---
        form_layout = QHBoxLayout()
        vbox_lon = QVBoxLayout()
        vbox_lon.addWidget(QLabel("Longitude (ex: 2.3522) :"))
        self.input_lon = QDoubleSpinBox()
        self.input_lon.setRange(-180.0, 180.0)
        self.input_lon.setDecimals(6)
        self.input_lon.setValue(2.3522)
        vbox_lon.addWidget(self.input_lon)

        vbox_lat = QVBoxLayout()
        vbox_lat.addWidget(QLabel("Latitude (ex: 48.8566) :"))
        self.input_lat = QDoubleSpinBox()
        self.input_lat.setRange(-90.0, 90.0)
        self.input_lat.setDecimals(6)
        self.input_lat.setValue(48.8566)
        vbox_lat.addWidget(self.input_lat)

        form_layout.addLayout(vbox_lon)
        form_layout.addLayout(vbox_lat)
        layout.addLayout(form_layout)

        # --- Bouton de Prédiction ---
        self.btn_predire = QPushButton("Prédire le Secteur")
        self.btn_predire.setStyleSheet("background-color: #28a745; color: white; padding: 10px; font-weight: bold;")
        self.btn_predire.clicked.connect(self.faire_prediction)
        layout.addWidget(self.btn_predire)

        # --- Affichage du Résultat ---
        self.lbl_resultat = QLabel("En attente de prédiction...")
        self.lbl_resultat.setStyleSheet("font-size: 14px; margin: 15px 0; color: #333;")
        self.lbl_resultat.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_resultat)

        line_bot = QFrame();
        line_bot.setFrameShape(QFrame.HLine);
        layout.addWidget(line_bot)

        # --- Bouton d'affichage Carte ---
        self.btn_carte = QPushButton("Visualiser sur la carte (Centres de zones)")
        self.btn_carte.setStyleSheet("background-color: #0078D7; color: white; padding: 10px;")
        self.btn_carte.clicked.connect(self.afficher_carte_rapide)
        self.btn_carte.setEnabled(False)
        layout.addWidget(self.btn_carte)

        self.setLayout(layout)

    def lister_et_charger_modeles(self):
        """Recherche les modèles K-Means existants, génère le k=6 s'il n'existe pas, et peuple le menu."""
        chemin_modele_defaut = os.path.join(DOSSIER_MODELES, f"modele_kmeans_k{self.k_cible}.pkl")

        # 1. On s'assure que le modèle conseillé (k=6) existe
        if not os.path.exists(chemin_modele_defaut):
            self.lbl_resultat.setText("Création du modèle conseillé k=6 en cours...")
            QApplication.processEvents()
            model_defaut = KMeans(n_clusters=self.k_cible, n_init="auto", random_state=42)
            model_defaut.fit(self.coords)
            joblib.dump(model_defaut, chemin_modele_defaut)
            self.lbl_resultat.setText("Modèle conseillé généré avec succès.")

        # 2. On scanne le dossier pour trouver TOUS les modèles K-Means
        fichiers_modeles = glob.glob(os.path.join(DOSSIER_MODELES, "modele_kmeans_k*.pkl"))

        self.combo_modeles.blockSignals(True)  # Évite de déclencher des événements pendant le remplissage
        self.combo_modeles.clear()

        # 3. On extrait les valeurs de 'k' pour trier et afficher joliment
        modeles_trouves = []
        for chemin in fichiers_modeles:
            nom_fichier = os.path.basename(chemin)
            # Extrait le nombre 'k' du nom du fichier
            try:
                k_val = int(nom_fichier.replace("modele_kmeans_k", "").replace(".pkl", ""))
                texte_affichage = f"K-Means ({k_val} secteurs) - Recommandé" if k_val == self.k_cible else f"K-Means ({k_val} secteurs)"
                modeles_trouves.append((k_val, texte_affichage, chemin))
            except ValueError:
                pass  # Ignore les fichiers mal nommés

        # On trie par ordre de clusters (k) croissant
        modeles_trouves.sort(key=lambda x: x[0])

        # 4. On ajoute les éléments au menu déroulant
        index_defaut = 0
        for i, (k_val, texte, chemin) in enumerate(modeles_trouves):
            # On stocke le chemin complet dans les "données cachées" (userData) du ComboBox
            self.combo_modeles.addItem(texte, chemin)
            if k_val == self.k_cible:
                index_defaut = i

        self.combo_modeles.blockSignals(False)

        # 5. On sélectionne le modèle k=6 par défaut
        self.combo_modeles.setCurrentIndex(index_defaut)
        self.changer_modele()  # Charge le modèle en mémoire

    def changer_modele(self):
        """Charge en mémoire le modèle sélectionné dans le menu déroulant."""
        chemin_selectionne = self.combo_modeles.currentData()
        if chemin_selectionne and os.path.exists(chemin_selectionne):
            self.model = joblib.load(chemin_selectionne)
            self.lbl_resultat.setText(f"Prêt à prédire avec {self.combo_modeles.currentText()}.")
            self.btn_carte.setEnabled(False)  # On désactive la carte tant qu'on n'a pas fait de prédiction

    def faire_prediction(self):
        if self.model is None: return

        self.last_lon = self.input_lon.value()
        self.last_lat = self.input_lat.value()

        prediction = self.model.predict([[self.last_lon, self.last_lat]])
        self.predicted_cluster = int(prediction[0])

        couleur = COULEURS[self.predicted_cluster % len(COULEURS)]
        self.lbl_resultat.setText(
            f"Cette borne appartiendra au <b>Cluster {self.predicted_cluster}</b><br>(Couleur associée : {couleur})")
        self.btn_carte.setEnabled(True)

    def afficher_carte_rapide(self):
        # La magie ici : self.model.cluster_centers_ s'adapte automatiquement
        # au modèle actuellement chargé (qu'il ait 6, 12 ou 50 clusters !)
        nom_fichier = "carte_prediction_rapide.html"
        chemin_absolu = os.path.abspath(os.path.join(DOSSIER_CARTES, nom_fichier))

        carte = folium.Map(location=[self.last_lat, self.last_lon], zoom_start=6, tiles='OpenStreetMap')
        centres = self.model.cluster_centers_

        # Adapter le rayon d'influence visuel selon le nombre de clusters (k)
        # Si beaucoup de clusters, les cercles doivent être plus petits
        nb_clusters = len(centres)
        rayon_visuel = 40000 if nb_clusters <= 10 else 15000

        for num_cluster, centre in enumerate(centres):
            lon_c, lat_c = centre[0], centre[1]
            couleur_centre = COULEURS[num_cluster % len(COULEURS)]

            folium.Circle(
                location=[lat_c, lon_c],
                radius=rayon_visuel,
                color=couleur_centre,
                fill=True,
                fill_opacity=0.2,
                popup=f"Centre de gravité du Cluster {num_cluster}"
            ).add_to(carte)

        couleur_prediction = COULEURS[self.predicted_cluster % len(COULEURS)]
        folium.Marker(
            location=[self.last_lat, self.last_lon],
            popup=f"<b>Nouvelle Borne Prédite</b><br>Secteur : {self.predicted_cluster}",
            icon=folium.Icon(color=couleur_prediction, icon='info-sign')
        ).add_to(carte)

        carte.save(chemin_absolu)
        webbrowser.open('file://' + chemin_absolu)