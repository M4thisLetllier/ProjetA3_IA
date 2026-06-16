import os
import webbrowser
import folium
from folium.plugins import MarkerCluster, HeatMap

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                             QLabel, QPushButton, QMessageBox, QFrame)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from configuration import DOSSIER_CARTES, DOSSIER_GRAPHE

class VisualisationHub(QWidget):
    def __init__(self,df):
        super().__init__()
        self.df = df

        self.init_ui()
        self.charger_donnees()

    def init_ui(self):
        self.setWindowTitle("Hub de Visualisation des Données IRVE")
        self.setFixedSize(450, 450)
        layout = QVBoxLayout()

        # Titre
        titre = QLabel("Tableau de Bord - Cartes & Graphiques")
        titre.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        titre.setAlignment(Qt.AlignCenter)
        layout.addWidget(titre)

        info_cartes = QLabel("Cartographies interactives :")
        info_cartes.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info_cartes)

        # --- Bouton 1 : Carte des Implantations ---
        self.btn_implantations = QPushButton("Ouvrir la Carte des Implantations (Clusters)")
        self.btn_implantations.setStyleSheet("background-color: #2196F3; color: white; padding: 10px;")
        self.btn_implantations.clicked.connect(self.gerer_carte_implantations)
        layout.addWidget(self.btn_implantations)

        # --- Bouton 2 : Carte de Chaleur (Rapide) ---
        self.btn_chaleur_rap = QPushButton("Ouvrir la Heatmap (Recharge Rapide)")
        self.btn_chaleur_rap.setStyleSheet("background-color: #F44336; color: white; padding: 10px;")
        self.btn_chaleur_rap.clicked.connect(self.gerer_carte_chaleur_rapide)
        layout.addWidget(self.btn_chaleur_rap)

        # --- Bouton 3 : Carte de Chaleur (Globale) ---
        self.btn_chaleur_glob = QPushButton("Ouvrir la Heatmap (Réseau Global)")
        self.btn_chaleur_glob.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        self.btn_chaleur_glob.clicked.connect(self.gerer_carte_chaleur_globale)
        layout.addWidget(self.btn_chaleur_glob)

        # Séparateur visuel
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        layout.addWidget(line)

        info_graphes = QLabel("Analyses statistiques :")
        info_graphes.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info_graphes)

        # --- Bouton 4 : Graphique Diagramme ---
        self.btn_graph_diag = QPushButton("Afficher le Diagramme de répartition")
        self.btn_graph_diag.setStyleSheet("background-color: #607D8B; color: white; padding: 10px;")
        self.btn_graph_diag.clicked.connect(lambda: self.ouvrir_image("Diagramme.png"))
        layout.addWidget(self.btn_graph_diag)

        # --- Bouton 5 : Graphique Puissance ---
        self.btn_graph_puis = QPushButton("Afficher le Graphique des Puissances")
        self.btn_graph_puis.setStyleSheet("background-color: #9C27B0; color: white; padding: 10px;")
        self.btn_graph_puis.clicked.connect(lambda: self.ouvrir_image("Puissance.png"))
        layout.addWidget(self.btn_graph_puis)

        self.setLayout(layout)

    # =========================================================
    # GESTION DES DONNÉES
    # =========================================================
    def charger_donnees(self):
        """Charge et nettoie les données uniquement si nécessaire."""
        colonnes_interet = ['nom_station', 'implantation_station', 'lat', 'lon']
        self.df = self.df[colonnes_interet]

        # Charte graphique
        couleurs = {
            'Voirie': '#2196F3',
            'Parking privé à usage public': '#4CAF50',
            'Parking public': '#FF9800',
            'Station dédiée à la recharge rapide': '#F44336',
            'Parking privé réservé à la clientèle': '#9C27B0',
        }
        self.df['couleur_hex'] = self.df['implantation_station'].map(couleurs).fillna('#757575')
        return True

    # =========================================================
    # MÉTHODES POUR LES CARTES
    # =========================================================
    def gerer_carte_implantations(self):
        nom_fichier = "carte_implantations.html"
        chemin_absolu = os.path.abspath(os.path.join(DOSSIER_CARTES, nom_fichier))

        if not os.path.exists(chemin_absolu):
            self.btn_implantations.setText("Génération en cours...")
            QApplication.processEvents()

            carte = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles='CartoDB Positron')
            cluster_moteur = MarkerCluster(disableClusteringAtZoom=15).add_to(carte)

            for lat, lon, imp, hex_c in zip(self.df['lat'], self.df['lon'], self.df['implantation_station'],
                                            self.df['couleur_hex']):
                folium.CircleMarker(
                    location=[lat, lon], radius=4, color=hex_c, fill=True, fill_color=hex_c, fill_opacity=0.7,
                    popup=folium.Popup(f"<b>Type :</b> {imp}", max_width=300)
                ).add_to(cluster_moteur)

            carte.save(chemin_absolu)
            self.btn_implantations.setText("Ouvrir la Carte des Implantations (Clusters)")

        webbrowser.open('file://' + chemin_absolu)

    def gerer_carte_chaleur_rapide(self):
        nom_fichier = "carte_chaleur_rapide.html"
        chemin_absolu = os.path.abspath(os.path.join(DOSSIER_CARTES, nom_fichier))

        if not os.path.exists(chemin_absolu):
            self.btn_chaleur_rap.setText("Génération en cours...")
            QApplication.processEvents()

            df_fast = self.df[self.df['implantation_station'] == 'Station dédiée à la recharge rapide']
            carte = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles='CartoDB Dark_Matter')
            points_rapides = df_fast[['lat', 'lon']].values.tolist()

            HeatMap(
                points_rapides, radius=9, blur=14, max_zoom=11,
                gradient={0.2: 'navy', 0.5: 'cyan', 0.8: 'yellow', 1.0: 'crimson'}
            ).add_to(carte)

            carte.save(chemin_absolu)
            self.btn_chaleur_rap.setText("Ouvrir la Heatmap (Recharge Rapide)")

        webbrowser.open('file://' + chemin_absolu)

    def gerer_carte_chaleur_globale(self):
        nom_fichier = "carte_chaleur_globale.html"
        chemin_absolu = os.path.abspath(os.path.join(DOSSIER_CARTES, nom_fichier))

        if not os.path.exists(chemin_absolu):
            self.btn_chaleur_glob.setText("Génération en cours...")
            QApplication.processEvents()

            carte = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles='CartoDB Dark_Matter')
            points_globaux = self.df[['lat', 'lon']].values.tolist()

            HeatMap(
                points_globaux, radius=7, blur=10, max_zoom=11,
                gradient={0.2: 'blue', 0.4: 'lime', 0.7: 'yellow', 1.0: 'red'}
            ).add_to(carte)

            carte.save(chemin_absolu)
            self.btn_chaleur_glob.setText("Ouvrir la Heatmap (Réseau Global)")

        webbrowser.open('file://' + chemin_absolu)

    # =========================================================
    # MÉTHODES POUR LES GRAPHIQUES PNG
    # =========================================================
    def ouvrir_image(self, nom_image : str):
        """Ouvre une image PNG avec la visionneuse par défaut de l'OS."""
        chemin_absolu = os.path.abspath(os.path.join(DOSSIER_GRAPHE, nom_image))

        if os.path.exists(chemin_absolu):
            # QDesktopServices ouvre le fichier avec le programme par défaut de Windows/Mac/Linux
            url = QUrl.fromLocalFile(chemin_absolu)
            QDesktopServices.openUrl(url)
        else:
            QMessageBox.warning(self, "Fichier Introuvable",
                                f"L'image '{nom_image}' est introuvable dans le dossier '{DOSSIER_GRAPHE}'.\n"
                                "Veuillez vérifier qu'elle a bien été générée en amont.")
