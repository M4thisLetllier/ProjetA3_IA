import sys
import os
import webbrowser
import numpy as np
import joblib
import folium
from sklearn.cluster import DBSCAN

from configuration import DOSSIER_CARTES, DOSSIER_MODELES, COULEURS
from PyQt5.QtWidgets import ( QWidget, QVBoxLayout,
                             QSlider, QLabel, QPushButton)
from PyQt5.QtCore import Qt



class DbscanPage(QWidget):
    def __init__(self, df_coords, parent_df):
        super().__init__()
        self.coords = df_coords
        self.df = parent_df
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Analyse DBSCAN (Densité)")
        self.setFixedSize(500, 300)
        layout = QVBoxLayout()

        self.label = QLabel("Rayon de recherche (Eps) : 10 km")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # Slider config pour sauts de 10 (1 à 10 -> multiplié par 10)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 10)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.update_label)
        layout.addWidget(self.slider)

        self.info = QLabel("DBSCAN définit les zones selon la proximité réelle.\nLes points isolés seront en noir.")
        self.info.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.info)

        self.btn_map = QPushButton("Générer / Ouvrir la Carte DBSCAN")
        self.btn_map.setStyleSheet("background-color: #17a2b8; color: white; padding: 10px;")
        self.btn_map.clicked.connect(self.open_map)
        layout.addWidget(self.btn_map)

        self.setLayout(layout)

    def update_label(self, val):
        self.label.setText(f"Rayon de recherche (Eps) : {val * 10} km")

    def open_map(self):
        eps_km = self.slider.value() * 10
        chemin_filename = os.path.join(DOSSIER_CARTES, f"carte_dbscan_eps{eps_km}.html")
        chemin_modele = os.path.join(DOSSIER_MODELES, f"modele_dbscan_eps{eps_km}.pkl")

        if not os.path.exists(chemin_modele):
            eps_rad = eps_km / 6371.0
            print("Entrainement du modele ...")
            db = DBSCAN(eps=eps_rad, min_samples=3, metric='haversine', algorithm='ball_tree')
            # On sauvegarde le modèle sur le disque
            joblib.dump(db, chemin_modele)
            print("Modèle sauvegardé avec succès.")
        else :
            print("Chargement du modele pré-existant")
            db = joblib.load(chemin_modele)

        if not os.path.exists(chemin_filename):
            print("Creation de la carte ...")
            self.df['cluster_db'] = db.fit_predict(np.radians(self.coords))
            self.save_folium_map(self.df, 'cluster_db', chemin_filename)
            print("Carte Sauvegardée")

        webbrowser.open('file://' + chemin_filename)

    def save_folium_map(self, df, col_cluster, path):
        m = folium.Map(location=[46.5, 2.5], zoom_start=6)
        print("Ajout des marqueurs ...")
        for _, r in df.dropna(subset=['lon', 'lat']).iterrows():
            cid = int(r[col_cluster])
            color = 'black' if cid == -1 else COULEURS[cid % len(COULEURS)]
            folium.CircleMarker([r['lat'], r['lon']], radius=4, color=color, fill=True).add_to(m)
        m.save(path)