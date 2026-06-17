import sys
import webbrowser
import pandas as pd

from Besoin_client_2.Kmeans_Hub import KmeansPage
from Besoin_client_2.DbScan_Hub import DbscanPage
from Besoin_client_2.Prediction_hub import PredictionPage
from Besoin_client_1.Visualisation_Hub import VisualisationHub
from besoin_client_3.Graphe_besoin3_Hub import HubJustification
from besoin_client_3.PredictionB3_Hub import HubPrediction
from Besoin_client_4.graphe_besoin4_hub import  HubJustificationB4
from Besoin_client_4.prediction import RegressionPage
from Besoin_client_4.PredictionB4_Hub import HubPredictionPuissance

from configuration import creation_dossier
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QLabel, QPushButton, QMessageBox, QFrame)
from PyQt5.QtCore import Qt




class MainHub(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Panel d'Analyse des Bornes de Recharge")
        self.setFixedSize(800, 800)

        # Simulation chargement données
        self.charger_donnees()

        layout = QVBoxLayout()
        title = QLabel("Menu Principal Analysis")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 20px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # ---------------------------
        ## Besoin client 1
        # ---------------------------
        info_b1 = QLabel("Visualisation :")
        info_b1.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info_b1)

        self.btn_visualisation = QPushButton("Ouvrir l'Outil de visualisation")
        self.btn_visualisation.setFixedHeight(50)
        self.btn_visualisation.clicked.connect(self.open_visualisation)
        layout.addWidget(self.btn_visualisation)

        # ---------------------------
        ## Besoin client 2
        # ---------------------------
        info_b2 = QLabel("Clusturing :")
        info_b2.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info_b2)

        self.btn_kmeans = QPushButton("Ouvrir Analyse K-MEANS")
        self.btn_kmeans.setFixedHeight(50)
        self.btn_kmeans.clicked.connect(self.open_kmeans)
        layout.addWidget(self.btn_kmeans)

        self.btn_dbscan = QPushButton("Ouvrir Analyse DBSCAN")
        self.btn_dbscan.setFixedHeight(50)
        self.btn_dbscan.clicked.connect(self.open_dbscan)
        layout.addWidget(self.btn_dbscan)

        self.btn_predict = QPushButton("Ouvrir l'Outil de Prédiction (K-Means)")
        self.btn_predict.setFixedHeight(50)
        self.btn_predict.clicked.connect(self.open_prediction)
        layout.addWidget(self.btn_predict)

        # ---------------------------
        ## Besoin client 3
        # ---------------------------

        info_b3 = QLabel("Machine Learning type d'implantation :")
        info_b3.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info_b3)

        self.btn_graphe_b3 = QPushButton("Ouvrir la justification du choix des variables")
        self.btn_graphe_b3.setFixedHeight(50)
        self.btn_graphe_b3.clicked.connect(self.open_graphe_b3)
        layout.addWidget(self.btn_graphe_b3)

        self.btn_predict_b3 = QPushButton("Ouvrir l'outil de prediction type d'implantation")
        self.btn_predict_b3.setFixedHeight(50)
        self.btn_predict_b3.clicked.connect(self.open_predict_b3)
        layout.addWidget(self.btn_predict_b3)

        # ---------------------------
        ## Besoin client 4
        # ---------------------------
        info_b4 = QLabel("Machine Learning Puissance :")
        info_b4.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info_b4)

        self.btn_graphe_b4 = QPushButton("Ouvrir la justification du choix des variables")
        self.btn_graphe_b4.setFixedHeight(50)
        self.btn_graphe_b4.clicked.connect(self.open_graphe_b4)
        layout.addWidget(self.btn_graphe_b4)

        self.btn_predict_b4 = QPushButton("Ouvrir l'outil de prediction de la puissance")
        self.btn_predict_b4.setFixedHeight(50)
        self.btn_predict_b4.clicked.connect(self.open_predict_b4)
        layout.addWidget(self.btn_predict_b4)

        footer = QLabel("D'autres analyses seront ajoutées ultérieurement...")
        footer.setStyleSheet("color: gray; font-size: 10px;")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)

        self.setLayout(layout)

    def charger_donnees(self):
        self.df = pd.read_csv("IRVE_clean_FINAL.csv")
        self.coords = self.df[['lon', 'lat']].dropna()

    def open_visualisation(self):
        self.vs_win = VisualisationHub(self.df)
        self.vs_win.show()

    def open_kmeans(self):
        self.km_win = KmeansPage(self.coords, self.df)
        self.km_win.show()

    def open_dbscan(self):
        self.db_win = DbscanPage(self.coords, self.df)
        self.db_win.show()

    def open_prediction(self):
        self.pred_win = PredictionPage(self.coords)
        self.pred_win.show()

    def open_graphe_b3(self):
        self.graphe_b3_win = HubJustification(self.df)
        self.graphe_b3_win.show()

    def open_predict_b3(self):
        self.pred_b3_win = HubPrediction(self.df)
        self.pred_b3_win.show()

    def open_graphe_b4(self):
        self.graphe_b4_win = HubJustificationB4(self.df)
        self.graphe_b4_win.show()

    def open_predict_b4(self):
        self.pred_b4_win = HubPredictionPuissance(self.df)
        self.pred_b4_win.show()


if __name__ == "__main__":
    creation_dossier()
    app = QApplication(sys.argv)
    hub = MainHub()
    hub.show()
    sys.exit(app.exec_())