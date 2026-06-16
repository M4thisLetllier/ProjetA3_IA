import sys
import webbrowser
import pandas as pd

from Besoin_client_2.Kmeans_Hub import KmeansPage
from Besoin_client_2.DbScan_Hub import DbscanPage
from Besoin_client_2.Prediction_hub import PredictionPage

from configuration import creation_dossier
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QLabel, QPushButton, QMessageBox, QFrame)
from PyQt5.QtCore import Qt




class MainHub(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Panel d'Analyse des Bornes de Recharge")
        self.setFixedSize(800, 300)

        # Simulation chargement données
        self.charger_donnees()

        layout = QVBoxLayout()
        title = QLabel("Menu Principal Analysis")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 20px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

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

        footer = QLabel("D'autres analyses seront ajoutées ultérieurement...")
        footer.setStyleSheet("color: gray; font-size: 10px;")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)

        self.setLayout(layout)

    def charger_donnees(self):
        self.df = pd.read_csv("IRVE_clean_FINAL.csv")
        self.coords = self.df[['lon', 'lat']].dropna()

    def open_kmeans(self):
        self.km_win = KmeansPage(self.coords, self.df)
        self.km_win.show()

    def open_dbscan(self):
        self.db_win = DbscanPage(self.coords, self.df)
        self.db_win.show()

    def open_prediction(self):
        self.pred_win = PredictionPage(self.coords)
        self.pred_win.show()


if __name__ == "__main__":
    creation_dossier()
    app = QApplication(sys.argv)
    hub = MainHub()
    hub.show()
    sys.exit(app.exec_())