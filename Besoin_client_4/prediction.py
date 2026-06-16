import sys
import webbrowser
import pandas as pd

# Imports initiaux pour le Besoin 2
from Besoin_client_2.Kmeans_Hub import KmeansPage
from Besoin_client_2.DbScan_Hub import DbscanPage
from Besoin_client_2.Prediction_hub import PredictionPage

# Si vous créez des fichiers séparés pour les Besoins 3 et 4, décommentez et adaptez ces lignes :
# from Besoin_client_3.Classification_Hub import ClassificationPage
# from Besoin_client_4.Regression_Hub import RegressionPage

from configuration import creation_dossier
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QLabel, QPushButton, QMessageBox, QFrame)
from PyQt5.QtCore import Qt


class MainHub(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Panel d'Analyse des Bornes de Recharge")
        # Augmentation de la hauteur de la fenêtre (de 300 à 420) pour accueillir les nouveaux boutons
        self.setFixedSize(800, 420)

        # Chargement des données
        self.charger_donnees()

        layout = QVBoxLayout()
        title = QLabel("Menu Principal d'Analyse IRVE")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 15px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # --- SECTION : BESOIN CLIENT 2 (Clustering) ---
        self.btn_kmeans = QPushButton("Ouvrir Analyse K-MEANS (Besoin 2)")
        self.btn_kmeans.setFixedHeight(45)
        self.btn_kmeans.clicked.connect(self.open_kmeans)
        layout.addWidget(self.btn_kmeans)

        self.btn_dbscan = QPushButton("Ouvrir Analyse DBSCAN (Besoin 2)")
        self.btn_dbscan.setFixedHeight(45)
        self.btn_dbscan.clicked.connect(self.open_dbscan)
        layout.addWidget(self.btn_dbscan)

        self.btn_predict = QPushButton("Ouvrir l'Outil de Prédiction (K-Means - Besoin 2)")
        self.btn_predict.setFixedHeight(45)
        self.btn_predict.clicked.connect(self.open_prediction)
        layout.addWidget(self.btn_predict)

        # --- SECTION : BESOIN CLIENT 3 (Classification) ---
        self.btn_classification = QPushButton("Ouvrir Classification - Type d'Implantation (Besoin 3)")
        self.btn_classification.setFixedHeight(45)
        # Style légèrement bleuté pour le différencier du Besoin 2
        self.btn_classification.setStyleSheet("background-color: #e3f2fd; font-weight: bold; border: 1px solid #90caf9; border-radius: 4px;")
        self.btn_classification.clicked.connect(self.open_classification)
        layout.addWidget(self.btn_classification)

        # --- SECTION : BESOIN CLIENT 4 (Régression) ---
        self.btn_regression = QPushButton("Ouvrir Régression - Puissance Nominale (Besoin 4)")
        self.btn_regression.setFixedHeight(45)
        # Style légèrement verdâtre pour le différencier
        self.btn_regression.setStyleSheet("background-color: #f1f8e9; font-weight: bold; border: 1px solid #c5e1a5; border-radius: 4px;")
        self.btn_regression.clicked.connect(self.open_regression)
        layout.addWidget(self.btn_regression)

        footer = QLabel("Système d'Analyse et de Machine Learning IRVE — FISE3 2026")
        footer.setStyleSheet("color: gray; font-size: 10px; margin-top: 10px;")
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

    # --- Nouvelles méthodes de redirection pour les Besoins 3 et 4 ---
    def open_classification(self):
        # On passe le dataframe au cas où votre page nécessite d'extraire des listes de variables (ex: types de prises)
        self.class_win = ClassificationPage(self.df)
        self.class_win.show()

    def open_regression(self):
        self.reg_win = RegressionPage(self.df)
        self.reg_win.show()


# =============================================================================
# SQUELETTES DES NOUVELLES PAGES (À compléter ou à externaliser dans vos modules)
# =============================================================================

class ClassificationPage(QWidget):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.setWindowTitle("Besoin 3 : Prédiction du Type d'Implantation")
        self.setFixedSize(500, 350)
        
        layout = QVBoxLayout()
        label = QLabel("Interface de Classification (Random Forest)")
        label.setStyleSheet("font-size: 14px; font-weight: bold; color: #1565c0;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        info = QLabel("Ici, intégrez vos champs de saisie (booléens de prises, gratuité,\ncoordonnées GPS...) pour interroger votre modèle 'machinelearning.py'.")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)
        
        # Exemple de bouton de traitement
        self.btn_run = QPushButton("Prédire l'implantation")
        self.btn_run.setFixedHeight(40)
        layout.addWidget(self.btn_run)
        
        self.setLayout(layout)


class RegressionPage(QWidget):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.setWindowTitle("Besoin 4 : Estimation de la Puissance Nominale")
        self.setFixedSize(500, 350)
        
        layout = QVBoxLayout()
        label = QLabel("Interface de Régression (Random Forest Regressor)")
        label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2e7d32;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        info = QLabel("Ici, placez les listes déroulantes ou curseurs (type d'implantation,\nconnecteurs requis...) pour estimer la puissance en kW avec 'machinelearning4.py'.")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)
        
        # Exemple de bouton de traitement
        self.btn_run = QPushButton("Estimer la Puissance (kW)")
        self.btn_run.setFixedHeight(40)
        layout.addWidget(self.btn_run)
        
        self.setLayout(layout)


if __name__ == "__main__":
    creation_dossier()
    app = QApplication(sys.argv)
    hub = MainHub()
    hub.show()
    sys.exit(app.exec_())