import os
import numpy as np
import joblib
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit
from PyQt5.QtCore import Qt

# Importations nécessaires pour calculer les métriques réelles du modèle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RegressionPage(QWidget):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.model = None  # Stockage du modèle en mémoire
        
        self.setWindowTitle("Besoin 4 : Estimation de la Puissance Nominale")
        self.setFixedSize(650, 480)
        
        layout = QVBoxLayout()
        
        title_label = QLabel("Outil d'Évaluation Réelle du Modèle (Besoin 4)")
        title_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #2e7d32; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Bouton d'action
        self.btn_run = QPushButton("Calculer les performances réelles du modèle (.pkl)")
        self.btn_run.setFixedHeight(40)
        self.btn_run.setStyleSheet("font-weight: bold; background-color: #aed581;")
        self.btn_run.clicked.connect(self.gerer_modele)
        layout.addWidget(self.btn_run)
        
        # Zone de texte pour les vrais résultats
        self.txt_resultats = QTextEdit()
        self.txt_resultats.setReadOnly(True)
        self.txt_resultats.setStyleSheet("background-color: #f1f8e9; font-family: Consolas, Courier; font-size: 11px; border: 1px solid #c5e1a5;")
        layout.addWidget(self.txt_resultats)
        
        self.setLayout(layout)

    def gerer_modele(self):
        # Nom exact exporté par machinelearning4.py
        nom_fichier_modele = "modele_prediction_puissance.pkl" 
        
        # 1. Vérification si déjà en mémoire vive
        if self.model is not None:
            self.txt_resultats.append("\n [INFO] Modèle déjà présent en mémoire. Recalcul instantané...")
            self.calculer_et_afficher_metriques()
            return

        # 2. Vérification de l'existence physique du fichier .pkl
        if os.path.exists(nom_fichier_modele):
            self.txt_resultats.setText(f" Chargement du fichier '{nom_fichier_modele}'...")
            try:
                self.model = joblib.load(nom_fichier_modele) 
                self.txt_resultats.append("✓ Modèle Random Forest chargé avec succès !")
                self.calculer_et_afficher_metriques()
            except Exception as e:
                self.txt_resultats.append(f"❌ Erreur lors du chargement : {str(e)}")
        else:
            self.txt_resultats.setText(
                f"❌ Erreur : Le fichier '{nom_fichier_modele}' est introuvable.\n\n"
                "Veuillez d'abord exécuter votre script 'machinelearning4.py' "
                "afin de générer et sauvegarder le modèle entraîné."
            )

    def calculer_et_afficher_metriques(self):
        try:
            self.txt_resultats.append(" Calcul des indicateurs de performance en cours sur la base de données...")
            
            # Reprise stricte des filtres et colonnes de votre script machinelearning4.py
            TARGET = "puissance_nominale"
            df_clean = self.df.dropna(subset=[TARGET, "lon", "lat"])
            
            BOOL_COLS = [
                "prise_type_ef", "prise_type_2", "prise_type_combo_ccs", 
                "prise_type_chademo", "prise_type_autre", "gratuit", "paiement_acte", 
                "paiement_cb", "paiement_autre", "reservation", "station_deux_roues", 
                "cable_t2_attache"
            ]
            CAT_COLS = ["condition_acces", "accessibilite_pmr", "implantation_station"]
            NUM_COLS = ["nbre_pdc", "lon", "lat"]
            
            X = df_clean[BOOL_COLS + CAT_COLS + NUM_COLS]
            y = df_clean[TARGET]
            
            # 3. Extraction des prédictions issues DIRECTEMENT du modèle
            y_pred = self.model.predict(X)
            
            # 4. Calcul des vraies métriques mathématiques
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            
            # Récupération de paramètres internes du modèle pour prouver la lecture
            nb_arbres = self.model.named_steps['regressor'].n_estimators
            
            # 5. Construction dynamique de la chaîne de texte
            resultats_format_texte = (
                "\n" + "="*55 + "\n"
                "  RÉSULTATS EXTRAITS DU MODÈLE ENREGISTRÉ\n"
                f" Type de modèle : Random Forest Regressor ({nb_arbres} arbres)\n"
                + "="*55 + "\n\n"
                "  INDICATEURS CALCULÉS SUR VOS DONNÉES :\n"
                " -----------------------------------------------------\n"
                f" • Erreur Absolue Moyenne (MAE) : {mae:.2f} kW\n"
                f"   (En moyenne, le modèle se trompe de {mae:.2f} kW par borne)\n\n"
                f" • Racine de l'Erreur (RMSE)    : {rmse:.2f} kW\n\n"
                f" • Coefficient de dét. (R²)    : {r2:.4f}\n"
                f"   (Le modèle explique {r2*100:.2f}% de la variance de la puissance)\n"
                " -----------------------------------------------------\n"
                "  STATUT : Évaluation dynamique effectuée avec succès."
            )
            self.txt_resultats.append(resultats_format_texte)
            
        except Exception as e:
            self.txt_resultats.append(f"\n❌ Erreur lors de l'extraction des performances : {str(e)}")