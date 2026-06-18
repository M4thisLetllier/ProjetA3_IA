import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Imports scikit-learn pour l'entraînement
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QMessageBox, QFrame, QDialog, QTextEdit,
                             QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QGroupBox)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

# Importation de votre fichier de configuration
from configuration import DOSSIER_MODELES, DOSSIER_GRAPHE_B4, creation_dossier

# =============================================================================
# DICTIONNAIRE DES EXPLICATIONS MÉTIER (VISIONNEUSE RÉGRESSION)
# =============================================================================
EXPLICATIONS = {
    "graphe_regression_prediction_vs_realite.png": {
        "titre": "Performances de l'IA : Prédiction vs Réalité",
        "texte": (
            "<b>Que montre ce graphique ?</b><br>"
            "Ce nuage de points compare la puissance théorique prédite par l'IA (en ordonnée) avec "
            "la puissance réelle de la borne sur le terrain (en abscisse).<br><br>"
            "<b>Analyse :</b><br>"
            "La ligne rouge en pointillés représente la <b>prédiction parfaite</b> (où la prédiction est exactement égale à la réalité). "
            "Plus les points bleus sont regroupés et proches de cette ligne rouge, plus le modèle est précis. "
            "Les points qui s'éloignent de cette ligne représentent les erreurs d'estimation de l'algorithme."
        )
    }
}


# =============================================================================
# FENÊTRE VISIONNEUSE (Pour le graphique de régression)
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

        # Zone Image
        self.lbl_image = QLabel("Chargement de l'image...")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: white; border: 1px solid #ccc;")

        pixmap = QPixmap(self.chemin_image)
        pixmap_scale = pixmap.scaled(700, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_image.setPixmap(pixmap_scale)

        layout_principal.addWidget(self.lbl_image, stretch=7)

        # Zone Texte
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

        btn_fermer = QPushButton("Fermer")
        btn_fermer.setStyleSheet("padding: 10px; background-color: #e74c3c; color: white; font-weight: bold;")
        btn_fermer.clicked.connect(self.close)
        layout_texte.addWidget(btn_fermer)

        layout_principal.addLayout(layout_texte, stretch=3)
        self.setLayout(layout_principal)


# =============================================================================
# HUB DE PRÉDICTION DE PUISSANCE PRINCIPAL
# =============================================================================
class HubPredictionPuissance(QWidget):
    def __init__(self,df):
        super().__init__()
        self.df = df.copy()
        self.chemin_modele = os.path.join(DOSSIER_MODELES, "modele_prediction_puissance.pkl")
        self.chemin_graphique = os.path.join(DOSSIER_GRAPHE_B4, "graphe_regression_prediction_vs_realite.png")

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("IA : Prédiction de la Puissance Nominale (Besoin 4)")
        self.resize(600, 750)
        layout_principal = QVBoxLayout()

        titre = QLabel("Simulateur de Puissance de Borne")
        titre.setStyleSheet("font-size: 18px; font-weight: bold; color: #d35400;")
        titre.setAlignment(Qt.AlignCenter)
        layout_principal.addWidget(titre)

        # --- SECTION 1 : Paramètres Numériques & Géo ---
        groupe_num = QGroupBox("Caractéristiques Numériques et Géographiques")
        layout_num = QGridLayout()

        layout_num.addWidget(QLabel("Longitude :"), 0, 0)
        self.spin_lon = QDoubleSpinBox()
        self.spin_lon.setRange(-10.0, 20.0)
        self.spin_lon.setDecimals(4)
        self.spin_lon.setValue(4.8320)
        layout_num.addWidget(self.spin_lon, 0, 1)

        layout_num.addWidget(QLabel("Latitude :"), 0, 2)
        self.spin_lat = QDoubleSpinBox()
        self.spin_lat.setRange(40.0, 55.0)
        self.spin_lat.setDecimals(4)
        self.spin_lat.setValue(45.7640)
        layout_num.addWidget(self.spin_lat, 0, 3)

        layout_num.addWidget(QLabel("Nombre de PDC :"), 1, 0)
        self.spin_pdc = QSpinBox()
        self.spin_pdc.setRange(1, 50)
        self.spin_pdc.setValue(4)
        layout_num.addWidget(self.spin_pdc, 1, 1)

        groupe_num.setLayout(layout_num)
        layout_principal.addWidget(groupe_num)

        # --- SECTION 2 : Paramètres Catégoriels ---
        groupe_cat = QGroupBox("Environnement et Accès")
        layout_cat = QGridLayout()

        layout_cat.addWidget(QLabel("Condition d'accès :"), 0, 0)
        self.combo_cond = QComboBox()
        self.combo_cond.addItems(["Accès libre", "Accès réservé", "Accès payant"])
        layout_cat.addWidget(self.combo_cond, 0, 1)

        layout_cat.addWidget(QLabel("Accessibilité PMR :"), 0, 2)
        self.combo_pmr = QComboBox()
        self.combo_pmr.addItems(["Accessible non réservé", "Non accessible", "Réservé PMR", "Accessibilité inconnue"])
        layout_cat.addWidget(self.combo_pmr, 0, 3)

        layout_cat.addWidget(QLabel("Type d'implantation :"), 1, 0)
        self.combo_impl = QComboBox()
        self.combo_impl.addItems([
            "Station dédiée à la recharge rapide",
            "Voirie",
            "Parking public",
            "Parking privé à usage public",
            "Parking privé réservé à la clientèle"
        ])
        layout_cat.addWidget(self.combo_impl, 1, 1, 1, 3)  # S'étend sur plusieurs colonnes

        groupe_cat.setLayout(layout_cat)
        layout_principal.addWidget(groupe_cat)

        # --- SECTION 3 : Booléens (Équipements & Services) ---
        groupe_bool = QGroupBox("Équipements et Services")
        layout_bool = QGridLayout()

        self.checkboxes = {
            "prise_type_ef": QCheckBox("Prise EF"),
            "prise_type_2": QCheckBox("Prise Type 2"),
            "prise_type_combo_ccs": QCheckBox("Combo CCS"),
            "prise_type_chademo": QCheckBox("CHAdeMO"),
            "prise_type_autre": QCheckBox("Autre Prise"),
            "cable_t2_attache": QCheckBox("Câble T2 attaché"),
            "gratuit": QCheckBox("Gratuit"),
            "paiement_acte": QCheckBox("Paiement Acte"),
            "paiement_cb": QCheckBox("Paiement CB"),
            "paiement_autre": QCheckBox("Autre Paiement"),
            "reservation": QCheckBox("Réservation possible"),
            "station_deux_roues": QCheckBox("Station 2 Roues")
        }

        row, col = 0, 0
        for key, cb in self.checkboxes.items():
            layout_bool.addWidget(cb, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1

        # Sélection par défaut typique d'une borne rapide
        self.checkboxes["prise_type_combo_ccs"].setChecked(True)
        self.checkboxes["paiement_cb"].setChecked(True)

        groupe_bool.setLayout(layout_bool)
        layout_principal.addWidget(groupe_bool)

        layout_principal.addWidget(QFrame(frameShape=QFrame.HLine))

        # --- SECTION 4 : Boutons d'Action & Résultat ---
        self.btn_predire = QPushButton("Prédire la Puissance Théorique")
        self.btn_predire.setStyleSheet(
            "background-color: #d35400; color: white; padding: 12px; font-size: 14px; font-weight: bold;")
        self.btn_predire.clicked.connect(self.faire_prediction)
        layout_principal.addWidget(self.btn_predire)

        self.lbl_resultat = QLabel("En attente des données...")
        self.lbl_resultat.setAlignment(Qt.AlignCenter)
        self.lbl_resultat.setStyleSheet(
            "font-size: 16px; margin: 10px; color: #333; padding: 10px; border: 2px dashed #ccc;")
        layout_principal.addWidget(self.lbl_resultat)

        self.btn_graphique = QPushButton("Afficher le Graphique des Performances (Prédit vs Réel)")
        self.btn_graphique.setStyleSheet("background-color: #2980b9; color: white; padding: 10px;")
        self.btn_graphique.clicked.connect(self.afficher_graphique)
        layout_principal.addWidget(self.btn_graphique)

        self.setLayout(layout_principal)

    # =========================================================================
    # LOGIQUE DE PRÉDICTION & ENTRAÎNEMENT
    # =========================================================================
    def faire_prediction(self):
        if not os.path.exists(self.chemin_modele):
            reponse = QMessageBox.question(
                self, "Modèle introuvable",
                "Le modèle de Régression n'existe pas encore. L'entraînement va démarrer automatiquement.\nVoulez-vous continuer ?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reponse == QMessageBox.Yes:
                succes = self.entrainer_modele()
                if not succes:
                    return
            else:
                return

        # Construction du dictionnaire d'entrée
        caracteristiques = {
            "lon": self.spin_lon.value(),
            "lat": self.spin_lat.value(),
            "nbre_pdc": self.spin_pdc.value(),
            "condition_acces": self.combo_cond.currentText(),
            "accessibilite_pmr": self.combo_pmr.currentText(),
            "implantation_station": self.combo_impl.currentText()
        }

        for key, cb in self.checkboxes.items():
            caracteristiques[key] = 1 if cb.isChecked() else 0

        # Chargement et Prédiction
        try:
            modele = joblib.load(self.chemin_modele)
            df_input = pd.DataFrame([caracteristiques])
            prediction = modele.predict(df_input)[0]

            self.lbl_resultat.setText(
                f"Puissance théorique estimée par l'IA :<br><b style='color:#d35400; font-size: 22px;'>{prediction:.2f} kW</b>")
            self.lbl_resultat.setStyleSheet(
                "font-size: 16px; margin: 10px; color: #333; padding: 10px; border: 2px solid #d35400; background-color: #fdf2e9;")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la prédiction : {str(e)}")

    def entrainer_modele(self):

        self.btn_predire.setText("Entraînement du modèle de régression... Patientez...")
        self.btn_predire.setEnabled(False)
        self.lbl_resultat.setText("Optimisation de la forêt aléatoire en cours...")
        QApplication.processEvents()

        try:
            # 1. Chargement données
            TARGET = "puissance_nominale"
            self.df = self.df.dropna(subset=[TARGET, "lon", "lat"])

            BOOL_COLS = list(self.checkboxes.keys())
            CAT_COLS = ["condition_acces", "accessibilite_pmr", "implantation_station"]
            NUM_COLS = ["nbre_pdc", "lon", "lat"]

            X = self.df[BOOL_COLS + CAT_COLS + NUM_COLS]
            y = self.df[TARGET]

            # 2. Split & Pipeline (Pas de 'stratify' pour la régression)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), NUM_COLS),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS),
                    ('bool', 'passthrough', BOOL_COLS)
                ]
            )

            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
            ])

            # 3. Entraînement (GridSearchCV avec R2)
            param_grid = {'regressor__n_estimators': [50], 'regressor__max_depth': [10, 20]}
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            # 4. Évaluation & Graphique
            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            plt.figure(figsize=(9, 7))
            plt.scatter(y_test, y_pred, alpha=0.3, color='#185FA5', s=12)
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            plt.plot(lims, lims, color='red', linestyle='--', lw=2, label="Prédiction Parfaite (Y = X)")
            plt.title(f"Validation B4 : Prédit vs Réel (R² = {r2:.2f})", fontsize=12, fontweight='bold')
            plt.xlabel('Valeurs Réelles (kW)')
            plt.ylabel('Valeurs Prédites (kW)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.gca().spines[['top', 'right']].set_visible(False)
            plt.tight_layout()

            # Sauvegardes
            plt.savefig(self.chemin_graphique, bbox_inches="tight")
            plt.close()
            joblib.dump(best_model, self.chemin_modele)

            # 5. Restauration de l'UI
            self.btn_predire.setText("Prédire la Puissance Théorique")
            self.btn_predire.setEnabled(True)
            self.lbl_resultat.setText("Modèle de régression généré avec succès ! Prêt à calculer.")
            QMessageBox.information(self, "Succès", f"Entraînement terminé !\nCoefficient R² du modèle : {r2:.2f}")
            return True

        except Exception as e:
            self.btn_predire.setText("Prédire la Puissance Théorique")
            self.btn_predire.setEnabled(True)
            QMessageBox.critical(self, "Erreur fatale", f"L'entraînement a échoué :\n{str(e)}")
            return False

    def afficher_graphique(self):
        if not os.path.exists(self.chemin_graphique):
            reponse_graphique = QMessageBox.question(
                    self, "Graphique Introuvable",
                    "Le graphique de la regression n'existe pas encore. Le chargement du modele et le tracet du graphique vont commencé automatiquement.\nVoulez-vous continuer ?",
                    QMessageBox.Yes | QMessageBox.No
                )
            if reponse_graphique == QMessageBox.Yes:
                if not os.path.exists(self.chemin_modele):
                    reponse_modele = QMessageBox.question(
                        self, "Modèle introuvable",
                        "Le modèle de Régression n'existe pas encore. L'entraînement va démarrer automatiquement.\nVoulez-vous continuer ?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reponse_modele == QMessageBox.Yes:
                        succes = self.entrainer_modele()
                        if not succes:
                            return
                    else:
                        return
                try:
                    modele = joblib.load(self.chemin_modele)
                    y_test = self.df["puissance_nominale"]
                    # Listes exactes validées par ton script de justification
                    BOOL_COLS = [
                        "prise_type_ef", "prise_type_2", "prise_type_combo_ccs",
                        "prise_type_chademo", "prise_type_autre", "gratuit", "paiement_acte",
                        "paiement_cb", "paiement_autre", "reservation", "station_deux_roues",
                        "cable_t2_attache"
                    ]

                    CAT_COLS = ["condition_acces", "accessibilite_pmr", "implantation_station"]

                    NUM_COLS = ["nbre_pdc", "lon", "lat"]

                    X = self.df[BOOL_COLS + CAT_COLS + NUM_COLS]
                    y_pred = modele.predict(X)
                    plt.figure(figsize=(9, 7))
                    # Tracé des points Réel vs Prédit
                    plt.scatter(y_test, y_pred, alpha=0.3, color='#185FA5', s=12)

                    # Ajout de la ligne idéale Y = X (Prédiction parfaite)
                    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
                    plt.plot(lims, lims, color='red', linestyle='--', lw=2, label="Prédiction Parfaite (Y = X)")

                    plt.title('Validation du Modèle B4 : Puissance Prédite vs Puissance Réelle', fontsize=12,
                              fontweight='bold')
                    plt.xlabel('Valeurs Réelles (kW)', fontsize=10)
                    plt.ylabel('Valeurs Prédites (kW)', fontsize=10)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.gca().spines[['top', 'right']].set_visible(False)
                    plt.tight_layout()

                    # Sauvegarde du graphique dans le dossier 'besoin4'
                    plt.savefig(self.chemin_graphique, dpi=150)
                except Exception as e:
                    QMessageBox.critical(self, "Erreur", f"Erreur lors de l'ouverture du modele : {str(e)}")
            else :
                return

        nom_fichier = "graphe_regression_prediction_vs_realite.png"
        self.viewer = ViewerGraphe(self.chemin_graphique, nom_fichier)
        self.viewer.exec_()

