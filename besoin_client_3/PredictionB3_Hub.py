import sys
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Imports scikit-learn pour l'entraînement
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QMessageBox, QFrame, QDialog, QTextEdit,
                             QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QGroupBox)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

# Importation de votre fichier de configuration
from configuration import DOSSIER_MODELES, DOSSIER_GRAPHE_B3, creation_dossier

# =============================================================================
# DICTIONNAIRE DES EXPLICATIONS MÉTIER (VISIONNEUSE)
# =============================================================================
EXPLICATIONS = {
    "matrice_confusion_implantation.png": {
        "titre": "Performances de l'IA : Matrice de Confusion",
        "texte": (
            "<b>Que montre ce graphique ?</b><br>"
            "La matrice de confusion compare les choix faits par l'Intelligence Artificielle "
            "(Prédiction) avec la réalité du terrain, sur un jeu de test qu'elle n'avait jamais vu.<br><br>"
            "<b>Analyse :</b><br>"
            "La diagonale principale (de haut en gauche à en bas à droite) représente les <b>bonnes prédictions</b>. "
            "Plus la couleur est foncée, plus le modèle est sûr et performant sur cette catégorie. "
            "Les cases en dehors de cette diagonale vous montrent où le modèle hésite "
            "(ex: confondre un parking privé et un parking public)."
        )
    }
}


# =============================================================================
# FENÊTRE VISIONNEUSE (Pour la matrice de confusion)
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
# HUB DE PRÉDICTION PRINCIPAL
# =============================================================================
class HubPrediction(QWidget):
    def __init__(self,df):
        super().__init__()
        creation_dossier()  # S'assure que les dossiers existent

        self.df = df.copy()
        self.chemin_modele = os.path.join(DOSSIER_MODELES, "modele_prediction_implantation.pkl")
        self.chemin_matrice = os.path.join(DOSSIER_GRAPHE_B3, "matrice_confusion_implantation.png")

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("IA : Prédiction du Type d'Implantation (Besoin 3)")
        self.resize(550, 750)
        layout_principal = QVBoxLayout()

        titre = QLabel("Simulateur d'Implantation de Borne")
        titre.setStyleSheet("font-size: 18px; font-weight: bold; color: #2980b9;")
        titre.setAlignment(Qt.AlignCenter)
        layout_principal.addWidget(titre)

        # --- SECTION 1 : Paramètres Numériques & Géo ---
        groupe_num = QGroupBox("Caractéristiques Numériques et Géographiques")
        layout_num = QGridLayout()

        layout_num.addWidget(QLabel("Longitude :"), 0, 0)
        self.spin_lon = QDoubleSpinBox();
        self.spin_lon.setRange(-10.0, 20.0);
        self.spin_lon.setDecimals(4);
        self.spin_lon.setValue(4.8320)
        layout_num.addWidget(self.spin_lon, 0, 1)

        layout_num.addWidget(QLabel("Latitude :"), 0, 2)
        self.spin_lat = QDoubleSpinBox();
        self.spin_lat.setRange(40.0, 55.0);
        self.spin_lat.setDecimals(4);
        self.spin_lat.setValue(45.7640)
        layout_num.addWidget(self.spin_lat, 0, 3)

        layout_num.addWidget(QLabel("Puissance (kW) :"), 1, 0)
        self.spin_puissance = QDoubleSpinBox();
        self.spin_puissance.setRange(0.0, 500.0);
        self.spin_puissance.setValue(150.0)
        layout_num.addWidget(self.spin_puissance, 1, 1)

        layout_num.addWidget(QLabel("Nombre de PDC :"), 1, 2)
        self.spin_pdc = QSpinBox();
        self.spin_pdc.setRange(1, 50);
        self.spin_pdc.setValue(4)
        layout_num.addWidget(self.spin_pdc, 1, 3)

        groupe_num.setLayout(layout_num)
        layout_principal.addWidget(groupe_num)

        # --- SECTION 2 : Paramètres Catégoriels ---
        groupe_cat = QGroupBox("Conditions d'Accès")
        layout_cat = QHBoxLayout()

        layout_cat.addWidget(QLabel("Condition d'accès :"))
        self.combo_cond = QComboBox()
        self.combo_cond.addItems(["Accès libre", "Accès réservé", "Accès payant"])
        layout_cat.addWidget(self.combo_cond)

        layout_cat.addWidget(QLabel("PMR :"))
        self.combo_pmr = QComboBox()
        self.combo_pmr.addItems(["Accessible non réservé", "Non accessible", "Réservé PMR", "Accessibilité inconnue"])
        layout_cat.addWidget(self.combo_pmr)

        groupe_cat.setLayout(layout_cat)
        layout_principal.addWidget(groupe_cat)

        # --- SECTION 3 : Booléens (Équipements & Services) ---
        groupe_bool = QGroupBox("Équipements et Services")
        layout_bool = QGridLayout()

        # Dictionnaire pour créer facilement les CheckBoxes
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

        # Disposition en grille 3 colonnes
        row, col = 0, 0
        for key, cb in self.checkboxes.items():
            layout_bool.addWidget(cb, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1

        # Sélection par défaut pour correspondre à l'exemple ultra-rapide
        self.checkboxes["prise_type_2"].setChecked(True)
        self.checkboxes["prise_type_combo_ccs"].setChecked(True)
        self.checkboxes["paiement_acte"].setChecked(True)
        self.checkboxes["paiement_cb"].setChecked(True)
        self.checkboxes["cable_t2_attache"].setChecked(True)

        groupe_bool.setLayout(layout_bool)
        layout_principal.addWidget(groupe_bool)

        layout_principal.addWidget(QFrame(frameShape=QFrame.HLine))

        # --- SECTION 4 : Boutons d'Action & Résultat ---
        self.btn_predire = QPushButton("Lancer la Prédiction")
        self.btn_predire.setStyleSheet(
            "background-color: #27ae60; color: white; padding: 12px; font-size: 14px; font-weight: bold;")
        self.btn_predire.clicked.connect(self.faire_prediction)
        layout_principal.addWidget(self.btn_predire)

        self.lbl_resultat = QLabel("En attente des données...")
        self.lbl_resultat.setAlignment(Qt.AlignCenter)
        self.lbl_resultat.setStyleSheet(
            "font-size: 16px; margin: 10px; color: #333; padding: 10px; border: 2px dashed #ccc;")
        layout_principal.addWidget(self.lbl_resultat)

        self.btn_matrice = QPushButton("Afficher la Matrice de Confusion du Modèle")
        self.btn_matrice.setStyleSheet("background-color: #8e44ad; color: white; padding: 10px;")
        self.btn_matrice.clicked.connect(self.afficher_matrice)
        layout_principal.addWidget(self.btn_matrice)

        self.setLayout(layout_principal)

    # =========================================================================
    # LOGIQUE DE PRÉDICTION & ENTRAÎNEMENT
    # =========================================================================
    def faire_prediction(self):
        # 1. Vérification de l'existence du modèle
        if not os.path.exists(self.chemin_modele):
            reponse = QMessageBox.question(
                self, "Modèle introuvable",
                "Le modèle d'IA n'existe pas encore. L'entraînement va démarrer automatiquement et peut prendre quelques minutes.\nVoulez-vous continuer ?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reponse == QMessageBox.Yes:
                succes = self.entrainer_modele()
                if not succes:
                    return
            else:
                return

        # 2. Construction du dictionnaire d'entrée
        caracteristiques = {
            "lon": self.spin_lon.value(),
            "lat": self.spin_lat.value(),
            "puissance_nominale": self.spin_puissance.value(),
            "nbre_pdc": self.spin_pdc.value(),
            "condition_acces": self.combo_cond.currentText(),
            "accessibilite_pmr": self.combo_pmr.currentText(),
        }

        # Ajout des valeurs booléennes (1 si coché, 0 sinon)
        for key, cb in self.checkboxes.items():
            caracteristiques[key] = 1 if cb.isChecked() else 0

        # 3. Chargement et Prédiction
        try:
            modele = joblib.load(self.chemin_modele)
            df_input = pd.DataFrame([caracteristiques])
            prediction = modele.predict(df_input)[0]

            self.lbl_resultat.setText(
                f"L'IA prédit une implantation de type :<br><b style='color:#e74c3c; font-size: 20px;'>{prediction}</b>")
            self.lbl_resultat.setStyleSheet(
                "font-size: 16px; margin: 10px; color: #333; padding: 10px; border: 2px solid #27ae60; background-color: #e8f8f5;")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la prédiction : {str(e)}")

    def entrainer_modele(self):
        """Réplique la logique de machinelearning.py"""
        self.btn_predire.setText("Entraînement de l'IA en cours... Patientez...")
        self.btn_predire.setEnabled(False)
        self.lbl_resultat.setText("Calculs de la forêt aléatoire en cours...")
        QApplication.processEvents()  # Force l'interface à se mettre à jour

        try:
            # 1. Chargement données
            TARGET = "implantation_station"

            BOOL_COLS = list(self.checkboxes.keys())
            CAT_COLS = ["condition_acces", "accessibilite_pmr"]
            NUM_COLS = ["puissance_nominale", "nbre_pdc", "lon", "lat"]

            X = self.df[BOOL_COLS + CAT_COLS + NUM_COLS]
            y = self.df[TARGET]

            # 2. Split & Pipeline
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), NUM_COLS),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS),
                    ('bool', 'passthrough', BOOL_COLS)
                ]
            )

            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
            ])

            # 3. Entraînement
            param_grid = {'classifier__n_estimators': [50], 'classifier__max_depth': [10, 20]}  # Réduit pour l'UI
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            # 4. Sauvegardes
            joblib.dump(best_model, self.chemin_modele)

            y_pred = best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)

            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_,
                        yticklabels=best_model.classes_)
            plt.title('Matrice de Confusion : Prédiction des Implantations')
            plt.ylabel('Réalité')
            plt.xlabel('Prédiction')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(self.chemin_matrice, bbox_inches="tight")
            plt.close()

            # 5. Restauration de l'UI
            self.btn_predire.setText("Lancer la Prédiction")
            self.btn_predire.setEnabled(True)
            self.lbl_resultat.setText("Modèle généré avec succès ! Prêt à prédire.")
            QMessageBox.information(self, "Succès", "Entraînement terminé et sauvegardé.")
            return True

        except Exception as e:
            self.btn_predire.setText("Lancer la Prédiction")
            self.btn_predire.setEnabled(True)
            QMessageBox.critical(self, "Erreur fatale", f"L'entraînement a échoué :\n{str(e)}")
            return False

    def afficher_matrice(self):
        if not os.path.exists(self.chemin_matrice):
            QMessageBox.warning(self, "Image introuvable",
                                "La matrice de confusion n'existe pas. Vous devez lancer une prédiction (et générer le modèle) au moins une fois.")
            return

        nom_fichier = "matrice_confusion_implantation.png"
        self.viewer = ViewerGraphe(self.chemin_matrice, nom_fichier)
        self.viewer.exec_()