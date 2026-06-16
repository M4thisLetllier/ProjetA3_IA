import os
import webbrowser
from sklearn.cluster import KMeans
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                             QSlider, QLabel, QPushButton, QMessageBox)
import matplotlib.pyplot as plt
from configuration import DOSSIER_CARTES, DOSSIER_MODELES
from PyQt5.QtCore import Qt
from Part2_Clusturing.Kmeans_module import silhouette,calinski,davies, creation_modele_Kmeans,creation_carte_kmeans


class KmeansPage(QWidget):
    def __init__(self, df_coords, parent_df):
        super().__init__()

        # --- Configuration de la fenêtre ---
        self.setWindowTitle("Gestionnaire de Clusters K-Means")
        self.setFixedSize(600, 400)

        # Chargement des données (à adapter avec votre vrai fichier CSV)
        self.coords = df_coords
        self.df = parent_df

        # --- Création des éléments de l'interface (Widgets) ---
        self.layout = QVBoxLayout()

        # Label d'instruction
        self.label_info = QLabel("Sélectionnez le nombre de clusters (k) : 5")
        self.label_info.setAlignment(Qt.AlignCenter)
        self.label_info.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        self.layout.addWidget(self.label_info)

        # Slider (de 1 à 186)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(186)
        self.slider.setValue(5)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.mettre_a_jour_label)
        self.layout.addWidget(self.slider)

        # Bouton d'ouverture / génération
        self.btn_ouvrir = QPushButton("Ouvrir la carte")
        self.btn_ouvrir.setStyleSheet("padding: 10px; font-size: 14px; background-color: #0078D7; color: white;")
        self.btn_ouvrir.clicked.connect(self.gerer_carte)
        self.layout.addWidget(self.btn_ouvrir)

        #Bonton de silhouette
        self.btn_silhouette = QPushButton("Trouver le meilleur 'k' (Silhouette)")
        self.btn_silhouette.setStyleSheet("padding: 8px; background-color: #28a745; color: white;")
        self.btn_silhouette.clicked.connect(self.calculer_meilleur_k_silhouette)
        self.layout.addWidget(self.btn_silhouette)

        # Bonton de calinski
        self.btn_calinski = QPushButton("Trouver le meilleur 'k' (Calinski)")
        self.btn_calinski.setStyleSheet("padding: 8px; background-color: #28a745; color: white;")
        self.btn_calinski.clicked.connect(self.calculer_meilleur_k_calinski)
        self.layout.addWidget(self.btn_calinski)

        # Bonton de davies
        self.btn_davies = QPushButton("Trouver le meilleur 'k' (Davies)")
        self.btn_davies.setStyleSheet("padding: 8px; background-color: #28a745; color: white;")
        self.btn_davies.clicked.connect(self.calculer_meilleur_k_davies)
        self.layout.addWidget(self.btn_davies)

        #Bonton d'affichage du coude
        self.btn_inertie = QPushButton("Afficher le graphique du Coude (Inertie)")
        self.btn_inertie.setStyleSheet("padding: 8px; background-color: #ffc107; color: black;")
        self.btn_inertie.clicked.connect(self.afficher_graphique_inertie)
        self.layout.addWidget(self.btn_inertie)

        # Application du layout
        self.setLayout(self.layout)

    def mettre_a_jour_label(self):
        """Met à jour le texte quand le slider bouge."""
        valeur = self.slider.value()
        self.label_info.setText(f"Sélectionnez le nombre de clusters (k) : {valeur}")

    def gerer_carte(self):
        """Vérifie si la carte existe, la génère si besoin, puis l'ouvre."""
        self.btn_ouvrir.setText("Génération en cours...")
        QApplication.processEvents()  # Force la mise à jour de l'interface

        k = self.slider.value()
        nom_fichier = f"carte_kmeans_k{k}.html"
        chemin_fichier = os.path.join(DOSSIER_CARTES, nom_fichier)
        chemin_modele = os.path.join(DOSSIER_MODELES, f"modele_kmeans_k{k}.pkl")
        chemin_absolu_fichier = os.path.abspath(chemin_fichier)
        chemin_absolu_modele = os.path.abspath(chemin_modele)

        # --- ÉTAPE 1 : GESTION DU MODÈLE ---
        df_kmeans = creation_modele_Kmeans(chemin_absolu_modele, self.coords, k, self.df)

        # --- ÉTAPE 2 : GÉNÉRATION DE LA CARTE ---
        # Vérification : Le fichier existe-t-il ?
        if not os.path.exists(chemin_absolu_fichier):
            try:
                creation_carte_kmeans(df_kmeans, k)
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur lors de la génération : {str(e)}")
                self.btn_ouvrir.setText("Ouvrir la carte")
                return

        # Ouverture de la carte dans le navigateur web par défaut
        self.btn_ouvrir.setText("Ouvrir la carte")
        webbrowser.open('file://' + chemin_absolu_fichier)

    def calculer_meilleur_k_silhouette(self):
        """Calcule le score de Silhouette pour k allant de 2 à 10 et garde le meilleur."""

        self.btn_silhouette.setText("Calcul en cours...")
        QApplication.processEvents()

        meilleur_k, meilleur_score = silhouette(self.coords)

        # Remise à zéro du bouton
        self.btn_silhouette.setText("Trouver le meilleur 'k' (Silhouette)")

        # On met automatiquement à jour le slider avec la meilleure valeur trouvée !
        self.slider.setValue(meilleur_k)

        QMessageBox.information(
            self,
            "Résultat",
            f"Le meilleur nombre de clusters trouvé est : {meilleur_k}\n"
            f"Score de Silhouette : {meilleur_score:.3f}\n\n"
            "(Le curseur a été mis à jour automatiquement !)"
        )

    def calculer_meilleur_k_calinski(self):
        """Trouve le meilleur 'k' très rapidement avec l'indice de Calinski-Harabasz."""
        self.btn_calinski.setText("Calcul en cours...")
        if len(self.coords) < 3:
            QMessageBox.warning(self, "Erreur", "Pas assez de données pour l'analyse.")
            return

        # On peut réutiliser le même bouton ou en créer un nouveau dans l'interface

        QApplication.processEvents()

        meilleur_k ,meilleur_score= calinski(self.coords)

        # Mise à jour automatique du curseur
        self.slider.setValue(meilleur_k)

        QMessageBox.information(
            self,
            "Résultat Rapide",
            f"Le meilleur nombre de clusters trouvé est : {meilleur_k}\n"
            f"Score de Calinski-Harabasz : {meilleur_score:.1f}\n\n"
            "(Le curseur a été mis à jour automatiquement !)"
        )
        self.btn_calinski.setText("Trouver le meilleur 'k' (Calinski)")

    def calculer_meilleur_k_davies(self):
        """Trouve le meilleur 'k' en minimisant l'indice de Davies-Bouldin."""
        self.btn_davies.setText("Calcul en cours...")

        # Si vous ajoutez un bouton dédié, vous pouvez changer son texte ici
        QApplication.processEvents()


        meilleur_k,meilleur_score = davies(self.coords)

        # Mise à jour automatique du curseur (slider)
        self.slider.setValue(meilleur_k)

        QMessageBox.information(
            self,
            "Résultat Davies-Bouldin",
            f"Le meilleur nombre de clusters trouvé est : {meilleur_k}\n"
            f"Score de Davies-Bouldin : {meilleur_score:.3f}\n"
            "(Le curseur a été mis à jour automatiquement !)"
        )
        self.btn_davies.setText("Trouver le meilleur 'k' (Davies)")

    def afficher_graphique_inertie(self):
        """Calcule l'inertie pour k de 1 à 15 et affiche la courbe avec Matplotlib."""
        if len(self.coords) < 2:
            QMessageBox.warning(self, "Erreur", "Pas assez de données.")
            return

        self.btn_inertie.setText("Génération du graphique...")
        QApplication.processEvents()

        inerties = []
        k_max = 15
        valeurs_k = range(1, k_max + 1)

        for k in valeurs_k:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.coords)
            inerties.append(kmeans.inertia_)

        self.btn_inertie.setText("Afficher le graphique du Coude (Inertie)")

        # --- Création de la fenêtre Matplotlib ---
        plt.figure(figsize=(8, 5))
        plt.plot(valeurs_k, inerties, marker='o', linestyle='-', color='b')
        plt.title("Méthode du Coude (Évolution de l'Inertie)")
        plt.xlabel("Nombre de clusters (k)")
        plt.ylabel("Inertie (Somme des distances au carré)")
        plt.xticks(valeurs_k)  # Pour forcer l'affichage des entiers sur l'axe X
        plt.grid(True, linestyle='--', alpha=0.7)

        # plt.show() va ouvrir une fenêtre Matplotlib par-dessus votre application PyQt
        plt.show()
