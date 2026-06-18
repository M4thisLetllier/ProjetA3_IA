import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os
import pandas as pd

def analyse_globale_kmeans(coords, k_max=15, dossier_sauvegarde="graphe"):
    """
    Exécute les 4 métriques de K-Means, calcule un score global,
    et affiche une planche de 4 graphiques.
    """
    print(f"Lancement de l'analyse globale de k=2 à k={k_max}...")

    valeurs_k = list(range(2, k_max + 1))
    inerties = []
    silhouettes = []
    calinskis = []
    davies_vals = []

    # 1. Calcul de toutes les métriques pour chaque K
    taille_echantillon = min(2000, len(coords))  # Optimisation pour la Silhouette

    for k in valeurs_k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(coords)
        labels = kmeans.labels_

        inerties.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(coords, labels, sample_size=taille_echantillon, random_state=42))
        calinskis.append(calinski_harabasz_score(coords, labels))
        davies_vals.append(davies_bouldin_score(coords, labels))

    # 2. Identification des meilleurs scores individuels (pour les points rouges)
    meilleur_k_sil = valeurs_k[np.argmax(silhouettes)]
    meilleur_k_cal = valeurs_k[np.argmax(calinskis)]
    meilleur_k_dav = valeurs_k[np.argmin(davies_vals)]

    # 3. CALCUL DU SCORE COMPOSITE GLOBAL (Le "Juge de Paix")
    # Fonction locale pour ramener une liste de valeurs entre 0 (pire) et 1 (meilleur)
    def normaliser(tableau, inverser=False):
        t_min, t_max = np.min(tableau), np.max(tableau)
        if t_max == t_min: return np.zeros(len(tableau))
        norm = (tableau - t_min) / (t_max - t_min)
        return 1 - norm if inverser else norm

    # On normalise les 3 scores (l'inertie n'est pas incluse car elle descend toujours mathématiquement)
    norm_sil = normaliser(silhouettes, inverser=False)
    norm_cal = normaliser(calinskis, inverser=False)
    norm_dav = normaliser(davies_vals, inverser=True)  # True car Davies cherche le minimum

    # Le score final est la somme des 3 notes (sur 3 points maximum)
    scores_globaux = norm_sil + norm_cal + norm_dav
    meilleur_k_global = valeurs_k[np.argmax(scores_globaux)]

    print(f"-> Analyse terminée. Le meilleur nombre de clusters global calculé est : k={meilleur_k_global}")

    # 4. CRÉATION DES 4 GRAPHIQUES
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Analyse K-Means - Meilleur K Global estimé : {meilleur_k_global}", fontsize=16, fontweight='bold',
                 color='#2c3e50')

    # Graphique 1 : Méthode du Coude (Inertie)
    axes[0, 0].plot(valeurs_k, inerties, marker='o', linestyle='-', color='#3498db')
    axes[0, 0].set_title("Méthode du Coude (Inertie - Plus bas = meilleur)")
    axes[0, 0].set_xlabel("Nombre de clusters (k)")
    # On met un marqueur spécifique pour montrer où le score global a tranché sur l'inertie
    idx_global = valeurs_k.index(meilleur_k_global)
    axes[0, 0].plot(meilleur_k_global, inerties[idx_global], marker='*', color='black', markersize=12,
                    label="Choix Global")
    axes[0, 0].legend()

    # Graphique 2 : Silhouette
    axes[0, 1].plot(valeurs_k, silhouettes, marker='o', linestyle='-', color='#2ecc71')
    axes[0, 1].plot(meilleur_k_sil, np.max(silhouettes), 'ro', markersize=10, label=f"Max local (k={meilleur_k_sil})")
    axes[0, 1].set_title("Score de Silhouette (Plus haut = meilleur)")
    axes[0, 1].legend()

    # Graphique 3 : Calinski-Harabasz
    axes[1, 0].plot(valeurs_k, calinskis, marker='o', linestyle='-', color='#f39c12')
    axes[1, 0].plot(meilleur_k_cal, np.max(calinskis), 'ro', markersize=10, label=f"Max local (k={meilleur_k_cal})")
    axes[1, 0].set_title("Indice de Calinski-Harabasz (Plus haut = meilleur)")
    axes[1, 0].legend()

    # Graphique 4 : Davies-Bouldin
    axes[1, 1].plot(valeurs_k, davies_vals, marker='o', linestyle='-', color='#9b59b6')
    axes[1, 1].plot(meilleur_k_dav, np.min(davies_vals), 'ro', markersize=10, label=f"Min local (k={meilleur_k_dav})")
    axes[1, 1].set_title("Indice de Davies-Bouldin (Plus bas = meilleur)")
    axes[1, 1].legend()

    # Finitions visuelles pour tous les graphiques
    for ax in axes.flatten():
        ax.set_xticks(valeurs_k)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()

    # Sauvegarde optionnelle
    os.makedirs(dossier_sauvegarde, exist_ok=True)
    chemin_sauvegarde = os.path.join(dossier_sauvegarde, "analyse_multicriteres_kmeans.png")
    plt.savefig(chemin_sauvegarde, dpi=150, bbox_inches="tight")
    plt.close()  # On ferme pour éviter l'affichage direct dans la console

    # On retourne le meilleur K calculé et le chemin de l'image pour l'afficher dans PyQt
    return meilleur_k_global, chemin_sauvegarde

if __name__ == "__main__":
    df = pd.read_csv("../IRVE_clean_FINAL.csv")
    coords = df[['lon', 'lat']].dropna()
    print("debut annalyse")
    retour =analyse_globale_kmeans(coords,dossier_sauvegarde="./")
    print(retour)
