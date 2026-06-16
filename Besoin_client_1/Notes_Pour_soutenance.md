1. Pour optimiser l'utilisation de la mémoire RAM, je n'ai pas chargé l'intégralité du dataset pour la carte. J'ai isolé uniquement la latitude, la longitude et la colonne implantation\_station.



2\. La Carte par Type d'Implantation (Les Clusters)C'est ici qu'on règle le problème des 70 000 points. Nous allons utiliser un MarkerCluster.  L'argument pour l'oral : "Pour le premier visuel, afficher 70 000 marqueurs indépendants aurait rendu la carte illisible et figé le navigateur. J'ai implémenté un algorithme de MarkerCluster. Cela regroupe dynamiquement les bornes par région quand on dézoome, et révèle les types d'implantation précis quand on zoome sur une ville, garantissant une navigation fluide."



3\. Si le jury te demande pourquoi tu as choisi ces couleurs, tu pourras justifier : "Le rouge attire l'œil sur les stations de recharge rapide (enjeu majeur actuel), le bleu représente le domaine public (voirie), le vert les parkings publics classiques, et les tons chauds (orange/violet) désignent le domaine privé (parkings privés).



4\. Le graphique de gauche montre que le réseau s'étend jusqu'à 400 kW pour la charge ultra-rapide. Cependant, la médiane très marquée à 22 kW prouve que le réseau IRVE français reste aujourd'hui massivement constitué de bornes de charge standard destinées au stationnement urbain de moyenne durée. C'est d'ailleurs pour cela que j'ai généré le second graphique à droite, qui fait un focus sur cette tranche de 0 à 50 kW, là où se trouve l'écrasante majorité de notre base de données.



5\. Mon premier graphique et ma médiane montrent que 50% du volume des bornes est inférieur à 22 kW. C'est la recharge urbaine du quotidien. Mais une voiture électrique doit aussi pouvoir traverser la France ! C'est pourquoi on trouve des valeurs extrêmes à 150, 250 ou 400 kW dans notre fichier. Ces bornes ultra-rapides sont minoritaires en quantité, mais majeures sur le plan stratégique. C'est exactement pour cette raison que j'ai créé une Heatmap filtrée uniquement sur ces stations de charge rapide : pour vérifier qu'elles couvrent bien nos autoroutes !



6\. J'ai simplement fait appel à la méthode .median() native de la bibliothèque Pandas sur ma colonne de puissance. Pandas s'occupe de trier la distribution et d'isoler la valeur centrale, que j'injecte ensuite dans Matplotlib pour tracer la ligne verticale de repère



7\. Les petites puissances (inférieures à 22 kW) ne suffisent pas à atteindre les 50 % de l'effectif total. C'est en rentrant dans le bloc massif des bornes de 22 kW qu'on franchit le cap de la moitié de la base de données, fixant ainsi la médiane à 22.

