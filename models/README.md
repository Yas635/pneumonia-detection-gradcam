# Modèle

Le fichier du modèle entraîné (`DenseNet_FineTuned.h5`) n'est pas versionné : il est trop lourd pour un dépôt Git classique.

Pour lancer l'application :

1. Entraînez le modèle (DenseNet121 avec une tête de classification Normal / Pneumonie), ou récupérez votre propre fichier `.h5`.
2. Placez-le dans ce dossier sous le nom `DenseNet_FineTuned.h5`, ou indiquez son chemin avec la variable d'environnement `MODEL_PATH`.

Sans modèle, l'application démarre quand même et affiche un message d'erreur clair lors d'une prédiction.
