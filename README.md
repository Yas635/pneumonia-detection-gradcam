# PneumoPredict : détection de pneumonie par deep learning

Application web Django qui classe des radiographies thoraciques (**Normal** / **Pneumonie**) avec un modèle **DenseNet121** adapté (fine-tuning), explique chaque prédiction avec **Grad-CAM** et génère un **rapport PDF**.

> ⚠️ Projet pédagogique réalisé à HEM Casablanca (2024/2025). Ce n'est pas un dispositif médical et il ne remplace pas l'avis d'un médecin.

## Fonctionnalités

- Envoi d'une radiographie avec le nom et l'âge du patient
- Prédiction Normal / Pneumonie par DenseNet121
- Carte de chaleur **Grad-CAM** et superposition sur l'image pour voir les zones qui ont influencé la décision
- Rapport PDF téléchargeable (image, prédiction, Grad-CAM)
- Tableau de bord : nombre de prédictions, part de pneumonies, répartition par tranche d'âge, historique

## Résultats

D'après le rapport du projet, sur l'ensemble de test : précision 92 %, rappel 89 %, F1-score 90 %.
Le rapport ne précise pas la taille ni la répartition des classes de l'ensemble de test : ces chiffres sont donc à prendre comme indicatifs.

## Modèle

DenseNet121 pré-entraîné sur ImageNet, dont les couches finales ont été remplacées par une couche dense de classification avec dropout. Entrée : image en niveaux de gris redimensionnée en 224×224, normalisée entre 0 et 1.
Le fichier `.h5` n'est pas inclus (voir [`models/README.md`](models/README.md)).

## Installation

```bash
python -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # puis adaptez les valeurs
python manage.py migrate
python manage.py runserver
```

Placez votre modèle dans `models/DenseNet_FineTuned.h5` (ou définissez `MODEL_PATH`), puis ouvrez http://127.0.0.1:8000.

Tests (sans TensorFlow ni modèle) :

```bash
python manage.py test
```

## Structure

```
pneumonia_project/   configuration Django (variables d'environnement)
predictions/
├── ml.py            chargement du modèle, prétraitement, Grad-CAM
├── views.py         prédiction, rapport PDF, tableau de bord
├── forms.py         formulaire validé (âge, taille et type de l'image)
├── models.py        modèle Submission
└── templates/       pages HTML et modèle du rapport PDF
models/              emplacement du modèle .h5 (non versionné)
```

## Limites et pistes d'amélioration

- Ensemble de données limité : le modèle peut mal généraliser à d'autres hôpitaux ou appareils.
- Grad-CAM est moins précis sur les petites zones.
- Les images et les noms envoyés sont stockés localement (`media/`, base SQLite) : à anonymiser avant tout usage réel.
- Pistes : jeu de données plus varié, Guided Grad-CAM, déploiement en ligne.

## Auteurs

Akram Ahrardi, Yassine Ahrimch, Oussama Bounjoum. Encadrement : M. Hernafi et Mme Lanbouri.

## Références

- Selvaraju et al., « Grad-CAM : Visual Explanations from Deep Networks via Gradient-based Localization », 2017
- Huang et al., « Densely Connected Convolutional Networks » (DenseNet), 2017
