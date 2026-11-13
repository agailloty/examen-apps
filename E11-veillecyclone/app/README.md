# VeilleCyclone : application d'inférence du sujet E11

Probabilité qu'un système tropical devienne un ouragan majeur (sujet E11, courtier en réassurance).

L'application suit la maquette du sujet (maquette.html à la racine du sujet) :
saisie métier, indicateurs de décision, avertissements permanents.

## Lancer l'application

Depuis la racine du sujet (le dossier qui contient `app/`) :

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

## Modèle embarqué

Le meilleur modèle entraîné par le notebook de correction est exposé dans
`app/models/` sous la forme `model_E11.joblib`, committé tel quel : l'application
le charge directement au lancement.

Modèle de la correction : régression logistique sur relevés et variations, correction E11.

## Déposer un autre modèle

La barre latérale accepte le dépôt de n'importe quel joblib similaire
(bouton « Déposer un modèle joblib »). Le modèle déposé remplace celui de la
correction pour toute la session ; un contrôle de conformité vérifie
`predict` / `predict_proba` et une prévision sur une ligne du contrat avant
de l'activer, et un bouton permet de revenir au modèle de la correction.

## Contrat du modèle

| Colonne | Rôle |
| --- | --- |
| mois | mois de formation (1-12) |
| lat0, lon0 | position du premier relevé |
| vent0, pres0 | vent (nœuds) et pression (mb) au premier relevé (manquants acceptés) |
| vent24, pres24 | relevés à 24 h, manquants si le système est mort avant |

Sortie : probabilité d'ouragan majeur (vent final d'au moins 96 nœuds),
classe 1. Le pipeline embarqué ajoute les variations à 24 h ; l'application
redéfinit cette fonction pour recharger le joblib.

## Mode contrôle

La barre latérale propose un mode contrôle qui rejoue les tests
d'acceptation du sujet sur le modèle chargé : ligne du contrat, valeurs
manquantes ou jamais vues, valeurs extrêmes, vitesse (100 lignes en moins
d'une seconde).
