# AtelierRadar : application d'inférence du sujet E8

Probabilité de défaillance de la machine au dernier cycle (sujet E8, PME d'usinage CNC).

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
`app/models/` sous la forme `model_E8.joblib`, committé tel quel : l'application
le charge directement au lancement.

Modèle de la correction : forêt aléatoire équilibrée (200 arbres), correction E8.

## Déposer un autre modèle

La barre latérale accepte le dépôt de n'importe quel joblib similaire
(bouton « Déposer un modèle joblib »). Le modèle déposé remplace celui de la
correction pour toute la session ; un contrôle de conformité vérifie
`predict` / `predict_proba` et une prévision sur une ligne du contrat avant
de l'activer, et un bouton permet de revenir au modèle de la correction.

## Contrat du modèle

| Colonne | Rôle |
| --- | --- |
| Type | type de produit : L, M ou H (jamais vu accepté) |
| Air_temperature, Process_temperature | températures (K) |
| Rotational_speed | vitesse de rotation (tr/min) |
| Torque | couple (Nm) |
| Tool_wear | usure d'outil (min), manquante acceptée |

Sortie : probabilité d'au moins un des cinq modes de défaillance, classe 1.
Le pipeline embarqué ajoute puissance mécanique et écart de températures ;
l'application redéfinit cette fonction pour recharger le joblib.

## Mode contrôle

La barre latérale propose un mode contrôle qui rejoue les tests
d'acceptation du sujet sur le modèle chargé : ligne du contrat, valeurs
manquantes ou jamais vues, valeurs extrêmes, vitesse (100 lignes en moins
d'une seconde).
