# PovertyMap : application d'inférence du sujet E4

Taux de pauvreté par pays, mesures réelles et estimations (sujet E4, organisation d'aide au développement).

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
`app/models/` sous la forme `model_E4.joblib`, committé tel quel : l'application
le charge directement au lancement.

Modèle de la correction : forêt aléatoire (200 arbres, feuille 5), correction E4.

## Déposer un autre modèle

La barre latérale accepte le dépôt de n'importe quel joblib similaire
(bouton « Déposer un modèle joblib »). Le modèle déposé remplace celui de la
correction pour toute la session ; un contrôle de conformité vérifie
`predict` / `predict_proba` et une prévision sur une ligne du contrat avant
de l'activer, et un bouton permet de revenir au modèle de la correction.

## Contrat du modèle

| Colonne | Rôle |
| --- | --- |
| NY.GDP.PCAP.PP.CD ... EN.POP.DNST | les 11 indicateurs du panel (manquants acceptés) |
| region | région Banque mondiale (jamais vue acceptée) |
| iso3 | code pays ISO 3 (jamais vu accepté) |

Sortie : taux de pauvreté au seuil de 3,00 $/jour (PPA 2021), en points,
borné [0, 100]. L'application nécessite le panel du sujet :
`data/E4-poverty/panel_E4.csv` (à côté du dossier app/, déjà dans le dépôt).

## Mode contrôle

La barre latérale propose un mode contrôle qui rejoue les tests
d'acceptation du sujet sur le modèle chargé : ligne du contrat, valeurs
manquantes ou jamais vues, valeurs extrêmes, vitesse (100 lignes en moins
d'une seconde).
