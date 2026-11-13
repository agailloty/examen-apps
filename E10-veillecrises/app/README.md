# VeilleCrises : application d'inférence du sujet E10

Probabilité de crise bancaire l'année suivante (sujet E10, cellule macroprudentielle régionale).

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
`app/models/` sous la forme `model_E10.joblib`, committé tel quel : l'application
le charge directement au lancement.

Modèle de la correction : HistGradientBoosting équilibré, correction E10.

## Déposer un autre modèle

La barre latérale accepte le dépôt de n'importe quel joblib similaire
(bouton « Déposer un modèle joblib »). Le modèle déposé remplace celui de la
correction pour toute la session ; un contrôle de conformité vérifie
`predict` / `predict_proba` et une prévision sur une ligne du contrat avant
de l'activer, et un bouton permet de revenir au modèle de la correction.

## Contrat du modèle

| Colonne | Rôle |
| --- | --- |
| pays | un des 13 pays du jeu (jamais vu accepté) |
| inflation, exch_usd | inflation annuelle, taux de change (manquants acceptés) |
| sovereign_external_debt_usd, gdp_weighted_default | dette extérieure, défaut pondéré (manquants acceptés) |
| currency_crises (0, 1 ou 2), inflation_crises, systemic_crisis | états de crise au relevé |

Sortie : probabilité de crise bancaire à l'année t+1, classe 1. L'année
n'entre pas dans le modèle.

## Mode contrôle

La barre latérale propose un mode contrôle qui rejoue les tests
d'acceptation du sujet sur le modèle chargé : ligne du contrat, valeurs
manquantes ou jamais vues, valeurs extrêmes, vitesse (100 lignes en moins
d'une seconde).
