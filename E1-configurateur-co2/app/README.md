# ConfigCo2 : application d'inférence du sujet E1

Anticiper le malus CO2 d'une configuration avant la commande (sujet E1, courtier en leasing).

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
`app/models/` sous la forme `model_E1.joblib`, committé tel quel : l'application
le charge directement au lancement.

Modèle de la correction : forêt aléatoire (60 arbres), correction E1.

## Déposer un autre modèle

La barre latérale accepte le dépôt de n'importe quel joblib similaire
(bouton « Déposer un modèle joblib »). Le modèle déposé remplace celui de la
correction pour toute la session ; un contrôle de conformité vérifie
`predict` / `predict_proba` et une prévision sur une ligne du contrat avant
de l'activer, et un bouton permet de revenir au modèle de la correction.

## Contrat du modèle

| Colonne | Rôle | Plage |
| --- | --- | --- |
| mass_kg | masse prête à rouler, pleins faits | 500-4000 kg |
| engine_power_kw | puissance maximale | kW |
| engine_capacity_cm3 | cylindrée | cm3 |
| fuel_type | essence, diesel, hybride essence, hybride diesel | 4 valeurs |
| hybrid | motorisation hybride (déduit du carburant) | booléen |
| cylinder_count | nombre de cylindres | entier |
| make | marque (jamais vue acceptée) | texte |

Sortie : CO2 WLTP en g/km, borné au contrat [50, 500]. Le malus de l'exercice
(seuil 118 g/km, 95 EUR par gramme) est calculé par l'application.

## Mode contrôle

La barre latérale propose un mode contrôle qui rejoue les tests
d'acceptation du sujet sur le modèle chargé : ligne du contrat, valeurs
manquantes ou jamais vues, valeurs extrêmes, vitesse (100 lignes en moins
d'une seconde).
