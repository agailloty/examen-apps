# CreditRadar : application d'inférence du sujet E6

Probabilité de défaut d'une demande de crédit (sujet E6, banque de détail taïwanaise).

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
`app/models/` sous la forme `model_E6.joblib`, committé tel quel : l'application
le charge directement au lancement.

Modèle de la correction : forêt aléatoire équilibrée (200 arbres), correction E6.

## Déposer un autre modèle

La barre latérale accepte le dépôt de n'importe quel joblib similaire
(bouton « Déposer un modèle joblib »). Le modèle déposé remplace celui de la
correction pour toute la session ; un contrôle de conformité vérifie
`predict` / `predict_proba` et une prévision sur une ligne du contrat avant
de l'activer, et un bouton permet de revenir au modèle de la correction.

## Contrat du modèle

| Colonne | Rôle |
| --- | --- |
| LIMIT_BAL | montant demandé (dollar TW) |
| SEX, EDUCATION, MARRIAGE | variables sensibles (codes 0, 5, 6 possibles) |
| AGE | âge |
| PAY_0, PAY_2 ... PAY_6 | retards des six derniers mois (valeurs brutes) |
| BILL_AMT1 ... BILL_AMT6 | factures des six derniers mois |
| PAY_AMT1 ... PAY_AMT6 | remboursements des six derniers mois |

Sortie : probabilité de défaut à l'échéance suivante, classe 1. Le pipeline
embarqué nettoie les codes rares et ajoute les variables dérivées du sujet ;
l'application redéfinit ces fonctions pour recharger le joblib.

## Mode contrôle

La barre latérale propose un mode contrôle qui rejoue les tests
d'acceptation du sujet sur le modèle chargé : ligne du contrat, valeurs
manquantes ou jamais vues, valeurs extrêmes, vitesse (100 lignes en moins
d'une seconde).
