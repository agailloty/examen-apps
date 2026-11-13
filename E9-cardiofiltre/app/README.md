# CardioFiltre : application d'inférence du sujet E9

Probabilité d'atteinte coronarienne au stade de la consultation (sujet E9, réseau de centres de santé).

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
`app/models/` sous la forme `model_E9.joblib`, committé tel quel : l'application
le charge directement au lancement.

Modèle de la correction : régression logistique équilibrée, correction E9.

## Déposer un autre modèle

La barre latérale accepte le dépôt de n'importe quel joblib similaire
(bouton « Déposer un modèle joblib »). Le modèle déposé remplace celui de la
correction pour toute la session ; un contrôle de conformité vérifie
`predict` / `predict_proba` et une prévision sur une ligne du contrat avant
de l'activer, et un bouton permet de revenir au modèle de la correction.

## Contrat du modèle

| Colonne | Rôle |
| --- | --- |
| centre | cleveland, longbeach ou switzerland (jamais vu accepté) |
| age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope | variables de consultation (manquants acceptés) |

Sortie : probabilité de rétrécissement coronarien supérieur à 50 %, classe 1.
Les examens complémentaires (ca, thal) sont interdits par le contrat :
l'application ne les saisit pas.

## Mode contrôle

La barre latérale propose un mode contrôle qui rejoue les tests
d'acceptation du sujet sur le modèle chargé : ligne du contrat, valeurs
manquantes ou jamais vues, valeurs extrêmes, vitesse (100 lignes en moins
d'une seconde).
