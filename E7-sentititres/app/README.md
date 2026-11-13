# SentiTitres : application d'inférence du sujet E7

Sentiment d'un titre de presse envers la valeur (sujet E7, société de gestion européenne).

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
`app/models/` sous la forme `model_E7.joblib`, committé tel quel : l'application
le charge directement au lancement.

Modèle de la correction : TF-IDF caractères (3-5) + régression logistique, correction E7.

## Déposer un autre modèle

La barre latérale accepte le dépôt de n'importe quel joblib similaire
(bouton « Déposer un modèle joblib »). Le modèle déposé remplace celui de la
correction pour toute la session ; un contrôle de conformité vérifie
`predict` / `predict_proba` et une prévision sur une ligne du contrat avant
de l'activer, et un bouton permet de revenir au modèle de la correction.

## Contrat du modèle

| Colonne | Rôle |
| --- | --- |
| titre | titre de presse en anglais (texte brut) |
| valeur | valeur concernée (ignorée par le modèle) |

Sortie : classe parmi negative / neutral / positive (predict), avec
probabilités par classe (predict_proba). L'application extrait les mots les
plus influents du pipeline TF-IDF + régression logistique de la correction.

## Mode contrôle

La barre latérale propose un mode contrôle qui rejoue les tests
d'acceptation du sujet sur le modèle chargé : ligne du contrat, valeurs
manquantes ou jamais vues, valeurs extrêmes, vitesse (100 lignes en moins
d'une seconde).
