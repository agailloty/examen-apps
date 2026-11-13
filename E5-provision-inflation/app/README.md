# ProvisionInflation : application d'inférence du sujet E5

Prévoir l'inflation de l'année suivante pour provisionner (sujet E5, trésorerie d'un groupe industriel).

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
`app/models/` sous la forme `model_E5.joblib`, committé tel quel : l'application
le charge directement au lancement.

Modèle de la correction : forêt aléatoire du mélange 70/30, correction E5.

## Déposer un autre modèle

La barre latérale accepte le dépôt de n'importe quel joblib similaire
(bouton « Déposer un modèle joblib »). Le modèle déposé remplace celui de la
correction pour toute la session ; un contrôle de conformité vérifie
`predict` / `predict_proba` et une prévision sur une ligne du contrat avant
de l'activer, et un bouton permet de revenir au modèle de la correction.

## Contrat du modèle

| Colonne | Rôle |
| --- | --- |
| infl, infl_lag1 | inflation courante et retardée (momentum calculé) |
| FM.LBL.BMNY.ZS, NY.GDP.MKTP.KD.ZG, NE.EXP.GNFS.ZS, BX.KLT.DINV.WD.GD.ZS, FR.INR.LEND | indicateurs (manquants acceptés) |
| year | année de la ligne observée |
| region, iso3 | région et code pays |

Sortie du joblib : la forêt seule (cible tronquée [0, 50]). L'application
applique le mélange retenu par la correction : prévision = 70 % forêt +
30 % persistance, puis l'intervalle (quantiles des résidus du test) et le
risque de dépasser 10 %. Panel requis : `data/E5-inflation/panel_E5.csv`.

## Mode contrôle

La barre latérale propose un mode contrôle qui rejoue les tests
d'acceptation du sujet sur le modèle chargé : ligne du contrat, valeurs
manquantes ou jamais vues, valeurs extrêmes, vitesse (100 lignes en moins
d'une seconde).
