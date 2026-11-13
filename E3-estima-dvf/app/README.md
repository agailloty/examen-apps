# EstimaDVF : application d'inférence du sujet E3

Estimation immobilière argumentée par des ventes réelles (sujet E3, réseau d'agences).

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
`app/models/` sous la forme `model_E3.joblib`, committé tel quel : l'application
le charge directement au lancement.

Modèle de la correction : forêt aléatoire (100 arbres), correction E3.

## Déposer un autre modèle

La barre latérale accepte le dépôt de n'importe quel joblib similaire
(bouton « Déposer un modèle joblib »). Le modèle déposé remplace celui de la
correction pour toute la session ; un contrôle de conformité vérifie
`predict` / `predict_proba` et une prévision sur une ligne du contrat avant
de l'activer, et un bouton permet de revenir au modèle de la correction.

## Contrat du modèle

| Colonne | Rôle | Remarques |
| --- | --- | --- |
| surface_reelle_bati | surface bâtie | m2 |
| nombre_pieces | nombre de pièces | entier |
| surface_terrain | surface du terrain | manquant accepté (appartement) |
| lat, lon | coordonnées de la commune | référentiel app/data/communes_E3.csv |
| type_local | Appartement ou Maison | autres valeurs acceptées sans erreur |
| nature_mutation | nature de la mutation | « Vente » dans le domaine couvert |

Sortie : prix au m2 en EUR, borné au contrat [200, 12 000]. L'application
ajoute l'intervalle, le prix total et les ventes comparables (données DVF
embarquées dans app/data/ventes_E3.csv, pas d'appel réseau).

## Mode contrôle

La barre latérale propose un mode contrôle qui rejoue les tests
d'acceptation du sujet sur le modèle chargé : ligne du contrat, valeurs
manquantes ou jamais vues, valeurs extrêmes, vitesse (100 lignes en moins
d'une seconde).
