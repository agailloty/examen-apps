# Vigiroute : application d'inférence du sujet E2

Gravité attendue d'un accident selon ses circonstances (sujet E2, métropole de 600 000 habitants).

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
`app/models/` sous la forme `model_E2.joblib`, committé tel quel : l'application
le charge directement au lancement.

Modèle de la correction : HistGradientBoosting pondéré (120 rondes), correction E2.

## Déposer un autre modèle

La barre latérale accepte le dépôt de n'importe quel joblib similaire
(bouton « Déposer un modèle joblib »). Le modèle déposé remplace celui de la
correction pour toute la session ; un contrôle de conformité vérifie
`predict` / `predict_proba` et une prévision sur une ligne du contrat avant
de l'activer, et un bouton permet de revenir au modèle de la correction.

## Contrat du modèle

| Colonne | Rôle | Remarques |
| --- | --- | --- |
| age_cond | âge du conducteur | manquant accepté |
| heure | heure de l'accident | 0-23 |
| jour_semaine | jour de la semaine | 1 (lundi) à 7 (dimanche) |
| atmosphere | météo au moment des faits | libellés BAAC (« Pluie legere »...) |
| lum | luminosité | libellés BAAC |
| catr | catégorie de route | libellés BAAC |
| vma | vitesse maximale autorisée | manquante acceptée |
| type_veh | type de véhicule | libellés BAAC (« Voiture legere »...) |
| secu | équipement de sécurité | libellés BAAC |
| dep | département | jamais vu accepté |
| trajet | motif du déplacement | libellés BAAC (« Domicile - travail »...) |

Sortie : probabilité que l'accident soit grave (au moins un tué ou un blessé
hospitalisé), classe 1.

## Mode contrôle

La barre latérale propose un mode contrôle qui rejoue les tests
d'acceptation du sujet sur le modèle chargé : ligne du contrat, valeurs
manquantes ou jamais vues, valeurs extrêmes, vitesse (100 lignes en moins
d'une seconde).
