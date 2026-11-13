# Applications Streamlit d'examen (E1 à E11)

Onze applications d'inférence Streamlit, une par sujet d'examen, chacune
embarquant son modèle scikit-learn entraîné (`app/models/model_E*.joblib`).
Chaque application expose une saisie métier, la prévision du modèle embarqué,
des indicateurs de décision, et accepte le dépôt d'un autre modèle joblib
respectant le même contrat.

| Sujet | Application | Prévision |
|---|---|---|
| E1 | [ConfigCo2](E1-configurateur-co2/app/app.py) | Malus CO2 d'une configuration avant la commande (courtier en leasing) |
| E2 | [Vigiroute](E2-vigiroute/app/app.py) | Gravité attendue d'un accident selon ses circonstances |
| E3 | [EstimaDVF](E3-estima-dvf/app/app.py) | Estimation immobilière argumentée par des ventes réelles |
| E4 | [PovertyMap](E4-povertymap/app/app.py) | Taux de pauvreté par pays, mesures réelles et estimations |
| E5 | [ProvisionInflation](E5-provision-inflation/app/app.py) | Inflation de l'année suivante pour provisionner |
| E6 | [CreditRadar](E6-creditradar/app/app.py) | Probabilité de défaut d'une demande de crédit |
| E7 | [SentiTitres](E7-sentititres/app/app.py) | Sentiment d'un titre de presse envers la valeur |
| E8 | [AtelierRadar](E8-atelierradar/app/app.py) | Probabilité de défaillance de la machine au dernier cycle |
| E9 | [CardioFiltre](E9-cardiofiltre/app/app.py) | Probabilité d'atteinte coronarienne au stade de la consultation |
| E10 | [VeilleCrises](E10-veillecrises/app/app.py) | Probabilité de crise bancaire l'année suivante |
| E11 | [VeilleCyclone](E11-veillecyclone/app/app.py) | Probabilité qu'un système tropical devienne un ouragan majeur |

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run E7-sentititres/app/app.py
```

## Données

Les données embarquées sont agrégées et issues de sources publiques :
registre des émissions CO2 des voitures neuves (Commission européenne),
demandes de valeurs foncières aggregées par commune (DGFiP), indicateurs
mondiaux (Banque mondiale), relevés HURDAT2 (NOAA) et jeux publics de
référence. Aucune donnée personnelle.

## Déploiement Streamlit Community Cloud

Créer une application par sujet, en pointant le chemin principal sur
`<sujet>/app/app.py`. Les dépendances sont installées depuis le
`requirements.txt` à la racine.
