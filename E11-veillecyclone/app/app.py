# -*- coding: utf-8 -*-
"""VeilleCyclone : probabilité qu'un système tropical devienne un ouragan majeur.

Application Streamlit d'inférence du sujet E11 (courtier en réassurance aux
Petites Antilles). Le courtier saisit les relevés des premières 24 heures
d'un système en formation, l'application affiche la probabilité de majoration
et la recommandation de placement au seuil du portefeuille, comme décrit
dans la maquette du sujet.

Le modèle chargé par défaut est le meilleur modèle entraîné par le notebook
de correction (régression logistique sur les relevés et leurs variations à
24 h, PR-AUC = 0,461 sur le jeu caché contre 0,209 pour la constante).
Le pipeline embarqué ajoute les variables dérivées du sujet. Tout autre
joblib respectant le contrat du sujet peut être déposé dans la barre
latérale.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- Configuration du sujet E11 -------------------------------------------
SUJET = "E11"
MODELE_JOBLIB = "model_E11.joblib"
NOM_MODELE_CORRECTION = "régression logistique sur relevés et variations, correction E11"
TYPE = "classification"
CLIP = None
TAUX_BASE = 17.0               # % de systèmes devenus majeurs dans le bassin
SEUIL_PLACEMENT = 30.0         # %, seuil du portefeuille
MOIS = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet",
        "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
FEATURES = ["mois", "lat0", "lon0", "vent0", "pres0", "vent24", "pres24"]


# --- Fonctions du pipeline (requis au chargement du joblib) -----------------
# Le pipeline de la correction référence ajoute_derivees définie dans le
# __main__ du notebook : l'application la redéfinit à l'identique pour que
# joblib puisse recharger le modèle.


def ajoute_derivees(X):
    "Tendance d'intensification et chute de pression sur 24 h, dans la fenêtre."
    X = X.copy()
    X["dvent24"] = X["vent24"] - X["vent0"]
    X["dpres24"] = X["pres24"] - X["pres0"]
    X["mois"] = X["mois"].astype(str)
    return X



def ligne_controle() -> pd.DataFrame:
    base = {"mois": 8, "lat0": 12.4, "lon0": 58.1, "vent0": 35.0,
            "pres0": 1006.0, "vent24": 55.0, "pres24": 997.0}
    return pd.DataFrame([base])[FEATURES]


def cas_controle() -> list[tuple[str, pd.DataFrame, str]]:
    """Tests d'acceptation publics du sujet E11."""
    def ligne(**surcharges):
        base = {"mois": 9, "lat0": 14.0, "lon0": 60.0, "vent0": 40.0,
                "pres0": 1005.0, "vent24": 50.0, "pres24": 1000.0}
        base.update(surcharges)
        return pd.DataFrame([base])[FEATURES]

    cent = pd.concat([ligne()] * 100, ignore_index=True)
    return [
        ("Chargement et probabilité dans [0, 1]", ligne(), "proba"),
        ("Relevés de 24 h manquants (système mort avant)",
         ligne(vent24=np.nan, pres24=np.nan), "proba"),
        ("Pression manquante au premier relevé", ligne(pres0=np.nan), "proba"),
        ("Cyclone déjà intense (vent 130 nœuds, pression 940)",
         ligne(vent0=130.0, pres0=940.0, vent24=140.0, pres24=930.0), "proba"),
        ("Vitesse : 100 lignes en moins d'une seconde", cent, "vitesse"),
    ]


# ---------------------------------------------------------------------------
# Chargement du modèle : correction par défaut, dépôt joblib au choix.
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODELE_DEFAUT = APP_DIR / "models" / MODELE_JOBLIB


@st.cache_resource(show_spinner=False)
def charge_modele_correction() -> object:
    """Charge le modèle exactement tel qu'entraîné par le notebook de correction."""
    if not MODELE_DEFAUT.exists():
        st.error(
            f"{MODELE_DEFAUT.name} est absent du dossier models/ de l'application."
        )
        st.stop()
    return joblib.load(MODELE_DEFAUT)


def valide_modele(modele: object) -> tuple[bool, str]:
    """Contrôle de conformité au contrat : méthode de prévision et bornes."""
    if TYPE == "classification" and not hasattr(modele, "predict_proba"):
        return False, "le modèle n'expose pas predict_proba (contrat de classification)"
    if not hasattr(modele, "predict"):
        return False, "le modèle n'expose pas predict"
    try:
        predit(modele, ligne_controle())
    except Exception as exc:  # noqa: BLE001 - le mode contrôle rapporte tout
        return False, f"échec de prévision sur une ligne du contrat : {exc}"
    return True, ""


def predit(modele: object, entree: pd.DataFrame) -> np.ndarray:
    """Prévision au contrat : probabilité de la classe 1, étiquette ou valeur bornée."""
    if TYPE == "classification":
        proba = np.asarray(modele.predict_proba(entree)[:, 1], dtype=float)
        return np.clip(proba, 0.0, 1.0)
    if TYPE == "etiquettes":
        return modele.predict(entree)
    valeurs = np.asarray(modele.predict(entree), dtype=float).ravel()
    if CLIP is not None:
        valeurs = np.clip(valeurs, CLIP[0], CLIP[1])
    return valeurs


def selection_modele() -> tuple[object, str, str]:
    """Modèle actif : dépôt uploadé s'il y en a un, sinon celui de la correction."""
    if st.session_state.get("modele_depose") is not None:
        return st.session_state["modele_depose"], st.session_state["nom_depose"], "dépôt"
    return charge_modele_correction(), NOM_MODELE_CORRECTION, "correction"


with st.sidebar:
    st.subheader("Modèle")
    modele_actif, nom_actif, origine = selection_modele()
    st.caption(f"Modèle chargé : {nom_actif} ({origine})")
    if st.session_state.get("modele_depose") is not None:
        if st.button("Revenir au modèle de la correction"):
            st.session_state["modele_depose"] = None
            st.session_state["nom_depose"] = None
            st.rerun()

    depot = st.file_uploader(
        "Déposer un modèle joblib similaire (model_"
        f"{SUJET.lower()}.joblib)",
        type=["joblib"],
        help="Tout joblib respectant le contrat du sujet est accepté : "
        "il remplace le modèle de la correction pour toute la session.",
    )
    if depot is not None:
        try:
            candidat = joblib.load(io.BytesIO(depot.getvalue()))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Chargement impossible : {exc}")
        else:
            ok, message = valide_modele(candidat)
            if not ok:
                st.error(f"Modèle hors contrat ({message}) : il n'est pas retenu.")
            elif st.session_state.get("nom_depose") != depot.name:
                st.session_state["modele_depose"] = candidat
                st.session_state["nom_depose"] = depot.name
                st.rerun()

    with st.expander("Mode contrôle (tests d'acceptation du contrat)"):
        if st.button("Exécuter le mode contrôle", use_container_width=True):
            lancer_controle()


def lancer_controle() -> None:
    """Tests d'acceptation publics, sur le modèle actuellement chargé."""
    resultats = []
    essais = cas_controle()
    for nom, entree, attendu in essais:
        debut = time.perf_counter()
        try:
            sortie = predit(modele_actif, entree)
            duree = time.perf_counter() - debut
            conforme = True
            if attendu == "vitesse":
                conforme = bool(duree < 1.0)
            elif attendu == "bornes" and CLIP is not None:
                conforme = bool((sortie >= CLIP[0] - 1e-9).all() and (sortie <= CLIP[1] + 1e-9).all())
            elif attendu == "proba":
                conforme = bool((sortie >= 0.0).all() and (sortie <= 1.0).all())
            elif attendu == "etiquettes":
                conforme = bool(set(np.asarray(sortie).astype(str)) <= set(ETIQUETTES))
            resultats.append((nom, "OK" if conforme else "ÉCHEC", f"{duree * 1000:.0f} ms"))
        except Exception as exc:  # noqa: BLE001
            resultats.append((nom, "ÉCHEC", str(exc)[:120]))
    st.dataframe(
        pd.DataFrame(resultats, columns=["Test", "Statut", "Durée / détail"]),
        hide_index=True, use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Interface : saisie des relevés 24 h, puis résultats.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="VeilleCyclone", page_icon="🌀", layout="wide")
st.title("VeilleCyclone")
st.caption("Probabilité qu'un système tropical devienne un ouragan majeur "
           "(sujet E11, courtier en réassurance aux Petites Antilles).")

with st.form("systeme"):
    st.subheader("Saisie, fenêtre de décision (24 premières heures)")
    mois = st.selectbox("Mois de formation", MOIS, index=7)
    c1, c2, c3 = st.columns(3)
    with c1:
        lat0 = st.number_input("Latitude du premier relevé (degrés nord)",
                               min_value=0.0, max_value=45.0, value=12.4, step=0.1)
        lon0 = st.number_input("Longitude du premier relevé (degrés ouest)",
                               min_value=30.0, max_value=100.0, value=58.1, step=0.1)
    with c2:
        vent0 = st.number_input("Vent au premier relevé (nœuds)", min_value=0,
                                max_value=200, value=35)
        pres0_connue = st.checkbox("Pression au premier relevé renseignée", value=True)
        pres0 = st.number_input("Pression au premier relevé (millibars)", min_value=850,
                                max_value=1030, value=1006) if pres0_connue else np.nan
    with c3:
        vent24_connu = st.checkbox("Relevé de 24 h présent", value=True,
                                   help="Manquant si le système est mort avant 24 h.")
        vent24 = st.number_input("Vent au relevé de 24 h (nœuds)", min_value=0,
                                 max_value=200, value=55) if vent24_connu else np.nan
        pres24_connue = st.checkbox("Pression de 24 h renseignée", value=True)
        pres24 = st.number_input("Pression au relevé de 24 h (millibars)", min_value=850,
                                 max_value=1030, value=997) if pres24_connue else np.nan
    st.caption("Rien au-delà de 24 heures n'est envoyé au modèle : le marché de la "
               "réassurance se ferme à l'approche de la menace.")
    st.form_submit_button("Évaluer le système", use_container_width=True, type="primary")

entree = pd.DataFrame([{
    "mois": float(MOIS.index(mois) + 1), "lat0": float(lat0), "lon0": float(lon0),
    "vent0": float(vent0), "pres0": float(pres0) if pres0_connue else np.nan,
    "vent24": float(vent24) if vent24_connu else np.nan,
    "pres24": float(pres24) if pres24_connue else np.nan,
}])
proba = float(predit(modele_actif, entree)[0])
ecart = proba * 100.0 - TAUX_BASE
recommandation = ("Placer la couverture complémentaire"
                  if proba * 100.0 >= SEUIL_PLACEMENT else "Attendre le prochain relevé")

st.subheader("Résultats")
k1, k2, k3 = st.columns(3)
k1.metric("Probabilité d'ouragan majeur", f"{proba * 100:.0f} %",
          help="vent final d'au moins 96 nœuds sur sa vie")
k2.metric("Écart au taux de base", f"{ecart:+.0f} points",
          help=f"taux de base du bassin : {TAUX_BASE:.0f} %")
k3.metric("Recommandation", recommandation,
          help=f"seuil du portefeuille : {SEUIL_PLACEMENT:.0f} % ; avant la "
               "fermeture du marché")

st.progress(min(proba, 1.0), text=f"probabilité {proba*100:.0f} % contre taux de base {TAUX_BASE:.0f} %")

st.markdown("##### Grille de l'énoncé (hypothèses chiffrées à défendre)")
st.table(pd.DataFrame({
    "Élément": ["Couverture non placée face à un majeur",
                "Couverture placée pour rien",
                "Seuil de placement du courtier"],
    "Coût retenu": ["scénario 2017 : dizaines de milliards de dollars de dégâts "
                    "aux Caraïbes, une compagnie locale en danger",
                    "prime non consommée, quelques pourcents du portefeuille",
                    f"{SEUIL_PLACEMENT:.0f} %, choisi face à cette grille"],
}))
st.warning("Avertissement permanent de l'application : outil pédagogique "
           "d'assurance, pas un avis officiel du National Hurricane Center ni une "
           "consigne de sécurité.")
st.caption("Énoncé : README.md du dossier. Avec le modèle de référence, tous les "
           "systèmes affichent 17 % : le tri vient du modèle chargé, évalué sur "
           "les tempêtes de 1996-2025, qu'il n'a jamais vues.")
