# -*- coding: utf-8 -*-
"""VeilleCrises : probabilité de crise bancaire l'année suivante.

Application Streamlit d'inférence du sujet E10 (cellule macroprudentielle
régionale). L'économiste décrit le dernier relevé annuel du pays,
l'application affiche la probabilité de crise bancaire à l'année suivante,
l'écart au taux de base et la décision d'examen au seuil du comité, comme
décrit dans la maquette du sujet.

Le modèle chargé par défaut est le meilleur modèle entraîné par le notebook
de correction (HistGradientBoosting équilibré, PR-AUC = 0,585 sur le test
1991-2014 contre 0,208 pour la constante). Tout autre joblib respectant le
contrat du sujet peut être déposé dans la barre latérale.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- Configuration du sujet E10 -------------------------------------------
SUJET = "E10"
MODELE_JOBLIB = "model_E10.joblib"
NOM_MODELE_CORRECTION = "HistGradientBoosting équilibré, correction E10"
TYPE = "classification"
CLIP = None
TAUX_BASE = 8.9                # % d'années en crise bancaire dans le jeu
SEUIL_COMITE = 25.0            # %, seuil de déclenchement du comité
PAYS = ["Algeria", "Angola", "Central African Republic", "Egypt", "Ivory Coast",
        "Kenya", "Mauritius", "Morocco", "Nigeria", "South Africa", "Tunisia",
        "Zambia", "Zimbabwe"]
FEATURES = ["pays", "inflation", "exch_usd", "sovereign_external_debt_usd",
            "gdp_weighted_default", "currency_crises", "inflation_crises",
            "systemic_crisis"]


def ligne_controle() -> pd.DataFrame:
    base = {"pays": "Zambia", "inflation": 18.7, "exch_usd": 3640.0,
            "sovereign_external_debt_usd": 6.3, "gdp_weighted_default": 0.17,
            "currency_crises": 0.0, "inflation_crises": 1.0, "systemic_crisis": 0.0}
    return pd.DataFrame([base])[FEATURES]


def cas_controle() -> list[tuple[str, pd.DataFrame, str]]:
    """Tests d'acceptation publics du sujet E10."""
    def ligne(**surcharges):
        base = {"pays": "Kenya", "inflation": 8.0, "exch_usd": 90.0,
                "sovereign_external_debt_usd": 3.0, "gdp_weighted_default": 0.10,
                "currency_crises": 0.0, "inflation_crises": 0.0, "systemic_crisis": 0.0}
        base.update(surcharges)
        return pd.DataFrame([base])[FEATURES]

    cent = pd.concat([ligne()] * 100, ignore_index=True)
    return [
        ("Chargement et probabilité dans [0, 1]", ligne(), "proba"),
        ("Pays jamais vu (« France »)", ligne(pays="France"), "proba"),
        ("Inflation et dette manquantes",
         ligne(inflation=np.nan, sovereign_external_debt_usd=np.nan,
               gdp_weighted_default=np.nan), "proba"),
        ("Hyperinflation (inflation 4 000 %)", ligne(inflation=4000.0), "proba"),
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
# Interface : saisie du dernier relevé, puis résultats.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="VeilleCrises", page_icon="🏦", layout="wide")
st.title("VeilleCrises")
st.caption("Probabilité de crise bancaire l'année suivante "
           "(sujet E10, cellule macroprudentielle régionale).")

with st.form("releve"):
    st.subheader("Saisie du dernier relevé (année t)")
    pays = st.selectbox("Pays", PAYS)
    c1, c2, c3 = st.columns(3)
    with c1:
        inflation_connue = st.checkbox("Inflation annuelle renseignée", value=True)
        inflation = st.number_input("Inflation annuelle (%)", min_value=-20.0,
                                    max_value=5000.0, value=18.7, step=0.1) \
            if inflation_connue else np.nan
        exch = st.number_input("Taux de change contre le dollar", min_value=0.0,
                               max_value=1e7, value=3640.0, step=1.0)
    with c2:
        dette_connue = st.checkbox("Dette extérieure souveraine renseignée", value=True)
        dette = st.number_input("Dette extérieure souveraine (md USD)", min_value=0.0,
                                max_value=500.0, value=6.3, step=0.1) \
            if dette_connue else np.nan
        defaut_connue = st.checkbox("Défaut pondéré par le PIB renseigné", value=True)
        defaut = st.number_input("Défaut pondéré par le PIB (part)", min_value=0.0,
                                 max_value=2.0, value=0.17, step=0.01) \
            if defaut_connue else np.nan
    with c3:
        crise_change = st.selectbox("Crise de change au relevé",
                                    ["Non (0)", "Oui (1)", "Sévere (2)"])
        crise_inflation = st.selectbox("Crise d'inflation au relevé", ["Non (0)", "Oui (1)"])
        crise_systemique = st.selectbox("Crise systémique au relevé", ["Non (0)", "Oui (1)"])
    st.caption("L'application n'envoie pas l'année : le modèle juge le risque sur "
               "les variables, pas sur la chronologie.")
    st.form_submit_button("Évaluer le pays", use_container_width=True, type="primary")

entree = pd.DataFrame([{
    "pays": pays, "inflation": float(inflation) if inflation_connue else np.nan,
    "exch_usd": float(exch),
    "sovereign_external_debt_usd": float(dette) if dette_connue else np.nan,
    "gdp_weighted_default": float(defaut) if defaut_connue else np.nan,
    "currency_crises": float(crise_change.split(" ")[1].strip("()")),
    "inflation_crises": 1.0 if crise_inflation == "Oui (1)" else 0.0,
    "systemic_crisis": 1.0 if crise_systemique == "Oui (1)" else 0.0,
}])
proba = float(predit(modele_actif, entree)[0])
ecart = proba * 100.0 - TAUX_BASE
decision = "Déclencher l'examen" if proba * 100.0 >= SEUIL_COMITE else "Poursuivre la veille"

st.subheader("Résultats")
k1, k2, k3 = st.columns(3)
k1.metric("Probabilité de crise l'an prochain", f"{proba * 100:.0f} %",
          help="crise bancaire à l'année t+1")
k2.metric("Écart au taux de base", f"{ecart:+.0f} points",
          help=f"taux de base du jeu : {TAUX_BASE} %")
k3.metric("Décision au seuil affiché", decision,
          help=f"seuil du comité : {SEUIL_COMITE:.0f} % ; équipe pays mobilisée")

st.progress(min(proba, 1.0), text=f"probabilité {proba*100:.0f} % contre taux de base {TAUX_BASE} %")

st.markdown("##### Grille de l'énoncé (hypothèses chiffrées à défendre)")
st.table(pd.DataFrame({
    "Élément": ["Une crise bancaire manquée",
                "Un examen pays déclenché à tort",
                "Seuil de déclenchement du comité"],
    "Coût retenu": ["plusieurs points de PIB et sauvetages publics",
                    "une équipe d'analystes pendant un trimestre",
                    f"{SEUIL_COMITE:.0f} %, choisi face à cette grille"],
}))
st.warning("Avertissement permanent de l'application : estimation statistique, pas "
           "une certitude ; réviser les sources officielles avant toute décision "
           "publique.")
st.caption("Énoncé : README.md du dossier. Avec le modèle de référence, tous les "
           "pays affichent 8,9 % : le tri des examens vient du modèle chargé, "
           "évalué sur 1991-2014, des années qu'il n'a jamais vues.")
