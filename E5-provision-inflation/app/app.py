# -*- coding: utf-8 -*-
"""ProvisionInflation : prévoir l'inflation de l'année suivante pour provisionner.

Application Streamlit d'inférence du sujet E5 (trésorerie d'un groupe
industriel). Le trésorier choisit un pays du portefeuille émergent,
l'application affiche l'historique, la prévision du mélange (forêt +
persistance), l'intervalle issu des résidus du test, le risque de dépasser
10 %, la provision recommandée et la comparaison avec la persistance, comme
décrit dans la maquette du sujet.

Le modèle chargé par défaut est la forêt entraînée par le notebook de
correction (MAE brute 5,0 sur le test contre 5,3 pour la persistance) ;
l'application applique le mélange retenu par la correction : prévision =
70 % forêt + 30 % persistance. Tout autre joblib respectant le contrat du
sujet peut être déposé dans la barre latérale.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- Configuration du sujet E5 --------------------------------------------
SUJET = "E5"
MODELE_JOBLIB = "model_E5.joblib"
NOM_MODELE_CORRECTION = "forêt aléatoire du mélange 70/30, correction E5"
TYPE = "regression"
CLIP = None                    # la forêt est bornée par sa cible tronquée [0, 50]
POIDS_FORET = 0.7              # mélange retenu par la correction
POIDS_PERSISTANCE = 0.3
INTERVALLE_RESIDUS = (-4.1, 5.8)   # quantiles 10 % / 90 % des résidus du test
RISQUE_10 = [                   # P(inflation suivante > 10 %) par tranche de prévision
    (0.0, 5.0, "4 %"), (5.0, 10.0, "14 %"), (10.0, 20.0, "52 %"),
    (20.0, 50.0, "100 %"), (50.0, float("inf"), "100 %"),
]
EMERGENTS = ["ARG", "BRA", "MEX", "TUR", "RUS", "UKR", "IND", "IDN", "ZAF", "NGA",
             "EGY", "GHA", "KEN", "ETH", "TZA", "PAK", "BGD", "VNM", "PHL", "THA",
             "MYS", "COL", "CHL", "PER", "POL", "CZE", "HUN", "ROU", "MAR", "TUN",
             "BOL", "ECU", "URY", "LKA", "KAZ"]
NUM = ["infl", "infl_lag1", "momentum", "FM.LBL.BMNY.ZS", "NY.GDP.MKTP.KD.ZG",
       "NE.EXP.GNFS.ZS", "BX.KLT.DINV.WD.GD.ZS", "FR.INR.LEND", "year"]
CAT = ["region", "iso3"]
FEATURES = NUM + CAT
CIBLE = "infl_next"


def prevision_melangee(modele: object, entree: pd.DataFrame) -> np.ndarray:
    """Mélange retenu : 70 % forêt + 30 % persistance (inflation courante)."""
    foret = predit(modele, entree)                     # partie forêt, par le loader
    infl = pd.to_numeric(entree["infl"], errors="coerce").fillna(0.0).values
    return POIDS_FORET * foret + POIDS_PERSISTANCE * infl


def risque_depasser_10(prevision: float) -> str:
    for basse, haute, etiquette in RISQUE_10:
        if basse <= prevision < haute:
            return etiquette
    return "100 %"


def ligne_controle() -> pd.DataFrame:
    base = {"infl": 58.5, "infl_lag1": 53.9, "momentum": 4.6, "FM.LBL.BMNY.ZS": 90.0,
            "NY.GDP.MKTP.KD.ZG": 2.5, "NE.EXP.GNFS.ZS": 60.0,
            "BX.KLT.DINV.WD.GD.ZS": 3.0, "FR.INR.LEND": 30.0, "year": 2024.0,
            "region": "Europe & Central Asia", "iso3": "TUR"}
    return pd.DataFrame([base])[FEATURES]


def cas_controle() -> list[tuple[str, pd.DataFrame, str]]:
    """Tests d'acceptation publics du sujet E5."""
    def ligne(**surcharges):
        base = {"infl": 4.2, "infl_lag1": 3.9, "momentum": 0.3, "FM.LBL.BMNY.ZS": 40.0,
                "NY.GDP.MKTP.KD.ZG": 4.0, "NE.EXP.GNFS.ZS": 25.0,
                "BX.KLT.DINV.WD.GD.ZS": 2.0, "FR.INR.LEND": 9.0, "year": 2024.0,
                "region": "Latin America & Caribbean", "iso3": "BRA"}
        base.update(surcharges)
        return pd.DataFrame([base])[FEATURES]

    cent = pd.concat([ligne()] * 100, ignore_index=True)
    return [
        ("Chargement et prévision d'une ligne", ligne(), "fini"),
        ("Pays jamais vu (« ZZZ »)", ligne(iso3="ZZZ", region="Autre"), "fini"),
        ("Tous les indicateurs manquants",
         ligne(**{c: np.nan for c in NUM[:-1]}), "fini"),
        ("Inflation extrême (300 %, pays en hyperinflation)",
         ligne(infl=300.0, infl_lag1=250.0, momentum=50.0), "fini"),
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
# Interface : choix du pays, historique, prévision, provision, persistance.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ProvisionInflation", page_icon="💸", layout="wide")
st.title("ProvisionInflation")
st.caption("Prévoir l'inflation de l'année suivante pour provisionner "
           "(sujet E5, trésorerie d'un groupe industriel).")

DATA_DIR = APP_DIR.parent / "data" / "E5-inflation"


@st.cache_data(show_spinner=False)
def charge_panel() -> pd.DataFrame:
    panel = pd.read_csv(DATA_DIR / "panel_E5.csv")
    panel = panel.rename(columns={"FM.LBL.BMNY.GD.ZS": "FM.LBL.BMNY.ZS"})
    panel["momentum"] = panel["infl"] - panel["infl_lag1"]
    return panel


panel = charge_panel()
if panel.empty:
    st.error("Panel absent : data/E5-inflation/panel_E5.csv est attendu à côté "
             "du dossier de l'application.")
    st.stop()

pays = st.selectbox("Pays du portefeuille (une trentaine de pays émergents)",
                    EMERGENTS, index=EMERGENTS.index("TUR"))
if st.button("Afficher la prévision", type="primary"):
    hist_pays = panel[panel["iso3"] == pays].sort_values("year").reset_index(drop=True)
    derniere = hist_pays.dropna(subset=["infl"]).iloc[-1]
    annee_precedente = int(derniere["year"])
    annee_prevue = annee_precedente + 1

    st.markdown(f"##### Historique d'inflation de {pays}, en %")
    serie = hist_pays.dropna(subset=["infl"]).set_index("year")["infl"]
    st.line_chart(serie, height=260)

    entree = derniere[FEATURES].to_frame().T.astype(object)
    entree["year"] = float(annee_precedente)   # la ligne de l'année observée prévoit l'année suivante
    prevision = float(prevision_melangee(modele_actif, entree)[0])
    persistance = float(derniere["infl"])
    basse = prevision + INTERVALLE_RESIDUS[0]
    haute = prevision + INTERVALLE_RESIDUS[1]
    risque = risque_depasser_10(prevision)

    st.subheader(f"Prévision {annee_prevue}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prévision du mélange", f"{prevision:.1f} %",
              help=f"{POIDS_FORET:.0%} forêt + {POIDS_PERSISTANCE:.0%} persistance, "
                   "mélange retenu par la correction")
    k2.metric("Intervalle", f"{max(0.0, basse):.1f} à {max(basse, haute):.1f}",
              help="quantiles 10 % / 90 % des résidus du jeu de test de la correction")
    k3.metric("Risque de dépasser 10 %", risque,
              help="fréquence observée sur le jeu de test, dans la tranche de la prévision")
    k4.metric("Persistance (référence)", f"{persistance:.1f} %",
              help=f"inflation de {annee_precedente} recopiée pour {annee_prevue}")

    st.markdown("##### Provision recommandée")
    marge = st.slider("Marge de provision, en points", 0, 20, 10,
                      help="points ajoutés à la prévision, réglables par le trésorier")
    provision = prevision + marge
    c1, c2 = st.columns(2)
    c1.metric(f"Provision recommandée pour {annee_prevue}", f"{provision:.1f} %",
              help=f"prévision {prevision:.1f} % plus une marge de {marge} points")
    c2.metric("Écart persistance - provision", f"{persistance - provision:+.1f} points",
              help="négatif : la persistance provisionne trop ; positif : trop peu")

    st.caption("La persistance suit la crise au lieu de l'anticiper : sur un pays "
               "qui s'emballe, elle provisionne trop peu, et la provision trop "
               "mince se paie en marge perdue.")
    st.caption("Les données sont embarquées dans l'application : pas d'appel réseau "
               "au moment de la prévision. Les chiffres affichés viennent du modèle "
               "livré avec l'application.")
