# -*- coding: utf-8 -*-
"""AtelierRadar : probabilité de défaillance de la machine au dernier cycle.

Application Streamlit d'inférence du sujet E8 (PME d'usinage CNC). Le chef
d'atelier décrit le dernier cycle de la machine (type de produit,
températures, vitesse de rotation, couple, usure d'outil), l'application
affiche la probabilité d'au moins un des cinq modes de défaillance,
l'écart au taux de base et la décision au seuil de coût réglable, comme
décrit dans la maquette du sujet.

Le modèle chargé par défaut est le meilleur modèle entraîné par le notebook
de correction (forêt aléatoire à 200 arbres équilibrée, PR-AUC = 0,830 sur le
test contre 0,034 pour la constante) ; le pipeline embarqué ajoute les
variables dérivées du sujet (puissance mécanique, écart de températures).
Tout autre joblib respectant le contrat du sujet peut être déposé dans la
barre latérale.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- Configuration du sujet E8 --------------------------------------------
SUJET = "E8"
MODELE_JOBLIB = "model_E8.joblib"
NOM_MODELE_CORRECTION = "forêt aléatoire équilibrée (200 arbres), correction E8"
TYPE = "classification"
CLIP = None
TAUX_BASE = 3.4               # % de cycles défaillants dans le jeu
SEUIL_COUT_DEFAUT = 5.0       # % ; grille de la correction : 500 / 10 000
TYPES_PRODUIT = ["L", "M", "H"]


# --- Fonctions du pipeline (requis au chargement du joblib) -----------------
# Le pipeline de la correction référence ajoute_derivees définie dans le
# __main__ du notebook : l'application la redéfinit à l'identique pour que
# joblib puisse recharger le modèle.


def ajoute_derivees(X):
    "Variables physiques du sujet, calculées dans le pipeline, jamais chez le client."
    X = X.copy()
    X["puissance_W"] = X["Torque"] * X["Rotational_speed"] * 2 * np.pi / 60
    X["ecart_process_air"] = X["Process_temperature"] - X["Air_temperature"]
    return X



def ligne_controle() -> pd.DataFrame:
    base = {"Type": "L", "Air_temperature": 298.4, "Process_temperature": 309.1,
            "Rotational_speed": 1521.0, "Torque": 58.7, "Tool_wear": 231.0}
    return pd.DataFrame([base])


def cas_controle() -> list[tuple[str, pd.DataFrame, str]]:
    """Tests d'acceptation publics du sujet E8."""
    def ligne(**surcharges):
        base = {"Type": "L", "Air_temperature": 298.4, "Process_temperature": 309.1,
                "Rotational_speed": 1521.0, "Torque": 58.7, "Tool_wear": 231.0}
        base.update(surcharges)
        return pd.DataFrame([base])

    cent = pd.concat([ligne()] * 100, ignore_index=True)
    return [
        ("Chargement et probabilité dans [0, 1]", ligne(), "proba"),
        ("Type de produit jamais vu (« X »)", ligne(Type="X"), "proba"),
        ("Usure d'outil manquante", ligne(Tool_wear=np.nan), "proba"),
        ("Valeur extrême (rotation 20 000 tr/min)", ligne(Rotational_speed=20000.0), "proba"),
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
# Interface : saisie du dernier cycle, puis résultats.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AtelierRadar", page_icon="⚙️", layout="wide")
st.title("AtelierRadar")
st.caption("Probabilité de défaillance de la machine au dernier cycle "
           "(sujet E8, PME d'usinage CNC).")

with st.form("cycle"):
    st.subheader("Saisie du dernier cycle")
    c1, c2, c3 = st.columns(3)
    with c1:
        type_produit = st.radio("Type de produit", TYPES_PRODUIT, horizontal=True)
        air = st.number_input("Température de l'air (K)", min_value=290.0,
                              max_value=305.0, value=298.4, step=0.1)
        process = st.number_input("Température du process (K)", min_value=305.0,
                                  max_value=315.0, value=309.1, step=0.1)
    with c2:
        rotation = st.number_input("Vitesse de rotation (tr/min)", min_value=1100,
                                   max_value=3000, value=1521, step=1)
        couple = st.number_input("Couple (Nm)", min_value=3.0, max_value=80.0,
                                 value=58.7, step=0.1)
    with c3:
        usure_connue = st.checkbox("Usure d'outil renseignée", value=True,
                                   help="La valeur manquante est acceptée par le contrat.")
        usure = st.number_input("Usure d'outil (min)", min_value=0, max_value=300,
                                value=231, step=1) if usure_connue else np.nan
    st.form_submit_button("Évaluer la machine", use_container_width=True, type="primary")

entree = pd.DataFrame([{
    "Type": type_produit, "Air_temperature": float(air),
    "Process_temperature": float(process), "Rotational_speed": float(rotation),
    "Torque": float(couple), "Tool_wear": float(usure) if usure_connue else np.nan,
}])
proba = float(predit(modele_actif, entree)[0])
ecart = proba * 100.0 - TAUX_BASE

seuil = st.slider("Seuil de coût toléré, en %", 1, 30, int(SEUIL_COUT_DEFAUT),
                  help="grille de la correction : intervention planifiée 500 € contre "
                       "10 000 € l'heure d'arrêt surprise, soit 5 %. La maquette "
                       "illustre un réglage à 15 %.")
decision = "Intervenir" if proba * 100.0 >= seuil else "Surveiller"

st.subheader("Résultats")
k1, k2, k3 = st.columns(3)
k1.metric("Probabilité de défaillance", f"{proba * 100:.0f} %",
          help="au moins un des cinq modes, sur le cycle")
k2.metric("Écart au taux de base", f"{ecart:+.0f} points",
          help=f"taux de base du jeu : {TAUX_BASE} %")
k3.metric("Décision au seuil affiché", decision,
          help=f"seuil de coût toléré : {seuil} % ; intervention planifiée si dépassement")

st.progress(min(proba, 1.0), text=f"probabilité {proba*100:.0f} % contre taux de base {TAUX_BASE} %")

st.markdown("##### Grille de l'énoncé, en euros (ordres de grandeur illustratifs)")
st.table(pd.DataFrame({
    "Élément": ["Une intervention planifiée de bonne heure",
                "Une heure d'arrêt surprise",
                "Une vérification sur fausse alerte"],
    "Coût retenu": ["500 €", "10 000 €", "150 €"],
}))
st.warning("Avertissement permanent de l'application : jeu de données synthétique, "
           "ordres de grandeur illustratifs, pas une consigne de sécurité réelle.")
st.caption("Énoncé : README.md du dossier. Avec le modèle de référence, toutes les "
           "machines affichent 3,4 % : le tri des interventions vient du modèle chargé.")
