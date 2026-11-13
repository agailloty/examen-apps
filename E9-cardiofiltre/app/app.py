# -*- coding: utf-8 -*-
"""CardioFiltre : probabilité d'atteinte coronarienne au stade de la consultation.

Application Streamlit d'inférence du sujet E9 (réseau de centres de santé
primaire). Le généraliste décrit le patient avec les seules données de la
consultation (les examens complémentaires ca et thal sont interdits),
l'application affiche la probabilité de rétrécissement coronarien, l'écart
au taux de base de Cleveland et la catégorie d'adressage au seuil clinique,
comme décrit dans la maquette du sujet.

Le modèle chargé par défaut est le meilleur modèle entraîné par le notebook
de correction (régression logistique équilibrée, AUROC = 0,888 en validation
contre 0,50 pour la constante, spécificité 0,82 à sensibilité 80 %) ; le pipeline embarqué impute les
manquants et one-hot le centre. Tout autre joblib respectant le contrat du
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

# --- Configuration du sujet E9 --------------------------------------------
SUJET = "E9"
MODELE_JOBLIB = "model_E9.joblib"
NOM_MODELE_CORRECTION = "régression logistique équilibrée, correction E9"
TYPE = "classification"
CLIP = None
TAUX_BASE = 45.9               # % d'atteinte dans le fichier Cleveland
CENTRES = {"cleveland": "Cleveland", "longbeach": "Long Beach",
           "switzerland": "Suisse"}
CLINIQUES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
             "thalach", "exang", "oldpeak", "slope"]
DOULEURS = {1: "Angine typique (1)", 2: "Angine atypique (2)",
            3: "Douleur non angineuse (3)", 4: "Absence de douleur (4)"}
PENTES = {0: "Montante (0)", 1: "Plate (1)", 2: "Descendante (2)"}


def ligne_controle() -> pd.DataFrame:
    base = {"centre": "cleveland", "age": 57.0, "sex": 1.0, "cp": 1.0,
            "trestbps": 140.0, "chol": 289.0, "fbs": 1.0, "restecg": 1.0,
            "thalach": 123.0, "exang": 1.0, "oldpeak": 2.5, "slope": 2.0}
    return pd.DataFrame([base])[["centre"] + CLINIQUES]


def cas_controle() -> list[tuple[str, pd.DataFrame, str]]:
    """Tests d'acceptation publics du sujet E9."""
    def ligne(**surcharges):
        base = {"centre": "cleveland", "age": 57.0, "sex": 1.0, "cp": 1.0,
                "trestbps": 140.0, "chol": 289.0, "fbs": 1.0, "restecg": 1.0,
                "thalach": 123.0, "exang": 1.0, "oldpeak": 2.5, "slope": 2.0}
        base.update(surcharges)
        return pd.DataFrame([base])[["centre"] + CLINIQUES]

    cent = pd.concat([ligne()] * 100, ignore_index=True)
    return [
        ("Chargement et probabilité dans [0, 1]", ligne(), "proba"),
        ("Centre jamais vu (« hongrie »)", ligne(centre="hongrie"), "proba"),
        ("Cholestérol manquant", ligne(chol=np.nan), "proba"),
        ("Douleur thoracique manquante et données extrêmes",
         ligne(cp=np.nan, age=90.0, trestbps=220.0, thalach=60.0, oldpeak=6.0), "proba"),
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
# Interface : saisie du patient, puis résultats et seuil clinique.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CardioFiltre", page_icon="🩺", layout="wide")
st.title("CardioFiltre")
st.caption("Probabilité d'atteinte coronarienne au stade de la consultation "
           "(sujet E9, réseau de centres de santé primaire).")

with st.form("patient"):
    st.subheader("Saisie du patient")
    centre = st.selectbox("Centre de recrutement", list(CENTRES.values()))
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Âge", min_value=20, max_value=110, value=57, step=1)
        sexe = st.radio("Sexe", ["Homme", "Femme"], horizontal=True)
        cp_connue = st.checkbox("Douleur thoracique renseignée", value=True)
        cp = st.selectbox("Douleur thoracique", list(DOULEURS.values()), index=0) \
            if cp_connue else np.nan
        tension_connue = st.checkbox("Tension au repos renseignée", value=True)
        tension = st.number_input("Tension au repos (mmHg)", min_value=80, max_value=250,
                                  value=140) if tension_connue else np.nan
        chol_connue = st.checkbox("Cholestérol renseigné", value=True)
        chol = st.number_input("Cholestérol (mg/dl)", min_value=80, max_value=700,
                               value=289) if chol_connue else np.nan
    with c2:
        fbs = st.radio("Glycémie à jeun supérieure à 120", ["Oui", "Non"], horizontal=True)
        restecg = st.selectbox("Électrocardiogramme de repos",
                               ["Normal (0)", "Anomalie (1)", "Hypertrophie (2)"],
                               index=1)
        thalach = st.number_input("Fréquence cardiaque maximale (bpm)", min_value=60,
                                  max_value=230, value=123)
        exang = st.radio("Angine à l'effort", ["Oui", "Non"], horizontal=True)
    with c3:
        oldpeak = st.number_input("Dépression ST (oldpeak)", min_value=0.0,
                                  max_value=10.0, value=2.5, step=0.1)
        slope_connue = st.checkbox("Pente ST renseignée", value=True)
        slope = st.selectbox("Pente ST", list(PENTES.values()), index=2) \
            if slope_connue else np.nan
    st.caption("Les examens complémentaires (ca, thal) ne sont pas envoyés : la "
               "décision se prend au stade de la consultation.")
    st.form_submit_button("Évaluer le patient", use_container_width=True, type="primary")

entree = pd.DataFrame([{
    "centre": [k for k, v in CENTRES.items() if v == centre][0],
    "age": float(age), "sex": 1.0 if sexe == "Homme" else 0.0,
    "cp": float([k for k, v in DOULEURS.items() if v == cp][0]) if cp_connue else np.nan,
    "trestbps": float(tension) if tension_connue else np.nan,
    "chol": float(chol) if chol_connue else np.nan,
    "fbs": 1.0 if fbs == "Oui" else 0.0,
    "restecg": float(restecg.split("(")[1].rstrip(")")),
    "thalach": float(thalach),
    "exang": 1.0 if exang == "Oui" else 0.0,
    "oldpeak": float(oldpeak),
    "slope": float([k for k, v in PENTES.items() if v == slope][0]) if slope_connue else np.nan,
}])

proba = float(predit(modele_actif, entree)[0])
ecart = proba * 100.0 - TAUX_BASE
if proba >= 0.60:
    categorie = "Adresser"
elif proba >= 0.20:
    categorie = "Surveiller"
else:
    categorie = "Rassurer"

st.subheader("Résultats")
k1, k2, k3 = st.columns(3)
k1.metric("Probabilité d'atteinte", f"{proba * 100:.0f} %",
          help="rétrécissement coronarien supérieur à 50 %")
k2.metric("Écart au taux de base", f"{ecart:+.0f} points",
          help=f"taux de base de Cleveland : {TAUX_BASE} %")
k3.metric("Catégorie au seuil clinique", categorie,
          help="supérieure à 60 % : adresser ; 20 à 60 % : surveiller ; sous 20 % : rassurer")

st.markdown("##### Seuil clinique affiché")
st.table(pd.DataFrame({
    "Règle de décision": ["Sensibilité minimale exigée",
                           "Spécificité au seuil courant (correction)",
                           "Catégories d'adressage"],
    "Valeur affichée": ["90 % des vrais cas adressés", "0,82 à sensibilité 80 %",
                        "supérieure à 60 % : adresser ; 20 à 60 % : surveiller ; "
                        "sous 20 % : rassurer"],
}))

st.progress(min(proba, 1.0), text=f"probabilité {proba*100:.0f} % contre taux de base {TAUX_BASE} %")
st.warning("Avertissement permanent de l'application : outil pédagogique de dépistage, "
           "ne remplace pas un avis médical, jamais utilisé pour un vrai patient.")
st.caption("Énoncé : README.md du dossier. Avec le modèle de référence, tous les "
           "patients affichent 45,9 % : le tri des adressages vient du modèle chargé, "
           "et le jeu caché est un centre jamais vu.")
