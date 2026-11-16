# -*- coding: utf-8 -*-
"""Vigiroute : gravité attendue d'un accident selon ses circonstances.

Application Streamlit d'inférence du sujet E2 (métropole de 600 000
habitants). Le service prévention décrit la circonstance à instruire,
l'application affiche la probabilité que l'accident soit grave (au moins un
tué ou un blessé hospitalisé), l'écart au taux de base et la décision au seuil
de coût affiché, comme décrit dans la maquette du sujet.

Le modèle chargé par défaut est le meilleur modèle entraîné par le notebook
de correction (HistGradientBoosting pondéré, 120 rondes, PR-AUC = 0,670 sur
2023-2024 contre 0,361 pour la constante). Tout autre joblib respectant le
contrat du sujet peut être déposé dans la barre latérale.
"""
from __future__ import annotations

import inspect
import io
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- Configuration du sujet E2 ---------------------------------------------
SUJET = "E2"
MODELE_JOBLIB = "model_E2.joblib"
NOM_MODELE_CORRECTION = "HistGradientBoosting pondéré (120 rondes), correction E2"
TYPE = "classification"
CLIP = None
TAUX_BASE = 18.0        # % d'accidents graves dans la base (énoncé)
SEUIL_VISITE = 0.20     # seuil de coût toléré : visiter le site si p >= 20 %
JOURS = {1: "Lundi", 2: "Mardi", 3: "Mercredi", 4: "Jeudi",
         5: "Vendredi", 6: "Samedi", 7: "Dimanche"}
ATMOSPHERES = ["Normale", "Pluie legere", "Pluie forte", "Neige - gresle",
               "Brouillard - fumee", "Vent fort - tempete", "Temps eblouissant",
               "Temps couvert", "Autre"]
LUMINOSITES = ["Plein jour", "Crepuscule ou aube", "Nuit sans eclairage public",
               "Nuit avec eclairage public", "Nuit avec eclairage non allume", "Autre"]
CATEGORIES_ROUTE = ["Autoroute", "Route nationale", "Route departementale",
                    "Voie communale", "Hors reseau public", "Parc de stationnement",
                    "Zone de rencontre", "Autre"]
TYPES_VEHICULE = ["Voiture legere", "Vehicule utilitaire", "Cyclomoteur", "Scooter",
                  "Motocyclette", "Velocipede", "Poids lourd", "Bus"]
EQUIPEMENTS = ["Ceinture", "Casque", "Aucun", "Inconnu", "Dispositif enfant",
               "Gilet reflectissant", "Airbag 2RM", "Gants 2RM", "Autre equipement", "Autre"]
MOTIFS = ["Domicile - travail", "Courses", "Utilisation professionnelle",
          "Promenade - loisirs", "Autre"]


def ligne_controle() -> pd.DataFrame:
    return pd.DataFrame([{
        "age_cond": 40.0, "heure": 14.0, "jour_semaine": 3, "atmosphere": "Normale",
        "lum": "Plein jour", "catr": "Voie communale", "vma": 50.0,
        "type_veh": "Voiture legere", "secu": "Ceinture", "dep": "59",
        "trajet": "Domicile - travail",
    }])


def cas_controle() -> list[tuple[str, pd.DataFrame, str]]:
    """Tests d'acceptation publics du sujet E2."""
    def ligne(**surcharges):
        base = {"age_cond": 40.0, "heure": 14.0, "jour_semaine": 3, "atmosphere": "Normale",
                "lum": "Plein jour", "catr": "Voie communale", "vma": 50.0,
                "type_veh": "Voiture legere", "secu": "Ceinture", "dep": "59",
                "trajet": "Domicile - travail"}
        base.update(surcharges)
        return pd.DataFrame([base])

    cent = pd.concat([ligne()] * 100, ignore_index=True)
    return [
        ("Chargement et probabilité dans [0, 1]", ligne(), "proba"),
        ("Département jamais vu (« 999 »)", ligne(dep="999"), "proba"),
        ("Vitesse maximale manquante", ligne(vma=np.nan), "proba"),
        ("Âge du conducteur manquant", ligne(age_cond=np.nan), "proba"),
        ("Vitesse : 100 lignes en moins d'une seconde", cent, "vitesse"),
    ]


# ---------------------------------------------------------------------------
# Chargement du modèle : correction par défaut, dépôt joblib au choix.
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODELE_DEFAUT = APP_DIR / "models" / MODELE_JOBLIB


_CHILD_ATTRS = (
    "best_estimator_",
    "estimator_",
    "base_estimator_",
    "final_estimator_",
    "regressor_",
    "classifier_",
)


def _iter_all_estimators(modele: object, seen: set[int] | None = None):
    if seen is None:
        seen = set()
    if modele is None or isinstance(modele, (str, bytes, int, float, bool)):
        return
    obj_id = id(modele)
    if obj_id in seen:
        return
    seen.add(obj_id)
    yield modele

    steps = getattr(modele, "steps", None)
    if isinstance(steps, (list, tuple)):
        for item in steps:
            if isinstance(item, (list, tuple)) and len(item) > 1:
                yield from _iter_all_estimators(item[1], seen)

    for attr_name in ("transformers_", "transformers", "transformer_list"):
        items = getattr(modele, attr_name, None)
        if isinstance(items, (list, tuple)):
            for item in items:
                if isinstance(item, (list, tuple)) and len(item) > 1:
                    yield from _iter_all_estimators(item[1], seen)

    for attr_name in ("estimators_", "estimators"):
        items = getattr(modele, attr_name, None)
        if isinstance(items, (list, tuple)):
            for item in items:
                est = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else item
                yield from _iter_all_estimators(est, seen)

    for attr_name in _CHILD_ATTRS:
        child = getattr(modele, attr_name, None)
        if child is not None:
            yield from _iter_all_estimators(child, seen)

    calibrated = getattr(modele, "calibrated_classifiers_", None)
    if isinstance(calibrated, (list, tuple)):
        for cal in calibrated:
            for sub_attr in ("estimator", "base_estimator"):
                sub = getattr(cal, sub_attr, None)
                if sub is not None:
                    yield from _iter_all_estimators(sub, seen)


def _harmonize_loaded_model(modele: object) -> object:
    """Harmonise les attributs manquants entre versions de scikit-learn apres joblib.load."""
    for est in _iter_all_estimators(modele):
        cls = type(est)
        cls_name = cls.__name__
        if not hasattr(est, "__dict__"):
            continue
        try:
            sig = inspect.signature(cls.__init__)
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                if param.default is inspect.Parameter.empty:
                    continue
                if not hasattr(est, param_name):
                    try:
                        setattr(est, param_name, param.default)
                    except Exception:
                        pass
        except (ValueError, TypeError):
            pass

        if cls_name == "LogisticRegression" and not hasattr(est, "multi_class"):
            try:
                est.multi_class = "auto"
            except Exception:
                pass
        if cls_name == "OneHotEncoder":
            if hasattr(est, "sparse_output") and not hasattr(est, "sparse"):
                try:
                    est.sparse = est.sparse_output
                except Exception:
                    pass
            elif hasattr(est, "sparse") and not hasattr(est, "sparse_output"):
                try:
                    est.sparse_output = est.sparse
                except Exception:
                    pass
        if cls_name == "SimpleImputer" and not hasattr(est, "_fill_dtype"):
            stats = getattr(est, "statistics_", None)
            if stats is not None and hasattr(stats, "dtype"):
                try:
                    est._fill_dtype = stats.dtype
                except Exception:
                    pass
        if any(
            k in cls_name
            for k in ("Tree", "Forest", "GradientBoosting", "ExtraTrees", "HistGradientBoosting")
        ) and not hasattr(est, "monotonic_cst"):
            try:
                est.monotonic_cst = None
            except Exception:
                pass
    return modele


@st.cache_resource(show_spinner=False)
def charge_modele_correction() -> object:
    """Charge le modèle exactement tel qu'entraîné par le notebook de correction."""
    if not MODELE_DEFAUT.exists():
        st.error(
            f"{MODELE_DEFAUT.name} est absent du dossier models/ de l'application."
        )
        st.stop()
    return _harmonize_loaded_model(joblib.load(MODELE_DEFAUT))


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
            candidat = _harmonize_loaded_model(joblib.load(io.BytesIO(depot.getvalue())))
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
# Interface : saisie de la circonstance, puis résultats.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Vigiroute", page_icon="🚧", layout="wide")
st.title("Vigiroute")
st.caption("Gravité attendue d'un accident selon ses circonstances "
           "(sujet E2, métropole de 600 000 habitants).")

st.subheader("Saisie de la circonstance")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Âge du conducteur", min_value=14, max_value=100, value=34, step=1)
    heure = st.slider("Heure", min_value=0, max_value=23, value=22)
    jour = st.selectbox("Jour de la semaine", list(JOURS.values()), index=4)
    atmosphere = st.selectbox("Météo", ATMOSPHERES, index=2)
with c2:
    lum = st.selectbox("Luminosité", LUMINOSITES, index=2)
    catr = st.selectbox("Catégorie de route", CATEGORIES_ROUTE, index=1)
    vma_connue = st.checkbox("Vitesse maximale autorisée renseignée", value=True,
                             help="La valeur manquante est acceptée par le contrat.")
    vma = st.slider("Vitesse maximale autorisée (km/h)", 10, 160, 80) if vma_connue else None
with c3:
    type_veh = st.selectbox("Type de véhicule", TYPES_VEHICULE, index=0,
                            help="« Automobile » dans la maquette = voiture légère.")
    secu = st.selectbox("Équipement de sécurité", EQUIPEMENTS, index=0)
    trajet = st.selectbox("Motif du déplacement", MOTIFS, index=0)
    dep = st.text_input("Département (code, ex. 69)", value="69",
                        help="Un département jamais vu à l'entraînement est accepté.")

if st.button("Évaluer la circonstance", type="primary", use_container_width=True):
    entree = pd.DataFrame([{
        "age_cond": float(age), "heure": float(heure),
        "jour_semaine": float([k for k, v in JOURS.items() if v == jour][0]),
        "atmosphere": atmosphere, "lum": lum, "catr": catr,
        "vma": float(vma) if vma is not None else np.nan,
        "type_veh": type_veh, "secu": secu, "dep": dep.strip() or "inconnu",
        "trajet": trajet,
    }])
    proba = float(predit(modele_actif, entree)[0])
    ecart = (proba * 100.0) - TAUX_BASE
    decision = "Visiter le site" if proba >= SEUIL_VISITE else "Ne pas visiter"

    st.subheader("Résultats")
    k1, k2, k3 = st.columns(3)
    k1.metric("Probabilité d'accident grave", f"{proba * 100:.0f} %",
              help="grave : au moins un tué ou un blessé hospitalisé")
    k2.metric("Écart au taux de base", f"{ecart:+.0f} points",
              help=f"taux de base de la base : {TAUX_BASE:.0f} %")
    k3.metric("Décision au seuil affiché", decision,
              help=f"seuil de coût toléré : {SEUIL_VISITE * 100:.0f} %")

    st.markdown("##### Grille de l'énoncé, en euros")
    st.table(pd.DataFrame({
        "Élément": ["Un tué (valeur statistique administrative)",
                    "Un blessé hospitalisé", "Un blessé léger", "Une visite de site"],
        "Coût retenu": ["3 000 000 €", "100 000 €", "20 000 €", "8 000 €"],
    }))
    st.error("Avertissement permanent de l'application : le modèle prévoit la gravité "
             "des accidents déclarés dans la base, pas le risque de circuler à un "
             "endroit donné.")
    st.caption("Énoncé : README.md du dossier. Avec le modèle de référence, toutes les "
               "circonstances affichent 18 % : la hiérarchisation vient du modèle chargé.")
