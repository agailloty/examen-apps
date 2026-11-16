# -*- coding: utf-8 -*-
"""SentiTitres : sentiment d'un titre de presse envers la valeur.

Application Streamlit d'inférence du sujet E7 (société de gestion
européenne). L'analyste colle un titre, l'application affiche la classe
prédite (negative / neutral / positive), les probabilités par classe, la
position dans la file de revue et les mots les plus influents de la
décision, comme décrit dans la maquette du sujet.

Le modèle chargé par défaut est le meilleur modèle entraîné par le notebook
de correction (TF-IDF caractères 3-5 + régression logistique équilibrée,
macro-F1 = 0,72 sur le test). Tout autre joblib respectant le contrat du
sujet peut être déposé dans la barre latérale.
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

# --- Configuration du sujet E7 --------------------------------------------
SUJET = "E7"
MODELE_JOBLIB = "model_E7.joblib"
NOM_MODELE_CORRECTION = "TF-IDF caractères (3-5) + régression logistique, correction E7"
TYPE = "etiquettes"
CLIP = None
ETIQUETTES = ["negative", "neutral", "positive"]
LIBELLES = {"negative": "Négatif", "neutral": "Neutre", "positive": "Positif"}
TITRE_EXEMPLE = ("Operating profit fell to EUR 35.4 mn from EUR 68.8 mn in 2007 , "
                 "including vessel sales gain of EUR 12.3 mn .")


def ligne_controle() -> pd.DataFrame:
    return pd.DataFrame([{"titre": TITRE_EXEMPLE, "valeur": "Cargotec"}])


def cas_controle() -> list[tuple[str, pd.DataFrame, str]]:
    """Tests d'acceptation publics du sujet E7."""
    def ligne(titre: str, valeur: str = "Cargotec") -> pd.DataFrame:
        return pd.DataFrame([{"titre": titre, "valeur": valeur}])

    cent = pd.concat([ligne(TITRE_EXEMPLE)] * 100, ignore_index=True)
    return [
        ("Chargement et classe prédite", ligne(TITRE_EXEMPLE), "etiquettes"),
        ("Titre jamais vu (invention)", ligne("The company launched a surprising new product"), "etiquettes"),
        ("Titre vide", ligne("", "Inconnu"), "etiquettes"),
        ("Valeur jamais vue", ligne(TITRE_EXEMPLE, "ZZZ"), "etiquettes"),
        ("Vitesse : 100 titres en moins d'une seconde", cent, "vitesse"),
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
# Interface : saisie du titre, classe, probabilités, file de revue, mots influents.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SentiTitres", page_icon="📰", layout="wide")
st.title("SentiTitres")
st.caption("Sentiment d'un titre de presse envers la valeur "
           "(sujet E7, société de gestion européenne).")

with st.form("titre"):
    st.subheader("Saisie du titre")
    titre = st.text_area("Titre de presse (texte brut, en anglais : la langue du corpus)",
                         value=TITRE_EXEMPLE, height=140)
    c1, c2 = st.columns(2)
    with c1:
        valeur = st.text_input("Valeur concernée (optionnelle, jamais vue possible)",
                               value="Cargotec")
    with c2:
        parution = st.selectbox("Date de parution", ["Ce matin", "Hier", "Cette semaine"],
                                help="Illustration de la maquette : la date n'entre pas "
                                     "dans le modèle.")
    st.form_submit_button("Classer le titre", use_container_width=True, type="primary")

titre = (titre or "").strip() or " "
entree = pd.DataFrame([{"titre": titre, "valeur": valeur}])
classes = list(modele_actif.classes_) if hasattr(modele_actif, "classes_") else ETIQUETTES
probas = modele_actif.predict_proba(entree)[0]
indice = int(np.argmax(probas))
classe = str(classes[indice])

st.subheader("Résultats")
k1, k2 = st.columns(2)
k1.metric("Classe prédite", f"{LIBELLES.get(classe, classe)}",
          help="envers la valeur, pas la tonalité du texte")
ordre_file = {"negative": "À lire en premier", "neutral": "File standard",
              "positive": "Bonne nouvelle"}
k2.metric("Position dans la file de revue",
          ordre_file.get(classe, "File standard"),
          help="classe négative en tête, ordonnée par probabilité")

st.markdown("##### Probabilités par classe")
tableau_proba = pd.DataFrame({
    "Classe": [LIBELLES.get(c, c) for c in classes],
    "Probabilité": [f"{p * 100:.0f} %" for p in probas],
}).sort_values("Probabilité", ascending=False, key=lambda s: s.str.rstrip(" %").str.replace(",", ".").astype(float))
st.table(tableau_proba.reset_index(drop=True))

st.markdown("##### Mots les plus influents de la décision")
try:
    tfidf = modele_actif.named_steps["select"].named_transformers_["texte"]
    clf = modele_actif.named_steps["clf"]
    vecteur = tfidf.transform([titre]).toarray().ravel()
    indice_classe = list(clf.classes_).index(classe) if classe in list(clf.classes_) else indice
    contributions = vecteur * clf.coef_[indice_classe]
    presents = np.where(vecteur > 0)[0]
    vocabulaire = np.array(tfidf.get_feature_names_out())
    if len(presents):
        tri = presents[np.argsort(-np.abs(contributions[presents]))][:8]
        mots = pd.DataFrame({
            "Mot ou expression": vocabulaire[tri],
            "Poids dans la décision": [f"{contributions[i]:+.2f}".replace(".", ",")
                                        for i in tri],
            "Lecture": ["fort" if abs(contributions[i]) >= 0.5 else "modéré"
                        for i in tri],
        })
        st.table(mots.reset_index(drop=True))
        st.caption("Poids logistique du n-gramme dans la classe prédite : positif "
                   "pousse vers la classe, négatif pousse contre.")
    else:
        st.caption("Aucun n-gramme du vocabulaire n'apparaît dans ce titre.")
except Exception:  # noqa: BLE001 - extraction indisponible sur un modèle déposé
    st.caption("Extraction des mots influents indisponible pour ce modèle : elle "
               "nécessite le pipeline TF-IDF + régression logistique de la correction.")

st.warning("Avertissement permanent de l'application : outil pédagogique, corpus "
           "sous licence non commerciale, pas un conseil d'investissement.")
st.caption("Énoncé : README.md du dossier. Avec le modèle de référence, tous les "
           "titres affichent neutre : le tri de la file de revue vient du modèle "
           "livré par le groupe.")
