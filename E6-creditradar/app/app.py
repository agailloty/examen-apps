# -*- coding: utf-8 -*-
"""CreditRadar : probabilité de défaut d'une demande de crédit.

Application Streamlit d'inférence du sujet E6 (banque de détail taïwanaise).
Le conseiller décrit la demande, l'application affiche la probabilité de
défaut à l'échéance suivante, l'écart au taux de base, la décision au seuil
de coût affiché et la limite suggérée, comme décrit dans la maquette du
sujet.

Le modèle chargé par défaut est le meilleur modèle entraîné par le notebook
de correction (forêt aléatoire à 200 arbres équilibrée, PR-AUC = 0,552 sur le
test contre 0,221 pour la constante) ; le pipeline embarqué nettoie les codes
rares et ajoute les variables dérivées du sujet. Tout autre joblib
respectant le contrat du sujet peut être déposé dans la barre latérale.
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

# --- Configuration du sujet E6 --------------------------------------------
SUJET = "E6"
MODELE_JOBLIB = "model_E6.joblib"
NOM_MODELE_CORRECTION = "forêt aléatoire équilibrée (200 arbres), correction E6"
TYPE = "classification"
CLIP = None
TAUX_BASE = 22.1              # % de défaut dans le fichier UCI
SEUIL_COUT_DEFAUT = 30.0      # %, seuil de coût toléré (réglable dans l'interface)
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT = [f"PAY_AMT{i}" for i in range(1, 7)]
SENSIBLES = ["SEX", "EDUCATION", "MARRIAGE"]
EDUCATIONS = {1: "Études supérieures (1)", 2: "Université (2)", 3: "Lycée (3)",
              4: "Autres (4)", 0: "Code rare (0)", 5: "Code rare (5)", 6: "Code rare (6)"}
MARIAGES = {1: "Marié (1)", 2: "Célibataire (2)", 3: "Autres (3)", 0: "Code rare (0)"}


# --- Fonctions du pipeline (requis au chargement du joblib) -----------------
# Le pipeline de la correction référence nettoie et ajoute_derivees définies
# dans le __main__ du notebook : l'application les redéfinit à l'identique
# pour que joblib puisse recharger le modèle.


def nettoie(X):
    "Regroupe les codes non documentés : EDUCATION 0, 5, 6 -> 4 ; MARRIAGE 0 -> 3."
    X = X.copy()
    X["EDUCATION"] = X["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    X["MARRIAGE"] = X["MARRIAGE"].replace({0: 3})
    return X


def ajoute_derivees(X):
    "Variables dérivées attendues par la chaîne de valeur du sujet."
    X = X.copy()
    X["utilisation"] = X["BILL_AMT1"] / (X["LIMIT_BAL"] + 1)
    X["rembourse_ratio"] = X["PAY_AMT1"] / (X["BILL_AMT2"].abs() + 1)
    return X



def ligne_controle() -> pd.DataFrame:
    base = {"LIMIT_BAL": 120000.0, "SEX": 2.0, "EDUCATION": 2.0, "MARRIAGE": 2.0,
            "AGE": 29.0, "PAY_0": 2.0, "BILL_AMT1": 38500.0, "PAY_AMT1": 1200.0}
    for i in range(2, 7):
        base[f"PAY_{i}"] = 2.0
    for i in range(2, 7):
        base[f"BILL_AMT{i}"] = 38500.0
        base[f"PAY_AMT{i}"] = 1200.0
    return pd.DataFrame([base])[["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
                             + PAY + BILL + PAYAMT]


def cas_controle() -> list[tuple[str, pd.DataFrame, str]]:
    """Tests d'acceptation publics du sujet E6."""
    def ligne(**surcharges):
        base = {"LIMIT_BAL": 120000.0, "SEX": 2.0, "EDUCATION": 2.0, "MARRIAGE": 2.0,
                "AGE": 29.0, "PAY_0": 0.0, "BILL_AMT1": 38500.0, "PAY_AMT1": 1200.0}
        for i in range(2, 7):
            base[f"PAY_{i}"] = 0.0
        for i in range(2, 7):
            base[f"BILL_AMT{i}"] = 30000.0
            base[f"PAY_AMT{i}"] = 1000.0
        base.update(surcharges)
        return pd.DataFrame([base])[["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
                                 + PAY + BILL + PAYAMT]

    cent = pd.concat([ligne()] * 100, ignore_index=True)
    return [
        ("Chargement et probabilité dans [0, 1]", ligne(), "proba"),
        ("Codes rares d'éducation (0, 5, 6) et de statut (0)",
         ligne(EDUCATION=5.0, MARRIAGE=0.0), "proba"),
        ("Retard extrême (PAY_0 = 9 mois)", ligne(PAY_0=9.0), "proba"),
        ("Aucun historique facturé (BILL = 0, PAY_AMT = 0)",
         ligne(**{f"BILL_AMT{i}": 0.0 for i in range(1, 7)}
               | {f"PAY_AMT{i}": 0.0 for i in range(1, 7)}), "proba"),
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
# Interface : saisie de la demande, puis résultats.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CreditRadar", page_icon="💳", layout="wide")
st.title("CreditRadar")
st.caption("Probabilité de défaut de la demande de crédit "
           "(sujet E6, banque de détail taïwanaise).")

with st.form("demande"):
    st.subheader("Saisie de la demande")
    c1, c2, c3 = st.columns(3)
    with c1:
        montant = st.number_input("Montant du crédit demandé ($ TW)", min_value=10000,
                                  max_value=1000000, value=120000, step=10000)
        sexe = st.radio("Sexe (variable sensible, voir énoncé)", ["Femme", "Homme"],
                        horizontal=True)
        education = st.selectbox("Éducation (codes 0, 5, 6 possibles)",
                                 list(EDUCATIONS.values()), index=1)
        mariage = st.selectbox("Statut marital (code 0 possible)",
                               list(MARIAGES.values()), index=1)
    with c2:
        age = st.number_input("Âge", min_value=18, max_value=100, value=29, step=1)
        pay0 = st.slider("Retard le plus récent (PAY_0), en mois", -2, 9, 2,
                         help="0 : à jour ; 1 et plus : mois de retard.")
        bill1 = st.number_input("Facture du dernier mois ($ TW)", min_value=0,
                                max_value=2000000, value=38500, step=1000)
        payamt1 = st.number_input("Remboursement du dernier mois ($ TW)", min_value=0,
                                  max_value=2000000, value=1200, step=100)
    with st.expander("Historique complet des six mois (retards, factures, remboursements)"):
        pays_hist, bills, payamts = [], [], []
        for i in range(2, 7):
            pays_hist.append(st.slider(f"Retard PAY_{i} (mois)", -2, 9, 2, key=f"pay{i}",
                                       help="Valeurs brutes, telles qu'envoyées au modèle."))
        for i in range(2, 7):
            bills.append(st.number_input(f"Facture BILL_AMT{i} ($ TW)", min_value=0,
                                         max_value=2000000, value=38500, step=1000,
                                         key=f"bill{i}"))
        for i in range(2, 7):
            payamts.append(st.number_input(f"Remboursement PAY_AMT{i} ($ TW)", min_value=0,
                                           max_value=2000000, value=1200, step=100,
                                           key=f"payamt{i}"))
        st.caption("L'application envoie au modèle l'historique complet des six mois, "
                   "avec les valeurs brutes.")
    st.form_submit_button("Évaluer la demande", use_container_width=True, type="primary")

entree = pd.DataFrame([{ 
    "LIMIT_BAL": float(montant), "SEX": 2.0 if sexe == "Femme" else 1.0,
    "EDUCATION": float([k for k, v in EDUCATIONS.items() if v == education][0]),
    "MARRIAGE": float([k for k, v in MARIAGES.items() if v == mariage][0]),
    "AGE": float(age), "PAY_0": float(pay0), "BILL_AMT1": float(bill1),
    "PAY_AMT1": float(payamt1),
    **{f"PAY_{i}": float(v) for i, v in zip(range(2, 7), pays_hist)},
    **{f"BILL_AMT{i}": float(v) for i, v in zip(range(2, 7), bills)},
    **{f"PAY_AMT{i}": float(v) for i, v in zip(range(2, 7), payamts)},
}])

proba = float(predit(modele_actif, entree)[0])
ecart = proba * 100.0 - TAUX_BASE

seuil = st.slider("Seuil de coût toléré, en %", 5, 60, int(SEUIL_COUT_DEFAUT),
                  help="règle de décision de l'énoncé : accepter sous la moitié du "
                       "seuil, limiter jusqu'au double, refuser au delà")
limite_suggeree = montant * (1.0 - proba)
if proba * 100.0 < seuil / 2.0:
    decision, detail = "Accepter", "probabilité sous la moitié du seuil"
elif proba * 100.0 < 2.0 * seuil:
    decision = f"Limiter (limite suggérée : {limite_suggeree:,.0f} $ TW)".replace(",", " ")
    detail = "probabilité entre la moitié du seuil et son double"
else:
    decision, detail = "Refuser", "probabilité au delà du double du seuil"

st.subheader("Résultats")
k1, k2, k3 = st.columns(3)
k1.metric("Probabilité de défaut", f"{proba * 100:.0f} %",
          help="défaut à l'échéance suivante")
k2.metric("Écart au taux de base", f"{ecart:+.0f} points",
          help=f"taux de base du fichier : {TAUX_BASE:.1f} %")
k3.metric("Décision au seuil affiché", decision, help=detail)

st.progress(min(proba, 1.0), text=f"probabilité {proba*100:.0f} % contre taux de base {TAUX_BASE:.1f} %")

st.markdown("##### Grille de l'énoncé, en dollars taïwanais")
st.table(pd.DataFrame({
    "Élément": ["Un défaut subi (créance exposée)",
                "La marge annuelle d'un bon client",
                "Le traitement d'une demande"],
    "Coût retenu (hypothèse)": ["40 000 $ TW", "800 $ TW", "20 $ TW"],
}))
st.warning("Avertissement permanent de l'application : score pédagogique, sans usage "
           "réel de crédit ; les variables sensibles ne conduisent pas la décision.")
st.caption("Énoncé : README.md du dossier. Avec le modèle de référence, toutes les "
           "demandes affichent 22,1 % : le tri des demandes vient du modèle chargé.")
