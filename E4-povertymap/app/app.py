# -*- coding: utf-8 -*-
"""PovertyMap : taux de pauvreté par pays, mesures réelles et estimations.

Application Streamlit d'inférence du sujet E4 (organisation d'aide au
développement). Le responsable de programme choisit l'année et bascule entre
les mesures réelles et les estimations du modèle ; l'application affiche la
carte, la fiche pays (estimation, dernière vraie enquête) et la distribution
régionale, comme décrit dans la maquette du sujet.

Le modèle chargé par défaut est le meilleur modèle entraîné par le notebook
de correction (forêt aléatoire à 200 arbres, MAE = 2,6 points sur le test
contre 3,3 pour la persistance). Tout autre joblib respectant le contrat du
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

# --- Configuration du sujet E4 --------------------------------------------
SUJET = "E4"
MODELE_JOBLIB = "model_E4.joblib"
NOM_MODELE_CORRECTION = "forêt aléatoire (200 arbres, feuille 5), correction E4"
TYPE = "regression"
CLIP = (0.0, 100.0)           # bornes du contrat, en points de pauvreté
CIBLE = "SI.POV.DDAY"         # taux de pauvreté au seuil de 3,00 dollars/jour
INDS = ["NY.GDP.PCAP.PP.CD", "NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG", "NE.EXP.GNFS.ZS",
        "BX.KLT.DINV.WD.GD.ZS", "EG.ELC.ACCS.ZS", "IT.NET.USER.ZS", "SL.UEM.TOTL.ZS",
        "SP.DYN.TFRT.IN", "SP.POP.GROW", "EN.POP.DNST"]
FEATURES = INDS + ["region", "iso3"]
LIGNE_PAUVRETE = "3,00 $/jour (PPA 2021)"


def ligne_controle() -> pd.DataFrame:
    base = {"iso3": "FRA", "region": "Europe & Central Asia",
            "NY.GDP.PCAP.PP.CD": 45000.0, "NY.GDP.MKTP.KD.ZG": 1.5,
            "FP.CPI.TOTL.ZG": 2.1, "NE.EXP.GNFS.ZS": 30.0,
            "BX.KLT.DINV.WD.GD.ZS": 2.0, "EG.ELC.ACCS.ZS": 100.0,
            "IT.NET.USER.ZS": 85.0, "SL.UEM.TOTL.ZS": 7.5,
            "SP.DYN.TFRT.IN": 1.8, "SP.POP.GROW": 0.3, "EN.POP.DNST": 120.0}
    return pd.DataFrame([base])[FEATURES]


def cas_controle() -> list[tuple[str, pd.DataFrame, str]]:
    """Tests d'acceptation publics du sujet E4."""
    def ligne(**surcharges):
        base = {"iso3": "ETH", "region": "Sub-Saharan Africa",
                "NY.GDP.PCAP.PP.CD": 2500.0, "NY.GDP.MKTP.KD.ZG": 6.0,
                "FP.CPI.TOTL.ZG": 20.0, "NE.EXP.GNFS.ZS": 10.0,
                "BX.KLT.DINV.WD.GD.ZS": 3.0, "EG.ELC.ACCS.ZS": 55.0,
                "IT.NET.USER.ZS": 25.0, "SL.UEM.TOTL.ZS": 5.0,
                "SP.DYN.TFRT.IN": 4.0, "SP.POP.GROW": 2.5, "EN.POP.DNST": 100.0}
        base.update(surcharges)
        return pd.DataFrame([base])[FEATURES]

    cent = pd.concat([ligne()] * 100, ignore_index=True)
    return [
        ("Chargement et estimation d'une ligne", ligne(), "bornes"),
        ("Pays jamais vu (« ZZZ »)", ligne(iso3="ZZZ", region="Autre"), "bornes"),
        ("Région jamais vue (« Mars »)",
         ligne(region="Mars", **{"NY.GDP.PCAP.PP.CD": np.nan}), "bornes"),
        ("Tous les indicateurs manquants",
         ligne(**{c: np.nan for c in INDS}), "bornes"),
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
# Interface : année et bascule mesures/estimations, carte et fiche pays.
# ---------------------------------------------------------------------------
import plotly.express as px

st.set_page_config(page_title="PovertyMap", page_icon="🌍", layout="wide")
st.title("PovertyMap")
st.caption("Taux de pauvreté par pays, mesures réelles et estimations "
           "(sujet E4, organisation d'aide au développement).")

DATA_DIR = APP_DIR.parent / "data" / "E4-poverty"


@st.cache_data(show_spinner=False)
def charge_panel() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "panel_E4.csv")


panel = charge_panel()
if panel.empty:
    st.error("Panel absent : data/E4-poverty/panel_E4.csv est attendu à côté du "
             "dossier de l'application.")
    st.stop()

annees = sorted(int(a) for a in panel["year"].dropna().unique())


@st.cache_data(show_spinner=False)
def estimations(annee: int, nom_modele: str) -> pd.DataFrame:
    """Estimation du modèle pour tous les pays disposant d'une ligne cette année."""
    lignes = panel[panel["year"] == annee].copy()
    if lignes.empty:
        lignes = panel[panel["iso3"].isin(panel["iso3"].unique())]
        lignes = lignes.sort_values("year").groupby("iso3").tail(1)
    prev = predit(modele_actif, lignes[FEATURES])
    lignes["estimation"] = prev
    return lignes[["iso3", "region", "estimation"]].copy()


def tranche_label(valeur: float) -> str:
    if valeur >= 30.0:
        return "Forte pauvreté estimée"
    if valeur >= 10.0:
        return "Pauvreté modérée estimée"
    return "Pauvreté faible estimée"


with st.sidebar:
    st.subheader("Affichage")
    annee = st.select_slider("Année", options=annees, value=annees[-1])
    mode = st.radio("Afficher", ["Estimations du modèle", "Mesures réelles"],
                    index=0, help="L'autre position affiche les mesures réelles, "
                                   "quand elles existent.")
    st.caption(f"Ligne de pauvreté : {LIGNE_PAUVRETE}")

st.info("Une estimation n'est pas une mesure. Le bandeau est permanent : les "
        "estimations affichées dans cette application ne remplacent pas les "
        "enquêtes ménages officielles.")

tab = estimations(int(annee), f"{nom_actif} ({origine})")
hist = panel[panel[CIBLE].notna() & (panel["year"] <= annee)] \
    .sort_values("year").groupby("iso3").tail(1)[["iso3", "year", CIBLE]]
hist.columns = ["iso3", "annee_enquete", "mesure"]

pays_choisi = st.selectbox("Pays sélectionné", sorted(tab["iso3"].unique()))
ligne_pays = tab[tab["iso3"] == pays_choisi].iloc[0]
hist_pays = hist[hist["iso3"] == pays_choisi]

c1, c2 = st.columns([2, 3])
with c1:
    st.markdown(f"### {pays_choisi}")
    if mode == "Estimations du modèle":
        st.metric(f"Estimation {annee}", f"{ligne_pays.estimation:.1f} %",
                  help=f"part de la population sous {LIGNE_PAUVRETE}")
        st.caption(tranche_label(float(ligne_pays.estimation)))
    else:
        if hist_pays.empty:
            st.metric(f"Mesure {annee}", "aucune")
            st.caption("Ce pays n'a aucune enquête dans le panel : basculer sur "
                       "les estimations du modèle.")
        else:
            st.metric(f"Dernière vraie enquête {int(hist_pays.annee_enquete.iloc[0])}",
                      f"{hist_pays.mesure.iloc[0]:.1f} %",
                      help="la mesure réelle la plus récente, à l'année choisie ou avant")
    if not hist_pays.empty and mode == "Estimations du modèle":
        anciennete = int(annee) - int(hist_pays.annee_enquete.iloc[0])
        st.caption(f"Dernière vraie enquête : {int(hist_pays.annee_enquete.iloc[0])} "
                   f"(ancienneté : {anciennete} ans) ; mesure "
                   f"{hist_pays.mesure.iloc[0]:.1f} % contre "
                   f"{ligne_pays.estimation:.1f} % estimés.")
    st.caption("Mesure ou estimation ? Le statut est affiché dans la bascule et "
               "l'export.")
with c2:
    if mode == "Estimations du modèle":
        carte = tab.copy()
        valeur, titre_carte = "estimation", f"Estimations {annee} (% de la population)"
    else:
        carte = hist.copy()
        carte = carte.rename(columns={"mesure": "estimation"})
        valeur, titre_carte = "estimation", f"Mesures réelles, dernière enquête à l'année {annee} ou avant (%)"
    fig = px.choropleth(carte, locations="iso3", locationmode="ISO-3", color=valeur,
                        hover_name="iso3", color_continuous_scale="YlOrRd",
                        range_color=(0, 60), title=titre_carte)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=420)
    st.plotly_chart(fig, use_container_width=True)

st.markdown(f"##### Distribution régionale, {mode.lower()} {annee}")
fond_region = tab if mode == "Estimations du modèle" else hist.rename(columns={"mesure": "estimation"})
regionale = (fond_region.groupby("region")["estimation"].median()
             .sort_values(ascending=False).reset_index())
regionale.columns = ["Région", "Médiane (%)"]
st.table(regionale.round(1).reset_index(drop=True))

export = tab.rename(columns={"estimation": "estimation_pct"})
export["mesure_disponible"] = export["iso3"].isin(hist["iso3"])
csv = export.to_csv(index=False).encode("utf-8")
st.download_button("Exporter le tableau des estimations (CSV)", csv,
                   file_name=f"estimations_pauvrete_{annee}.csv", mime="text/csv")

st.caption("Énoncé : README.md du dossier. Avec le modèle de référence "
           "(persistance), un pays sans enquête afficherait sa dernière valeur "
           "publiée : la valeur ajoutée du groupe se lit dans l'écart entre les "
           "deux chiffres.")
