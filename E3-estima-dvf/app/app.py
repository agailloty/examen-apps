# -*- coding: utf-8 -*-
"""EstimaDVF : estimation immobilière argumentée par des ventes réelles.

Application Streamlit d'inférence du sujet E3 (réseau d'agences). Le
négociateur saisit le bien en barre latérale (type, commune, surface, pièces,
terrain), l'argumentaire s'affiche en zone principale : prix au m² estimé,
intervalle annoncé au vendeur, prix total et ventes réelles comparables, comme
décrit dans la maquette du sujet.

Le modèle chargé par défaut est le meilleur modèle entraîné par le notebook
de correction (forêt aléatoire à 100 arbres, erreur médiane 14,1 % sur le test
contre 15,8 % pour la référence communale). Tout autre joblib respectant le
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

# --- Configuration du sujet E3 --------------------------------------------
SUJET = "E3"
MODELE_JOBLIB = "model_E3.joblib"
NOM_MODELE_CORRECTION = "forêt aléatoire (100 arbres), correction E3"
TYPE = "regression"
CLIP = (200.0, 12000.0)       # bornes du contrat, en EUR par m²
TYPES_LOCAL = ["Appartement", "Maison"]
MARGE_INTERVALLE = 0.11        # marge d'incertitude annoncée au vendeur
NATURE_MUTATION = "Vente"      # le modèle n'a vu que des ventes


def ligne_controle() -> pd.DataFrame:
    base = {"surface_reelle_bati": 68.0, "nombre_pieces": 3.0, "surface_terrain": np.nan,
            "lat": 45.75, "lon": 4.85, "type_local": "Appartement",
            "nature_mutation": NATURE_MUTATION}
    return pd.DataFrame([base])


def cas_controle() -> list[tuple[str, pd.DataFrame, str]]:
    """Tests d'acceptation publics du sujet E3."""
    def ligne(**surcharges):
        base = {"surface_reelle_bati": 68.0, "nombre_pieces": 3.0, "surface_terrain": np.nan,
                "lat": 45.75, "lon": 4.85, "type_local": "Appartement",
                "nature_mutation": NATURE_MUTATION}
        base.update(surcharges)
        return pd.DataFrame([base])

    cent = pd.concat([ligne()] * 100, ignore_index=True)
    return [
        ("Chargement et estimation d'une ligne", ligne(), "bornes"),
        ("Type de local jamais vu (« Château »)", ligne(type_local="Château"), "bornes"),
        ("Surface de terrain manquante (appartement)", ligne(), "bornes"),
        ("Valeurs manquantes (surface, pièces)",
         ligne(surface_reelle_bati=np.nan, nombre_pieces=np.nan, surface_terrain=np.nan),
         "bornes"),
        ("Valeur extrême (surface 1 500 m²)",
         ligne(surface_reelle_bati=1500.0, nombre_pieces=12.0,
               surface_terrain=8000.0, type_local="Maison"), "bornes"),
        ("Vitesse : 100 lignes en moins d'une seconde", cent, "vitesse"),
    ]


# ---------------------------------------------------------------------------
# Chargement du modèle : correction par défaut, dépôt joblib au choix.
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODELE_DEFAUT = APP_DIR / "models" / MODELE_JOBLIB
MODELE_ZIP = APP_DIR / "models" / "model_E3.zip"


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
    if not MODELE_DEFAUT.exists() and MODELE_ZIP.exists():
        import zipfile
        with zipfile.ZipFile(MODELE_ZIP, "r") as zf:
            zf.extractall(MODELE_DEFAUT.parent)
    if not MODELE_DEFAUT.exists():
        st.error(
            f"{MODELE_DEFAUT.name} (ou {MODELE_ZIP.name}) est absent du dossier models/ de l'application."
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
# Interface : saisie du bien en barre latérale, argumentaire en zone principale.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="EstimaDVF", page_icon="🏠", layout="wide")
st.title("EstimaDVF")
st.caption("Estimation immobilière argumentée par des ventes réelles "
           "(sujet E3, réseau d'agences).")

DATA_APP = APP_DIR / "data"
FICHIER_COMMUNES = DATA_APP / "communes_E3.csv"
FICHIER_VENTES = DATA_APP / "ventes_E3.csv"


@st.cache_data(show_spinner=False)
def charge_communes() -> pd.DataFrame:
    if FICHIER_COMMUNES.exists():
        return pd.read_csv(FICHIER_COMMUNES)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def charge_ventes() -> pd.DataFrame:
    if FICHIER_VENTES.exists():
        return pd.read_csv(FICHIER_VENTES)
    return pd.DataFrame()


communes = charge_communes()
ventes = charge_ventes()

# --- Saisie du bien, en barre latérale, à la maquette -----------------------
with st.sidebar:
    st.subheader("Saisie du bien")
    type_local = st.radio("Type de local", TYPES_LOCAL, horizontal=True)
    if not communes.empty:
        etiquettes = {f"{r.nom} ({int(r.code_commune)})": r for r in communes.itertuples()}
        choix = st.selectbox("Commune", list(etiquettes))
        commune = etiquettes[choix]
        lat, lon = float(commune.lat), float(commune.lon)
        prix_median_communal = float(commune.prix_median)
    else:
        st.caption("Référentiel des communes absent : saisie manuelle des coordonnées.")
        lat = st.number_input("Latitude", min_value=41.0, max_value=52.0, value=45.75, step=0.001)
        lon = st.number_input("Longitude", min_value=-6.0, max_value=10.0, value=4.85, step=0.001)
        prix_median_communal = None
    surface = st.number_input("Surface bâtie (m²)", min_value=5, max_value=1500, value=68, step=1)
    pieces = st.number_input("Nombre de pièces", min_value=1, max_value=20, value=3, step=1)
    avec_terrain = st.checkbox("Surface de terrain renseignée",
                               value=(type_local == "Maison"),
                               help="Laisser vide pour un appartement : le manquant est "
                                    "accepté par le contrat.")
    terrain = st.number_input("Surface du terrain (m²)", min_value=0, max_value=100000,
                              value=250 if avec_terrain else 0) if avec_terrain else np.nan
    trimestre = st.selectbox("Date de négociation",
                              ["1er trimestre 2025", "2e trimestre 2025",
                               "3e trimestre 2025", "4e trimestre 2025"], index=1)
    st.caption("La date oriente l'argumentaire commercial ; le contrat du modèle "
               "ne comprend pas la date.")
    estimer = st.button("Estimer", type="primary", use_container_width=True)

if not estimer:
    st.info("Le négociateur ajuste la saisie face au vendeur, l'argumentaire se "
            "met à jour dans cette zone après « Estimer ».")
    st.stop()

entree = pd.DataFrame([{
    "surface_reelle_bati": float(surface), "nombre_pieces": float(pieces),
    "surface_terrain": float(terrain) if avec_terrain else np.nan,
    "lat": lat, "lon": lon, "type_local": type_local,
    "nature_mutation": NATURE_MUTATION,
}])
prix_m2 = float(predit(modele_actif, entree)[0])
total = prix_m2 * surface

st.subheader("Estimation")
k1, k2, k3 = st.columns(3)
k1.metric("Prix au m² estimé", f"{prix_m2:,.0f} €/m²".replace(",", " "),
          help="sortie du modèle, bornée au contrat [200, 12 000] €/m²")
if prix_median_communal is not None:
    k2.metric("Référence communale", f"{prix_median_communal:,.0f} €/m²".replace(",", " "),
              delta=f"{prix_m2 - prix_median_communal:+,.0f} €/m²".replace(",", " "),
              help="médiane des ventes de la commune : la valeur ajoutée du modèle se "
                   "lit dans l'écart à cette référence.")
else:
    k2.metric("Référence communale", "n.d.")
k3.metric("Prix total du bien", f"{total:,.0f} €".replace(",", " "),
          help=f"{surface} m² multipliés par l'estimation")

basse, haute = prix_m2 * (1 - MARGE_INTERVALLE), prix_m2 * (1 + MARGE_INTERVALLE)
st.caption(f"Intervalle annoncé au vendeur : {basse:,.0f} à {haute:,.0f} €/m² "
           f"(marge d'incertitude de {MARGE_INTERVALLE*100:.0f} %).".replace(",", " "))

# --- Argumentaire : les ventes réelles les plus proches ---------------------
if not ventes.empty:
    candidats = ventes.copy()
    if not communes.empty:
        meme_commune = candidats[candidats["code_commune"] == int(commune.code_commune)] \
            if not communes.empty and "code_commune" in candidats else candidats.iloc[0:0]
        fond = meme_commune if len(meme_commune) >= 5 else candidats
    else:
        fond = candidats
    fond = fond.copy()
    fond["ecart_surface"] = (fond["surface_reelle_bati"] - surface).abs()
    proches = fond.nsmallest(5, "ecart_surface")
    if len(proches):
        st.markdown("##### Les ventes réelles les plus proches (argumentaire du mandat)")
        tableau = proches.rename(columns={
            "nom": "Commune", "surface_reelle_bati": "Surface (m²)",
            "valeur_fonciere": "Prix de vente", "prix_m2": "Prix au m²",
            "date_mutation": "Date",
        })[["Commune", "Surface (m²)", "Prix de vente", "Prix au m²", "Date"]]
        for col in ("Prix de vente", "Prix au m²"):
            tableau[col] = tableau[col].map(lambda v: f"{v:,.0f} €".replace(",", " "))
        st.table(tableau.reset_index(drop=True))

# --- Domaine couvert --------------------------------------------------------
avertissements = []
if surface < 9 or surface > 600:
    avertissements.append(f"la surface ({surface} m²) sort de la plage courante des ventes")
if not communes.empty:
    lat_min, lat_max = communes["lat"].min(), communes["lat"].max()
    lon_min, lon_max = communes["lon"].min(), communes["lon"].max()
    if not (lat_min - 0.2 <= lat <= lat_max + 0.2 and lon_min - 0.2 <= lon <= lon_max + 0.2):
        avertissements.append("la commune sort des départements couverts "
                              "(75, 44, 49)")
if avertissements:
    st.warning("Saisie hors du domaine couvert : " + " ; ".join(avertissements)
               + ". L'estimation est donnée à titre indicatif.")
else:
    st.success("Domaine couvert : cette saisie est dans le domaine d'entraînement.")

st.caption("Énoncé : README.md du dossier. Le modèle ignore l'identité de la rue : "
           "la localisation entre par les coordonnées de la commune.")
