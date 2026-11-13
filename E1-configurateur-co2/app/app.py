# -*- coding: utf-8 -*-
"""ConfigCo2 : anticiper le malus CO2 avant la commande d'une voiture.

Application Streamlit d'inférence du sujet E1 (courtier en leasing). Le
commercial saisit la configuration demandée par son client, l'application
affiche le CO2 WLTP prévu, le malus estimé et l'écart à la cible moyenne des
constructeurs, comme décrit dans la maquette du sujet.

Le modèle chargé par défaut est le meilleur modèle entraîné par le notebook
de correction (forêt aléatoire à 60 arbres, MAE = 4,5 g/km sur 2025 contre
13,0 pour la baseline log-masse + puissance). Tout autre joblib respectant
le contrat du sujet peut être déposé dans la barre latérale.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- Configuration du sujet E1 --------------------------------------------
SUJET = "E1"
MODELE_JOBLIB = "model_E1.joblib"
NOM_MODELE_CORRECTION = "forêt aléatoire (60 arbres), correction E1"
TYPE = "regression"
CLIP = (50.0, 500.0)          # bornes du contrat, en g/km
TAUX_MALUS_EUROS_PAR_GRAMME = 95.0
SEUIL_MALUS = 118.0           # g/km, barème de l'exercice
CIBLE_CONSTRUCTEURS = 93.6    # g/km, cible moyenne de l'énoncé
MARQUES = [
    "Abarth", "Alfa Romeo", "Audi", "Bmw", "Citroen", "Cupra", "Dacia", "Ds",
    "Fiat", "Ford", "Honda", "Hyundai", "Jaguar", "Jeep", "Kia", "Lancia",
    "Land Rover", "Lexus", "Mazda", "Mercedes-Benz", "Mini", "Mitsubishi",
    "Nissan", "Opel", "Peugeot", "Porsche", "Renault", "Seat", "Skoda",
    "Smart", "Ssangyong", "Subaru", "Suzuki", "Toyota", "Volvo",
    "Volkswagen", "Škoda",
]
CARBURANTS = ["essence", "diesel", "hybride essence", "hybride diesel"]


def ligne_controle() -> pd.DataFrame:
    base = {"mass_kg": 1400.0, "engine_power_kw": 85.0, "engine_capacity_cm3": 1500.0,
            "fuel_type": "essence", "hybrid": False, "cylinder_count": 4.0, "make": "Autre"}
    return pd.DataFrame([base])


def cas_controle() -> list[tuple[str, pd.DataFrame, str]]:
    """Tests d'acceptation publics du sujet E1."""
    def ligne(**surcharges):
        base = {"mass_kg": 1400.0, "engine_power_kw": 85.0, "engine_capacity_cm3": 1500.0,
                "fuel_type": "essence", "hybrid": False, "cylinder_count": 4.0, "make": "Peugeot"}
        base.update(surcharges)
        return pd.DataFrame([base])

    cent = pd.concat([ligne()] * 100, ignore_index=True)
    return [
        ("Chargement et prévision d'une ligne", ligne(), "bornes"),
        ("Marque jamais vue (« Autre »)", ligne(make="Autre"), "bornes"),
        ("Valeurs manquantes (masse, cylindrée, cylindres)",
         ligne(mass_kg=np.nan, engine_capacity_cm3=np.nan, cylinder_count=np.nan), "bornes"),
        ("Valeur extrême (masse 3 500 kg)",
         ligne(mass_kg=3500.0, engine_power_kw=200.0, engine_capacity_cm3=3000.0,
               fuel_type="diesel", hybrid=True, cylinder_count=6.0), "bornes"),
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
# Interface : écran de saisie de la configuration, puis résultats.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ConfigCo2", page_icon="🚗", layout="wide")
st.title("ConfigCo2")
st.caption("Anticiper le malus CO2 avant la commande d'une voiture (sujet E1, courtier en leasing).")

DATA_DIR = APP_DIR.parent / "data" / "E1-configurateur-co2"


@st.cache_data(show_spinner=False)
def charge_donnees() -> pd.DataFrame:
    """Copie figée des homologations (réassemblée depuis les parties texte)."""
    complet = DATA_DIR / "co2-voitures-europe.csv"
    if complet.exists():
        return pd.read_csv(complet)
    parties = sorted(DATA_DIR.glob("co2-voitures-europe.part*of3.csv"))
    if not parties:
        return pd.DataFrame()
    complet.write_bytes(b"".join(p.read_bytes() for p in parties))
    return pd.read_csv(complet)


with st.form("configuration"):
    st.subheader("Saisie de la configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        marque_choisie = st.selectbox(
            "Marque", MARQUES + ["Autre (saisie libre)"],
            index=MARQUES.index("Peugeot") if "Peugeot" in MARQUES else 0,
            help="La valeur « Autre » est possible si la marque n'est pas dans la liste.")
        marque = (st.text_input("Nom de la marque (Autre)")
                  if marque_choisie == "Autre (saisie libre)" else marque_choisie) or "Autre"
        carburant = st.selectbox("Carburant", CARBURANTS,
                                 index=CARBURANTS.index("hybride essence"))
        cylindree = st.number_input("Cylindrée (cm³) - la taille du moteur",
                                    min_value=500, max_value=8000, value=1199, step=1)
    with col2:
        puissance = st.number_input("Puissance (kW) - l'énergie maximale que le moteur délivre",
                                    min_value=20, max_value=800, value=96, step=1)
        masse = st.number_input("Masse (kg) - poids prêt à rouler, pleins faits",
                                min_value=500, max_value=4000, value=1395, step=1)
        cylindres = st.number_input("Nombre de cylindres", min_value=1, max_value=16,
                                    value=3, step=1)
    st.caption("Le champ « hybride » est déduit du carburant : une motorisation "
               "« hybride essence » ou « hybride diesel » est hybride par définition.")
    st.form_submit_button("Prévoir le CO2 et le malus", use_container_width=True, type="primary")

entree = pd.DataFrame([{
    "mass_kg": float(masse),
    "engine_power_kw": float(puissance),
    "engine_capacity_cm3": float(cylindree),
    "fuel_type": carburant,
    "hybrid": carburant.startswith("hybride"),
    "cylinder_count": float(cylindres),
    "make": marque,
}])

co2 = float(predit(modele_actif, entree)[0])
malus = max(0.0, co2 - SEUIL_MALUS) * TAUX_MALUS_EUROS_PAR_GRAMME
ecart_cible = co2 - CIBLE_CONSTRUCTEURS

st.subheader("Résultats")
k1, k2, k3 = st.columns(3)
k1.metric("CO2 WLTP prévu", f"{co2:.0f} g/km",
          delta=f"{ecart_cible:+.0f} g vs cible {CIBLE_CONSTRUCTEURS:.1f}".replace("+", "+") if ecart_cible else None,
          help="mesure normalisée des émissions, bornée au contrat [50, 500] g/km")
k2.metric("Malus estimé", f"{malus:,.0f} €".replace(",", " "),
          help=f"{max(0.0, co2 - SEUIL_MALUS):.0f} g au dessus du seuil de "
               f"{SEUIL_MALUS:.0f} g/km, à {TAUX_MALUS_EUROS_PAR_GRAMME:.0f} € le gramme")
k3.metric("Écart à la cible constructeurs", f"{ecart_cible:+.0f} g",
          help="cible moyenne : 93,6 g/km")

# Échelle des émissions, à la façon de la maquette.
gauche, droite = 40.0, 260.0
position = min(max(co2, gauche), droite)
p1 = (CIBLE_CONSTRUCTEURS - gauche) / (droite - gauche) * 100
p2 = (SEUIL_MALUS - gauche) / (droite - gauche) * 100
p3 = (position - gauche) / (droite - gauche) * 100
st.markdown(
    f"""<div style="position:relative;height:56px;margin:8px 0">
    <div style="position:absolute;top:20px;left:0;right:0;height:14px;border-radius:7px;
    background:linear-gradient(90deg,#b7e4c7 0%,#b7e4c7 {p1}%,#fde68a {p1}%,#fca5a5 100%)"></div>
    <div style="position:absolute;top:12px;left:{p1}%;width:2px;height:30px;background:#374151"></div>
    <div style="position:absolute;top:44px;left:{p1}%;transform:translateX(-50%);
    font-size:11px;color:#374151">cible {CIBLE_CONSTRUCTEURS:.1f}</div>
    <div style="position:absolute;top:12px;left:{p2}%;width:2px;height:30px;
    background:#374151;border-left:2px dashed #374151"></div>
    <div style="position:absolute;top:44px;left:{p2}%;transform:translateX(-50%);
    font-size:11px;color:#374151">seuil malus {SEUIL_MALUS:.0f}</div>
    <div style="position:absolute;top:8px;left:{p3}%;width:3px;height:38px;background:#0b7a4b"></div>
    <div style="position:absolute;top:-8px;left:{p3}%;transform:translateX(-50%);
    font-size:11px;color:#0b7a4b"><b>prévu {co2:.0f}</b></div></div>""",
    unsafe_allow_html=True,
)

donnees = charge_donnees()
if not donnees.empty:
    with st.expander("Les configurations réelles les plus proches (argumentaire commercial)", expanded=True):
        fond = donnees.dropna(subset=["mass_kg", "engine_power_kw"]).copy()
        fond["d2"] = ((fond["mass_kg"] - masse) / 400.0) ** 2 \
            + ((fond["engine_power_kw"] - puissance) / 100.0) ** 2
        proches = fond.nsmallest(10, "d2")
        st.dataframe(
            proches.rename(columns={
                "make": "Marque", "fuel_type": "Carburant", "mass_kg": "Masse (kg)",
                "engine_power_kw": "Puissance (kW)", "ewltp_gkm": "CO2 mesuré (g/km)",
            })[["Marque", "Carburant", "Masse (kg)", "Puissance (kW)", "CO2 mesuré (g/km)"]],
            hide_index=True, use_container_width=True,
        )
        if len(proches):
            mini, maxi = float(proches["ewltp_gkm"].min()), float(proches["ewltp_gkm"].max())
            st.info(
                "Argumentaire commercial : « des véhicules comparables, déjà homologués, "
                f"émettent entre {mini:.0f} et {maxi:.0f} g/km »."
            )

st.info(
    "Énoncé : README.md du dossier. Malus calculé avec le barème de l'exercice : "
    f"seuil {SEUIL_MALUS:.0f} g/km, environ {TAUX_MALUS_EUROS_PAR_GRAMME:.0f} € "
    "par gramme au dessus."
)
