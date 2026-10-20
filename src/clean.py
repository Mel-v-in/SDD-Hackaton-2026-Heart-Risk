"""Decodage BRFSS 2022 : donnees brutes -> features propres.

`clean` est une fonction pure (aucun fit), appliquee a l'identique a train, extra et test.

Les regles sont derivees des distributions reelles des valeurs (le codebook CDC n'etait pas
accessible lors de l'ecriture) et du codebook BRFSS 2022 tel que connu :
  - variables Oui/Non : 1 = oui, 2 = non, 7 = ne sait pas, 9 = refus
  - variables a 77/99 : 77 = ne sait pas, 99 = refus (7 et 9 sont alors des modalites reelles)
  - variables "nombre de jours" : 88 = aucun
  - variables calculees `_XXX` (CDC) : 9 = ne sait pas / refus / manquant
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- Colonnes administratives / screening telephonique / poids : sans lien causal ---
ADMIN_DROP = [
    "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU", "QSTVER", "QSTLANG",
    "CTELENM1", "PVTRESD1", "COLGHOUS", "STATERE1", "CELPHON1", "LADULT1", "COLGSEX1", "NUMADULT",
    "LANDSEX1", "NUMMEN", "NUMWOMEN", "RESPSLCT", "SAFETIME", "CTELNUM1", "CELLFON5", "CADULT1",
    "CELLSEX1", "PVTRESD3", "CCLGHOUS", "CSTATE1", "LANDLINE", "HHADULT",
    "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_DUALUSE", "_DUALCOR",
    "_LLCPWT2", "_CLLCPWT", "CAGEG",  # poids / groupe d'age de l'echantillon cellulaire
]
# Poids d'enquete : conserves a part pour les estimations ponderees en dataviz, jamais comme feature.
WEIGHT_COL = "_LLCPWT"

# Colonnes a plus de 95 % de NaN dans train (EDA du 20/10/2026), AUC univariee <= 0,52 : sans signal.
SPARSE_DROP = (
    "COLGHOUS COLGSEX1 CCLGHOUS NUMPHON4 LASTSIG4 VCLNTES2 BLDSTFIT SDNATES1 DIABTYPE INSULIN1 "
    "CHKHEMO3 EYEEXAM1 DIABEYE1 DIABEDU1 FEETSORE TOLDCFS HAVECFS WORKCFS IMFVPLA3 HPVADVC4 HPVADSHT "
    "SHINGLE2 COPDCOGH COPDFLEM COPDBRTH COPDBTST COPDSMOK CNCRDIFF CNCRAGE CNCRTYP2 CSRVTRT3 CSRVDOC1 "
    "CSRVSUM CSRVRTRN CSRVINST CSRVINSR CSRVDEIN CSRVCLIN CSRVPAIN CSRVCTL2 PSATEST1 PSATIME1 PCPSARS2 "
    "PSASUGST PCSTALK1 CDHOUSE CDASSIST CDHELP CDSOCIAL CDDISCUS CRGVREL4 CRGVLNG1 CRGVHRS1 CRGVPRB3 "
    "CRGVALZD CRGVPER1 CRGVHOU1 MARJSMOK MARJEAT MARJVAPE MARJDAB MARJOTHR USEMRJN4 STOPSMK2 MENTCIGS "
    "MENTECIG GUNLOAD LOADULK2 RCSXBRTH CASTHNO2 HADSEX PFPPRVN4 TYPCNTR9 BRTHCNT4 WHEREGET NOBCUSE8 "
    "BCPREFER"
).split()

# Poids et taille declares (unites melangees, 9999 = refus) : on prefere WTKG3, HTM4, _BMI5 calcules par le CDC.
# Dates MMYYYY (777777/999999) : peu informatives, supprimees.
RAW_DROP = ["WEIGHT2", "HEIGHT3", "FLSHTMY3", "HIVTSTD3", "COVIDFS1", "COVIDSE1"]

# Variables "nombre de jours / de verres" : 88 = aucun -> 0 ; 77 = ne sait pas, 99 = refus -> NaN
ZERO_88 = ["PHYSHLTH", "MENTHLTH", "POORHLTH", "DRNK3GE5", "CHILDREN", "MARIJAN1"]
# Quantites numeriques : 77 / 99 -> NaN
NUM_77_99 = ["SLEPTIM1", "AVEDRNK3", "MAXDRNKS"]
# 7 est une modalite reelle (pas "ne sait pas") quand la variable n'a pas de code 77
KEEP_7 = {"EMPLOY1", "_RACE1", "_RACEPR1", "_INCOMG1"}  # 7 = modalite reelle (retraite, multiracial, hispanique, revenu 200k+)
# Ordinales dont la modalite "jamais" / "aucun" (8) doit etre replacee sur l'echelle
ORDINAL_8 = {"CHECKUP1": 5, "LASTDEN4": 6, "RMVTETH4": 0}

# Binaires (Oui/Non) utilises pour les comptages de features
COMORBID_YES = ["CVDSTRK3", "CHCCOPD3", "CHCKDNY2", "HAVARTH4", "ASTHMA3", "CHCSCNC1", "CHCOCNC1", "ADDEPEV3"]
LIMITATIONS = ["DEAF", "BLIND", "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON"]


def _is_dk_code(col: str, values: set) -> tuple[bool, bool]:
    """Retourne (7 est DK, 9 est refus) d'une variable categorielle selon ses codes presents."""
    if 77 in values or 99 in values:
        return False, False
    return col not in KEEP_7, True


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Decode un DataFrame BRFSS brut (train, extra ou test). Conserve ID, TARGET et _LLCPWT s'ils existent."""
    out = df.copy()

    # 1. comptage des reponses "ne sait pas / refus" sur les questions coeur (avant decodage)
    core = ["GENHLTH", "CVDSTRK3", "DIABETE4", "CHCKDNY2", "CHCCOPD3", "HAVARTH4", "ASTHMA3", "EXERANY2",
            "SMOKE100", "ADDEPEV3", "CHCSCNC1", "CHCOCNC1", "DIFFWALK", "MEDCOST1", "PERSDOC3"]
    core = [c for c in core if c in out]
    out["n_dk_core"] = out[core].isin([7, 9]).sum(axis=1).astype("float32")

    # 2. suppression des colonnes inutiles
    drop = [c for c in ADMIN_DROP + SPARSE_DROP + RAW_DROP if c in out]
    out = out.drop(columns=drop)

    # 3. variables numeriques specifiques
    for c in ZERO_88:
        if c in out:
            v = out[c]
            out[c] = v.mask(v.isin([77, 99])).where(v != 88, 0)
    for c in NUM_77_99:
        if c in out:
            out[c] = out[c].mask(out[c].isin([77, 99]))
    if "DIABAGE4" in out:
        out["DIABAGE4"] = out["DIABAGE4"].mask(out["DIABAGE4"].isin([98, 99]))
    for c in ("LCSFIRST", "LCSLAST", "LCSNUMCG"):
        if c in out:
            out[c] = out[c].mask(out[c] >= 777)
    if "ALCDAY4" in out:  # 1xx = jours/semaine, 2xx = jours/mois, 888 = aucun
        a = out["ALCDAY4"]
        days = pd.Series(np.nan, index=a.index, dtype="float32")
        wk = (a >= 101) & (a <= 199)
        mo = (a >= 201) & (a <= 299)
        days[wk] = (a[wk] - 100) * 30.0 / 7.0
        days[mo] = a[mo] - 200
        days[a == 888] = 0
        out["ALCDAY4"] = days

    # 4. variables calculees CDC en unites x100
    if "WTKG3" in out:
        out["WTKG3"] = out["WTKG3"] / 100.0
    if "_BMI5" in out:
        out["_BMI5"] = out["_BMI5"] / 100.0
    if "_DRNKWK2" in out:
        out["_DRNKWK2"] = out["_DRNKWK2"].mask(out["_DRNKWK2"] >= 99900) / 100.0
    if "DROCDY4_" in out:
        out["DROCDY4_"] = out["DROCDY4_"].mask(out["DROCDY4_"] >= 900)
    if "_AGEG5YR" in out:
        out["_AGEG5YR"] = out["_AGEG5YR"].mask(out["_AGEG5YR"] == 14)
    if "_AGE65YR" in out:
        out["_AGE65YR"] = out["_AGE65YR"].mask(out["_AGE65YR"] == 3)
    for c in ("_PRACE2", "_MRACE2", "_CRACE2", "_CPRACE2"):
        if c in out:
            out[c] = out[c].mask(out[c].isin([77, 99]))

    # 5. DIABETE4 : ordinal propre (0 non, 1 prediabete / gestationnel seul, 2 oui) + indicateur
    if "DIABETE4" in out:
        d = out["DIABETE4"].map({3: 0, 4: 1, 2: 1, 1: 2})
        out["diab_status"] = d.astype("float32")
        out = out.drop(columns=["DIABETE4"])

    # 6. recodages generiques pour les colonnes categorielles restantes
    special = set(ZERO_88 + NUM_77_99 + ["DIABAGE4", "LCSFIRST", "LCSLAST", "LCSNUMCG", "ALCDAY4", "WTKG3", "_BMI5",
                                          "_DRNKWK2", "DROCDY4_", "_AGE80", "HTIN4", "HTM4", "_YRSSMOK", "_PACKDAY",
                                          "_PACKYRS", "_YRSQUIT", "_STATE", "ID", "TARGET", WEIGHT_COL,
                                          "n_dk_core", "diab_status", "_AGEG5YR", "_AGE65YR", "_PRACE2",
                                          "_MRACE2", "_CRACE2", "_CPRACE2"])
    for c in list(out.columns):
        if c in special:
            continue
        s = out[c]
        vals = set(pd.unique(s.dropna()).tolist())
        if not vals or max(vals) > 99 or len(vals) > 13:
            continue
        if c in ORDINAL_8:
            s = s.mask(s.isin([7, 9])).where(s != 8, ORDINAL_8[c])
            out[c] = s
            continue
        dk7, rf9 = _is_dk_code(c, vals)
        if 77 in vals or 99 in vals:
            s = s.mask(s.isin([77, 99]))
        else:
            if dk7:
                s = s.mask(s == 7)
            if rf9:
                s = s.mask(s == 9)
        # Oui/Non (1/2) -> 1/0 pour les colonnes sans autre modalite
        left = set(pd.unique(s.dropna()).tolist())
        if left and left <= {1, 2} and not c.startswith("_"):
            s = (2 - s)  # 1 -> 1, 2 -> 0
        out[c] = s

    return out.astype({c: "float32" for c in out.columns if c not in ("ID", "TARGET") and out[c].dtype == np.float64})
