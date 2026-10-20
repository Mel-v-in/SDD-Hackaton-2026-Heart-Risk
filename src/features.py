"""Feature engineering (n_comorbid, n_limitations, interactions, statut tabagique...).

Meme principe que clean.py : fonction pure appliquee a l'identique a train, extra et test.
Entree : DataFrame deja passe par `clean`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.clean import COMORBID_YES, LIMITATIONS


def _ones(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.concat([(df[c] == 1) for c in cols if c in df], axis=1).astype("float32")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    f = {}

    # comptages "storytellable"
    com = _ones(out, COMORBID_YES)
    f["n_comorbid"] = com.sum(axis=1) + (out["diab_status"] == 2).astype("float32")
    f["n_limitations"] = _ones(out, LIMITATIONS).sum(axis=1)

    age = out["_AGE80"]
    male = out["SEXVAR"]  # 1 = homme, 0 = femme apres clean
    ever_smoker = out["SMOKE100"]
    diab = (out["diab_status"] == 2).astype("float32")
    gen = out["GENHLTH"]
    f["age_x_male"] = age * male
    f["age_x_smoker"] = age * ever_smoker
    f["age_x_diab"] = age * diab
    f["age_x_genhlth"] = age * gen
    f["age_x_comorbid"] = age * f["n_comorbid"]
    f["male_x_genhlth"] = male * gen
    f["comorbid_x_limit"] = f["n_comorbid"] * f["n_limitations"]

    # tabac / poids / sommeil / alcool
    f["obese"] = (out["_BMI5"] >= 30).astype("float32").where(out["_BMI5"].notna())
    f["sleep_short"] = (out["SLEPTIM1"] < 6).astype("float32").where(out["SLEPTIM1"].notna())
    f["sleep_long"] = (out["SLEPTIM1"] > 9).astype("float32").where(out["SLEPTIM1"].notna())
    f["pack_years_x_age"] = out["_PACKYRS"] * age

    # sante declaree
    f["fair_poor_health"] = (gen >= 4).astype("float32").where(gen.notna())
    f["phys_poor"] = (out["PHYSHLTH"] >= 14).astype("float32").where(out["PHYSHLTH"].notna())
    f["ment_poor"] = (out["MENTHLTH"] >= 14).astype("float32").where(out["MENTHLTH"].notna())
    f["retired_or_unable"] = out["EMPLOY1"].isin([7, 8]).astype("float32").where(out["EMPLOY1"].notna())
    f["low_income"] = (out["INCOME3"] <= 4).astype("float32").where(out["INCOME3"].notna())

    # structure des reponses manquantes (modules posés selon l'age / l'Etat, skip patterns)
    f["n_nan_total"] = out.isna().sum(axis=1).astype("float32")
    f["smoke_skip_nan"] = out["SMOKDAY2"].isna().astype("float32")

    return pd.concat([out, pd.DataFrame(f, index=out.index).astype("float32")], axis=1)
