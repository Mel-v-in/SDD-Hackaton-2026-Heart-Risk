"""Chargement des donnees avec cache parquet (float32 pour tenir en memoire).

Premier appel : lecture du CSV (pyarrow), reduction des float64 en float32,
ecriture de data/<nom>.parquet. Appels suivants : lecture directe du parquet.

Les identifiants (ID, SEQNO) et tout entier > 2**24 restent en float64/int64 :
float32 ne represente pas exactement ces valeurs et creerait des doublons.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv

from src.config import DATA_DIR

NAMES = ("train", "extra", "test", "sample_submission")
KEEP_WIDE = ("ID", "SEQNO")


def _shrink(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        s = df[col]
        # colonne entierement vide : pyarrow la lit en objet, pandas la lirait en float
        if s.dtype == object and s.isna().all():
            df[col] = np.full(len(df), np.nan, dtype=np.float32)
            continue
        if s.dtype != np.float64 or col in KEEP_WIDE:
            continue
        vals = s.to_numpy()
        finite = vals[np.isfinite(vals)]
        if finite.size and np.all(finite == np.round(finite)) and np.abs(finite).max() > 2**24:
            continue
        df[col] = s.astype(np.float32)
    return df


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    # Dans les CSV, TARGET est ecrit True/False (sample_submission compris) : on travaille en 0/1.
    if "TARGET" in df.columns and df["TARGET"].dtype == bool:
        df["TARGET"] = df["TARGET"].astype("int8")
    return df


def _read_csv(name: str) -> pd.DataFrame:
    return _finalize(_shrink(pacsv.read_csv(DATA_DIR / f"{name}.csv").to_pandas()))


def load(name: str, refresh: bool = False) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    if refresh or not path.exists():
        df = _read_csv(name)
        df.to_parquet(path, compression="zstd", index=False)
        return df
    return _finalize(pd.read_parquet(path))


if __name__ == "__main__":
    for name in NAMES:
        t0 = time.time()
        existed = (DATA_DIR / f"{name}.parquet").exists()
        df = load(name)
        counts = df.dtypes.astype(str).value_counts().to_dict()
        print(f"{name:18s} {'(cache)' if existed else '(converti)'} {df.shape} "
              f"{df.memory_usage(deep=True).sum() / 1e6:6.0f} Mo  {counts}  {time.time() - t0:.0f}s", flush=True)
