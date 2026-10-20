"""Experiences E1 (donnees brutes) et E2 (decodage + features) : LightGBM, CV 5 folds sur train."""
from __future__ import annotations

import sys
import time

import numpy as np

from src.config import OUTPUT_DIR
from src.load import load
from src.train import (cv_lgbm, feature_columns, get_folds, get_matrix, log_experiment, summarize)


def main(which: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    tr_raw = load("train")
    y = tr_raw["TARGET"].to_numpy()
    folds = get_folds(y)
    print("prevalence", y.mean(), "positifs par fold", [int(y[folds == k].sum()) for k in range(5)], flush=True)

    if which == "raw":
        # Avant decodage : toutes les colonnes brutes sauf identifiants / TARGET
        cols = [c for c in tr_raw.columns if c not in ("ID", "TARGET")]
        X, Xt = tr_raw[cols], load("test")[cols]
        name, notes = "lgbm_raw", "colonnes brutes, aucun decodage"
    else:
        tr = get_matrix("train")
        te = get_matrix("test")
        cols = feature_columns(tr)
        X, Xt = tr[cols], te[cols]
        name, notes = "lgbm_clean_feat", "decodage BRFSS + features"
    t0 = time.time()
    oof, pt, its = cv_lgbm(X, y, folds, Xt)
    res = summarize(y, oof, folds)
    print(name, res, "iters", its, f"{time.time() - t0:.0f}s", flush=True)
    np.save(OUTPUT_DIR / f"oof_{name}.npy", oof)
    np.save(OUTPUT_DIR / f"test_{name}.npy", pt)
    log_experiment(name, res, len(cols), notes + f" | iters={its}")


if __name__ == "__main__":
    main(sys.argv[1])
