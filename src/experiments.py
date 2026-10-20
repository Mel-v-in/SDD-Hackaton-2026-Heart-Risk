"""Experiences LightGBM nommees : `python -m src.experiments <nom> [<nom> ...]`.

Chaque experience : CV 5 folds sur train (folds partages), OOF + test sauvegardes, ligne dans experiments.csv.
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np

from src.config import OUTPUT_DIR
from src.load import load
from src.train import cv_lgbm, feature_columns, get_folds, get_matrix, log_experiment, summarize

warnings.filterwarnings("ignore")

REG1 = dict(num_leaves=15, min_child_samples=200, learning_rate=0.03, feature_fraction=0.5, reg_lambda=10.0)
REG2 = dict(num_leaves=31, min_child_samples=300, learning_rate=0.03, feature_fraction=0.4, reg_lambda=20.0)
REG3 = dict(num_leaves=7, min_child_samples=300, learning_rate=0.04, feature_fraction=0.5, reg_lambda=10.0)

# nom -> (params, drop_state, use_extra, extra_weight, notes)
CONFIGS = {
    "lgbm_reg1": (REG1, False, False, 1.0, "num_leaves=15, mcs=200, ff=0.5, l2=10"),
    "lgbm_reg2": (REG2, False, False, 1.0, "num_leaves=31, mcs=300, ff=0.4, l2=20"),
    "lgbm_reg3": (REG3, False, False, 1.0, "num_leaves=7, mcs=300, ff=0.5, l2=10"),
    "lgbm_reg1_nostate": (REG1, True, False, 1.0, "reg1 sans colonnes Etat (reference pour extra)"),
    "lgbm_reg1_extra": (REG1, True, True, 1.0, "reg1 sans colonnes Etat + extra (poids 1) dans les folds"),
    "lgbm_reg1_extra05": (REG1, True, True, 0.5, "reg1 sans colonnes Etat + extra (poids 0.5)"),
}


def run(name: str) -> None:
    params, drop_state, use_extra, ew, notes = CONFIGS[name]
    tr, te = get_matrix("train"), get_matrix("test")
    y = load("train")["TARGET"].to_numpy()
    folds = get_folds(y)
    cols = feature_columns(tr, drop_state=drop_state)
    extra = None
    if use_extra:
        ex = get_matrix("extra")
        extra = (ex[cols], ex["TARGET"].to_numpy())
    t0 = time.time()
    oof, pt, its = cv_lgbm(tr[cols], y, folds, te[cols], extra=extra, params=params, extra_weight=ew, verbose=False)
    res = summarize(y, oof, folds)
    print(name, res, "iters", its, f"{time.time() - t0:.0f}s", flush=True)
    np.save(OUTPUT_DIR / f"oof_{name}.npy", oof)
    np.save(OUTPUT_DIR / f"test_{name}.npy", pt)
    log_experiment(name, res, len(cols), notes + f" | iters={its}")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
