"""CatBoost / XGBoost, avec extra et sans colonnes Etat : `python -m src.run_gbdt cat|xgb`."""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np

from src.config import OUTPUT_DIR
from src.load import load
from src.train import (cv_catboost, cv_xgboost, feature_columns, get_folds, get_matrix, log_experiment, summarize)

warnings.filterwarnings("ignore")


def main() -> None:
    kind = sys.argv[1]
    tr, te, ex = get_matrix("train"), get_matrix("test"), get_matrix("extra")
    y = load("train")["TARGET"].to_numpy()
    folds = get_folds(y)
    cols = feature_columns(tr, drop_state=True)
    extra = (ex[cols], ex["TARGET"].to_numpy())
    fn = {"cat": cv_catboost, "xgb": cv_xgboost}[kind]
    t0 = time.time()
    oof, pt, its = fn(tr[cols], y, folds, te[cols], extra=extra)
    res = summarize(y, oof, folds)
    name = {"cat": "catboost_extra", "xgb": "xgboost_extra"}[kind]
    print(name, res, "iters", its, f"{time.time() - t0:.0f}s", flush=True)
    np.save(OUTPUT_DIR / f"oof_{name}.npy", oof)
    np.save(OUTPUT_DIR / f"test_{name}.npy", pt)
    log_experiment(name, res, len(cols), f"sans colonnes Etat + extra | iters={its}")


if __name__ == "__main__":
    main()
