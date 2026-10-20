"""Baseline regression logistique (imputation mediane + indicateurs de NaN + standardisation), CV 5 folds."""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import N_FOLDS, OUTPUT_DIR
from src.load import load
from src.train import feature_columns, get_folds, get_matrix, log_experiment, summarize

warnings.filterwarnings("ignore")


def pipe(C: float):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=C, max_iter=300, solver="lbfgs"))


def main() -> None:
    tr, te = get_matrix("train"), get_matrix("test")
    y = load("train")["TARGET"].to_numpy()
    folds = get_folds(y)
    cols = [c for c in feature_columns(tr) if tr[c].notna().any() and tr[c].nunique() > 1]
    for C in (0.01,):
        oof, pt = np.zeros(len(tr)), np.zeros(len(te))
        t0 = time.time()
        for k in range(N_FOLDS):
            a, b = np.where(folds != k)[0], np.where(folds == k)[0]
            m = pipe(C).fit(tr.iloc[a][cols], y[a])
            oof[b] = m.predict_proba(tr.iloc[b][cols])[:, 1]
            pt += m.predict_proba(te[cols])[:, 1] / N_FOLDS
        res = summarize(y, oof, folds)
        print(f"logreg C={C}", res, f"{time.time() - t0:.0f}s", flush=True)
        np.save(OUTPUT_DIR / "oof_logreg.npy", oof)
        np.save(OUTPUT_DIR / "test_logreg.npy", pt)
        log_experiment("logreg", res, len(cols), f"imputation mediane + indicateurs NaN, C={C}")


if __name__ == "__main__":
    main()
