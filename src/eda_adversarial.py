"""EDA 2 : adversarial validation (un modele devine de quel jeu vient une ligne).

AUC ~ 0.5 : les deux jeux se ressemblent. AUC >> 0.5 : decalage de distribution,
les features les plus importantes disent d'ou il vient.
Usage : python -m src.eda_adversarial [numeros des runs, ex. 1 2]
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import SEED
from src.load import load

ADMIN = ["IDATE", "IMONTH", "IDAY", "IYEAR", "FMONTH", "DISPCODE", "SEQNO", "_PSU"]
ALWAYS_DROP = ["ID", "TARGET"]
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=100,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, max_bin=63,
              num_threads=4, verbose=-1, seed=SEED)


def adversarial(a, b, label, drop=(), n=75_000, rounds=150, folds=3):
    t0 = time.time()
    A, B = a.sample(min(n, len(a)), random_state=SEED), b.sample(min(n, len(b)), random_state=SEED)
    cols = [c for c in A.columns if c not in ALWAYS_DROP and c not in drop]
    X = pd.concat([A[cols], B[cols]], ignore_index=True)
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    oof = np.zeros(len(y))
    for tri, vai in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y):
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tri], y[tri]), rounds)
        oof[vai] = m.predict(X.iloc[vai])
    auc = roc_auc_score(y, oof)
    m = lgb.train(PARAMS, lgb.Dataset(X, y), rounds)
    imp = pd.Series(m.feature_importance("gain"), cols).sort_values(ascending=False)
    top = ", ".join(f"{k} ({v / imp.sum():.0%})" for k, v in imp.head(6).items())
    print(f"[{label}] n={len(A)}+{len(B)}, {len(cols)} features | AUC={auc:.4f} | top gain: {top} | {time.time() - t0:.0f}s", flush=True)
    return auc


if __name__ == "__main__":
    runs = set(sys.argv[1:]) or {"1", "2", "3", "4", "5"}
    tr, ex, te = load("train"), load("extra"), load("test")
    if "1" in runs: adversarial(tr, te, "train vs test, toutes features")
    if "2" in runs: adversarial(tr, te, "train vs test, sans colonnes admin", drop=ADMIN)
    if "3" in runs: adversarial(ex, te, "extra vs test, toutes features")
    if "4" in runs: adversarial(ex, te, "extra vs test, sans _STATE ni admin", drop=ADMIN + ["_STATE"])
    if "5" in runs: adversarial(tr, ex, "train vs extra, toutes features")
