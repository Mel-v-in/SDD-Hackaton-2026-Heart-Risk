"""Entrainement : CV stratifiee sur train (+ extra optionnel), OOF, predictions test, choix du seuil.

Toutes les decisions (features, hyperparametres, seuil) se prennent sur le F1 OOF,
jamais sur le leaderboard public. Chaque experience est journalisee dans
outputs/experiments.csv (CV F1, seuil, AUC, PR-AUC).
"""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.clean import WEIGHT_COL, clean
from src.config import DATA_DIR, EXPERIMENTS_LOG, N_FOLDS, OUTPUT_DIR, SEED
from src.features import add_features
from src.load import load

THRESHOLDS = np.arange(0.05, 0.60, 0.005)
# Colonnes qui identifient l'Etat / la version du questionnaire : a retirer si on utilise extra
STATE_COLS = ["_STATE", "MSCODE", "_METSTAT", "_URBSTAT"]


# --------------------------------------------------------------------------- donnees
def get_matrix(name: str, refresh: bool = False) -> pd.DataFrame:
    """Donnees nettoyees + features pour train / extra / test (cache parquet)."""
    path = DATA_DIR / f"proc_{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = add_features(clean(load(name)))
    df.to_parquet(path, compression="zstd", index=False)
    return df


def feature_columns(df: pd.DataFrame, drop_state: bool = False) -> list[str]:
    drop = {"ID", "TARGET", WEIGHT_COL}
    if drop_state:
        drop |= set(STATE_COLS)
    return [c for c in df.columns if c not in drop]


def get_folds(y: np.ndarray) -> np.ndarray:
    """Affectation de fold (0..N_FOLDS-1) fixe, stratifiee, partagee par tous les modeles."""
    path = OUTPUT_DIR / "folds.npy"
    if path.exists():
        folds = np.load(path)
        if len(folds) == len(y):
            return folds
    folds = np.zeros(len(y), dtype=np.int8)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[va] = k
    np.save(path, folds)
    return folds


# --------------------------------------------------------------------------- metriques
def best_f1(y: np.ndarray, p: np.ndarray, grid: np.ndarray = THRESHOLDS) -> tuple[float, float]:
    """(F1 max, seuil) sur une grille. Calcul vectorise via tri des probabilites."""
    y = np.asarray(y).astype(bool)
    f1s = np.array([f1_score(y, p >= t) for t in grid])
    i = int(f1s.argmax())
    return float(f1s[i]), float(grid[i])


def summarize(y: np.ndarray, p: np.ndarray, folds: np.ndarray | None = None) -> dict:
    f1, thr = best_f1(y, p)
    out = {"f1": round(f1, 5), "thr": round(thr, 3), "auc": round(roc_auc_score(y, p), 5),
           "pr_auc": round(average_precision_score(y, p), 5)}
    if folds is not None:
        out["f1_folds"] = [round(f1_score(y[folds == k], p[folds == k] >= thr), 4) for k in range(N_FOLDS)]
    return out


def log_experiment(name: str, res: dict, n_features: int, notes: str = "") -> None:
    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "name": name, "cv_f1": res["f1"], "thr": res["thr"],
           "auc": res["auc"], "pr_auc": res["pr_auc"], "n_features": n_features, "notes": notes}
    df = pd.DataFrame([row])
    header = not EXPERIMENTS_LOG.exists()
    df.to_csv(EXPERIMENTS_LOG, mode="a", header=header, index=False)


# --------------------------------------------------------------------------- modeles
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, reg_lambda=5.0,
                  n_estimators=4000, random_state=SEED, verbose=-1, n_jobs=2)


def cv_lgbm(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray, X_test: pd.DataFrame | None = None,
            extra: tuple[pd.DataFrame, np.ndarray] | None = None, params: dict | None = None,
            es_rounds: int = 100, verbose: bool = True, extra_weight: float = 1.0):
    """CV LightGBM. `extra` = (X_extra, y_extra) ajoute au train de chaque fold (jamais a la validation)."""
    import lightgbm as lgb

    p = {**LGB_PARAMS, **(params or {})}
    oof = np.zeros(len(X), dtype=np.float64)
    pred_test = np.zeros(len(X_test)) if X_test is not None else None
    best_iters = []
    for k in range(N_FOLDS):
        t0 = time.time()
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        Xtr, ytr = X.iloc[tr], y[tr]
        w = None
        if extra is not None:
            Xtr = pd.concat([Xtr, extra[0]], axis=0, ignore_index=True)
            ytr = np.concatenate([ytr, extra[1]])
            w = np.concatenate([np.ones(len(tr)), np.full(len(extra[1]), extra_weight)])
        m = lgb.LGBMClassifier(**p)
        m.fit(Xtr, ytr, sample_weight=w, eval_set=[(X.iloc[va], y[va])], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(es_rounds, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        if X_test is not None:
            pred_test += m.predict_proba(X_test)[:, 1] / N_FOLDS
        best_iters.append(int(m.best_iteration_))
        if verbose:
            print(f"  fold {k}: iters={m.best_iteration_} auc={roc_auc_score(y[va], oof[va]):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return oof, pred_test, best_iters


def _fold_train(X, y, tr, extra, extra_weight):
    Xtr, ytr = X.iloc[tr], y[tr]
    w = None
    if extra is not None:
        Xtr = pd.concat([Xtr, extra[0]], axis=0, ignore_index=True)
        ytr = np.concatenate([ytr, extra[1]])
        w = np.concatenate([np.ones(len(tr)), np.full(len(extra[1]), extra_weight)])
    return Xtr, ytr, w


def cv_catboost(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray, X_test: pd.DataFrame | None = None,
                params: dict | None = None, verbose: bool = True, extra=None, extra_weight: float = 1.0):
    from catboost import CatBoostClassifier

    p = dict(iterations=3000, learning_rate=0.1, depth=6, l2_leaf_reg=6.0, random_strength=1.0,
             bootstrap_type="Bernoulli", subsample=0.8, rsm=0.5, eval_metric="AUC", od_type="Iter", od_wait=120,
             random_seed=SEED, thread_count=2, verbose=0, allow_writing_files=False)
    p.update(params or {})
    oof = np.zeros(len(X))
    pred_test = np.zeros(len(X_test)) if X_test is not None else None
    best_iters = []
    for k in range(N_FOLDS):
        t0 = time.time()
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        m = CatBoostClassifier(**p)
        Xtr, ytr, w = _fold_train(X, y, tr, extra, extra_weight)
        m.fit(Xtr, ytr, sample_weight=w, eval_set=(X.iloc[va], y[va]), use_best_model=True)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        if X_test is not None:
            pred_test += m.predict_proba(X_test)[:, 1] / N_FOLDS
        best_iters.append(int(m.get_best_iteration()))
        if verbose:
            print(f"  fold {k}: iters={best_iters[-1]} auc={roc_auc_score(y[va], oof[va]):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return oof, pred_test, best_iters


def cv_xgboost(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray, X_test: pd.DataFrame | None = None,
               params: dict | None = None, verbose: bool = True, extra=None, extra_weight: float = 1.0):
    import xgboost as xgb

    p = dict(n_estimators=4000, learning_rate=0.07, max_depth=4, min_child_weight=20, subsample=0.8,
             colsample_bytree=0.5, reg_lambda=10.0, tree_method="hist", eval_metric="aucpr",
             early_stopping_rounds=120, random_state=SEED, n_jobs=2, verbosity=0)
    p.update(params or {})
    oof = np.zeros(len(X))
    pred_test = np.zeros(len(X_test)) if X_test is not None else None
    best_iters = []
    for k in range(N_FOLDS):
        t0 = time.time()
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        m = xgb.XGBClassifier(**p)
        Xtr, ytr, w = _fold_train(X, y, tr, extra, extra_weight)
        m.fit(Xtr, ytr, sample_weight=w, eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        if X_test is not None:
            pred_test += m.predict_proba(X_test)[:, 1] / N_FOLDS
        best_iters.append(int(m.best_iteration))
        if verbose:
            print(f"  fold {k}: iters={best_iters[-1]} auc={roc_auc_score(y[va], oof[va]):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return oof, pred_test, best_iters
