"""Blend final : poids optimises sur le PR-AUC OOF (moyenne des logits), seuil F1 sur OOF, ecrit outputs/final_blend.json.

Usage : python -m src.blend <modele> <modele> ...
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, f1_score

from src.config import N_FOLDS, OUTPUT_DIR
from src.load import load
from src.train import best_f1, get_folds, summarize


def _lg(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


def main() -> None:
    names = sys.argv[1:]
    y = load("train")["TARGET"].to_numpy()
    folds = get_folds(y)
    L = np.column_stack([_lg(np.load(OUTPUT_DIR / f"oof_{n}.npy")) for n in names])
    T = np.column_stack([_lg(np.load(OUTPUT_DIR / f"test_{n}.npy")) for n in names])

    def softmax(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    def loss(z):
        return -average_precision_score(y, expit(L @ softmax(z)))

    r = minimize(loss, np.zeros(len(names)), method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-6})
    w = softmax(r.x)
    oof, pt = expit(L @ w), expit(T @ w)
    res = summarize(y, oof, folds)
    # seuil : maximum de F1 sur une grille fine, lisse par moyenne mobile pour eviter un pic isole
    grid = np.arange(0.05, 0.60, 0.005)
    f1s = np.array([f1_score(y, oof >= t) for t in grid])
    sm = np.convolve(f1s, np.ones(5) / 5, mode="same")
    thr = float(grid[int(sm[2:-2].argmax()) + 2])
    out = {"models": names, "weights": [round(float(x), 4) for x in w], "threshold": round(thr, 3),
           "oof_f1_at_thr": round(float(f1_score(y, oof >= thr)), 5), "oof_best": res,
           "pos_rate_oof": round(float((oof >= thr).mean()), 4), "pos_rate_test": round(float((pt >= thr).mean()), 4),
           "thr_by_fold": [round(best_f1(y[folds == k], oof[folds == k])[1], 3) for k in range(N_FOLDS)]}
    np.save(OUTPUT_DIR / "oof_blend.npy", oof)
    np.save(OUTPUT_DIR / "test_blend.npy", pt)
    (OUTPUT_DIR / "final_blend.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
