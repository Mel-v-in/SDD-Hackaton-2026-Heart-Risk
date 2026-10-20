"""Blend d'OOF existants, seuil F1 choisi sur OOF, ecriture de la soumission avec controles.

Usage : python -m src.make_submission <nom_sortie> <modele[:poids]> [<modele[:poids]> ...] [--thr 0.2]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import f1_score

from src.config import N_FOLDS, OUTPUT_DIR
from src.load import load
from src.train import best_f1, get_folds, summarize


def blend(specs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    w, oof, test = [], [], []
    for s in specs:
        name, _, wt = s.partition(":")
        w.append(float(wt) if wt else 1.0)
        oof.append(np.load(OUTPUT_DIR / f"oof_{name}.npy"))
        test.append(np.load(OUTPUT_DIR / f"test_{name}.npy"))
    w = np.array(w) / sum(w)
    lg = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))   # moyenne des logits, identique a src/blend.py
    return (expit(sum(wi * lg(o) for wi, o in zip(w, oof))), expit(sum(wi * lg(t) for wi, t in zip(w, test))))


def main() -> None:
    args = sys.argv[1:]
    thr_fixed = None
    if "--thr" in args:
        i = args.index("--thr")
        thr_fixed = float(args[i + 1])
        args = args[:i] + args[i + 2:]
    out_name, specs = args[0], args[1:]
    y = load("train")["TARGET"].to_numpy()
    folds = get_folds(y)
    oof, pt = blend(specs)
    res = summarize(y, oof, folds)
    thr = thr_fixed if thr_fixed is not None else res["thr"]
    f1_at = f1_score(y, oof >= thr)
    print(f"blend {specs}\n  OOF F1={f1_at:.5f} @thr={thr:.3f}  (meilleur {res['f1']:.5f} @ {res['thr']:.3f})  "
          f"AUC={res['auc']:.5f}  PR-AUC={res['pr_auc']:.5f}")
    print("  F1 par fold au seuil retenu:", [round(f1_score(y[folds == k], oof[folds == k] >= thr), 4) for k in range(N_FOLDS)])
    print("  seuil optimal par fold:", [best_f1(y[folds == k], oof[folds == k])[1] for k in range(N_FOLDS)])

    # controles de coherence OOF <-> test
    pos_oof, pos_test = float((oof >= thr).mean()), float((pt >= thr).mean())
    print(f"  taux de positifs predits: OOF {pos_oof:.4f} | test {pos_test:.4f} | prevalence train {y.mean():.4f}")

    ids = load("test")["ID"].to_numpy()
    sample = load("sample_submission")
    assert len(ids) == len(sample) == len(pt) == 75000, "taille inattendue"
    assert (ids == sample["ID"].to_numpy()).all(), "IDs non alignes sur sample_submission"
    pred = (pt >= thr)
    sub = pd.DataFrame({"ID": ids, "TARGET": pred})  # format de sample_submission.csv : True / False
    sub.to_csv(OUTPUT_DIR / f"{out_name}.csv", index=False)
    pd.DataFrame({"ID": ids, "TARGET": pred.astype(int)}).to_csv(OUTPUT_DIR / f"{out_name}_01.csv", index=False)
    assert sub["ID"].is_unique and sub["TARGET"].isna().sum() == 0
    print(f"  ecrit: outputs/{out_name}.csv (True/False, comme sample_submission) et {out_name}_01.csv (0/1)"
          f" | {int(pred.sum())} positifs sur {len(pred)}")


if __name__ == "__main__":
    main()
