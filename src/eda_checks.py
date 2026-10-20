"""EDA 1 : integrite des donnees, nature de extra.csv, fuite eventuelle via ID/SEQNO."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

from src.load import load

tr, ex, te = load("train"), load("extra"), load("test")
sets = {"train": tr, "extra": ex, "test": te}

print("== Formes")
for n, d in sets.items():
    print(f"{n:6s} {d.shape}")

print("\n== Identifiants")
for n, d in sets.items():
    print(f"{n:6s} ID unique={d.ID.is_unique} [{d.ID.min()}..{d.ID.max()}] "
          f"SEQNO unique={d.SEQNO.is_unique} NaN ID/SEQNO={d.ID.isna().sum()}/{d.SEQNO.isna().sum()}")
for a, b in [("train", "extra"), ("train", "test"), ("extra", "test")]:
    print(f"ID commun {a}/{b}: {len(set(sets[a].ID) & set(sets[b].ID))}   "
          f"SEQNO commun {a}/{b}: {len(set(sets[a].SEQNO) & set(sets[b].SEQNO))}")

print("\n== Doublons de lignes (hors ID/TARGET, par hachage de lignes)")
hashes = {}
for n, d in sets.items():
    feats = d.drop(columns=[c for c in ("ID", "TARGET") if c in d.columns])
    hashes[n] = pd.util.hash_pandas_object(feats, index=False).to_numpy()
    print(f"{n:6s} doublons internes: {len(hashes[n]) - len(np.unique(hashes[n]))}")
for a, b in [("train", "extra"), ("train", "test"), ("extra", "test")]:
    print(f"lignes identiques {a}/{b}: {len(np.intersect1d(hashes[a], hashes[b]))}")
del hashes

print("\n== TARGET")
for n in ["train", "extra"]:
    d = sets[n]
    print(f"{n:6s} valeurs={sorted(d.TARGET.dropna().unique())} NaN={d.TARGET.isna().sum()} "
          f"positifs={int(d.TARGET.sum())} prevalence={d.TARGET.mean():.4f}")

print("\n== Distributions vs jeu de comparaison (chi2 ; TV = distance en variation totale)")
for col in ["_STATE", "IYEAR", "IMONTH", "FMONTH"]:
    for a, b in [("train", "test"), ("extra", "test"), ("train", "extra")]:
        ca, cb = sets[a][col].value_counts(), sets[b][col].value_counts()
        idx = ca.index.union(cb.index)
        ca, cb = ca.reindex(idx, fill_value=0), cb.reindex(idx, fill_value=0)
        p = chi2_contingency(np.vstack([ca, cb]))[1]
        tv = 0.5 * np.abs(ca / ca.sum() - cb / cb.sum()).sum()
        print(f"{col:7s} {a:5s} vs {b:5s}: k={len(idx):2d} TV={tv:.4f} p={p:.3f}")
print("IYEAR train:", tr.IYEAR.value_counts().to_dict(), "| extra:", ex.IYEAR.value_counts().to_dict(),
      "| test:", te.IYEAR.value_counts().to_dict())

print("\n== Signal de TARGET dans ID / SEQNO / _PSU (AUC, 0.5 = aucun signal)")
for n in ["train", "extra"]:
    d = sets[n]
    print(f"{n:6s} " + "  ".join(f"{c}={roc_auc_score(d.TARGET, d[c]):.4f}" for c in ["ID", "SEQNO", "_PSU", "_STATE"]))
print("ID croissant (train)?", tr.ID.is_monotonic_increasing, "| corr de rang ID~SEQNO:",
      round(tr[["ID", "SEQNO"]].corr(method="spearman").iloc[0, 1], 4),
      "| ID~_STATE:", round(tr[["ID", "_STATE"]].corr(method="spearman").iloc[0, 1], 4))
dec = pd.qcut(tr.ID.rank(method="first"), 10, labels=False)
print("prevalence par decile d'ID (train):", tr.groupby(dec).TARGET.mean().round(4).tolist())

print("\n== Couverture par etat (_STATE = code FIPS)")
st = {n: set(d._STATE.astype(int)) for n, d in sets.items()}
print(f"nb etats: train={len(st['train'])} test={len(st['test'])} extra={len(st['extra'])}")
print("etats de test inclus dans train:", st["test"] <= st["train"])
print("etats communs train/extra:", sorted(st["train"] & st["extra"]) or "aucun")
print("etats seulement dans extra:", sorted(st["extra"] - st["train"]))
print("part des lignes de extra dans un etat present dans train:", round(ex._STATE.astype(int).isin(st["train"]).mean(), 4))
pt = tr.groupby(tr._STATE.astype(int)).TARGET.agg(["mean", "size"])
pe = ex.groupby(ex._STATE.astype(int)).TARGET.agg(["mean", "size"])
print(f"prevalence par etat, train: min={pt['mean'].min():.3f} max={pt['mean'].max():.3f} | extra: min={pe['mean'].min():.3f} max={pe['mean'].max():.3f}")
print("plus gros etats train (code: n, prevalence):", {int(k): (int(r['size']), round(r['mean'], 3)) for k, r in pt.sort_values('size', ascending=False).head(5).iterrows()})
print("plus gros etats extra (code: n, prevalence):", {int(k): (int(r['size']), round(r['mean'], 3)) for k, r in pe.sort_values('size', ascending=False).head(5).iterrows()})

print("\n== SEQNO : signal brut ou simple effet d'etat ? (train)")
pct = tr.groupby("_STATE").SEQNO.rank(pct=True)
print(f"AUC SEQNO brut={roc_auc_score(tr.TARGET, tr.SEQNO):.4f} | "
      f"AUC du rang percentile de SEQNO dans l'etat={roc_auc_score(tr.TARGET, pct):.4f}")
