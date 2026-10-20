"""EDA 3 : valeurs manquantes et pouvoir predictif univarie (train uniquement).

AUC univariee sur valeurs BRUTES (codes 7/9/77/99... non decodes) : classement indicatif.
Ecrit le tableau complet dans outputs/eda_columns.csv (ignore par git).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import OUTPUT_DIR
from src.load import load

tr, ex, te = load("train"), load("extra"), load("test")
y = tr["TARGET"].to_numpy()
feats = [c for c in tr.columns if c not in ("ID", "TARGET")]

rows = []
for c in feats:
    x = tr[c]
    v = x.to_numpy(dtype=np.float64)
    nan = np.isnan(v)
    auc_v = 0.5 if nan.all() or np.nanmin(v) == np.nanmax(v) else roc_auc_score(y, np.where(nan, np.nanmin(v) - 1, v))
    auc_n = roc_auc_score(y, nan) if 0 < nan.sum() < len(nan) else 0.5
    rows.append(dict(col=c, nan_train=nan.mean(), nan_test=te[c].isna().mean(), nan_extra=ex[c].isna().mean(),
                     n_unique=x.nunique(), auc_value=auc_v, auc_isnan=auc_n))
t = pd.DataFrame(rows)
t["auc_value_dir"] = np.maximum(t.auc_value, 1 - t.auc_value)
t["auc_isnan_dir"] = np.maximum(t.auc_isnan, 1 - t.auc_isnan)
OUTPUT_DIR.mkdir(exist_ok=True)
t.to_csv(OUTPUT_DIR / "eda_columns.csv", index=False)

print(f"== Manquants (train, {len(feats)} features)")
bins = [(-0.001, 0), (0, 0.05), (0.05, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0)]
for lo, hi in bins:
    n = ((t.nan_train > lo) & (t.nan_train <= hi)).sum()
    print(f"  taux de NaN dans ({max(lo, 0):.2f}, {hi:.2f}] : {n} colonnes")
print(f"  colonnes 100 % NaN dans train: {(t.nan_train == 1).sum()} | dans test: {(t.nan_test == 1).sum()} | dans extra: {(t.nan_extra == 1).sum()}")
print(f"  ecart max de taux de NaN train vs test: {(t.nan_train - t.nan_test).abs().max():.4f}")
d = (t.nan_extra - t.nan_train)
print(f"  colonnes dont le taux de NaN differe de > 0.5 entre extra et train: {(d.abs() > 0.5).sum()}")

print("\n== Top 25 AUC univariee (valeurs brutes, orientee >= 0.5)")
top = t.sort_values("auc_value_dir", ascending=False).head(25)
for _, r in top.iterrows():
    print(f"  {r.col:12s} AUC={r.auc_value_dir:.3f}  NaN={r.nan_train:.0%}  n_unique={int(r.n_unique)}")
print(f"\nnb de features avec AUC univariee > 0.6: {(t.auc_value_dir > 0.6).sum()} | > 0.7: {(t.auc_value_dir > 0.7).sum()} | > 0.8: {(t.auc_value_dir > 0.8).sum()}")

print("\n== Top 10 AUC de l'indicateur 'est NaN' (manquant informatif)")
for _, r in t.sort_values("auc_isnan_dir", ascending=False).head(10).iterrows():
    print(f"  {r.col:12s} AUC(isnan)={r.auc_isnan_dir:.3f}  NaN={r.nan_train:.0%}")
