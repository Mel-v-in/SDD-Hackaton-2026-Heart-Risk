"""Construit notebooks/heart_risk.ipynb (storytelling) puis l'execute.

Usage : python notebooks/build_notebook.py   (depuis la racine du depot)
Les titres de figures sont calcules a partir des donnees : ils restent justes si les resultats changent.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# =====================================================================================================
md("""
# Repérer les cardiopathies avec quelques questions au téléphone

**SDD Hackathon 2026, Heart Risk** (BRFSS 2022, CDC)

> Dans cet échantillon d'adultes américains interrogés par téléphone, environ **1 personne sur 11** déclare avoir déjà eu
> un infarctus ou une maladie coronarienne. Peut-on les repérer **uniquement à partir de leurs réponses à l'enquête**,
> sans mesure de tension ni de cholestérol ? Et que nous apprend le modèle sur les facteurs qui comptent le plus ?

Ce notebook suit le fil suivant : les données et leur nettoyage, qui est à risque, ce que racontent les réponses manquantes,
le modèle et son point de fonctionnement, puis la lecture de ses décisions et ses limites.
""")

code("""
import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score)

from src.load import load
from src.train import get_matrix, feature_columns, get_folds, best_f1

# --- charte graphique : une couleur forte pour TARGET = 1, neutres pour le reste
RED, GREY, LIGHT, BG, DARK, BLUE = "#B22222", "#8C8C86", "#D9D4C7", "#F5F5F1", "#2B2B2B", "#3B5B7A"
mpl.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "figure.dpi": 110,
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
    "axes.grid": True, "grid.color": "#E4E0D6", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titleweight": "bold", "legend.frameon": False})

import textwrap
def headline(fig, title, sub=None, extra=0.0):
    # titre-conclusion en gras + sous-titre gris ; zone de trace placee sous le titre quelle que soit sa longueur
    w, h = fig.get_size_inches()
    t = textwrap.fill(title, width=max(40, int(w * 7.4)))
    used = (t.count("\\n") + 1) * 0.27 / h
    fig.text(0.01, 0.985, t, ha="left", va="top", fontsize=14, fontweight="bold", color=DARK, linespacing=1.25)
    if sub:
        fig.text(0.01, 0.985 - used - 0.012, sub, ha="left", va="top", fontsize=10.5, color=GREY)
        used += 0.30 / h
    fig.subplots_adjust(top=1 - 0.015 - used - (0.40 + extra) / h)

def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

raw = load("train")                     # donnees brutes (codes BRFSS non decodes)
Xc = get_matrix("train")                # donnees decodees + features (src/clean.py, src/features.py)
y = raw["TARGET"].to_numpy()
wt = raw["_LLCPWT"].to_numpy()          # poids d'enquete : estimations de prevalence uniquement
folds = get_folds(y)
blend = json.load(open(ROOT / "outputs" / "final_blend.json"))
oof = np.load(ROOT / "outputs" / "oof_blend.npy")
thr = blend["threshold"]
print(f"{len(raw):,} adultes, {Xc.shape[1]} variables decodees (sur {raw.shape[1]-2} brutes), prevalence {y.mean():.1%}")
""")

# =====================================================================================================
md("""
## 1. Le jeu de données : une enquête brute, qu'il faut décoder

Le BRFSS est une enquête téléphonique annuelle des CDC. Les réponses sont stockées sous forme de **codes** :
`7` ou `77` pour « ne sait pas », `9` ou `99` pour « refus », `88` pour « aucun »… Traités comme des nombres, ces codes
faussent tout calcul. Nous avons écrit un décodage (`src/clean.py`) appliqué à l'identique au jeu d'entraînement,
au jeu supplémentaire et au test.
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
# a) PHYSHLTH : jours de mauvaise sante physique sur 30 ; 88 = "aucun"
v = raw["PHYSHLTH"].value_counts().sort_index()
cols_ = [RED if k in (88, 77, 99) else LIGHT for k in v.index]
axes[0].bar(range(len(v)), v.values, color=cols_)
axes[0].set_xticks(range(len(v))); axes[0].set_xticklabels([int(k) for k in v.index], fontsize=6.5, rotation=90)
axes[0].set_title("Avant : jours de mauvaise santé physique", fontsize=10.5, loc="left")
axes[0].annotate("88 = « aucun jour »\\nlu comme 88 jours", xy=(list(v.index).index(88), v[88]), xytext=(12, v[88] * 0.8),
                 fontsize=9, color=RED, arrowprops=dict(arrowstyle="-", color=RED))
d = Xc["PHYSHLTH"].dropna()
axes[1].hist(d, bins=np.arange(-0.5, 31.5, 1), color=BLUE)
axes[1].set_title("Après : 88 devient 0, 77 et 99 manquants", fontsize=10.5, loc="left")
axes[1].set_xlabel("jours sur 30")
# c) EXERANY2 : activite physique, 1 oui, 2 non, 7/9 ne sait pas/refus
e = raw["EXERANY2"].value_counts().sort_index()
axes[2].bar([str(int(k)) for k in e.index], e.values, color=[LIGHT, LIGHT, RED, RED][:len(e)])
axes[2].set_title("EXERANY2 avant : 1 oui, 2 non, 7 et 9 parasites", fontsize=10.5, loc="left")
for a in axes: a.grid(axis="x", visible=False)
share = raw["EXERANY2"].isin([7, 9]).mean()
headline(fig, "Sans décodage, « aucun » vaut 88 jours et « refus » vaut 9", "Trois exemples de codes spéciaux du BRFSS et leur traitement", extra=0.3)
plt.show()
""")

md("""
Au-delà de ces recodages, le décodage (i) transforme les questions Oui/Non en 0/1, (ii) convertit `ALCDAY4` en jours par mois,
(iii) remplace poids et taille déclarés (unités mélangées) par les valeurs calculées par le CDC, et (iv) retire les colonnes
administratives et les questions de screening téléphonique. Dans l'ensemble, **cela change très peu la performance d'un
modèle à arbres** (qui apprend les codes tout seul), mais cela rend les variables lisibles, utilisables par une régression
logistique et interprétables dans la suite de ce notebook.
""")

# =====================================================================================================
md("""
## 2. Qui est à risque ?
""")

code("""
age_lab = {1:"18-24",2:"25-29",3:"30-34",4:"35-39",5:"40-44",6:"45-49",7:"50-54",8:"55-59",9:"60-64",10:"65-69",11:"70-74",12:"75-79",13:"80+"}
df = pd.DataFrame({"age": Xc["_AGEG5YR"], "male": Xc["SEXVAR"], "y": y}).dropna()
g = df.groupby(["age", "male"])["y"].agg(["sum", "count"]).reset_index()
g["p"] = g["sum"] / g["count"]; g["lo"], g["hi"] = wilson(g["sum"], g["count"])
fig, ax = plt.subplots(figsize=(10, 4.6))
for m, col, lab in [(1.0, RED, "Hommes"), (0.0, GREY, "Femmes")]:
    s = g[g.male == m]
    ax.plot(s.age, s.p * 100, color=col, lw=2.4, marker="o", ms=4)
    ax.fill_between(s.age, s.lo * 100, s.hi * 100, color=col, alpha=0.15, lw=0)
    ax.text(s.age.iloc[-1] + 0.15, s.p.iloc[-1] * 100, lab, color=col, fontweight="bold", va="center")
ax.set_xticks(list(age_lab)); ax.set_xticklabels(list(age_lab.values()), fontsize=8.5)
ax.set_ylabel("% ayant déjà eu une maladie cardiaque"); ax.set_xlim(0.7, 14)
r = g[g.age == 12].set_index("male")["p"]
headline(fig, f"À 75-79 ans, {r[1.0]:.0%} des hommes ont déjà eu une maladie cardiaque, contre {r[0.0]:.0%} des femmes",
         "Prévalence par tranche d'âge et par sexe (bande = intervalle de confiance à 95 %)")
plt.show()
""")

code("""
d = pd.DataFrame({"n": Xc["n_comorbid"].clip(upper=5), "y": y})
t = d.groupby("n")["y"].agg(["sum", "count"]); t["p"] = t["sum"] / t["count"]; t["lo"], t["hi"] = wilson(t["sum"], t["count"])
fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar([f"{int(i)}" if i < 5 else "5 et +" for i in t.index], t.p * 100, color=[LIGHT] * 5 + [RED], zorder=3)
bars[0].set_color(LIGHT)
for i, (idx, row) in enumerate(t.iterrows()):
    ax.text(i, row.p * 100 + 1.2, f"{row.p:.0%}", ha="center", fontweight="bold", color=DARK)
    ax.text(i, 1.5, f"n={int(row['count']):,}".replace(",", " "), ha="center", fontsize=8, color=GREY, zorder=4)
ax.set_xlabel("nombre de maladies chroniques déclarées (AVC, diabète, rein, BPCO, arthrite, asthme, cancers, dépression)")
ax.set_ylabel("% de cardiopathies"); ax.grid(axis="x", visible=False)
headline(fig, f"Chaque maladie chronique de plus fait grimper le risque : {t.p.iloc[0]:.0%} sans aucune, {t.p.iloc[-1]:.0%} avec cinq ou plus",
         "Prévalence selon le nombre de maladies chroniques (variable construite : n_comorbid)")
plt.show()
""")

code("""
gl = {1:"Excellente",2:"Très bonne",3:"Bonne",4:"Passable",5:"Mauvaise"}
d = pd.DataFrame({"g": Xc["GENHLTH"], "y": y}).dropna()
t = d.groupby("g")["y"].agg(["sum", "count"]); t["p"] = t["sum"] / t["count"]
fig, ax = plt.subplots(figsize=(9, 4.3))
ax.barh([gl[int(i)] for i in t.index][::-1], (t.p * 100).values[::-1], color=[RED, RED, LIGHT, LIGHT, LIGHT], zorder=3)
for i, (idx, row) in enumerate(t.iloc[::-1].iterrows()):
    ax.text(row.p * 100 + 0.6, i, f"{row.p:.0%}   (n={int(row['count']):,})".replace(",", " "), va="center", fontsize=9.5)
ax.set_xlabel("% de cardiopathies"); ax.grid(axis="y", visible=False); ax.set_xlim(0, t.p.max() * 100 * 1.3)
headline(fig, f"Les personnes qui jugent leur santé « mauvaise » sont {t.p.iloc[-1] / t.p.iloc[0]:.0f} fois plus touchées que celles qui la jugent « excellente »",
         "Prévalence selon la santé perçue (GENHLTH)")
plt.show()
""")

code("""
py = Xc["_PACKYRS"]
d = pd.DataFrame({"py": py, "ever": Xc["SMOKE100"], "y": y})
lab, p_, n_ = [], [], []
nev = d[d.ever == 0]; lab.append("Jamais\\nfumé"); p_.append(nev.y.mean()); n_.append(len(nev))
sm = d[(d.ever == 1) & d.py.notna()].copy()
sm["dec"] = pd.qcut(sm.py.rank(method="first"), 5, labels=False)
for k, s in sm.groupby("dec"):
    lab.append(f"{s.py.min():.0f}-{s.py.max():.0f}" if k < 4 else f"{s.py.min():.0f}+"); p_.append(s.y.mean()); n_.append(len(s))
fig, ax = plt.subplots(figsize=(9.5, 4.3))
ax.bar(lab, np.array(p_) * 100, color=[GREY] + [LIGHT] * 4 + [RED], zorder=3)
for i, (p, n) in enumerate(zip(p_, n_)):
    ax.text(i, p * 100 + 0.5, f"{p:.0%}", ha="center", fontweight="bold")
ax.set_xlabel("paquets-années (quintiles parmi les fumeurs ; 1 paquet-année = 1 paquet par jour pendant 1 an)")
ax.set_ylabel("% de cardiopathies"); ax.grid(axis="x", visible=False)
headline(fig, f"Plus on a fumé, plus le risque monte : {p_[-1]:.0%} chez les plus gros fumeurs, {p_[0]:.0%} chez ceux qui n'ont jamais fumé",
         "Prévalence selon l'exposition au tabac (non ajustée sur l'âge, qui joue aussi : les gros fumeurs sont plus âgés)")
plt.show()
""")

md("""
### Une carte, pondérée

Les prévalences par État utilisent le **poids d'enquête** (`_LLCPWT`) pour que chaque État reflète sa population adulte.
La carte combine le jeu d'entraînement et le jeu supplémentaire (qui couvre d'autres États). Les différences entre États
mêlent âge de la population, mode de vie et accès aux soins : elles ne sont pas standardisées sur l'âge.
""")

code("""
fips = {1:"AL",2:"AK",4:"AZ",5:"AR",6:"CA",8:"CO",9:"CT",10:"DE",11:"DC",12:"FL",13:"GA",15:"HI",16:"ID",17:"IL",18:"IN",19:"IA",20:"KS",
        21:"KY",22:"LA",23:"ME",24:"MD",25:"MA",26:"MI",27:"MN",28:"MS",29:"MO",30:"MT",31:"NE",32:"NV",33:"NH",34:"NJ",35:"NM",36:"NY",
        37:"NC",38:"ND",39:"OH",40:"OK",41:"OR",42:"PA",44:"RI",45:"SC",46:"SD",47:"TN",48:"TX",49:"UT",50:"VT",51:"VA",53:"WA",54:"WV",
        55:"WI",56:"WY",66:"GU",72:"PR",78:"VI"}
grid = {"AK":(0,0),"ME":(0,10),"WI":(1,5),"VT":(1,9),"NH":(1,10),"WA":(2,0),"ID":(2,1),"MT":(2,2),"ND":(2,3),"MN":(2,4),"IL":(2,5),"MI":(2,6),
        "NY":(2,7),"MA":(2,8),"OR":(3,0),"NV":(3,1),"WY":(3,2),"SD":(3,3),"IA":(3,4),"IN":(3,5),"OH":(3,6),"PA":(3,7),"NJ":(3,8),"CT":(3,9),
        "RI":(3,10),"CA":(4,0),"UT":(4,1),"CO":(4,2),"NE":(4,3),"MO":(4,4),"KY":(4,5),"WV":(4,6),"VA":(4,7),"MD":(4,8),"DE":(4,9),"AZ":(5,1),
        "NM":(5,2),"KS":(5,3),"AR":(5,4),"TN":(5,5),"NC":(5,6),"SC":(5,7),"DC":(5,8),"OK":(6,3),"LA":(6,4),"MS":(6,5),"AL":(6,6),"GA":(6,7),
        "HI":(7,0),"TX":(7,3),"FL":(7,8),"GU":(8,0),"PR":(8,1),"VI":(8,2)}
ex = load("extra")
allst = pd.concat([pd.DataFrame({"st": raw["_STATE"], "y": y, "w": wt}),
                   pd.DataFrame({"st": ex["_STATE"], "y": ex["TARGET"], "w": ex["_LLCPWT"]})])
allst["ab"] = allst.st.astype(int).map(fips)
sp = allst.groupby("ab").apply(lambda s: np.average(s.y, weights=s.w) * 100)
fig, ax = plt.subplots(figsize=(11, 6.2))
cmap = mpl.colors.LinearSegmentedColormap.from_list("risk", [LIGHT, "#E7A39B", RED, "#6E1111"])
vmin, vmax = sp.min(), sp.max()
for ab, (r_, c_) in grid.items():
    if ab in sp.index:
        v_ = sp[ab]; col = cmap((v_ - vmin) / (vmax - vmin))
        ax.add_patch(plt.Rectangle((c_, -r_), 0.94, 0.94, color=col, zorder=2))
        ax.text(c_ + 0.47, -r_ + 0.60, ab, ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if (v_ - vmin) / (vmax - vmin) > 0.45 else DARK, zorder=3)
        ax.text(c_ + 0.47, -r_ + 0.28, f"{v_:.1f}", ha="center", va="center", fontsize=8.5,
                color="white" if (v_ - vmin) / (vmax - vmin) > 0.45 else DARK, zorder=3)
ax.set_xlim(-0.1, 11.1); ax.set_ylim(-8.2, 1.1); ax.axis("off"); ax.grid(False)
hi, lo = sp.idxmax(), sp.idxmin()
headline(fig, f"La prévalence pondérée varie de {sp.min():.1f} % ({lo}) à {sp.max():.1f} % ({hi}) selon l'État",
         "Part d'adultes ayant déjà eu une maladie cardiaque, pondérée par _LLCPWT (carte en tuiles, non standardisée sur l'âge)")
sm_ = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin, vmax))
cb = fig.colorbar(sm_, ax=ax, fraction=0.025, pad=0.01); cb.set_label("%"); cb.outline.set_visible(False)
plt.show()
""")

code("""
inc = Xc["_INCOMG1"]  # 1 <15k ... 7 >=200k (9 manquant -> NaN)
agebin = pd.cut(Xc["_AGE80"], [17, 64, 80], labels=["18-64 ans", "65 ans et +"])
d = pd.DataFrame({"inc": inc, "age": agebin, "y": y}).dropna()
inc_lab = {1:"<15k",2:"15-25k",3:"25-35k",4:"35-50k",5:"50-100k",6:"100-200k",7:"200k+"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=False)
for ax, (a, col) in zip(axes, [("18-64 ans", GREY), ("65 ans et +", RED)]):
    s = d[d.age == a].groupby("inc")["y"].agg(["sum", "count"]); s["p"] = s["sum"] / s["count"]
    s["lo"], s["hi"] = wilson(s["sum"], s["count"])
    ax.errorbar(range(len(s)), s.p * 100, yerr=[(s.p - s.lo) * 100, (s.hi - s.p) * 100], color=col, marker="o", lw=2.2, capsize=3)
    ax.set_xticks(range(len(s))); ax.set_xticklabels([inc_lab[int(i)] for i in s.index], fontsize=8.5)
    ax.set_title(a, loc="left", fontsize=11); ax.set_xlabel("revenu annuel du foyer (USD)")
axes[0].set_ylabel("% de cardiopathies")
s0 = d[d.age == "18-64 ans"].groupby("inc")["y"].mean()
headline(fig, f"À âge comparable, le revenu reste un marqueur : {s0.iloc[0]:.0%} des moins de 65 ans les plus modestes, {s0.iloc[-1]:.0%} des plus aisés",
         "Prévalence selon le revenu du foyer, séparément avant et après 65 ans (le revenu est lié à l'âge ; barres = IC 95 %)", extra=0.3)
plt.show()
""")

# =====================================================================================================
md("""
## 3. Les données manquantes racontent aussi une histoire

Beaucoup de colonnes sont **vides pour la majorité des répondants**. Ce n'est pas du hasard : le BRFSS comprend un questionnaire
de base posé partout et des **modules optionnels** que chaque État choisit d'activer. Le graphique suivant montre, pour chaque État,
la part de réponses obtenues dans chaque module (les modules de dépistage, posés partout mais aux seules personnes éligibles, ne sont pas représentés).
""")

code("""
modules = {
 "COVID-19": "COVIDSMP COVIDPRM COVIDVA1 COVIDNU1".split(),
 "Pré-diabète": "PDIABTS1 PREDIAB2".split(),
 "Expériences de l'enfance (ACE)": "ACEDEPRS ACEDRINK ACEDRUGS ACEPRISN ACEDIVRC ACEPUNCH ACEHURT1 ACESWEAR ACETOUCH ACETTHEM ACEHVSEX ACEADSAF ACEADNED".split(),
 "Aidants, mémoire": "CIMEMLOS CAREGIV1 CRGVEXPT".split(),
 "Déterminants sociaux": "LSATISFY EMTSUPRT SDHISOLT SDHEMPLY FOODSTMP SDHFOOD1 SDHBILLS SDHUTILS SDHTRNSP SDHSTRE1".split(),
 "Cannabis": ["MARIJAN1"],
 "Alcool (modèles de consommation)": "ASBIALCH ASBIDRNK ASBIBING ASBIADVC ASBIRDUC".split(),
 "Armes à feu": ["FIREARM5"],
 "Genre et orientation": "RCSGEND1 RCSRLTN2 BIRTHSEX SOMALE SOFEMALE TRNSGNDR".split(),
 "Discrimination": "RRCLASS3 RRCOGNT2 RRTREAT RRATWRK2 RRHCARE4 RRPHYSM2".split(),
}
st = raw["_STATE"].astype(int).map(fips)
rows = {}
for name, cl in modules.items():
    cl = [c for c in cl if c in raw]
    rows[name] = raw[cl].notna().mean(axis=1).groupby(st).mean()
H = pd.DataFrame(rows)
H = H.loc[H.mean(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(12, 7.8))
im = ax.imshow(H.values, aspect="auto", cmap=mpl.colors.LinearSegmentedColormap.from_list("m", [BG, "#9BB3C9", BLUE]), vmin=0, vmax=1)
ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(H.columns, rotation=45, ha="right", fontsize=8.5)
ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(H.index, fontsize=8); ax.grid(False)
for s_ in ax.spines.values(): s_.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("part de réponses obtenues"); cb.outline.set_visible(False)
full = (H > 0.2).sum(axis=1)
headline(fig, f"Chaque État active ses propres modules : de {int(full.min())} à {int(full.max())} modules sur {H.shape[1]} réellement posés",
         "Part de réponses obtenues par module (colonnes) et par État (lignes), jeu d'entraînement")
plt.show()
""")

code("""
dk = Xc["n_dk_core"].clip(upper=3)
t = pd.DataFrame({"dk": dk, "y": y}).groupby("dk")["y"].agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(8.5, 4))
ax.bar(["0", "1", "2", "3 et +"], t["mean"] * 100, color=[LIGHT, LIGHT, LIGHT, RED], zorder=3)
for i, (idx, r_) in enumerate(t.iterrows()):
    ax.text(i, r_["mean"] * 100 + 0.3, f"{r_['mean']:.1%}", ha="center", fontweight="bold")
    ax.text(i, 0.4, f"n={int(r_['count']):,}".replace(",", " "), ha="center", fontsize=8, color=GREY, zorder=4)
ax.set_xlabel("nombre de réponses « ne sait pas » / « refus » parmi 15 questions de santé de base"); ax.set_ylabel("% de cardiopathies")
ax.grid(axis="x", visible=False)
headline(fig, "Ne pas répondre n'est pas neutre : la prévalence diffère nettement selon le nombre de « ne sait pas » ou de refus",
         "Prévalence selon le nombre de réponses manquantes aux 15 questions de santé de base (n_dk_core)")
plt.show()
""")

# =====================================================================================================
md("""
## 4. Le modèle

**Méthode.** Validation croisée stratifiée à 5 plis sur les 225 000 lignes de train (les seules représentatives du test : le jeu
supplémentaire vient d'autres États). Le jeu supplémentaire est ajouté aux seules parties d'entraînement de chaque pli (sans les
colonnes qui identifient l'État), ce qui améliore légèrement la généralisation. Le **seuil de décision** est choisi sur les prédictions
hors-pli (OOF), jamais sur le classement public. La métrique est le **F1** de la classe « maladie cardiaque ».
""")

code("""
OUT = ROOT / "outputs"
names = {"logreg": "Régression logistique", "lgbm_raw": "LightGBM, données brutes", "lgbm_clean_feat": "LightGBM, données décodées",
         "lgbm_reg1_extra": "LightGBM réglé + extra", "xgboost_extra": "XGBoost + extra", "catboost_extra": "CatBoost + extra", "blend": "Blend final"}
rows = []
for k, lab in names.items():
    f = OUT / f"oof_{k}.npy"
    if not f.exists(): continue
    p = np.load(f); f1_, t_ = best_f1(y, p)
    rows.append(dict(modele=lab, F1=f1_, seuil=t_, recall=recall_score(y, p >= t_), precision=precision_score(y, p >= t_),
                     PR_AUC=average_precision_score(y, p), AUC=roc_auc_score(y, p)))
M = pd.DataFrame(rows).set_index("modele")
display(M.style.format({"F1": "{:.4f}", "seuil": "{:.3f}", "recall": "{:.3f}", "precision": "{:.3f}", "PR_AUC": "{:.4f}", "AUC": "{:.4f}"}))
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
order = M.index[::-1]
for ax, col, ttl in zip(axes, ["F1", "PR_AUC"], ["F1 (au seuil optimal de chaque modèle)", "PR-AUC (indépendant du seuil)"]):
    v = M.loc[order, col]
    ax.barh(order, v, color=[RED if o == "Blend final" else LIGHT for o in order], zorder=3)
    for i, x in enumerate(v): ax.text(x + 0.002, i, f"{x:.3f}", va="center", fontsize=9)
    ax.set_xlim(v.min() - 0.03, v.max() + 0.03); ax.set_title(ttl, loc="left", fontsize=10.5); ax.grid(axis="y", visible=False)
d_ = M["F1"].iloc[-1] - M["F1"].iloc[0] if len(M) > 1 else 0
lo_, hi_ = M["F1"].idxmin(), M["F1"].idxmax()
headline(fig, f"Tous les modèles se tiennent de près : F1 de {M['F1'].min():.3f} ({lo_.lower()}) à {M['F1'].max():.3f} ({hi_.lower()})",
         "Comparaison des modèles en validation croisée 5 plis (OOF, 225 000 personnes)", extra=0.3)
plt.show()
""")

md("""
Le plafond est visible : malgré réglages, jeu supplémentaire et ensemble de modèles, le F1 reste autour de 0,43 (voir le tableau ci-dessus).
C'est cohérent avec l'information réellement disponible : **sans tension, cholestérol ni examens**, une part des cardiopathies reste
indiscernable de personnes saines du même profil déclaratif.
""")

code("""
prec, rec, tt = precision_recall_curve(y, oof)
f1c = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
r_op = recall_score(y, oof >= thr); p_op = precision_score(y, oof >= thr); f_op = f1_score(y, oof >= thr)
fig, ax = plt.subplots(figsize=(7.5, 5.6))
ax.plot(rec, prec, color=BLUE, lw=2.4)
ax.axhline(y.mean(), color=GREY, ls="--", lw=1); ax.text(0.99, y.mean() + 0.01, "hasard (prévalence)", ha="right", color=GREY, fontsize=9)
ax.scatter([r_op], [p_op], color=RED, s=90, zorder=5)
ax.annotate(f"point choisi : seuil {thr:.2f}\\nrecall {r_op:.0%}, précision {p_op:.0%}, F1 {f_op:.3f}", xy=(r_op, p_op),
            xytext=(r_op + 0.12, p_op + 0.20), fontsize=10, color=RED, fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED))
ax.set_xlabel("recall (part des malades repérés)"); ax.set_ylabel("précision (part des alertes justifiées)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
headline(fig, f"Pour repérer {r_op:.0%} des malades, environ {p_op:.0%} des alertes sont justifiées",
         f"Courbe précision-rappel du blend final (PR-AUC = {average_precision_score(y, oof):.3f}, hasard = {y.mean():.3f})")
plt.show()
""")

code("""
grid_ = np.arange(0.05, 0.60, 0.005)
P = np.array([precision_score(y, oof >= t) for t in grid_]); R = np.array([recall_score(y, oof >= t) for t in grid_])
F = np.array([f1_score(y, oof >= t) for t in grid_])
fig, ax = plt.subplots(figsize=(9.5, 5))
ax.plot(grid_, R, color=RED, lw=2.4); ax.plot(grid_, P, color=GREY, lw=2.4); ax.plot(grid_, F, color=BLUE, lw=2.4)
ax.axvline(thr, color=DARK, ls=":", lw=1.3)
ax.text(grid_[-1] + 0.005, R[-1], "recall", color=RED, fontweight="bold", va="center")
ax.text(grid_[-1] + 0.005, P[-1], "précision", color=GREY, fontweight="bold", va="center")
ax.text(grid_[-1] + 0.005, F[-1], "F1", color=BLUE, fontweight="bold", va="center")
fa = (1 - p_op) / p_op
ax.annotate(f"seuil {thr:.2f} : on détecte {r_op:.0%} des malades\\nau prix de {fa:.1f} fausses alertes par vrai positif",
            xy=(thr, r_op), xytext=(thr + 0.08, 0.80), fontsize=10, color=DARK, arrowprops=dict(arrowstyle="->", color=DARK))
ax.set_xlabel("seuil de décision"); ax.set_ylabel("score"); ax.set_xlim(0.05, 0.62); ax.set_ylim(0, 1)
headline(fig, "Le F1 se joue sur le seuil : on peut aussi viser un plus fort recall pour de la prévention, en acceptant plus de fausses alertes",
         "Recall, précision et F1 en fonction du seuil de décision (blend final, OOF)")
plt.show()
""")

code("""
tp = int(((oof >= thr) & (y == 1)).sum()); fp = int(((oof >= thr) & (y == 0)).sum())
fn = int(((oof < thr) & (y == 1)).sum()); tn = int(((oof < thr) & (y == 0)).sum())
cm = np.array([[tn, fp], [fn, tp]]); N = cm.sum()
fig, ax = plt.subplots(figsize=(7.4, 5.4))
ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap=mpl.colors.LinearSegmentedColormap.from_list("c", [BG, "#E7A39B"]), vmin=0, vmax=1)
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, f"{v:,}".replace(",", " ") + f"\\n{v / cm[i].sum():.0%} des {'sains' if i == 0 else 'malades'}", ha="center", va="center", fontsize=12, fontweight="bold", color=RED if i == j == 1 else DARK)
ax.set_xticks([0, 1]); ax.set_xticklabels(["prédit sain", "prédit malade"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["sain", "malade"], rotation=90, va="center")
ax.grid(False); [s_.set_visible(False) for s_ in ax.spines.values()]
k = 1000 / N
headline(fig, f"Sur 1000 personnes : {y.mean()*1000:.0f} ont eu une maladie cardiaque, {tp*k:.0f} sont repérées, {fn*k:.0f} manquées",
         f"et {fp*k:.0f} personnes en bonne santé reçoivent à tort une alerte (matrice de confusion, seuil {thr:.2f})")
plt.show()
""")

code("""
bins = np.quantile(oof, np.linspace(0, 1, 11)); idx = np.clip(np.digitize(oof, bins[1:-1]), 0, 9)
cal = pd.DataFrame({"p": oof, "y": y, "b": idx}).groupby("b").mean()
fig, ax = plt.subplots(figsize=(5.6, 5.4))
ax.plot([0, cal.p.max() * 1.05], [0, cal.p.max() * 1.05], color=GREY, ls="--", lw=1)
ax.plot(cal.p, cal.y, color=RED, marker="o", lw=2.2)
ax.set_xlabel("probabilité prédite (moyenne par décile)"); ax.set_ylabel("fréquence observée")
headline(fig, "Les probabilités sont bien calibrées : une probabilité de 30 % correspond à ~30 % de malades", "Courbe de calibration par déciles (blend, OOF)")
plt.show()
""")

code("""
import plotly.graph_objects as go, plotly.io as pio
pio.renderers.default = "notebook"
ts = np.round(np.arange(0.10, 0.50, 0.01), 2)
Pt = [precision_score(y, oof >= t) for t in ts]; Rt = [recall_score(y, oof >= t) for t in ts]; Ft = [f1_score(y, oof >= t) for t in ts]
fig = go.Figure()
for nm, vals, col in [("recall", Rt, RED), ("précision", Pt, GREY), ("F1", Ft, BLUE)]:
    fig.add_trace(go.Scatter(x=ts, y=vals, mode="lines", name=nm, line=dict(color=col, width=3)))
i0 = int(np.argmin(np.abs(ts - thr)))
for i, t in enumerate(ts):
    fig.add_trace(go.Scatter(x=[t] * 3, y=[Rt[i], Pt[i], Ft[i]], mode="markers+text", showlegend=False, visible=(i == i0),
                             marker=dict(size=11, color=[RED, GREY, BLUE]), text=[f"{Rt[i]:.0%}", f"{Pt[i]:.0%}", f"{Ft[i]:.2f}"], textposition="top center"))
steps = [dict(method="update", label=f"{t:.2f}", args=[{"visible": [True] * 3 + [j == i for j in range(len(ts))]}]) for i, t in enumerate(ts)]
fig.update_layout(sliders=[dict(active=i0, currentvalue=dict(prefix="Seuil : "), steps=steps, pad=dict(t=40))], paper_bgcolor=BG, plot_bgcolor=BG,
                  title="Faites glisser le seuil : ce que l'on gagne en recall, on le perd en précision", yaxis=dict(range=[0, 1], tickformat=".0%"),
                  xaxis_title="seuil de décision", height=480, margin=dict(t=70))
fig.show()
""")

# =====================================================================================================
md("""
## 5. Comment le modèle décide

Pour ouvrir la boîte noire, nous réentraînons un LightGBM sur les mêmes variables, **sans les produits de variables ni les recodages redondants** d'une même question
(pour que l'importance ne se disperse pas entre « âge » et « tranche d'âge »), et calculons les valeurs **SHAP** : la contribution
de chaque variable à la prédiction de chaque personne (en échelle log-odds : positif = pousse vers « cardiopathie »).
""")

code("""
import lightgbm as lgb, shap
from src.experiments import REG1
from src.train import LGB_PARAMS, STATE_COLS
REDUNDANT = set((
    "_AGEG5YR _AGE_G _AGE65YR _SEX _RFHLTH fair_poor_health _PHYS14D _MENT14D phys_poor ment_poor _RFBMI5 _BMI5CAT obese WTKG3 HTM4 HTIN4 "
    "_TOTINDA _EDUCAG _INCOMG1 low_income _RFSMOK3 _SMOKGRP smoke_skip_nan _RFBING6 _RFDRHV8 DRNKANY6 DROCDY4_ _DRNKWK2 retired_or_unable _EXTETH3 _ALTETH3 _DENVST3 "
    "sleep_short sleep_long n_nan_total _CHLDCNT _RACEG22 _RACEGR4 _RACEPR1 _PRACE2 _MRACE2 _PNEUMO3 _FLSHOT7 _AIDTST4 male_x_genhlth comorbid_x_limit pack_years_x_age "
).split())
cols = [c for c in feature_columns(Xc, drop_state=True) if not c.startswith("age_x") and c not in REDUNDANT]   # lisibilité : ni produits de variables, ni recodages redondants
n_iter = int(np.mean([801, 583, 622, 687, 499]))
m = lgb.LGBMClassifier(**{**LGB_PARAMS, **REG1, "n_estimators": n_iter, "n_jobs": 2}).fit(Xc[cols], y)
rs = np.random.RandomState(42); si = rs.choice(len(Xc), 12000, replace=False)
Xs = Xc.iloc[si][cols].reset_index(drop=True)
expl = shap.TreeExplainer(m)
sv = expl.shap_values(Xs)
sv = sv[1] if isinstance(sv, list) else sv
base = float(np.ravel(expl.expected_value)[-1])
LAB = {"_AGE80":"Âge","SEXVAR":"Sexe masculin","GENHLTH":"Santé perçue (1 excellente → 5 mauvaise)","n_comorbid":"Nb de maladies chroniques",
       "age_x_genhlth":"Âge × santé perçue","age_x_comorbid":"Âge × maladies chroniques","age_x_male":"Âge × homme","age_x_smoker":"Âge × fumeur","age_x_diab":"Âge × diabète",
       "diab_status":"Diabète (0 non, 1 pré, 2 oui)","CVDSTRK3":"AVC","CHCKDNY2":"Maladie rénale","CHCCOPD3":"BPCO","HAVARTH4":"Arthrite","DIFFWALK":"Difficulté à marcher",
       "_SMOKER3":"Statut tabagique","SMOKE100":"A fumé 100 cigarettes","_PACKYRS":"Paquets-années","_BMI5":"IMC","_TOTINDA":"Activité physique (1 oui)","EXERANY2":"Activité physique",
       "EMPLOY1":"Emploi","INCOME3":"Revenu","EDUCA":"Éducation","CHECKUP1":"Dernier bilan de santé","PHYSHLTH":"Jours de mauvaise santé physique","MENTHLTH":"Jours de mauvaise santé mentale",
       "n_limitations":"Nb de limitations","retired_or_unable":"Retraité ou inapte au travail","fair_poor_health":"Santé passable ou mauvaise","_AGEG5YR":"Tranche d'âge",
       "_AGE_G":"Groupe d'âge","_RFHLTH":"Santé perçue (indicateur)","MEDCOST1":"Renoncement aux soins pour raison de coût","_ALTETH3":"Dentition",
       "n_nan_total":"Nb de réponses manquantes","_PHYS14D":"Santé physique (jours)","PNEUVAC4":"Vaccin pneumonie","_PNEUMO3":"Vaccin pneumonie (65+)","_FLSHOT7":"Vaccin grippe (50+)",
       "WTKG3":"Poids (kg)","HTM4":"Taille (cm)","ADDEPEV3":"Dépression","ASTHMA3":"Asthme","CHCSCNC1":"Cancer de la peau","CHCOCNC1":"Autre cancer","MARITAL":"Situation maritale",
       "SLEPTIM1":"Heures de sommeil","ALCDAY4":"Jours d'alcool par mois","_DRNKWK2":"Verres par semaine","DECIDE":"Difficulté de concentration","DIFFALON":"Difficulté à sortir seul","LCSCTSC1":"Scanner thoracique déjà réalisé","LCSSCNCR":"Scanner pour dépister un cancer du poumon",
       "HADHYST2":"Hystérectomie","_EXTETH3":"Dents extraites (oui/non)","RMVTETH4":"Nb de dents extraites","PERSDOC3":"Médecin traitant (1-2 oui, 3 non)","_DENVST3":"Visite dentaire récente","LASTDEN4":"Dernière visite chez le dentiste","_HCVU652":"Couverture santé (18-64 ans)"}
lab = lambda c: LAB.get(c, c)
plt.figure(figsize=(9.5, 6.8))
shap.summary_plot(sv, Xs, feature_names=[lab(c) for c in cols], max_display=15, show=False, plot_size=None, color_bar_label="valeur de la variable")
fig = plt.gcf(); fig.patch.set_facecolor(BG)
for a in fig.axes: a.set_facecolor(BG)
imp = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)
headline(fig, f"« {lab(imp.index[0])} », « {lab(imp.index[1])} » et « {lab(imp.index[2])} » sont les variables qui pèsent le plus dans la prédiction",
         "Importance SHAP (top 15) : chaque point est une personne ; rouge = valeur élevée de la variable")
plt.show()
""")

md("""
Certaines variables, comme avoir déjà passé un scanner thoracique ou avoir un médecin traitant, jouent le rôle de **marqueurs de contact avec le système de soins** :
elles aident la prédiction (les personnes déjà malades ou âgées voient plus de médecins) sans être des causes de la maladie.
""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
for ax, c in zip(axes, ["_AGE80", "n_comorbid", "GENHLTH", "_BMI5"]):
    j = cols.index(c); x = Xs[c].to_numpy(); s = sv[:, j]; male = Xs["SEXVAR"].to_numpy()
    for mval, col, nm in [(0, GREY, "femmes"), (1, RED, "hommes")]:
        k = male == mval
        ax.scatter(x[k] + (np.random.RandomState(0).rand(k.sum()) - .5) * (0.25 if c in ("n_comorbid", "GENHLTH") else 0), s[k], s=5, alpha=0.35, color=col, label=nm)
    ax.axhline(0, color=DARK, lw=0.8); ax.set_xlabel(lab(c), fontsize=8.5); ax.set_title({"_AGE80": "Âge", "n_comorbid": "Maladies chroniques", "GENHLTH": "Santé perçue", "_BMI5": "IMC"}[c], loc="left", fontsize=10)
axes[0].set_ylabel("contribution SHAP (log-odds)"); axes[0].legend(markerscale=3)
if "_BMI5" in cols: axes[3].set_xlim(15, 55)
_j = cols.index("_AGE80"); _a = pd.DataFrame({"age": Xs["_AGE80"], "s": sv[:, _j]}); _a["b"] = (_a.age // 5) * 5
_m = _a.groupby("b")["s"].mean(); _cross = _m[_m > 0].index.min()
headline(fig, f"L'effet de l'âge ne devient défavorable qu'à partir de {int(_cross)} ans, puis s'amplifie ; l'IMC pèse peu et suit une courbe en cloche",
         "Dépendance SHAP pour quatre variables clés (contribution à la prédiction, en log-odds ; rouge = hommes, gris = femmes)", extra=0.3)
plt.show()
""")

code("""
def pick(mask, target=0.5):
    cand = np.where(mask)[0]
    p = m.predict_proba(Xc.iloc[cand][cols])[:, 1]
    return cand[np.argsort(np.abs(p - np.median(p)))[0]]
A = pick(((Xc.SEXVAR == 1) & Xc._AGE80.between(66, 70) & (Xc.SMOKE100 == 1) & (Xc.diab_status == 2) & (Xc.GENHLTH >= 3)).to_numpy())
B = pick(((Xc.SEXVAR == 0) & Xc._AGE80.between(43, 47) & (Xc.SMOKE100 == 0) & (Xc.diab_status == 0) & (Xc.GENHLTH <= 2) & (Xc.EXERANY2 == 1)).to_numpy())
for ix, ttl in [(A, "Persona A : homme d'environ 68 ans, fumeur, diabétique"), (B, "Persona B : femme d'environ 45 ans, active, non-fumeuse")]:
    row = Xc.iloc[[ix]][cols]; s_ = expl.shap_values(row); s_ = s_[1] if isinstance(s_, list) else s_
    pr = m.predict_proba(row)[0, 1]
    ex_ = shap.Explanation(values=s_[0], base_values=base, data=row.iloc[0].to_numpy(), feature_names=[lab(c) for c in cols])
    shap.plots.waterfall(ex_, max_display=9, show=False)
    fig = plt.gcf(); fig.set_size_inches(10.5, 5.6); fig.patch.set_facecolor(BG)
    for a in fig.axes: a.set_facecolor(BG)
    headline(fig, f"{ttl}, risque estimé à {pr:.1%}",
             f"réalité dans l'enquête : {'a eu une cardiopathie' if y[ix] else 'pas de cardiopathie'} ; contributions SHAP en log-odds (rouge = pousse vers la maladie)", extra=0.35)
    plt.show()
""")

md("""
### Une lecture « épidémiologique » : les odds ratios

Une régression logistique sur quelques variables lisibles donne une autre lecture, plus classique : de combien les **chances**
d'avoir une cardiopathie sont multipliées, toutes choses égales par ailleurs.
""")

code("""
import statsmodels.api as sm
d = pd.DataFrame({
    "Âge (par 10 ans)": Xc["_AGE80"] / 10, "Homme": Xc["SEXVAR"], "Santé perçue (par cran de dégradation)": Xc["GENHLTH"],
    "Diabète": (Xc["diab_status"] == 2).astype(float).where(Xc["diab_status"].notna()), "Antécédent d'AVC": Xc["CVDSTRK3"], "Maladie rénale": Xc["CHCKDNY2"],
    "BPCO": Xc["CHCCOPD3"], "Arthrite": Xc["HAVARTH4"], "Dépression": Xc["ADDEPEV3"], "A déjà fumé (100 cigarettes)": Xc["SMOKE100"],
    "IMC (par 5 kg/m²)": Xc["_BMI5"] / 5, "Activité physique (30 derniers jours)": Xc["EXERANY2"], "Difficulté à marcher": Xc["DIFFWALK"]})
ok = d.notna().all(axis=1)
res = sm.Logit(y[ok], sm.add_constant(d[ok])).fit(disp=0)
ci = np.exp(res.conf_int().drop("const")); orr = np.exp(res.params.drop("const"))
t = pd.DataFrame({"or": orr, "lo": ci[0], "hi": ci[1]}).sort_values("or")
fig, ax = plt.subplots(figsize=(9.5, 5.8))
ax.hlines(range(len(t)), t.lo, t.hi, color=GREY, lw=2); ax.scatter(t["or"], range(len(t)), color=[RED if v > 1 else BLUE for v in t["or"]], s=55, zorder=4)
ax.axvline(1, color=DARK, lw=1); ax.set_xscale("log"); ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index); ax.grid(axis="y", visible=False)
ax.set_xticks([0.8, 1, 1.5, 2, 3]); ax.set_xticklabels(["0,8", "1", "1,5", "2", "3"]); ax.minorticks_off(); ax.set_xlim(0.8, t.hi.max() * 1.35)
for i, v in enumerate(t["or"]): ax.text(t.hi.iloc[i] * 1.04, i, f"×{v:.2f}", va="center", fontsize=9)
ax.set_xlabel("odds ratio (échelle logarithmique), IC 95 %")
top = t.index[-1]
headline(fig, f"Toutes choses égales par ailleurs, « {top} » est l'antécédent le plus fortement associé : chances multipliées par {t['or'].iloc[-1]:.1f}",
         f"Régression logistique sur {int(ok.sum()):,} personnes avec données complètes sur 13 variables lisibles".replace(",", " "))
plt.show()
""")

# =====================================================================================================
md("""
## 6. Limites et conclusion

**Équité.** Un modèle peut être performant en moyenne et moins bon pour certains groupes. Voici le recall et la précision au seuil choisi,
par sexe, tranche d'âge et revenu.
""")

code("""
pred = oof >= thr
age = Xc["_AGE80"]
groups = {
  "Sexe": {"Femmes": Xc["SEXVAR"] == 0, "Hommes": Xc["SEXVAR"] == 1},
  "Âge": {"18-44": age < 45, "45-64": age.between(45, 64.99), "65+": age >= 65},
  "Revenu": {"< 35k": Xc["_INCOMG1"] <= 3, "35-100k": Xc["_INCOMG1"].isin([4, 5]), "100k+": Xc["_INCOMG1"] >= 6},
}
rows = []
for gname, gd in groups.items():
    for lab_, mk in gd.items():
        mk = mk.to_numpy(); yy, pp = y[mk], pred[mk]
        rows.append(dict(groupe=gname, sous_groupe=lab_, n=int(mk.sum()), prevalence=yy.mean(), recall=recall_score(yy, pp), precision=precision_score(yy, pp), f1=f1_score(yy, pp)))
E = pd.DataFrame(rows)
display(E.style.format({"prevalence": "{:.1%}", "recall": "{:.2f}", "precision": "{:.2f}", "f1": "{:.2f}"}).hide(axis="index"))
fig, ax = plt.subplots(figsize=(10, 4.6))
yv = np.arange(len(E))[::-1]
ax.hlines(yv, E.precision, E.recall, color=LIGHT, lw=3)
ax.scatter(E.recall, yv, color=RED, s=60, label="recall", zorder=4); ax.scatter(E.precision, yv, color=GREY, s=60, label="précision", zorder=4)
ax.set_yticks(yv); ax.set_yticklabels([f"{a} : {b}" for a, b in zip(E.groupe, E.sous_groupe)]); ax.grid(axis="y", visible=False)
ax.legend(loc="lower right"); ax.set_xlim(0, 1)
worst = E.loc[E.recall.idxmin()]
headline(fig, f"Le recall est très inégal : de {E.recall.min():.0%} ({worst.groupe} {worst.sous_groupe}) à {E.recall.max():.0%} selon le groupe",
         "Recall et précision du blend au seuil choisi, par sous-groupe (OOF)")
plt.show()
""")

md("""
Cette différence est en grande partie **mécanique** : chez les jeunes la maladie est rare et peu prévisible à partir de réponses déclaratives,
chez les plus âgés la prévalence est élevée, donc il est plus facile d'« avoir raison ». Un seuil unique n'est pas forcément le bon choix
si l'on veut une couverture équitable entre groupes.

**Ce qu'il faut garder en tête**

- **Données déclaratives** : tout repose sur ce que les personnes disent d'elles-mêmes, avec les erreurs de mémoire et de désirabilité que cela suppose.
- **Corrélation n'est pas causalité** : le modèle repère des profils, il n'explique pas pourquoi une personne est tombée malade (SHAP décrit le modèle, pas la biologie).
- **Biais de survie** : l'enquête interroge des personnes vivantes ; celles qui ont succombé à un infarctus n'y figurent pas.
- **Variables absentes** : ni tension ni cholestérol dans ces données, deux des meilleurs prédicteurs cliniques connus.
- **Prévalence** : environ 9 % dans cet échantillon ; avec une autre prévalence, précision et seuil optimal changeraient.
- **Un signal administratif écarté** : le numéro d'ordre de la réponse (`SEQNO`) est faiblement corrélé à la cible. Nous ne l'avons **pas** utilisé, car il ne
  reflète aucune caractéristique de la personne et son usage serait contraire à l'esprit de la compétition.

""")

code("""
from IPython.display import Markdown, display
display(Markdown(f"**En une phrase.** Avec quelques réponses déclaratives, le modèle repère **{r_op:.0%}** des personnes concernées, avec **{p_op:.0%}** d'alertes justifiées "
                 f"(soit {fa:.1f} fausses alertes pour un vrai positif, F1 = {f_op:.3f}) : un outil de **priorisation** raisonnable pour orienter une prévention ciblée, mais pas un outil de diagnostic."))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
out = ROOT / "notebooks" / "heart_risk.ipynb"
nbf.write(nb, out)
print("ecrit", out)
if "--no-exec" not in sys.argv:
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=1800", str(out)], check=True, cwd=ROOT / "notebooks")
    print("execute")
