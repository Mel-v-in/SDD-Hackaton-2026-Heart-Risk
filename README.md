# SDD Hackathon 2026 : Heart Risk

Projet pour le hackathon de Machine Learning SDD 2026 (Kaggle) : prédire, à partir des réponses à l'enquête téléphonique BRFSS 2022 (CDC), si un adulte a déjà eu un infarctus ou une maladie coronarienne.

- **Tâche** : classification binaire (`TARGET`), prévalence d'environ 9 %
- **Métrique** : F1 sur la classe positive (soumission en classes, format `ID,TARGET`)
- **Deux livrables** : le meilleur F1 sur le leaderboard privé, et un notebook de data storytelling

## Structure

```
data/         données brutes et codebook (non versionnés)
src/          clean.py (décodage BRFSS), features.py, train.py (CV, métriques, modèles), config.py, load.py,
              experiments.py, run_baselines.py, run_gbdt.py, run_logreg.py (expériences), blend.py, make_submission.py
notebooks/    heart_risk.ipynb : le livrable de storytelling (exécuté, figures visibles) ; build_notebook.py le régénère
docs/         eda_findings.md (constats d'EDA), methode_et_resultats.md (méthode, résultats, limites)
outputs/      OOF, prédictions test, soumissions (non versionnés), experiments.csv (versionné)
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Données

Les données de la compétition ne sont pas dans le dépôt (trop volumineuses pour GitHub, et à ne pas redistribuer). Pour les récupérer, avec un token Kaggle dans `~/.kaggle/kaggle.json` (jamais dans le dépôt) :

```bash
kaggle competitions download -c sdd-hackaton-2026-heart-risk -p data/
unzip -o "data/*.zip" -d data/
```

## Principes de travail

- Même pipeline de nettoyage appliqué à train, extra et test.
- `random_state=42` partout, 5 folds stratifiés, validation sur les lignes de train (seules représentatives de test : extra vient d'autres États, voir `docs/eda_findings.md`).
- Aucun fit (imputer, scaler, target encoding) en dehors des folds d'entraînement.
- Seuil de décision choisi sur les prédictions OOF, jamais sur le leaderboard.
- Aucune donnée externe BRFSS jointe (règle de la compétition).

## Résultats

Journal complet des expériences : `outputs/experiments.csv`. Méthode, enseignements et limites : `docs/methode_et_resultats.md`.

CV 5 folds stratifiés sur les 225 000 lignes de train, F1 au seuil optimal de chaque modèle :

| Modèle | F1 OOF | Seuil | AUC | PR-AUC |
|---|---|---|---|---|
| Régression logistique (décodé) | 0,4177 | 0,195 | 0,8505 | 0,3682 |
| LightGBM, colonnes brutes | 0,4207 | 0,220 | 0,8527 | 0,3723 |
| LightGBM, décodé + features | 0,4204 | 0,210 | 0,8535 | 0,3740 |
| LightGBM régularisé, sans colonnes État | 0,4242 | 0,225 | 0,8543 | 0,3762 |
| LightGBM régularisé + extra | 0,4249 | 0,215 | 0,8551 | 0,3777 |
| XGBoost + extra | 0,4236 | 0,220 | 0,8546 | 0,3776 |
| CatBoost + extra | 0,4219 | 0,200 | 0,8537 | 0,3732 |
| **Blend des trois (poids égaux)** | **0,4266** | **0,215** | **0,8553** | **0,3789** |

Soumissions : `outputs/submission_final.csv` (blend, seuil 0,215) et `outputs/submission_alt.csv` (seuil 0,20, un peu plus de recall). Les fichiers `*_01.csv` contiennent les mêmes prédictions avec `TARGET` en 0/1 (le `sample_submission.csv` utilise `True/False`).

## Reproduire

```bash
python -m src.load
python -m src.run_baselines clean && python -m src.run_baselines raw
python -m src.experiments lgbm_reg1_nostate lgbm_reg1_extra
python -m src.run_gbdt xgb && python -m src.run_gbdt cat && python -m src.run_logreg
python -m src.blend lgbm_reg1_extra xgboost_extra catboost_extra
python -m src.make_submission submission_final lgbm_reg1_extra xgboost_extra catboost_extra
python -m src.make_submission submission_alt lgbm_reg1_extra xgboost_extra catboost_extra --thr 0.2
python notebooks/build_notebook.py
```
