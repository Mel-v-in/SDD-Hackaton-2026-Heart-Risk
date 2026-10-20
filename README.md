# SDD Hackathon 2026 : Heart Risk

Projet pour le hackathon de Machine Learning SDD 2026 (Kaggle) : prédire, à partir des réponses à l'enquête téléphonique BRFSS 2022 (CDC), si un adulte a déjà eu un infarctus ou une maladie coronarienne.

- **Tâche** : classification binaire (`TARGET`), prévalence d'environ 9 %
- **Métrique** : F1 sur la classe positive (soumission en classes 0/1, format `ID,TARGET`)
- **Deux livrables** : le meilleur F1 sur le leaderboard privé, et un notebook de data storytelling

## Structure

```
data/         données brutes et codebook (non versionnés)
src/          clean.py (décodage BRFSS), features.py, train.py, config.py
notebooks/    heart_risk.ipynb : le livrable de storytelling (exécuté, figures visibles)
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

Journal complet des expériences : `outputs/experiments.csv`.

| Modèle | F1 OOF | Seuil | AUC | PR-AUC |
|---|---|---|---|---|
| à remplir | | | | |
