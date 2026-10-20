# Constats d'EDA (20/10/2026)

Reproduction : `python -m src.load` (CSV vers parquet), puis `python -m src.eda_checks`, `python -m src.eda_adversarial`, `python -m src.eda_missing`.
Chiffres recoupés avec une lecture indépendante des CSV bruts.

## Données et intégrité

- train 225 000 x 325, test 75 000 x 324 (pas de `TARGET`), extra 145 132 x 325, sample_submission 75 000 x 2. Mêmes colonnes dans train et extra.
- `ID` unique dans chaque jeu et disjoints (train 0 à 224999, test 225000 à 299999, extra 300000 à 445131). Aucune ligne dupliquée, ni dans un jeu ni entre jeux.
- `TARGET` est écrit `True`/`False` dans les CSV (train, extra et sample_submission). Prévalence : 8,95 % dans train (20 139 positifs), 8,87 % dans extra (12 878). **extra est bien labellisé.**
- Aucune colonne de type object : `IDATE`, `IMONTH`, `IDAY`, `IYEAR`, `SEQNO` sont des chaînes de chiffres lues comme des entiers (`IDATE` perd son zéro initial). Toutes sont administratives.
- Le codebook HTML n'est pas dans `data/` : à récupérer (Kaggle ou CDC) avant le décodage BRFSS.

## extra.csv vient d'autres États

- Train et test couvrent les mêmes 36 États (codes FIPS 1 à 39). extra couvre 19 États (FIPS 40 à 78, dont Guam, Porto Rico, Îles Vierges) et un seul État commun avec train (Ohio, 0,86 % des lignes de extra).
- Adversarial validation (LightGBM, 3 folds, 75 000 + 75 000 lignes) : train vs test AUC 0,503 (0,502 sans colonnes administratives), donc indiscernables. extra vs test et train vs extra : AUC 1,000, y compris sans `_STATE` ni colonnes administratives (`_STSTR` prend le relais, puis `CAREGIV1`, `QSTVER`, `TRNSGNDR`).
- Prévalence par État comparable des deux côtés (5,9 % à 13 %).
- Conséquences proposées (à valider par expérience) :
  1. Valider uniquement sur les lignes de train, seules représentatives de test. extra sert au mieux de données d'entraînement supplémentaires dans les folds, avec un test du gain.
  2. Si extra est utilisé, retirer les features qui identifient l'État (`_STATE`, `_STSTR`, `_STRWT`, `QSTVER`...).
  3. Idée de storytelling : entraîner sur train, évaluer sur extra, soit la généralisation à des États jamais vus.

## Fuite éventuelle

- `ID` : AUC 0,5001 (train), 0,4844 (extra), prévalence par décile d'ID entre 8,7 % et 9,3 %. Les lignes sont mélangées, pas de signal.
- `SEQNO` (et `_PSU`, identique) : AUC 0,4387, soit 0,4354 pour son rang à l'intérieur de l'État. Signal réel, équivalent à une AUC d'environ 0,56. À ne pas exploiter (colonne administrative, risque de disqualification), à mentionner dans le notebook. Cause non vérifiée.

## Valeurs manquantes

- 323 features : 51 sans NaN, 40 entre 0 et 5 %, 49 entre 5 et 50 %, 55 entre 50 et 80 %, 51 entre 80 et 95 %, 77 au-dessus de 95 % (dont 3 entièrement vides dans train et test, 14 dans extra).
- Taux de NaN identiques entre train et test (écart maximal 0,5 point).
- Manquant informatif : `_FLSHOT7`, `_PNEUMO3`, `_ALTETH3` (62 % de NaN). L'AUC de l'indicateur "est NaN" (0,674) est presque celle de leur valeur. L'âge moyen vaut 44 ans quand la réponse manque contre 72 ans quand elle est renseignée : l'indicateur re-encode surtout l'âge (pas une coupure stricte, l'âge minimal renseigné est 19 ans).

## Pouvoir prédictif univarié (valeurs brutes, codes 7/9/77/99 non décodés)

- Meilleures AUC : `_AGE80` 0,742, `_AGEG5YR` 0,730, `_AGE_G` 0,721, `GENHLTH` 0,707, `EMPLOY1` 0,693.
- 4 features au-dessus de 0,7, 32 au-dessus de 0,6, aucune au-dessus de 0,8 : pas de fuite triviale.
- Tableau complet par colonne : `outputs/eda_columns.csv` (non versionné, régénéré par `src.eda_missing`).

## Format de soumission

- `sample_submission.csv` : `TARGET` vaut `False` pour les 75 000 lignes (IDs 225000 à 299999). Le brief parle de 0/1 : reproduire le format de l'exemple, ou vérifier la page d'évaluation Kaggle avant la première soumission.
