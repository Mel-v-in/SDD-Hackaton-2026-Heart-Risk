"""Decodage BRFSS 2022 : donnees brutes -> features propres.

`clean` doit etre une fonction pure, appliquee a l'identique a train, extra et test.
Les codes speciaux (7/9/77/99/88...) dependent de la variable : toujours verifier
dans le codebook CDC avant d'appliquer une regle generique.
"""
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError
