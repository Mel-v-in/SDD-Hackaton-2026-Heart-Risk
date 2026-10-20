"""Feature engineering (n_comorbid, n_limitations, interactions, statut tabagique...).

Meme principe que clean.py : fonction pure, appliquee a l'identique a train, extra et test.
"""
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError
