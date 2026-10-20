"""Constantes partagees par tout le projet."""
from pathlib import Path

SEED = 42
N_FOLDS = 5

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
EXPERIMENTS_LOG = OUTPUT_DIR / "experiments.csv"
