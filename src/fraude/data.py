"""Import du jeu de données Kaggle « Credit Card Fraud Detection » (mlg-ulb/creditcardfraud).

Usage typique (notebook) :

    from fraude.data import load_creditcard
    df = load_creditcard()          # télécharge au 1er appel, puis lit le cache local
    X, y = split_features_target(df)

En ligne de commande :

    python -m fraude.data            # télécharge + vérifie + affiche un résumé
    python -m fraude.data --force    # force un nouveau téléchargement

Le téléchargement passe par `kagglehub`. Le jeu étant public, aucun identifiant
n'est en principe requis ; si Kaggle en demande, définir KAGGLE_USERNAME et
KAGGLE_KEY (ou placer kaggle.json dans ~/.kaggle/).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
CSV_NAME = "creditcard.csv"

# Racine du dépôt : src/fraude/data.py -> ../../
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "Class"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURES = ["Time", *PCA_FEATURES, "Amount"]
EXPECTED_COLUMNS = [*FEATURES, TARGET]
EXPECTED_N_ROWS = 284_807
EXPECTED_N_FRAUDS = 492


def download_raw(force: bool = False) -> Path:
    """Télécharge le CSV brut depuis Kaggle dans data/raw/ et renvoie son chemin."""
    target = RAW_DIR / CSV_NAME
    if target.exists() and not force:
        return target

    import kagglehub  # import local : inutile si le cache existe déjà

    src_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET, force_download=force))
    src = src_dir / CSV_NAME
    if not src.exists():
        raise FileNotFoundError(f"{CSV_NAME} introuvable dans {src_dir}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    return target


def validate(df: pd.DataFrame) -> None:
    """Vérifie que le jeu correspond à la version de référence (colonnes, taille, nb de fraudes)."""
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Colonnes inattendues : {list(df.columns)}")
    if len(df) != EXPECTED_N_ROWS:
        raise ValueError(f"{len(df)} lignes au lieu de {EXPECTED_N_ROWS}")
    n_frauds = int(df[TARGET].sum())
    if n_frauds != EXPECTED_N_FRAUDS:
        raise ValueError(f"{n_frauds} fraudes au lieu de {EXPECTED_N_FRAUDS}")
    if df.isna().any().any():
        raise ValueError("Valeurs manquantes détectées")


def load_creditcard(
    force_download: bool = False, use_cache: bool = True, drop_duplicates: bool = False
) -> pd.DataFrame:
    """Renvoie le jeu complet sous forme de DataFrame validé.

    drop_duplicates=True supprime les 1 081 lignes strictement identiques à une
    autre (dont 19 fraudes) -> 283 726 transactions, 473 fraudes. Voir l'EDA
    (notebooks/01_eda.py) pour la justification.

    Un cache Parquet (data/processed/creditcard.parquet) est créé au premier
    appel : la relecture est alors bien plus rapide que le CSV (~150 Mo).
    Les doublons ne sont PAS supprimés ici : c'est un choix méthodologique
    à documenter dans l'analyse exploratoire.
    """
    parquet = PROCESSED_DIR / "creditcard.parquet"
    if use_cache and parquet.exists() and not force_download:
        df = pd.read_parquet(parquet)
        validate(df)
        return df.drop_duplicates(ignore_index=True) if drop_duplicates else df

    csv_path = download_raw(force=force_download)
    df = pd.read_csv(csv_path, dtype={c: "float64" for c in FEATURES} | {TARGET: "int8"})
    validate(df)

    if use_cache:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet, index=False)
    return df.drop_duplicates(ignore_index=True) if drop_duplicates else df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare les variables explicatives (X) de la cible (y = Class, 1 = fraude)."""
    return df[FEATURES], df[TARGET]


def summary(df: pd.DataFrame) -> str:
    n = len(df)
    n_fraud = int(df[TARGET].sum())
    n_dup = int(df.duplicated().sum())
    return (
        f"Transactions        : {n:,}\n"
        f"Fraudes             : {n_fraud:,} ({n_fraud / n:.4%})\n"
        f"Ratio légit./fraude : {(n - n_fraud) / n_fraud:.0f} : 1\n"
        f"Lignes dupliquées   : {n_dup:,}\n"
        f"Période (Time, h)   : {df['Time'].min() / 3600:.1f} -> {df['Time'].max() / 3600:.1f}\n"
        f"Montant max         : {df['Amount'].max():,.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="retélécharger depuis Kaggle")
    args = parser.parse_args()
    df = load_creditcard(force_download=args.force)
    print(summary(df))


if __name__ == "__main__":
    main()
