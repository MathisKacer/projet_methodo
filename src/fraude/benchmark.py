"""Protocole expérimental commun à toutes les méthodes de rééquilibrage.

Principes (cf. CLAUDE.md, « Règles méthodologiques ») :
- données dédoublonnées ; découpage train/test fait UNE fois, avant tout rééchantillonnage ;
- toutes les transformations et le rééchantillonnage sont dans un `imblearn.pipeline.Pipeline`,
  donc ajustés uniquement sur les plis d'entraînement (aucune observation synthétique
  n'est créée à partir de données de validation ou de test) ;
- validation croisée stratifiée répétée sur le train ; le test n'est utilisé qu'à la fin ;
- on conserve les probabilités prédites de chaque pli, avec l'index des transactions,
  pour étudier ensuite le seuil, les coûts et la typologie des fraudes sans réentraîner.

Exemple :

    from sklearn.linear_model import LogisticRegression
    from imblearn.over_sampling import SMOTE
    from fraude.benchmark import load_split, make_pipeline, evaluate, summarize

    split = load_split()
    res = evaluate(make_pipeline(LogisticRegression(max_iter=1000), SMOTE(random_state=42)),
                   split.X_train, split.y_train, name="logreg_smote")
    summarize({"logreg_smote": res})
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from fraude.data import PCA_FEATURES, PROJECT_ROOT, TARGET, load_creditcard

SEED = 42
TEST_SIZE = 0.2
DAY_SECONDS = 86_400
RESULTS_DIR = PROJECT_ROOT / "data" / "results"  # dans data/ : non versionné


# --------------------------------------------------------------------------- découpages

@dataclass
class Split:
    """Un découpage train/test. Les index d'origine des transactions sont conservés."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    kind: str

    def describe(self) -> pd.DataFrame:
        rows = {}
        for part, y in (("train", self.y_train), ("test", self.y_test)):
            rows[part] = {"transactions": len(y), "fraudes": int(y.sum()), "taux": y.mean()}
        return pd.DataFrame(rows).T


def load_split(test_size: float = TEST_SIZE, seed: int = SEED) -> Split:
    """Découpage principal : aléatoire, stratifié sur la classe."""
    df = load_creditcard(drop_duplicates=True)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y,
                                              random_state=seed)
    return Split(X_tr, X_te, y_tr, y_te, kind="stratifié aléatoire")


def load_temporal_split(cutoff: float = DAY_SECONDS) -> Split:
    """Découpage temporel (piste prioritaire n° 1) : entraînement sur le jour 1, test sur le jour 2.

    Pas de stratification possible : le taux de fraude peut différer entre les deux jours.
    """
    df = load_creditcard(drop_duplicates=True)
    X, y = df.drop(columns=TARGET), df[TARGET]
    train = df["Time"] < cutoff
    return Split(X[train], X[~train], y[train], y[~train], kind="temporel (jour 1 → jour 2)")


# --------------------------------------------------------------------------- pipelines

def _hour_cyclic(X):
    """Heure relative encodée sur le cercle (23 h et 0 h sont voisines)."""
    hour = (np.asarray(X, dtype=float)[:, 0] % DAY_SECONDS) / 3600
    angle = 2 * np.pi * hour / 24
    return np.column_stack([np.sin(angle), np.cos(angle)])


def make_preprocessor(use_hour: bool = False, scaler=None) -> ColumnTransformer:
    """Prétraitement retenu après l'EDA.

    - `Amount` : log1p puis mise à l'échelle ;
    - `V1`–`V28` : mise à l'échelle (les variances vont de 0,1 à 3,8) ;
    - `Time` brut est retiré (compteur monotone sur 2 jours, sans sens en découpage temporel) ;
      `use_hour=True` ajoute l'heure relative encodée en sin/cos.
    """
    scaler = scaler if scaler is not None else StandardScaler()
    blocks = [
        ("amount", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                             ("scale", clone(scaler))]), ["Amount"]),
        ("pca", clone(scaler), PCA_FEATURES),
    ]
    if use_hour:
        blocks.append(("hour", FunctionTransformer(
            _hour_cyclic, feature_names_out=lambda *_: np.array(["hour_sin", "hour_cos"])), ["Time"]))
    return ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=False)


def make_pipeline(model, sampler=None, use_hour: bool = False, scaler=None) -> Pipeline:
    """prétraitement → (rééchantillonnage) → modèle.

    Le sampler vient APRÈS la mise à l'échelle : SMOTE, ADASYN, ENN, NearMiss reposent sur
    des distances euclidiennes, qui seraient sinon dominées par les variables à forte variance.
    `sampler=None` = pas de rééchantillonnage (modèle de référence ou pondération des classes,
    à régler directement dans `model`, ex. `class_weight="balanced"`).
    """
    steps = [("prep", make_preprocessor(use_hour, scaler))]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return Pipeline(steps)


# --------------------------------------------------------------------------- métriques

def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Métriques indépendantes du seuil (PR-AUC, ROC-AUC) et au seuil donné."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# --------------------------------------------------------------------------- évaluation

@dataclass
class CVResult:
    """Résultat d'une évaluation.

    metrics     : une ligne par pli (repeat, fold, métriques, temps d'ajustement, taille
                  du train après rééchantillonnage).
    predictions : une ligne par (repeat, transaction de validation) : index d'origine,
                  y_true, proba, fold. Chaque transaction apparaît une fois par répétition.
    """
    name: str
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    params: dict = field(default_factory=dict)

    def save(self, directory: Path = RESULTS_DIR) -> None:
        directory = Path(directory) / self.name
        directory.mkdir(parents=True, exist_ok=True)
        self.metrics.to_parquet(directory / "metrics.parquet")
        self.predictions.to_parquet(directory / "predictions.parquet")
        pd.Series(self.params, dtype=str).to_frame("value").to_parquet(directory / "params.parquet")

    @classmethod
    def load(cls, name: str, directory: Path = RESULTS_DIR) -> "CVResult":
        d = Path(directory) / name
        params = pd.read_parquet(d / "params.parquet")["value"].to_dict()
        return cls(name, pd.read_parquet(d / "metrics.parquet"),
                   pd.read_parquet(d / "predictions.parquet"), params)


def _fit_fold(pipeline, X, y, tr, va, repeat, fold, threshold):
    """Ajuste le pipeline sur le pli d'entraînement et prédit le pli de validation.

    Les étapes sont enchaînées explicitement (exactement ce que fait `Pipeline.fit`) pour
    connaître la taille de l'échantillon réellement vu par le modèle après rééchantillonnage.
    """
    est = clone(pipeline)
    steps = est.named_steps
    t0 = time.perf_counter()
    Xt = steps["prep"].fit_transform(X.iloc[tr], y.iloc[tr])
    y_fit = y.iloc[tr].to_numpy()
    if "sampler" in steps:
        Xt, y_fit = steps["sampler"].fit_resample(Xt, y_fit)
    steps["model"].fit(Xt, y_fit)
    fit_time = time.perf_counter() - t0
    proba = steps["model"].predict_proba(steps["prep"].transform(X.iloc[va]))[:, 1]
    metrics = {"repeat": repeat, "fold": fold, **compute_metrics(y.iloc[va], proba, threshold),
               "fit_time_s": fit_time, "n_train_vu": len(y_fit),
               "n_fraudes_vues": int(np.sum(y_fit))}
    preds = pd.DataFrame({"repeat": repeat, "fold": fold, "index": X.index[va],
                          "y_true": y.iloc[va].to_numpy(), "proba": proba})
    return metrics, preds


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, name: str,
             n_splits: int = 5, n_repeats: int = 3, seed: int = SEED,
             threshold: float = 0.5, n_jobs: int = -1, save: bool = True) -> CVResult:
    """Validation croisée stratifiée répétée, identique pour toutes les méthodes.

    Les mêmes plis sont utilisés pour toutes les méthodes (même graine) : les comparaisons
    sont donc appariées, pli par pli.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    jobs = (delayed(_fit_fold)(pipeline, X, y, tr, va, i // n_splits, i % n_splits, threshold)
            for i, (tr, va) in enumerate(cv.split(X, y)))
    out = Parallel(n_jobs=n_jobs)(jobs)
    res = CVResult(
        name=name,
        metrics=pd.DataFrame([m for m, _ in out]),
        predictions=pd.concat([p for _, p in out], ignore_index=True),
        params={"n_splits": n_splits, "n_repeats": n_repeats, "seed": seed,
                "threshold": threshold, "pipeline": repr(pipeline)},
    )
    if save:
        res.save()
    return res


def quick_evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, name: str = "rapide",
                   seed: int = SEED, threshold: float = 0.5, save: bool = False) -> CVResult:
    """Évaluation rapide pour l'exploration : UN seul pli de validation (≈ 20 % du train).

    C'est exactement le premier pli de `evaluate` (même graine) : le résultat est donc
    directement comparable à la ligne (repeat=0, fold=0) de la validation complète.
    Le test n'est jamais utilisé.

    Attention : ≈ 76 fraudes en validation → la PR-AUC varie d'environ ± 0,05 d'un pli à
    l'autre. Un écart plus petit entre deux méthodes n'est pas interprétable ici ; il faut
    le confirmer avec `evaluate`. Pour aller vite, paralléliser dans le modèle (n_jobs=-1),
    puisqu'il n'y a qu'un pli.
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
    tr, va = next(cv.split(X, y))
    metrics, preds = _fit_fold(pipeline, X, y, tr, va, 0, 0, threshold)
    res = CVResult(name=name, metrics=pd.DataFrame([metrics]), predictions=preds,
                   params={"mode": "rapide (1 pli)", "seed": seed, "threshold": threshold,
                           "pipeline": repr(pipeline)})
    if save:
        res.save()
    return res


def evaluate_on_test(pipeline: Pipeline, split: Split, name: str,
                     threshold: float = 0.5) -> tuple[dict, pd.DataFrame]:
    """Ajuste sur tout le train et évalue sur le test. À n'utiliser qu'en fin d'étude
    (ou pour le découpage temporel, où il n'y a qu'un seul test possible)."""
    est = clone(pipeline).fit(split.X_train, split.y_train)
    proba = est.predict_proba(split.X_test)[:, 1]
    preds = pd.DataFrame({"index": split.X_test.index, "y_true": split.y_test.to_numpy(),
                          "proba": proba})
    return {"name": name, **compute_metrics(split.y_test, proba, threshold)}, preds


# --------------------------------------------------------------------------- synthèse

MAIN_METRICS = ["pr_auc", "roc_auc", "precision", "recall", "f1", "f2",
                "balanced_accuracy", "accuracy", "fit_time_s"]


def summarize(results: dict[str, CVResult], metrics: list[str] = MAIN_METRICS) -> pd.DataFrame:
    """Tableau moyenne et écart-type par méthode (sur tous les plis × répétitions)."""
    rows = {}
    for name, r in results.items():
        m = r.metrics[metrics]
        rows[name] = pd.concat([m.mean().add_suffix("_mean"), m.std().add_suffix("_std")])
    table = pd.DataFrame(rows).T
    return table[[f"{m}_{s}" for m in metrics for s in ("mean", "std")]]


def format_summary(table: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """Version lisible : « moyenne ± écart-type »."""
    metrics = [c[:-5] for c in table.columns if c.endswith("_mean")]
    return pd.DataFrame({m: table[f"{m}_mean"].round(digits).astype(str) + " ± "
                         + table[f"{m}_std"].round(digits).astype(str) for m in metrics})
