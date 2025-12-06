from __future__ import annotations

"""Train the XGBoost urgency prediction model.

This script trains an XGBoost classifier on 776-dimensional feature
vectors (768-dim MuRIL embeddings + 8 structured features) produced by
``feature_extraction.py``.

Urgency levels:
  * Critical
  * High
  * Medium
  * Low

Run as a script:
    python -m src.train_urgency_model
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from xgboost import XGBClassifier

RANDOM_SEED: int = 42

URGENCY_LEVELS: List[str] = ["Critical", "High", "Medium", "Low"]
ID_TO_URGENCY: Dict[int, str] = {i: u for i, u in enumerate(URGENCY_LEVELS)}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_features(root_dir: Path) -> Dict[str, np.ndarray]:
    """Load urgency features from the NPZ file.

    Args:
        root_dir: Project root directory.

    Returns:
        Dictionary containing feature matrices, labels, and IDs for each split.
    """
    npz_path = root_dir / "data" / "processed" / "urgency_features.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Urgency feature file not found at {npz_path}. Run feature_extraction.py first."
        )
    # allow_pickle=True is safe here because we generate the NPZ locally
    # and only store simple ID arrays as object dtype.
    data = np.load(npz_path, allow_pickle=True)
    return {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "id_train": data["id_train"],
        "X_val": data["X_val"],
        "y_val": data["y_val"],
        "id_val": data["id_val"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "id_test": data["id_test"],
    }


def train_and_save_urgency_model() -> None:
    """Train XGBoost urgency model and save it to disk."""
    root_dir = Path(__file__).resolve().parents[1]
    models_dir = root_dir / "models"
    data_path = root_dir / "data" / "civic_complaints.csv"

    models_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(root_dir)

    X_train = features["X_train"]
    y_train = features["y_train"]
    X_val = features["X_val"]
    y_val = features["y_val"]
    X_test = features["X_test"]
    y_test = features["y_test"]

    logger.info("Training set shape: %s, validation set shape: %s, test set shape: %s", X_train.shape, X_val.shape, X_test.shape)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(URGENCY_LEVELS),
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
    )

    logger.info("Starting XGBoost training for urgency prediction...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
        early_stopping_rounds=25,
    )

    # Combined train+val metrics for overall performance
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    def _eval_split(name: str, X: np.ndarray, y: np.ndarray) -> None:
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        logger.info("%s accuracy: %.4f", name, acc)
        logger.info("%s confusion matrix (rows=true, cols=pred):\n%s", name, confusion_matrix(y, preds))
        report = classification_report(
            y,
            preds,
            target_names=URGENCY_LEVELS,
            digits=4,
            zero_division=0,
        )
        logger.info("%s classification report:\n%s", name, report)

    _eval_split("Train+Val", X_trainval, y_trainval)
    _eval_split("Test", X_test, y_test)

    # Language-specific performance on the test set
    if data_path.exists():
        df = pd.read_csv(data_path)
        id_to_lang = df.set_index("complaint_id")["language"].to_dict()
        test_ids = features["id_test"]
        test_langs = [id_to_lang.get(cid, "Unknown") for cid in test_ids]

        preds_test = model.predict(X_test)
        for lang in sorted(set(test_langs)):
            mask = np.array([l == lang for l in test_langs])
            if mask.sum() == 0:
                continue
            lang_true = y_test[mask]
            lang_pred = preds_test[mask]
            lang_acc = accuracy_score(lang_true, lang_pred)
            logger.info(
                "Test language '%s': n=%d, accuracy=%.4f",
                lang,
                mask.sum(),
                lang_acc,
            )

    # Save model
    model_path = models_dir / "xgboost_urgency_predictor.pkl"
    from joblib import dump

    dump(model, model_path)
    logger.info("Saved XGBoost urgency model to %s", model_path)


def main() -> None:
    """CLI entry-point for urgency model training."""
    try:
        train_and_save_urgency_model()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to train urgency model: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover - CLI path
    main()
