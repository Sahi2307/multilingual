from __future__ import annotations

"""Train the MuRIL-based category classification model.

This script fine-tunes ``google/muril-base-cased`` for 3-way
classification of complaint categories:
  * Sanitation
  * Water Supply
  * Transportation

It uses the dataset prepared by ``data_preparation.py`` and saves the
fine-tuned model to ``models/muril_category_classifier/``.

Run as a script:
    python -m src.train_category_model
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

RANDOM_SEED: int = 42
MURIL_MODEL_NAME: str = "google/muril-base-cased"
MAX_LENGTH: int = 128
BATCH_SIZE: int = 8
NUM_EPOCHS: int = 5
LEARNING_RATE: float = 2e-5
WEIGHT_DECAY: float = 0.01

CATEGORIES: List[str] = ["Sanitation", "Water Supply", "Transportation"]
CATEGORY_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}
ID_TO_CATEGORY: Dict[int, str] = {i: c for c, i in CATEGORY_TO_ID.items()}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass
class ComplaintDataset(Dataset):
    """PyTorch dataset for complaint category classification.

    Attributes:
        texts: List of complaint texts.
        labels: List of integer-encoded labels.
        tokenizer: Hugging Face tokenizer.
    """

    texts: List[str]
    labels: List[int]
    tokenizer: AutoTokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = int(self.labels[idx])
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def encode_labels(labels: List[str]) -> List[int]:
    """Map string category labels to integer IDs.

    Args:
        labels: List of category names.

    Returns:
        List of integer labels.
    """
    return [CATEGORY_TO_ID[l] for l in labels]


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute evaluation metrics for the Trainer.

    Args:
        eval_pred: Tuple of (logits, labels).

    Returns:
        Dictionary containing accuracy, precision, recall, and weighted F1.
    """
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }


def train_and_save_category_model() -> None:
    """Fine-tune MuRIL for category classification and save the model."""
    root_dir = Path(__file__).resolve().parents[1]
    data_path = root_dir / "data" / "civic_complaints.csv"
    models_dir = root_dir / "models" / "muril_category_classifier"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run data_preparation.py first."
        )

    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    logger.info("Loaded dataset with %d rows from %s", len(df), data_path)

    tokenizer = AutoTokenizer.from_pretrained(MURIL_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MURIL_MODEL_NAME,
        num_labels=len(CATEGORIES),
    )

    def _build_split(split: str) -> ComplaintDataset:
        subset = df[df["split"] == split].reset_index(drop=True)
        texts = subset["text"].astype(str).tolist()
        labels = encode_labels(subset["category"].tolist())
        return ComplaintDataset(texts=texts, labels=labels, tokenizer=tokenizer)

    train_dataset = _build_split("train")
    val_dataset = _build_split("val")
    test_subset = df[df["split"] == "test"].reset_index(drop=True)
    test_dataset = ComplaintDataset(
        texts=test_subset["text"].astype(str).tolist(),
        labels=encode_labels(test_subset["category"].tolist()),
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=str(models_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=10,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0001)],
    )

    logger.info("Starting MuRIL fine-tuning for category classification...")
    trainer.train()

    logger.info("Evaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    logger.info("Validation metrics: %s", val_metrics)

    logger.info("Evaluating on test set...")
    test_predictions = trainer.predict(test_dataset)
    test_logits = test_predictions.predictions
    test_labels = test_predictions.label_ids
    test_preds = test_logits.argmax(axis=-1)

    acc = accuracy_score(test_labels, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="weighted", zero_division=0
    )
    logger.info("Test accuracy: %.4f, precision: %.4f, recall: %.4f, f1_weighted: %.4f", acc, precision, recall, f1)

    cm = confusion_matrix(test_labels, test_preds)
    logger.info("Test confusion matrix (rows=true, cols=pred):\n%s", cm)

    report = classification_report(
        test_labels,
        test_preds,
        target_names=CATEGORIES,
        digits=4,
        zero_division=0,
    )
    logger.info("Per-class classification report (test):\n%s", report)

    # Language-specific performance on test set
    test_langs = test_subset["language"].tolist()
    test_categories = test_subset["category"].tolist()

    for lang in sorted(set(test_langs)):
        mask = np.array([l == lang for l in test_langs])
        if mask.sum() == 0:
            continue
        lang_true = test_labels[mask]
        lang_pred = test_preds[mask]
        lang_acc = accuracy_score(lang_true, lang_pred)
        _, _, lang_f1, _ = precision_recall_fscore_support(
            lang_true, lang_pred, average="weighted", zero_division=0
        )
        logger.info(
            "Test language '%s': n=%d, accuracy=%.4f, f1_weighted=%.4f",
            lang,
            mask.sum(),
            lang_acc,
            lang_f1,
        )

    # Save the best model
    trainer.save_model(str(models_dir))
    logger.info("Saved fine-tuned MuRIL model to %s", models_dir)


def main() -> None:
    """CLI entry-point for category model training."""
    try:
        train_and_save_category_model()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to train category model: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover - CLI path
    main()
