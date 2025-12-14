from __future__ import annotations

"""SHAP-based explainability for the civic complaint system (Phase 2).

This module provides explainability for both:

* Category classification (MuRIL fine-tuned model)
* Urgency prediction (XGBoost model + 776-dim features)

Key features:
  * Word-level importance for the category prediction using
    :mod:`shap`'s text explainers.
  * Factor-wise importance for urgency (text embedding vs 8
    structured features) using :class:`shap.TreeExplainer`.
  * Natural-language summaries of the most important words/factors
    for each prediction.

The functions and classes defined here are designed to be called from
Streamlit pages and from the higher-level ``ComplaintProcessor`` that
will be implemented in a later phase.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import shap
import torch
import xgboost as xgb
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

from .feature_extraction import (  # type: ignore
    MURIL_MODEL_NAME,
    STRUCTURED_FEATURE_COLUMNS,
    MurilFeatureExtractor,
)

logger = logging.getLogger(__name__)

# Ensure SHAP does not spam progress bars (using new config API)
try:
    shap.set_config(show_progress=False)
except Exception:
    # Older SHAP versions may not support set_config; ignore safely.
    pass


CATEGORY_LABELS: List[str] = ["Sanitation", "Water Supply", "Transportation"]
URGENCY_LEVELS: List[str] = ["Critical", "High", "Medium", "Low"]


@dataclass
class CategoryExplanation:
    """Container for category-level SHAP explanation.

    Attributes:
        predicted_label: Human-readable category label.
        confidence: Probability associated with the predicted label.
        token_importances: List of ``{"token": str, "value": float}`` in
            the order they appear in the text.
        top_keywords: Top-k tokens by absolute SHAP value.
        shap_values: Raw SHAP values (1D array) aligned with tokens.
        tokens: List of tokens for the explanation.
        text: Original complaint text.
        nl_explanation: Natural-language summary of the reasoning.
    """

    predicted_label: str
    confidence: float
    token_importances: List[Dict[str, Any]]
    top_keywords: List[str]
    shap_values: np.ndarray
    tokens: List[str]
    text: str
    nl_explanation: str


@dataclass
class UrgencyExplanation:
    """Container for urgency-level SHAP explanation.

    Attributes:
        predicted_label: Predicted urgency label.
        confidence: Probability associated with the predicted label.
        feature_contributions: Mapping from feature name to SHAP value
            for the predicted class. Includes an aggregated
            ``text_embedding`` factor.
        factor_importance: Mapping from feature name to absolute
            contribution percentage (sums to 100).
        shap_values_vector: Raw SHAP values (1D array of length 776).
        expected_value: SHAP expected value for the predicted class.
        text: Original complaint text.
        structured_features: Mapping of the 8 structured features used.
        nl_explanation: Natural-language explanation of what drove
            the urgency decision.
    """

    predicted_label: str
    confidence: float
    feature_contributions: Dict[str, float]
    factor_importance: Dict[str, float]
    shap_values_vector: np.ndarray
    expected_value: float
    text: str
    structured_features: Dict[str, float]
    nl_explanation: str


class CategorySHAPExplainer:
    """SHAP explainer for the MuRIL category classifier.

    This uses a Hugging Face :func:`pipeline` combined with
    :class:`shap.Explainer` and a text masker to produce token-level
    attributions for each complaint.
    """

    def __init__(self, model_dir: Path) -> None:
        """Initialize the category explainer.

        Args:
            model_dir: Directory containing the fine-tuned MuRIL
                sequence classification model (as saved by
                ``train_category_model.py``).
        """
        self.model_dir = model_dir
        logger.info("Loading category model and tokenizer from %s", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir)
        )
        self.model.eval()

        # Pipeline for SHAP text explanations
        self.pipe = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=128,
            return_all_scores=True,
        )

        # SHAP text masker + explainer
        masker = shap.maskers.Text(self.pipe.tokenizer)
        self.explainer = shap.Explainer(self.pipe, masker)

    @torch.no_grad()
    def _predict_logits(self, text: str) -> np.ndarray:
        """Run the raw MuRIL classifier to obtain logits for a single text.

        Args:
            text: Complaint text.

        Returns:
            NumPy array of shape (num_classes,) with raw logits.
        """
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        outputs = self.model(**encoded)
        logits = outputs.logits[0].detach().cpu().numpy()
        return logits

    def explain(self, text: str, top_k: int = 5) -> CategoryExplanation:
        """Generate a SHAP-based explanation for category prediction.

        Args:
            text: Input complaint text.
            top_k: Number of top contributing tokens to highlight.

        Returns:
            :class:`CategoryExplanation` instance containing tokens,
            SHAP values, and a natural-language summary.
        """
        # Get prediction from raw logits for reliable label mapping
        logits = self._predict_logits(text)
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        pred_idx: int = int(np.argmax(probs))
        confidence: float = float(probs[pred_idx])
        predicted_label: str = CATEGORY_LABELS[pred_idx]

        # Compute SHAP values for the input text. For a text-classification
        # pipeline, SHAP can return values with different shapes depending on
        # version, typically (1, num_classes, num_tokens) or
        # (1, num_tokens, num_classes). We reduce this to a 1D array of
        # per-token attributions for the predicted class.
        shap_values = self.explainer([text])
        raw_vals = np.array(shap_values.values)
        tokens: List[str] = list(shap_values.data[0])

        if raw_vals.ndim == 2:
            # Shape (1, num_tokens)
            token_shap = raw_vals[0]
        elif raw_vals.ndim == 3:
            v0 = raw_vals[0]
            if v0.shape[0] == len(tokens):
                # Shape (num_tokens, num_classes)
                token_shap = v0[:, pred_idx]
            elif v0.shape[1] == len(tokens):
                # Shape (num_classes, num_tokens)
                token_shap = v0[pred_idx, :]
            else:  # pragma: no cover - defensive
                raise RuntimeError(
                    f"Unexpected SHAP values shape {v0.shape} for text length {len(tokens)}"
                )
        else:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Unexpected SHAP values ndim={raw_vals.ndim}; expected 2 or 3."
            )

        token_shap = np.array(token_shap, dtype=float)

        # Align tokens and SHAP values
        token_importances: List[Dict[str, Any]] = [
            {"token": tok, "value": float(val)}
            for tok, val in zip(tokens, token_shap)
        ]

        # Get top-k keywords by absolute attribution (excluding pure
        # whitespace tokens)
        indexed = [
            (i, t["token"], t["value"])
            for i, t in enumerate(token_importances)
            if t["token"].strip() != ""
        ]
        indexed.sort(key=lambda x: abs(x[2]), reverse=True)
        top_tokens = [tok for _, tok, _ in indexed[:top_k]]

        nl_explanation = self._build_natural_language_explanation(
            predicted_label, confidence, top_tokens
        )

        return CategoryExplanation(
            predicted_label=predicted_label,
            confidence=confidence,
            token_importances=token_importances,
            top_keywords=top_tokens,
            shap_values=token_shap,
            tokens=tokens,
            text=text,
            nl_explanation=nl_explanation,
        )

    @staticmethod
    def _build_natural_language_explanation(
        label: str, confidence: float, keywords: Sequence[str]
    ) -> str:
        """Create a human-friendly explanation sentence.

        Args:
            label: Predicted category label.
            confidence: Prediction confidence in [0, 1].
            keywords: Top contributing tokens.

        Returns:
            Short explanation sentence.
        """
        conf_pct = round(confidence * 100, 1)
        if not keywords:
            return (
                f"The complaint was classified as '{label}' with"
                f" {conf_pct}% confidence. The model did not find any"
                " highly distinctive keywords."
            )

        joined = ", ".join(f"'{k}'" for k in keywords)
        return (
            f"The complaint was classified as '{label}' with {conf_pct}%"
            f" confidence, mainly due to the presence of keywords like"
            f" {joined}."
        )


class UrgencySHAPExplainer:
    """SHAP explainer for the XGBoost urgency classifier.

    This explainer:
      * Recomputes the 768-dim MuRIL embedding for the input text.
      * Concatenates the 8 structured features to form a 776-dim
        vector, scaled using the stored :class:`StandardScaler`.
      * Uses :class:`shap.TreeExplainer` to obtain per-feature
        contributions for the predicted urgency class.
      * Aggregates the 768 text dimensions into a single
        ``text_embedding`` factor.
    """

    def __init__(
        self,
        xgb_model_path: Path,
        scaler_path: Path,
        muril_model_name: str = MURIL_MODEL_NAME,
    ) -> None:
        """Initialize the urgency explainer.

        Args:
            xgb_model_path: Path to ``xgboost_urgency_predictor.pkl``.
            scaler_path: Path to ``feature_scaler.pkl``.
            muril_model_name: Name of the MuRIL model to use for
                embeddings.
        """
        logger.info("Loading XGBoost urgency model from %s", xgb_model_path)
        
        # Load from pickle which contains model, scaler, label_encoder
        try:
            with open(xgb_model_path, "rb") as f:
                data = joblib.load(f)
            
            if isinstance(data, dict) and "model" in data:
                self.model = data["model"]
                self.label_encoder = data.get("label_encoder")
            else:
                self.model = data
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

        # Always load the fitted scaler from the separate file
        logger.info("Loading feature scaler from %s", scaler_path)
        try:
            self.scaler = joblib.load(scaler_path)
        except Exception as exc:
            logger.error("Failed to load scaler: %s", exc)
            raise

        logger.info("Initializing MuRIL feature extractor (%s)", muril_model_name)
        self.muril_extractor = MurilFeatureExtractor(model_name=muril_model_name)

        self.tree_explainer = shap.TreeExplainer(self.model)

    def _build_feature_vector(
        self,
        text: str,
        structured_features: Dict[str, float],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Construct a single 776-dim feature vector for a complaint.

        Args:
            text: Complaint text.
            structured_features: Mapping for the 8 structured
                features; must contain all names from
                :data:`STRUCTURED_FEATURE_COLUMNS`.

        Returns:
            Tuple of (feature_vector, features_dict) where
            ``feature_vector`` has shape (1, 776).
        """
        missing = [
            name
            for name in STRUCTURED_FEATURE_COLUMNS
            if name not in structured_features
        ]
        if missing:
            raise ValueError(
                "Missing structured features: " + ", ".join(missing)
            )

        # Text embedding (1, 768)
        emb = self.muril_extractor.encode([text])  # (1, 768)

        # Structured feature vector (1, 8)
        sf_values = np.array(
            [[float(structured_features[name]) for name in STRUCTURED_FEATURE_COLUMNS]],
            dtype=np.float32,
        )

        features = np.concatenate([emb, sf_values], axis=1)  # (1, 776)
        return features, structured_features

    def explain(
        self,
        text: str,
        structured_features: Dict[str, float],
    ) -> UrgencyExplanation:
        """Generate a SHAP-based explanation for urgency prediction.

        Args:
            text: Complaint text.
            structured_features: Dictionary with the 8 structured
                features used by the urgency model.

        Returns:
            :class:`UrgencyExplanation` with factor-wise SHAP
            contributions and a human-readable summary.
        """
        X_raw, sf_dict = self._build_feature_vector(text, structured_features)

        # Scale using the same StandardScaler as training
        X_scaled = self.scaler.transform(X_raw)

        # Predict probabilities using XGBoost Booster (trained with multi:softprob)
        dmatrix = xgb.DMatrix(X_scaled)
        probs = np.asarray(self.model.predict(dmatrix)).reshape(-1, len(URGENCY_LEVELS))[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        predicted_label = URGENCY_LEVELS[pred_idx]

        # SHAP values using TreeExplainer. For multi-class XGBoost,
        # ``shap_values_all`` can be either:
        #   * a list of arrays (one per class), shape (n_samples, n_features),
        #   * or a single array of shape (n_samples, n_classes, n_features).
        shap_values_all = self.tree_explainer.shap_values(X_scaled)
        expected_raw = self.tree_explainer.expected_value

        if isinstance(shap_values_all, list):
            # List-of-arrays API
            shap_vec = np.array(shap_values_all[pred_idx][0])
            if isinstance(expected_raw, (list, np.ndarray)):
                expected_value = float(expected_raw[pred_idx])
            else:
                expected_value = float(expected_raw)
        else:
            raw = np.array(shap_values_all)
            if raw.ndim == 3:
                # Either (n_samples, n_features, n_classes) or
                # (n_samples, n_classes, n_features). Infer which axis is
                # the feature axis from the known feature dimension.
                n_features = X_scaled.shape[1]
                if raw.shape[1] == n_features:
                    # Shape (n_samples, n_features, n_classes)
                    shap_vec = raw[0, :, pred_idx]
                elif raw.shape[2] == n_features:
                    # Shape (n_samples, n_classes, n_features)
                    shap_vec = raw[0, pred_idx, :]
                else:  # pragma: no cover - defensive
                    raise RuntimeError(
                        f"Unexpected SHAP values shape {raw.shape} for urgency model."
                    )
            elif raw.ndim == 2:
                # Shape (n_samples, n_features) for some configurations
                shap_vec = raw[0, :]
            else:  # pragma: no cover - defensive
                raise RuntimeError(
                    f"Unexpected SHAP values shape {raw.shape} for urgency model."
                )

            if isinstance(expected_raw, (list, np.ndarray)):
                expected_value = float(expected_raw[pred_idx])
            else:
                expected_value = float(expected_raw)

        if shap_vec.shape[0] != 776:
            raise RuntimeError(
                f"Expected 776 SHAP values, got {shap_vec.shape[0]} instead."
            )

        # Aggregate the 768 embedding dimensions into one factor
        text_contrib = float(shap_vec[:768].sum())
        structured_contribs: Dict[str, float] = {}
        for i, name in enumerate(STRUCTURED_FEATURE_COLUMNS):
            structured_contribs[name] = float(shap_vec[768 + i])

        # Combine into a feature->contribution mapping
        feature_contribs: Dict[str, float] = {"text_embedding": text_contrib}
        feature_contribs.update(structured_contribs)

        # Convert to absolute-importance percentages
        abs_vals = np.array([abs(v) for v in feature_contribs.values()])
        total_abs = abs_vals.sum() or 1.0
        factor_importance: Dict[str, float] = {}
        for key, val in feature_contribs.items():
            pct = float(abs(val) / total_abs * 100.0)
            factor_importance[key] = round(pct, 2)

        nl_explanation = self._build_natural_language_explanation(
            predicted_label, confidence, factor_importance
        )

        return UrgencyExplanation(
            predicted_label=predicted_label,
            confidence=confidence,
            feature_contributions=feature_contribs,
            factor_importance=factor_importance,
            shap_values_vector=shap_vec,
            expected_value=expected_value,
            text=text,
            structured_features={k: float(v) for k, v in sf_dict.items()},
            nl_explanation=nl_explanation,
        )

    @staticmethod
    def _build_natural_language_explanation(
        label: str,
        confidence: float,
        factors: Dict[str, float],
    ) -> str:
        """Create a high-level natural-language explanation for urgency.

        Args:
            label: Predicted urgency label.
            confidence: Prediction confidence.
            factors: Mapping of factor name to absolute contribution
                percentage.

        Returns:
            Short explanation string for citizens/officials.
        """
        conf_pct = round(confidence * 100, 1)
        # Sort factors by importance (descending)
        sorted_factors = sorted(
            factors.items(), key=lambda x: x[1], reverse=True
        )
        top_parts: List[str] = []
        for name, pct in sorted_factors[:4]:
            if pct <= 0.0:
                continue
            # Map internal feature names to human-readable phrases
            if name == "text_embedding":
                label_name = "the detailed description in the complaint text"
            elif name == "emergency_keyword_score":
                label_name = "the presence of emergency-related keywords"
            elif name == "severity_score":
                label_name = "the estimated severity score"
            elif name == "text_length":
                label_name = "the length of the complaint description"
            elif name == "affected_population":
                label_name = "the size of the affected population"
            elif name == "repeat_complaint_count":
                label_name = "how many times similar complaints were raised"
            elif name == "hour_of_day":
                label_name = "the time of day when the complaint was filed"
            elif name == "is_weekend":
                label_name = "whether the issue occurs on weekends"
            elif name == "is_monsoon_season":
                label_name = "whether it is the monsoon season"
            else:
                label_name = name.replace("_", " ")
            top_parts.append(f"{pct:.1f}% due to {label_name}")

        if not top_parts:
            return (
                f"The complaint was assigned '{label}' urgency with"
                f" {conf_pct}% confidence. The model did not find any"
                " clearly dominant risk factors."
            )

        joined = "; ".join(top_parts)
        return (
            f"The complaint was assigned '{label}' urgency with {conf_pct}%"
            f" confidence, mainly influenced by: {joined}."
        )


class ExplainabilityEngine:
    """High-level façade combining category and urgency explainers.

    This class is designed for convenient use within the
    ``ComplaintProcessor`` and Streamlit pages. It lazily loads models
    and SHAP explainers from the default project paths.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
    ) -> None:
        """Create an explainability engine.

        Args:
            project_root: Root of the project. If ``None``, this is
                inferred from the location of this file.
        """
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        models_dir = self.project_root / "models"

        self._category_explainer: Optional[CategorySHAPExplainer] = None
        self._urgency_explainer: Optional[UrgencySHAPExplainer] = None

        self._category_model_dir = models_dir / "muril_category_classifier"
        self._xgb_model_path = models_dir / "xgboost_urgency_predictor.pkl"
        self._scaler_path = models_dir / "feature_scaler.pkl"

    @property
    def category_explainer(self) -> CategorySHAPExplainer:
        """Lazily-initialized category SHAP explainer."""
        if self._category_explainer is None:
            self._category_explainer = CategorySHAPExplainer(
                self._category_model_dir
            )
        return self._category_explainer

    @property
    def urgency_explainer(self) -> UrgencySHAPExplainer:
        """Lazily-initialized urgency SHAP explainer."""
        if self._urgency_explainer is None:
            self._urgency_explainer = UrgencySHAPExplainer(
                xgb_model_path=self._xgb_model_path,
                scaler_path=self._scaler_path,
            )
        return self._urgency_explainer

    def explain_category(self, text: str, top_k: int = 5) -> CategoryExplanation:
        """Public helper to explain the category decision for a text."""
        return self.category_explainer.explain(text, top_k=top_k)

    def explain_urgency(
        self,
        text: str,
        structured_features: Dict[str, float],
    ) -> UrgencyExplanation:
        """Public helper to explain the urgency decision for a complaint."""
        return self.urgency_explainer.explain(text, structured_features)


__all__ = [
    "CategoryExplanation",
    "UrgencyExplanation",
    "CategorySHAPExplainer",
    "UrgencySHAPExplainer",
    "ExplainabilityEngine",
]
