from __future__ import annotations

"""Data preparation for the Explainable Multilingual Civic Complaint System.

This module generates a synthetic dataset of 200 civic complaints across
three categories (Sanitation, Water Supply, Transportation), three
languages (English, Hindi, Hinglish), and four urgency levels
(Critical, High, Medium, Low).

It also performs:
  * Text cleaning with basic PII removal while preserving Devanagari.
  * Simple rule-based language detection (English/Hindi/Hinglish).
  * Structured feature construction for 8-dimensional meta-features.
  * Stratified train/validation/test splitting (70/15/15).

The final CSV is saved to ``data/civic_complaints.csv`` and includes:
  * complaint_id, text, category, urgency, language
  * all 8 structured features
  * split (train/val/test)

Run as a script:
    python -m src.data_preparation
"""

import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RANDOM_SEED: int = 42

CATEGORIES: List[str] = ["Sanitation", "Water Supply", "Transportation"]
URGENCY_LEVELS: List[str] = ["Critical", "High", "Medium", "Low"]
LANGUAGES: List[str] = ["English", "Hindi", "Hinglish"]

CATEGORY_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}
URGENCY_TO_ID: Dict[str, int] = {u: i for i, u in enumerate(URGENCY_LEVELS)}

# (category, language) -> count ensuring totals:
#   * 66 Sanitation, 67 Water Supply, 67 Transportation
#   * 80 Hindi, 60 English, 60 Hinglish
#   * Total 200
COMBINATION_COUNTS: Dict[Tuple[str, str], int] = {
    ("Sanitation", "Hindi"): 27,
    ("Sanitation", "English"): 20,
    ("Sanitation", "Hinglish"): 19,
    ("Water Supply", "Hindi"): 26,
    ("Water Supply", "English"): 20,
    ("Water Supply", "Hinglish"): 21,
    ("Transportation", "Hindi"): 27,
    ("Transportation", "English"): 20,
    ("Transportation", "Hinglish"): 20,
}

# Emergency-related keywords spanning English, transliterated Hindi, and Hindi script
EMERGENCY_KEYWORDS: List[str] = [
    "urgent",
    "immediately",
    "emergency",
    "accident",
    "risk",
    "danger",
    "fire",
    "flood",
    "severe",
    "critical",
    "no water",
    "gas leak",
    "जानलेवा",
    "खतरा",
    "तुरंत",
    "turant",
    "bahut zyada",
]

DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")
EMAIL_REGEX = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE_REGEX = re.compile(r"\b\d{3,}\d+\b")
NAME_REGEX = re.compile(r"\b(Mr\.?|Mrs\.?|Ms\.?|Shri|Smt)\s+[A-Z][a-zA-Z]*\b")

HINGLISH_HINT_WORDS = {
    "pani",
    "sadak",
    "kachra",
    "gali",
    "nala",
    "bahut",
    "kharab",
    "jam",
    "risk",
    "school ke paas",
}

ENGLISH_HINT_WORDS = {
    "water",
    "supply",
    "garbage",
    "drainage",
    "road",
    "potholes",
    "bus",
    "traffic",
    "accident",
    "smell",
    "leak",
}

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -----------------------------------------------------------------------------
# Text templates
# -----------------------------------------------------------------------------

TEMPLATES = {
    "Water Supply": {
        "English": [
            "There has been no water supply in our area for the past {days} days, residents are suffering.",
            "Water pressure is very low in our building and tanks are not filling properly.",
            "Water pipeline leakage near our house is wasting a lot of clean water.",
        ],
        "Hindi": [
            "हमारे इलाके में पिछले {days} दिनों से पानी की सप्लाई बंद है, लोग बहुत परेशान हैं।",
            "पानी का प्रेशर बहुत कम है, टंकी बिल्कुल नहीं भर रही है।",
            "घर के पास पानी की पाइपलाइन से लीकेज हो रही है, बहुत पानी बर्बाद हो रहा है।",
        ],
        "Hinglish": [
            "Hamare area me {days} din se pani nahi aa raha, log bahut pareshaan hain.",
            "Building me pani ka pressure bahut low hai, tank bhar hi nahi raha.",
            "Ghar ke paas wali pipeline me leakage hai, bahut saara pani waste ho raha hai.",
        ],
    },
    "Sanitation": {
        "English": [
            "Garbage has not been collected from our street for a week and the smell is unbearable.",
            "The drain near our house is blocked and dirty water is overflowing on the road.",
            "Public dustbins are overflowing and stray animals are scattering the waste.",
        ],
        "Hindi": [
            "गली से पिछले कई दिनों से कूड़ा नहीं उठाया गया है, बदबू से रहना मुश्किल हो गया है।",
            "घर के पास नाली जाम है और गंदा पानी सड़क पर बह रहा है।",
            "सार्वजनिक डस्टबिन कचरे से भर गए हैं और जानवर कचरा फैला रहे हैं।",
        ],
        "Hinglish": [
            "Gali se ek hafte se kachra nahi uthaya gaya, smell se rehna mushkil ho gaya hai.",
            "Nali bilkul jam hai, ganda pani road pe overflow ho raha hai.",
            "Public dustbin overflow ho rahe hain, stray dogs kachra idhar-udhar phaila rahe hain.",
        ],
    },
    "Transportation": {
        "English": [
            "The main road near the school is full of potholes and accidents can happen anytime.",
            "Streetlights on the main road are not working, making it very unsafe at night.",
            "The bus stop has no proper shelter and people stand on the road in traffic.",
        ],
        "Hindi": [
            "स्कूल के पास वाली मुख्य सड़क पर बहुत गड्ढे हैं, कभी भी दुर्घटना हो सकती है।",
            "मुख्य सड़क पर लगी स्ट्रीटलाइट्स काम नहीं कर रही हैं, रात में बहुत खतरा रहता है।",
            "बस स्टॉप पर शेल्टर नहीं है, लोग सड़क पर खड़े होकर बस का इंतज़ार करते हैं।",
        ],
        "Hinglish": [
            "School ke paas wali main road par bahut potholes hain, kabhi bhi accident ho sakta hai.",
            "Main road ki streetlights kaam nahi kar rahi, raat me bahut khatra rehta hai.",
            "Bus stop pe proper shelter nahi है, log road par khade hokar bus ka wait karte हैं.",
        ],
    },
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean complaint text while preserving Devanagari script.

    This removes simple PII patterns (emails, phone numbers, honorific+name)
    and normalizes whitespace.

    Args:
        text: Raw complaint text.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    text = EMAIL_REGEX.sub("[EMAIL]", text)
    text = PHONE_REGEX.sub("[PHONE]", text)
    text = NAME_REGEX.sub("[NAME]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(text: str) -> str:
    """Detect whether text is English, Hindi, or Hinglish.

    Heuristics:
      * Any Devanagari characters -> Hindi.
      * Otherwise, if both Hinglish and English hint words appear -> Hinglish.
      * Otherwise -> English.

    Args:
        text: Cleaned complaint text.

    Returns:
        Detected language label in {"English", "Hindi", "Hinglish"}.
    """
    if not isinstance(text, str) or not text:
        return "English"

    if DEVANAGARI_REGEX.search(text):
        return "Hindi"

    lowered = text.lower()
    hinglish_hits = sum(1 for w in HINGLISH_HINT_WORDS if w in lowered)
    english_hits = sum(1 for w in ENGLISH_HINT_WORDS if w in lowered)

    if hinglish_hits > 0 and english_hits > 0:
        return "Hinglish"

    return "English"


def extract_emergency_keywords(text: str) -> List[str]:
    """Extract emergency keywords present in the complaint text.

    Args:
        text: Clean or raw complaint text.

    Returns:
        List of keyword substrings that were found.
    """
    lowered = text.lower()
    found: List[str] = [kw for kw in EMERGENCY_KEYWORDS if kw.lower() in lowered]
    return found


def compute_severity_score(urgency: str, emergency_keywords: List[str], text: str) -> float:
    """Compute a heuristic severity score in [0.0, 1.0].

    Args:
        urgency: Urgency level ("Critical", "High", "Medium", "Low").
        emergency_keywords: List of emergency keywords detected in the text.
        text: Complaint text.

    Returns:
        Float severity score between 0.0 and 1.0.
    """
    base_map = {
        "Critical": 0.8,
        "High": 0.6,
        "Medium": 0.4,
        "Low": 0.2,
    }
    base = base_map.get(urgency, 0.4)
    emergency_boost = min(0.1 * len(emergency_keywords), 0.2)
    length_boost = min(len(text.split()) / 100.0, 0.1)
    score = float(min(base + emergency_boost + length_boost, 1.0))
    return round(score, 3)


def sample_affected_population(urgency: str) -> int:
    """Sample affected_population category given urgency.

    0: few individuals, 1: one street, 2: neighborhood, 3: large area.

    Args:
        urgency: Urgency label.

    Returns:
        Encoded affected_population in {0, 1, 2, 3}.
    """
    if urgency == "Critical":
        weights = [0.1, 0.2, 0.3, 0.4]
    elif urgency == "High":
        weights = [0.2, 0.3, 0.3, 0.2]
    elif urgency == "Medium":
        weights = [0.4, 0.3, 0.2, 0.1]
    else:  # Low
        weights = [0.6, 0.25, 0.1, 0.05]
    return int(random.choices([0, 1, 2, 3], weights=weights, k=1)[0])


def sample_repeat_complaint_count(urgency: str) -> int:
    """Sample historical repeat complaint count given urgency.

    Args:
        urgency: Urgency label.

    Returns:
        Non-negative integer repeat complaint count.
    """
    if urgency in ("Critical", "High"):
        return int(np.random.poisson(lam=2))
    if urgency == "Medium":
        return int(np.random.poisson(lam=1))
    return int(np.random.poisson(lam=0.3))


def generate_complaint_text(category: str, language: str) -> str:
    """Generate a synthetic complaint text for a given category and language.

    Args:
        category: Complaint category name.
        language: One of "English", "Hindi", "Hinglish".

    Returns:
        Generated complaint text.
    """
    try:
        options = TEMPLATES[category][language]
    except KeyError as exc:  # pragma: no cover - config error
        raise ValueError(f"No template for ({category}, {language})") from exc

    template = random.choice(options)
    days = random.randint(1, 5)
    text = template.format(days=days)
    return text


def assign_urgency(df: pd.DataFrame) -> pd.DataFrame:
    """Assign urgency labels with the specified overall distribution.

    Distribution:
      * Critical: 20
      * High: 60
      * Medium: 80
      * Low: 40

    Args:
        df: DataFrame with synthetic complaints (no urgency column yet).

    Returns:
        DataFrame with an added ``urgency`` column.
    """
    total = len(df)
    expected = 20 + 60 + 80 + 40
    if total != expected:
        raise ValueError(f"Dataset size {total} does not match required {expected}.")

    urgency_labels: List[str] = (
        ["Critical"] * 20
        + ["High"] * 60
        + ["Medium"] * 80
        + ["Low"] * 40
    )
    random.shuffle(urgency_labels)
    df = df.copy()
    df["urgency"] = urgency_labels
    return df


def add_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 8 structured features for each complaint.

    Adds the following columns:
      * emergency_keyword_score
      * severity_score
      * text_length
      * affected_population
      * repeat_complaint_count
      * hour_of_day
      * is_weekend
      * is_monsoon_season

    Args:
        df: Input DataFrame with columns
            ``['complaint_id', 'text', 'category', 'urgency', 'language']``.

    Returns:
        DataFrame with new feature columns appended.
    """
    feature_records: List[Dict[str, float]] = []

    for _, row in df.iterrows():
        text: str = row["text"]
        urgency: str = row["urgency"]

        emergency_keywords = extract_emergency_keywords(text)
        emergency_keyword_score = 1 if emergency_keywords else 0
        severity_score = compute_severity_score(urgency, emergency_keywords, text)
        text_length = len(text.split())
        affected_population = sample_affected_population(urgency)
        repeat_count = sample_repeat_complaint_count(urgency)

        hour_of_day = random.randint(0, 23)
        is_weekend = int(random.choices([0, 1], weights=[5, 2], k=1)[0])
        is_monsoon = int(random.choices([0, 1], weights=[3, 2], k=1)[0])

        feature_records.append(
            {
                "emergency_keyword_score": emergency_keyword_score,
                "severity_score": severity_score,
                "text_length": text_length,
                "affected_population": affected_population,
                "repeat_complaint_count": repeat_count,
                "hour_of_day": hour_of_day,
                "is_weekend": is_weekend,
                "is_monsoon_season": is_monsoon,
            }
        )

    features_df = pd.DataFrame(feature_records)
    df = df.reset_index(drop=True)
    df = pd.concat([df, features_df], axis=1)
    return df


def stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """Perform stratified train/val/test split on category + urgency.

    Ensures:
      * Train: 70% (140 examples)
      * Validation: 15% (30 examples)
      * Test: 15% (30 examples)

    Args:
        df: DataFrame with ``category`` and ``urgency`` columns.

    Returns:
        DataFrame with an added ``split`` column in
        {"train", "val", "test"}.
    """
    df = df.copy()
    df["stratify_key"] = df["category"].astype(str) + "__" + df["urgency"].astype(str)

    splitter1 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=RANDOM_SEED
    )
    train_idx, temp_idx = next(splitter1.split(df, df["stratify_key"]))

    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    splitter2 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=RANDOM_SEED
    )
    val_idx, test_idx = next(splitter2.split(temp_df, temp_df["stratify_key"]))
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    df["split"] = ""
    df.loc[train_df.index, "split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.loc[test_df.index, "split"] = "test"

    # Sanity checks
    assert (df["split"] == "train").sum() == 140, "Train size must be 140."
    assert (df["split"] == "val").sum() == 30, "Validation size must be 30."
    assert (df["split"] == "test").sum() == 30, "Test size must be 30."

    df = df.drop(columns=["stratify_key"])
    return df


def build_and_save_dataset() -> pd.DataFrame:
    """Create synthetic complaints, compute features, and save CSV.

    Returns:
        Final DataFrame with all features and a ``split`` column.

    Side effects:
        * Writes raw CSV to ``data/raw/civic_complaints_raw.csv``.
        * Writes processed CSV to ``data/civic_complaints.csv``.
    """
    root_dir = Path(__file__).resolve().parents[1]
    data_dir = root_dir / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic complaints...")
    records = []
    complaint_counter = 1

    for (category, language), count in COMBINATION_COUNTS.items():
        for _ in range(count):
            text = generate_complaint_text(category, language)
            text = clean_text(text)
            records.append(
                {
                    "complaint_id": f"C{complaint_counter:04d}",
                    "text": text,
                    "category": category,
                    "language": language,
                }
            )
            complaint_counter += 1

    df = pd.DataFrame(records)

    # Save raw version
    raw_path = raw_dir / "civic_complaints_raw.csv"
    df.to_csv(raw_path, index=False)
    logger.info("Saved raw synthetic dataset to %s", raw_path)

    # Assign urgency labels
    df = assign_urgency(df)

    # Language detection sanity check
    logger.info("Running language detection sanity check...")
    df["detected_language"] = df["text"].apply(detect_language)
    lang_accuracy = (df["language"] == df["detected_language"]).mean()
    logger.info("Language detection self-check accuracy: %.2f%%", lang_accuracy * 100)

    # Structured features
    df = add_structured_features(df)

    # Stratified split
    df = stratified_split(df)

    # Final processed dataset
    final_path = data_dir / "civic_complaints.csv"
    df.to_csv(final_path, index=False)
    logger.info("Saved processed dataset with splits to %s", final_path)

    return df


def main() -> None:
    """CLI entry-point for data preparation."""
    try:
        build_and_save_dataset()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to build dataset: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover - CLI path
    main()
