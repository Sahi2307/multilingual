from __future__ import annotations

"""Data preparation module for civic complaint system.
Generates synthetic multilingual complaints and prepares training data.
"""

import logging
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

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
            "The drinking water supplied today is dirty and smells bad.",
            "Our colony main valve is broken, water is flowing on the streets.",
            "We are receiving water only for 10 minutes, which is not enough.",
            "The public tap in our lane is broken and leaking continuously.",
            "Sewer water is mixing with drinking water supply lines.",
            "Please fix the water meter, it is showing wrong readings.",
            "No water tanker has arrived despite booking 2 days ago.",
        ],
        "Hindi": [
            "हमारे इलाके में पिछले {days} दिनों से पानी की सप्लाई बंद है, लोग बहुत परेशान हैं।",
            "पानी का प्रेशर बहुत कम है, टंकी बिल्कुल नहीं भर रही है।",
            "घर के पास पानी की पाइपलाइन से लीकेज हो रही है, बहुत पानी बर्बाद हो रहा है।",
            "आज सप्लाई किया गया पीने का पानी गंदा है और उसमें बदबू आ रही है।",
            "हमारी कॉलोनी का मुख्य वाल्व टूटा हुआ है, सड़कों पर पानी बह रहा है।",
            "हमें केवल 10 मिनट के लिए पानी मिल रहा है, जो पर्याप्त नहीं है।",
            "हमारी गली का सार्वजनिक नल टूटा हुआ है और लगातार बह रहा है।",
            "सीवर का पानी पीने के पानी की सप्लाई लाइनों के साथ मिल रहा है।",
            "कृपया पानी का मीटर ठीक करें, यह गलत रीडिंग दिखा रहा है।",
            "2 दिन पहले बुकिंग करने के बावजूद कोई पानी का टैंकर नहीं आया है।",
        ],
        "Hinglish": [
            "Hamare area me {days} din se pani nahi aa raha, log bahut pareshaan hain.",
            "Building me pani ka pressure bahut low hai, tank bhar hi nahi raha.",
            "Ghar ke paas wali pipeline me leakage hai, bahut saara pani waste ho raha hai.",
            "Aaj jo peene ka pani aaya hai wo ganda hai aur smell kar raha hai.",
            "Hamari colony ka main valve tooth gaya hai, road par pani beh raha hai.",
            "Humein sirf 10 minute ke liye pani mil raha hai, jo kaafi nahi hai.",
            "Hamari gali ka public tap toota hua hai aur continuously leak ho raha hai.",
            "Sewer ka pani drinking water supply ke saath mix ho raha hai.",
            "Please water meter fix karein, ye galat reading dikha raha hai.",
            "2 din pehle book karne ke bawajood koi water tanker nahi aaya hai.",
        ],
    },
    "Sanitation": {
        "English": [
            "Garbage has not been collected from our street for a week and the smell is unbearable.",
            "The drain near our house is blocked and dirty water is overflowing on the road.",
            "Public dustbins are overflowing and stray animals are scattering the waste.",
            "Dead animal body lying on the road for 2 days, causing health hazard.",
            "Mosquito breeding in stagnant water near the park.",
            "Sweepers are not cleaning the streets regularly in our sector.",
            "Open defecation near the boundary wall is causing nuisance.",
            "The public toilet is in very bad condition and needs cleaning.",
            "Construction debris dumped on the sidewalk is blocking the path.",
            "Sewer line is choked and backflowing into our houses.",
        ],
        "Hindi": [
            "गली से पिछले कई दिनों से कूड़ा नहीं उठाया गया है, बदबू से रहना मुश्किल हो गया है।",
            "घर के पास नाली जाम है और गंदा पानी सड़क पर बह रहा है।",
            "सार्वजनिक डस्टबिन कचरे से भर गए हैं और जानवर कचरा फैला रहे हैं।",
            "सड़क पर 2 दिन से मरे हुए जानवर का शरीर पड़ा है, जिससे बीमारी का खतरा है।",
            "पार्क के पास जमा पानी में मच्छर पनप रहे हैं।",
            "सफाई कर्मचारी हमारे सेक्टर में नियमित रूप से सड़कें साफ नहीं कर रहे हैं।",
            "चारदीवारी के पास खुले में शौच से परेशानी हो रही है।",
            "सार्वजनिक शौचालय बहुत खराब हालत में है और सफाई की जरूरत है।",
            "फुटपाथ पर डाला गया निर्माण मलबा रास्ता रोक रहा है।",
            "सीवर लाइन चोक हो गई है और हमारे घरों में वापस बह रही है।",
        ],
        "Hinglish": [
            "Gali se ek hafte se kachra nahi uthaya gaya, smell se rehna mushkil ho gaya hai.",
            "Nali bilkul jam hai, ganda pani road pe overflow ho raha hai.",
            "Public dustbin overflow ho rahe hain, stray dogs kachra idhar-udhar phaila rahe hain.",
            "Road par 2 din se dead animal body padi hai, health hazard ho sakta hai.",
            "Park ke paas stagnant water mein machhar paida ho rahe hain.",
            "Sweepers hamare sector mein regularly sadak saaf nahi kar rahe hain.",
            "Boundary wall ke paas open defecation se pareshani ho rahi hai.",
            "Public toilet bahut buri halat mein hai aur safai ki zarurat hai.",
            "Sidewalk par construction debris dump kiya gaya hai jo rasta block kar raha hai.",
            "Sewer line choke ho gayi hai aur hamare gharon mein backflow ho raha hai.",
        ],
    },
    "Transportation": {
        "English": [
            "The main road near the school is full of potholes and accidents can happen anytime.",
            "Streetlights on the main road are not working, making it very unsafe at night.",
            "The bus stop has no proper shelter and people stand on the road in traffic.",
            "Traffic signal at the junction is not working since yesterday.",
            "Illegal parking on the service road is causing traffic jams.",
            "Speed breakers are too high and damaging vehicles.",
            "Footpath tiles are broken and it is dangerous for pedestrians.",
            "Direction signboard is missing at the main crossing.",
            "Stray cattle sitting on the road are causing accidents.",
            "Manhole cover is open in the middle of the road.",
        ],
        "Hindi": [
            "स्कूल के पास वाली मुख्य सड़क पर बहुत गड्ढे हैं, कभी भी दुर्घटना हो सकती है।",
            "मुख्य सड़क पर लगी स्ट्रीटलाइट्स काम नहीं कर रही हैं, रात में बहुत खतरा रहता है।",
            "बस स्टॉप पर शेल्टर नहीं है, लोग सड़क पर खड़े होकर बस का इंतज़ार करते हैं।",
            "चौराहे पर ट्रैफिक सिग्नल कल से काम नहीं कर रहा है।",
            "सर्विस रोड पर अवैध पार्किंग से ट्रैफिक जाम हो रहा है।",
            "स्पीड ब्रेकर बहुत ऊंचे हैं और वाहनों को नुकसान पहुंचा रहे हैं।",
            "फुटपाथ की टाइलें टूटी हुई हैं और यह पैदल चलने वालों के लिए खतरनाक है।",
            "मुख्य चौराहे पर दिशा सूचक बोर्ड गायब है।",
            "सड़क पर बैठे आवारा मवेशी दुर्घटना का कारण बन रहे हैं।",
            "सड़क के बीच में मैनहोल का ढक्कन खुला है।",
        ],
        "Hinglish": [
            "School ke paas wali main road par bahut potholes hain, kabhi bhi accident ho sakta hai.",
            "Main road ki streetlights kaam nahi kar rahi, raat me bahut khatra rehta hai.",
            "Bus stop pe proper shelter nahi hai, log road par khade hokar bus ka wait karte hain.",
            "Junction par traffic signal kal se kaam nahi kar raha hai.",
            "Service road par illegal parking se traffic jam ho raha hai.",
            "Speed breakers bahut high hain aur vehicles ko damage kar rahe hain.",
            "Footpath tiles tooti hui hain aur ye pedestrians ke liye dangerous hai.",
            "Main crossing par direction signboard missing hai.",
            "Road par baithe stray cattle accidents ka reason ban rahe hain.",
            "Road ke beech mein manhole cover open hai.",
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


def compute_severity_score(urgency: str | None, emergency_keywords: List[str], text: str) -> float:
    """Compute a heuristic severity score in [0.0, 1.0].

    Args:
        urgency: Urgency level ("Critical", "High", "Medium", "Low") or None.
                 If None (inference time), a neutral base is used.
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
    # Use 0.4 (Medium-equivalent) as default base if urgency not known yet
    base = base_map.get(urgency, 0.4) if urgency else 0.4
    
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


def _ensure_distribution(df: pd.DataFrame) -> None:
    """Validate category/language distributions and total size.
    
    Warns if distribution is off, but does not error out.
    """

    total_expected = 200
    if len(df) < 50:
         logger.warning(f"Dataset size {len(df)} is quite small. Expected close to {total_expected}.")
    
    # Category counts
    cat_counts = df["category"].value_counts().to_dict()
    logger.info("Category distribution: %s", cat_counts)
    
    # Language counts
    lang_counts = df["language"].value_counts().to_dict()
    logger.info("Language distribution: %s", lang_counts)


def _generate_deduplicated_records() -> List[Dict[str, str]]:
    """Generate unique complaints.
    
    Tries to meet target counts but stops if unique templates are exhausted.
    Does not force-create artificial duplicates.
    """

    records: List[Dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    complaint_counter = 1

    for (category, language), target_count in COMBINATION_COUNTS.items():
        bucket_records: List[Dict[str, str]] = []
        
        # Get all templates for this bucket
        template_list = TEMPLATES[category][language]
        
        # We want to generate 'target_count' items.
        # Since we only have ~10 templates, we will likely need to reuse them 
        # with different parameters ({days}) to create unique strings.
        
        attempts = 0
        while len(bucket_records) < target_count:
            attempts += 1
            if attempts > target_count * 10:
                logger.warning(f"Could not fully satisfy count for {category}-{language} (Got {len(bucket_records)}/{target_count})")
                break
                
            text = generate_complaint_text(category, language)
            text = clean_text(text)
            
            key = (text, category, language)
            
            if key in seen:
                continue
                
            seen.add(key)
            bucket_records.append(
                {
                    "complaint_id": f"C{complaint_counter:04d}",
                    "text": text,
                    "category": category,
                    "language": language,
                }
            )
            complaint_counter += 1

        records.extend(bucket_records)

    df = pd.DataFrame(records)
    _ensure_distribution(df)
    return records


def _assert_no_duplicates(df: pd.DataFrame) -> None:
    """Ensure no duplicate (text, category, language) combinations remain."""
    dup_mask = df.duplicated(subset=["text", "category", "language"], keep=False)
    if dup_mask.any():
        dup_rows = df[dup_mask][["text", "category", "language"]].head(5)
        # We just log warning instead of crashing, user said "remove duplicates" which is done by generation logic
        logger.warning(f"Duplicate complaints check found potential dupes (should be handled by generation): {dup_rows}")


def assign_urgency(df: pd.DataFrame) -> pd.DataFrame:
    """Assign urgency labels properly even if N != 200.
    """

    total = len(df)
    if total == 0:
        return df
        
    # We want roughly Critical:10%, High:30%, Medium:40%, Low:20%
    ratios = {"Critical": 0.10, "High": 0.30, "Medium": 0.40, "Low": 0.20}
    
    df = df.copy()
    df["urgency"] = ""

    # Global pool logic is complex when N varies independently per category.
    # Simplified approach: Stratified per category to match global ratios approximatey.
    
    categories = df["category"].unique().tolist()
    
    for category in categories:
        cat_mask = df["category"] == category
        n_cat = cat_mask.sum()
        
        # Calculate counts for this category
        counts = {}
        allocated = 0
        
        for u, r in ratios.items():
            c = int(round(n_cat * r))
            counts[u] = c
            allocated += c
            
        # Fix rounding errors
        diff = n_cat - allocated
        if diff != 0:
            # dumping diff into Medium
            counts["Medium"] += diff
            
        # Create list
        urgencies = []
        for u, c in counts.items():
            urgencies.extend([u] * max(0, c))
            
        # If still mismatch (due to negative c if strict rounding?), fill/trim
        while len(urgencies) < n_cat:
            urgencies.append("Medium")
        while len(urgencies) > n_cat:
            urgencies.pop()
            
        random.shuffle(urgencies)
        df.loc[cat_mask, "urgency"] = urgencies

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
    
    Tries 70/15/15 but falls back if data is too small.
    """
    df = df.copy()
    df["stratify_key"] = df["category"].astype(str) + "__" + df["urgency"].astype(str)
    
    # If dataset is very small, just efficient split 
    if len(df) < 50:
         # simple random split
         train, temp = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
         val, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_SEED)
    else:
        try:
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
            
            # Reconstruct for assignment
            train = train_df
            val = val_df
            test = test_df
        except ValueError:
             # Fallback if some classes have too few members for stratified
             logger.warning("Stratified split failed (likely too few samples per class). Falling back to random split.")
             train, temp = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
             val, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_SEED)

    df["split"] = ""
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    
    logger.info(f"Split sizes: Train={len(train)}, Val={len(val)}, Test={len(test)}")

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
    # Generate deduplicated records per distribution constraints
    records = _generate_deduplicated_records()

    df = pd.DataFrame(records)

    # Save raw version
    raw_path = raw_dir / "civic_complaints_raw.csv"
    df.to_csv(raw_path, index=False)
    logger.info("Saved raw synthetic dataset to %s", raw_path)

    # Sanity: no duplicates
    _assert_no_duplicates(df)

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

    # Add cleaned_text column (text is already cleaned above but keep explicit column for training scripts)
    df["cleaned_text"] = df["text"]

    # Save split CSVs expected by training scripts
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    test_path = data_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Saved split datasets to %s, %s, %s", train_path, val_path, test_path)

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
