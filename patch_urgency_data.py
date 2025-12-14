
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
root = Path(__file__).resolve().parent
sys.path.append(str(root))

from src.data_preparation import compute_severity_score, extract_emergency_keywords

def patch_csv():
    data_path = root / "data" / "civic_complaints.csv"
    if not data_path.exists():
        print("CSV not found.")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")

    # Recalculate severity_score with urgency=None
    new_scores = []
    for _, row in df.iterrows():
        text = str(row["text"])
        keywords = extract_emergency_keywords(text)
        # vital: pass None for urgency to simulate inference conditions
        score = compute_severity_score(None, keywords, text)
        new_scores.append(score)

    df["severity_score"] = new_scores
    print("Updated severity_score column (validation: mean=", df["severity_score"].mean(), ")")

    df.to_csv(data_path, index=False)
    print("Saved patched CSV.")

if __name__ == "__main__":
    patch_csv()
