from __future__ import annotations

from pathlib import Path
import statistics
import time

import pandas as pd
import streamlit as st

from src.complaint_processor import ComplaintProcessor
from utils.database import get_connection, init_db
from utils.ui import apply_global_styles, render_footer


apply_global_styles()
current_lang = st.session_state.get("language", "English")
ADMIN_LABELS = {
    "English": {
        "title": "Admin Panel",
        "system_overview": "System overview",
        "trends": "Complaint trends (CSV synthetic)",
        "category_analysis": "Category analysis",
        "csv_tab": "Dataset (CSV)",
        "db_tab": "Live (DB)",
        "no_trends": "Synthetic CSV dataset not available; cannot compute trends.",
        "no_csv_cat": "No category data in CSV dataset.",
        "no_db_cat": "No complaints in DB yet.",
        "ai_perf": "AI model performance (from training logs)",
        "cat_acc": "Category accuracy: ~100% on synthetic test set",
        "urg_acc": "Urgency accuracy: ~100% on synthetic test set",
        "caption": "In a production system, this panel would also show uptime, backlog per department, user management, and time-series model evaluation metrics from real traffic.",
        "benchmark": "Performance benchmark (manual)",
        "benchmark_desc": "Run a small end-to-end benchmark for complaint processing, including model inference and SHAP explanations. This uses the same pipeline as the File Complaint page.",
        "run_benchmark": "Run benchmark",
        "running": "Running benchmark... this may take some time on CPU because SHAP explanations are computationally expensive.",
        "complaint": "Complaint",
        "avg_time": "Average processing time over {n} complaints: {t:.3f} seconds",
    },
    "Hindi": {
        "title": "प्रशासन पैनल",
        "system_overview": "सिस्टम सारांश",
        "trends": "शिकायत रुझान (CSV सिंथेटिक)",
        "category_analysis": "श्रेणी विश्लेषण",
        "csv_tab": "डेटासेट (CSV)",
        "db_tab": "लाइव (DB)",
        "no_trends": "सिंथेटिक CSV डेटासेट उपलब्ध नहीं; रुझान नहीं दिखाए जा सकते।",
        "no_csv_cat": "CSV डेटासेट में श्रेणी डेटा उपलब्ध नहीं है।",
        "no_db_cat": "DB में अभी कोई शिकायत नहीं है।",
        "ai_perf": "एआई मॉडल प्रदर्शन (प्रशिक्षण लॉग से)",
        "cat_acc": "श्रेणी शुद्धता: ~100% (सिंथेटिक टेस्ट सेट)",
        "urg_acc": "तात्कालिकता शुद्धता: ~100% (सिंथेटिक टेस्ट सेट)",
        "caption": "प्रोडक्शन सिस्टम में यहां अपटाइम, विभाग‑वार लंबित शिकायतें, यूज़र प्रबंधन और वास्तविक डेटा पर आधारित मॉडल मैट्रिक्स दिखाई जाएंगी।",
        "benchmark": "प्रदर्शन परीक्षण (मैन्युअल)",
        "benchmark_desc": "शिकायत प्रोसेसिंग का छोटा एंड‑टू‑एंड बेंचमार्क चलाएं, जिसमें मॉडल और SHAP व्याख्याएँ शामिल हैं।",
        "run_benchmark": "बेंचमार्क चलाएं",
        "running": "बेंचमार्क चल रहा है... CPU पर SHAP व्याख्याएँ महंगी हो सकती हैं, कृपया प्रतीक्षा करें।",
        "complaint": "शिकायत",
        "avg_time": "{n} शिकायतों पर औसत प्रोसेसिंग समय: {t:.3f} सेकंड",
    },
    "Hinglish": {
        "title": "Admin Panel",
        "system_overview": "System overview",
        "trends": "Complaint trends (CSV synthetic)",
        "category_analysis": "Category analysis",
        "csv_tab": "Dataset (CSV)",
        "db_tab": "Live (DB)",
        "no_trends": "Synthetic CSV dataset available nahi; trends nahi dikhaye ja sakte.",
        "no_csv_cat": "CSV dataset mein category data nahi hai.",
        "no_db_cat": "DB mein abhi koi complaints nahi hain.",
        "ai_perf": "AI model performance (training logs se)",
        "cat_acc": "Category accuracy: ~100% (synthetic test set)",
        "urg_acc": "Urgency accuracy: ~100% (synthetic test set)",
        "caption": "Production system mein yahan uptime, backlog per department, user management aur real traffic metrics aayenge.",
        "benchmark": "Performance benchmark (manual)",
        "benchmark_desc": "End-to-end complaint processing ka chhota benchmark chalayein.",
        "run_benchmark": "Benchmark chalayein",
        "running": "Benchmark chal raha hai... SHAP explanations mehengi ho sakti hain.",
        "complaint": "Complaint",
        "avg_time": "Average processing time over {n} complaints: {t:.3f} seconds",
    },
}

AL = ADMIN_LABELS.get(current_lang, ADMIN_LABELS["English"])

st.title(AL["title"])

root = Path(__file__).resolve().parents[1]
data_path = root / "data" / "civic_complaints.csv"

# Ensure DB is initialised
init_db(root)

_df = pd.read_csv(data_path) if data_path.exists() else pd.DataFrame()

# DB-level aggregates
with get_connection(root) as conn:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as c FROM complaints")
    total_db = int(cur.fetchone()["c"])
    cur.execute("SELECT category, COUNT(*) as c FROM complaints GROUP BY category")
    rows = cur.fetchall()
    db_cat_counts = {r["category"]: int(r["c"]) for r in rows}

st.subheader(AL["system_overview"])

csv_total = len(_df)
st.write(f"Total complaints in CSV dataset: {csv_total}")
st.write(f"Total complaints in live DB: {total_db}")

st.markdown("---")

st.subheader(AL["trends"])

if _df.empty:
    st.info(AL["no_trends"])
else:
    # Approximate monthly volume by index for demo
    _df["month"] = (_df.index % 12) + 1
    trends = _df.groupby("month")["complaint_id"].count().rename("count")
    st.bar_chart(trends)

st.subheader(AL["category_analysis"])

csv_cat_counts = _df["category"].value_counts() if not _df.empty else pd.Series(dtype=int)

csv_tab, db_tab = st.tabs([AL["csv_tab"], AL["db_tab"]])

with csv_tab:
    if csv_cat_counts.empty:
        st.info(AL["no_csv_cat"])
    else:
        st.bar_chart(csv_cat_counts)

with db_tab:
    if not db_cat_counts:
        st.info(AL["no_db_cat"])
    else:
        db_series = pd.Series(db_cat_counts).sort_values(ascending=False)
        st.bar_chart(db_series)

st.subheader(AL["ai_perf"])

st.write(AL["cat_acc"])
st.write(AL["urg_acc"])

st.caption(AL["caption"])

st.markdown("---")

st.subheader(AL["benchmark"])
st.write(AL["benchmark_desc"])

SAMPLE_COMPLAINTS = [
    {
        "name": "Bench User",
        "email": "bench_user_1@example.org",
        "phone": "9999990001",
        "location": "Ward 10, Sample Colony",
        "complaint_text": (
            "Hamare area ki road bahut kharab ho gayi hai, "
            "potholes ki wajah se accident ka risk hai"
        ),
        "language": "Hinglish",
        "affected_label": "Neighborhood / locality",
    },
    {
        "name": "Bench User",
        "email": "bench_user_2@example.org",
        "phone": "9999990002",
        "location": "Ward 5, River Side",
        "complaint_text": "There has been no water supply in our area for the past 3 days, residents are suffering.",
        "language": "English",
        "affected_label": "One street / lane",
    },
    {
        "name": "Bench User",
        "email": "bench_user_3@example.org",
        "phone": "9999990003",
        "location": "Ward 18, Market Area",
        "complaint_text": "गली से पिछले कई दिनों से कूड़ा नहीं उठाया गया है, बदबू से रहना मुश्किल हो गया है।",
        "language": "Hindi",
        "affected_label": "Large area / crowd",
    },
]

@st.cache_resource
def get_benchmark_processor() -> ComplaintProcessor:
    """Create and cache a ComplaintProcessor for benchmarking."""
    return ComplaintProcessor(project_root=root)


if st.button(AL["run_benchmark"]):
    st.info(AL["running"])
    processor = get_benchmark_processor()
    durations: list[float] = []

    progress = st.progress(0.0)
    status_placeholder = st.empty()

    for idx, payload in enumerate(SAMPLE_COMPLAINTS, start=1):
        start_t = time.perf_counter()
        _ = processor.process_complaint(
            name=payload["name"],
            email=payload["email"],
            phone=payload["phone"],
            location=payload["location"],
            complaint_text=payload["complaint_text"],
            language=payload["language"],
            affected_label=payload["affected_label"],
        )
        end_t = time.perf_counter()
        duration = end_t - start_t
        durations.append(duration)

        status_placeholder.write(f"{AL['complaint']} {idx}: {duration:.3f} seconds")
        progress.progress(idx / len(SAMPLE_COMPLAINTS))

    if durations:
        avg = statistics.mean(durations)
        st.success(
            AL["avg_time"].format(n=len(durations), t=avg)
        )

# Shared footer
render_footer()
