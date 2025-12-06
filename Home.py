from __future__ import annotations

"""Main Streamlit entry point and Home page for the civic complaint system.

Run from the project root with::

    streamlit run Home.py

This single root page shows the landing dashboard (stats, how it works,
and quick actions). Detailed flows live in ``pages/`` as separate
Streamlit pages that you can navigate to from the sidebar or the quick
action buttons.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.ui import apply_global_styles, render_footer


def init_language() -> None:
    """Initialise the global language setting in ``st.session_state``."""
    if "language" not in st.session_state:
        st.session_state["language"] = "English"


def load_stats() -> tuple[int, int, float]:
    """Load simple statistics from the synthetic dataset.

    Returns ``(total_complaints, resolved_today, avg_resolution_days)``.
    Values are heuristic placeholders based on the CSV file.
    """

    root = Path(__file__).resolve().parent
    data_path = root / "data" / "civic_complaints.csv"
    if not data_path.exists():
        return 0, 0, 0.0

    df = pd.read_csv(data_path)
    total = len(df)
    resolved_today = int(total * 0.1)
    avg_resolution_days = 2.3
    return total, resolved_today, avg_resolution_days


# Simple multilingual label dictionary for the landing page
LABELS = {
    "English": {
        "app_title": "Explainable Multilingual Civic Complaint System",
        "overview": "Today's overview",
        "how_it_works": "How it works",
        "quick_actions": "Quick actions",
        "tagline": (
            "A simple, explainable portal for citizens to file complaints in "
            "English, Hindi, or Hinglish and track them through municipal offices."
        ),
        "step1_title": "1. Submit complaint",
        "step1_desc": "Citizens submit complaints in English, Hindi, or Hinglish with optional photos.",
        "step2_title": "2. AI classification",
        "step2_desc": "MuRIL and custom ML models analyze the complaint and predict category & urgency with explanations.",
        "step3_title": "3. Routing & tracking",
        "step3_desc": "The system routes complaints to the right department and lets citizens track status and queue position.",
        "file_button": "File Complaint",
        "track_button": "Track Complaint",
        "official_button": "Official Dashboard",
    },
    "Hindi": {
        "app_title": "व्याख्यात्मक बहुभाषी नागरिक शिकायत प्रणाली",
        "overview": "आज का सारांश",
        "how_it_works": "कैसे काम करता है",
        "quick_actions": "त्वरित कार्रवाइयाँ",
        "tagline": (
            "एक सरल, व्याख्यात्मक पोर्टल जहाँ नागरिक अंग्रेज़ी, हिंदी या हिंग्लिश में "
            "शिकायत दर्ज कर सकते हैं और उन्हें नगर निगम कार्यालयों तक ट्रैक कर सकते हैं।"
        ),
        "step1_title": "1. शिकायत दर्ज करें",
        "step1_desc": "नागरिक अंग्रेज़ी, हिंदी या हिंग्लिश में शिकायत और चाहें तो फोटो के साथ दर्ज करते हैं।",
        "step2_title": "2. एआई वर्गीकरण",
        "step2_desc": "MuRIL और अन्य मॉडल शिकायत की श्रेणी और तात्कालिकता का अनुमान लगाकर व्याख्या दिखाते हैं।",
        "step3_title": "3. रूटिंग और ट्रैकिंग",
        "step3_desc": "प्रणाली शिकायत को सही विभाग तक भेजती है और नागरिक को स्थिति व कतार स्थान दिखाती है।",
        "file_button": "शिकायत दर्ज करें",
        "track_button": "शिकायत ट्रैक करें",
        "official_button": "अधिकारी डैशबोर्ड",
    },
    "Hinglish": {
        "app_title": "Explainable Multilingual Civic Complaint System",
        "overview": "Aaj ka overview",
        "how_it_works": "Kaise kaam karta hai",
        "quick_actions": "Quick actions",
        "tagline": (
            "Ek simple portal jahan citizens English, Hindi ya Hinglish mein complaint "
            "file karke usse track kar sakte hain."
        ),
        "step1_title": "1. Complaint file karein",
        "step1_desc": "Citizens English, Hindi ya Hinglish mein complaint aur photos ke saath bhejte hain.",
        "step2_title": "2. AI classification",
        "step2_desc": "MuRIL aur ML models complaint ki category aur urgency predict karke explanation dete hain.",
        "step3_title": "3. Routing & tracking",
        "step3_desc": "System complaint ko sahi department tak bhejkar status aur queue position dikhata hai.",
        "file_button": "Complaint file karein",
        "track_button": "Complaint track karein",
        "official_button": "Official dashboard",
    },
}


# Shared styling and language state
apply_global_styles()
init_language()

st.sidebar.title("Settings")
lang = st.sidebar.selectbox(
    "Interface language",
    options=["English", "Hindi", "Hinglish"],
    index=["English", "Hindi", "Hinglish"].index(st.session_state["language"]),
)
st.session_state["language"] = lang

current_lang = st.session_state.get("language", "English")
labels = LABELS.get(current_lang, LABELS["English"])

# ---- Hero header ----
st.markdown(
    f"<h1 style='margin-bottom:0.25rem'>{labels['app_title']}</h1>",
    unsafe_allow_html=True,
)
st.write(labels["tagline"])

# Optional in-page language selector mirroring sidebar
col_lang, _ = st.columns([1, 3])
with col_lang:
    lang_choice = st.selectbox(
        "Language / भाषा / Language",
        options=["English", "Hindi", "Hinglish"],
        index=["English", "Hindi", "Hinglish"].index(current_lang),
    )
    st.session_state["language"] = lang_choice
    labels = LABELS.get(lang_choice, LABELS["English"])

st.markdown("---")

# ---- Quick stats dashboard ----
st.subheader(labels["overview"])

total_complaints, resolved_today, avg_resolution_days = load_stats()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total complaints", value=total_complaints)
with c2:
    st.metric("Resolved today", value=resolved_today)
with c3:
    st.metric("Avg. resolution time (days)", value=f"{avg_resolution_days:.1f}")

st.markdown("---")

# ---- How it works ----
st.subheader(labels["how_it_works"])

steps = [
    (labels["step1_title"], labels["step1_desc"]),
    (labels["step2_title"], labels["step2_desc"]),
    (labels["step3_title"], labels["step3_desc"]),
]

cols = st.columns(3)
for col, (title, desc) in zip(cols, steps):
    with col:
        st.markdown("<div class='civic-card'>", unsafe_allow_html=True)
        st.markdown(f"**{title}**")
        st.write(desc)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ---- Quick actions ----
st.subheader(labels["quick_actions"])

qa1, qa2, qa3 = st.columns(3)
with qa1:
    st.markdown("<div class='civic-card'>", unsafe_allow_html=True)
    if st.button(labels["file_button"], use_container_width=True):
        st.switch_page("pages/2_File_Complaint.py")
    st.markdown("</div>", unsafe_allow_html=True)
with qa2:
    st.markdown("<div class='civic-card'>", unsafe_allow_html=True)
    if st.button(labels["track_button"], use_container_width=True):
        st.switch_page("pages/4_Track_Complaint.py")
    st.markdown("</div>", unsafe_allow_html=True)
with qa3:
    st.markdown("<div class='civic-card'>", unsafe_allow_html=True)
    if st.button(labels["official_button"], use_container_width=True):
        st.switch_page("pages/5_Official_Dashboard.py")
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Footer ----
render_footer()
