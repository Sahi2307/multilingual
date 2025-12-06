from __future__ import annotations
import datetime as dt
import random
import re
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.complaint_processor import ComplaintProcessor, ProcessedComplaintResult
from utils.ui import apply_global_styles, render_footer


@st.cache_resource
def get_complaint_processor() -> ComplaintProcessor:
    """Create and cache a single :class:`ComplaintProcessor` instance."""
    root = Path(__file__).resolve().parents[1]
    return ComplaintProcessor(project_root=root)


def map_affected_population(label: str) -> int:
    """Map human-readable affected population to encoded value."""
    mapping = {
        "Few individuals": 0,
        "One street / lane": 1,
        "Neighborhood / locality": 2,
        "Large area / crowd": 3,
    }
    return mapping.get(label, 1)


def highlight_keywords(text: str, keywords) -> str:
    """Return markdown-formatted text with keywords bolded.

    Args:
        text: Original complaint text.
        keywords: Iterable of keyword/token strings.

    Returns:
        Markdown string where matching keywords are wrapped in ``**``.
    """
    highlighted = text
    for kw in keywords:
        clean_kw = kw.strip()
        if not clean_kw:
            continue
        try:
            pattern = re.escape(clean_kw)
            highlighted = re.sub(
                pattern,
                f"**{clean_kw}**",
                highlighted,
                flags=re.IGNORECASE,
            )
        except re.error:
            continue
    return highlighted


LABELS = {
    "English": {
        "title": "File a Complaint",
        "details": "Complaint details",
        "registered": "Complaint registered",
        "ai_expl": "AI explanations",
        "why_cat": "Why categorized as this?",
        "why_pri": "Why this priority?",
        "lang_label": "Language / भाषा / Language",
        "name": "Name",
        "email": "Email",
        "phone": "Phone",
        "location": "Location",
        "category_hint": "Category (optional hint)",
        "complaint_language": "Complaint language",
        "affected_population": "Affected population",
        "complaint_description": "Complaint description",
        "complaint_help": "Describe the issue in as much detail as possible.",
        "upload_photos": "Upload photos (optional, up to 3)",
        "submit": "Submit complaint",
        "error_no_text": "Please enter a complaint description before submitting.",
        "spinner_main": "Running MuRIL analysis, ML models, and saving your complaint...",
        "spinner_shap": "Generating explainable AI insights with SHAP...",
    },
    "Hindi": {
        "title": "शिकायत दर्ज करें",
        "details": "शिकायत विवरण",
        "registered": "शिकायत दर्ज हो गई है",
        "ai_expl": "एआई व्याख्या",
        "why_cat": "इस श्रेणी में क्यों रखा गया?",
        "why_pri": "यह प्राथमिकता क्यों?",
        "lang_label": "भाषा / Language",
        "name": "नाम",
        "email": "ईमेल",
        "phone": "फ़ोन",
        "location": "स्थान",
        "category_hint": "श्रेणी (वैकल्पिक संकेत)",
        "complaint_language": "शिकायत की भाषा",
        "affected_population": "प्रभावित आबादी",
        "complaint_description": "शिकायत का विवरण",
        "complaint_help": "समस्या को जितना हो सके विस्तार से लिखें।",
        "upload_photos": "फोटो अपलोड करें (वैकल्पिक, अधिकतम 3)",
        "submit": "शिकायत दर्ज करें",
        "error_no_text": "कृपया शिकायत का विवरण लिखने के बाद ही सबमिट करें।",
        "spinner_main": "MuRIL विश्लेषण, मॉडल और शिकायत सहेजी जा रही है...",
        "spinner_shap": "SHAP के माध्यम से व्याख्यात्मक एआई जानकारी तैयार की जा रही है...",
    },
    "Hinglish": {
        "title": "Complaint file karein",
        "details": "Complaint details",
        "registered": "Complaint register ho gayi hai",
        "ai_expl": "AI explanations",
        "why_cat": "Is category mein kyun?",
        "why_pri": "Yeh priority kyun?",
        "lang_label": "Language / भाषा / Language",
        "name": "Naam",
        "email": "Email",
        "phone": "Phone",
        "location": "Location",
        "category_hint": "Category (optional hint)",
        "complaint_language": "Complaint ki language",
        "affected_population": "Affected population",
        "complaint_description": "Complaint description",
        "complaint_help": "Issue ko detail mein describe karein.",
        "upload_photos": "Photos upload karein (optional, max 3)",
        "submit": "Complaint submit karein",
        "error_no_text": "Complaint description likhne ke baad hi submit karein.",
        "spinner_main": "MuRIL analysis, ML models aur complaint save ho rahi hai...",
        "spinner_shap": "SHAP se explainable AI insights ban rahe hain...",
    },
}

# Keep language in sync with global setting
current_lang = st.session_state.get("language", "English")
labels = LABELS.get(current_lang, LABELS["English"])

# Global look & feel + light page-specific styling
apply_global_styles()
st.markdown(
    """
    <style>
    .file-card {
        padding: 1.0rem 1.2rem;
        border-radius: 0.75rem;
        border: 1px solid #e0e4f2;
        background-color: #ffffff;
        box-shadow: 0 0 8px rgba(15, 23, 42, 0.05);
    }
    .file-section-title {
        font-weight: 600;
        color: #1f4e79;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<h1 class='file-section-title'>{labels['title']}</h1>", unsafe_allow_html=True)
col_lang, _ = st.columns([1, 3])
with col_lang:
    lang = st.selectbox(
        labels["lang_label"],
        options=["English", "Hindi", "Hinglish"],
        index=["English", "Hindi", "Hinglish"].index(current_lang),
    )
    st.session_state["language"] = lang
    current_lang = lang
    labels = LABELS.get(current_lang, LABELS["English"])

st.markdown("---")

st.subheader(labels["details"])

with st.form("file_complaint_form", clear_on_submit=False):
    with st.container():
        st.markdown("<div class='file-card'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input(labels["name"])
            email = st.text_input(labels["email"])
            phone = st.text_input(labels["phone"])
            location = st.text_input(labels["location"])

        with c2:
            ui_category = st.selectbox(
                labels["category_hint"],
                ["Let AI decide", "Sanitation", "Water Supply", "Transportation"],
            )
            ui_language = st.selectbox(
                labels["complaint_language"],
                ["English", "Hindi", "Hinglish"],
                index=["English", "Hindi", "Hinglish"].index(lang),
            )
            affected_label = st.selectbox(
                labels["affected_population"],
                [
                    "Few individuals",
                    "One street / lane",
                    "Neighborhood / locality",
                    "Large area / crowd",
                ],
            )

        complaint_text = st.text_area(
            labels["complaint_description"],
            height=180,
            max_chars=2000,
            help=labels["complaint_help"],
        )

        uploaded_photos = st.file_uploader(
            labels["upload_photos"],
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_photos and len(uploaded_photos) > 3:
        st.warning("Only the first 3 uploaded photos will be considered.")
        uploaded_photos = uploaded_photos[:3]

    submitted = st.form_submit_button(labels["submit"])

if submitted:
    if not complaint_text.strip():
        st.error(labels["error_no_text"])
        st.stop()

    processor = get_complaint_processor()

    with st.spinner(labels["spinner_main"]):
        result: ProcessedComplaintResult = processor.process_complaint(
            name=name,
            email=email,
            phone=phone,
            location=location,
            complaint_text=complaint_text,
            language=ui_language,
            affected_label=affected_label,
            category_hint=None if ui_category == "Let AI decide" else ui_category,
        )

    # Short spinner to reflect SHAP explanation generation (already done in pipeline)
    with st.spinner(labels["spinner_shap"]):
        pass

    complaint_id = result.complaint_id
    cat_exp = result.category_explanation
    urg_exp = result.urgency_explanation
    department = result.department_name
    queue_position = result.queue_position
    eta_text = result.eta_text

    st.markdown("---")
    st.subheader(labels["registered"])

    c_info, c_meta = st.columns([2, 2])
    with c_info:
        st.write(f"**Complaint ID:** {complaint_id}")
        st.write(f"**Detected category:** {cat_exp.predicted_label} ({cat_exp.confidence * 100:.1f}% confidence)")
        st.write(f"**Assigned urgency:** {urg_exp.predicted_label} ({urg_exp.confidence * 100:.1f}% confidence)")
    with c_meta:
        st.write(f"**Department routing:** {department}")
        st.write(f"**Queue position:** #{queue_position}")
        st.write(f"**Estimated response time:** {eta_text}")

    st.markdown("---")
    st.subheader(labels["ai_expl"])

    with st.expander(labels["why_cat"], expanded=True):
        highlighted_text = highlight_keywords(complaint_text, cat_exp.top_keywords)
        st.markdown(highlighted_text)
        st.caption(cat_exp.nl_explanation)

        # Optional: simple bar chart of token importances for top keywords
        top_tokens = [kw.strip() for kw in cat_exp.top_keywords if kw.strip()]
        if top_tokens:
            token_to_val: Dict[str, float] = {}
            for item in cat_exp.token_importances:
                tok = item["token"].strip()
                if tok in top_tokens:
                    token_to_val[tok] = abs(float(item["value"]))
            if token_to_val:
                token_df = (
                    pd.DataFrame({"token": list(token_to_val.keys()), "importance": list(token_to_val.values())})
                    .sort_values("importance", ascending=False)
                )
                st.bar_chart(token_df.set_index("token"))

    with st.expander(labels["why_pri"], expanded=True):
        factors_df = pd.DataFrame(
            {
                "factor": list(urg_exp.factor_importance.keys()),
                "importance": list(urg_exp.factor_importance.values()),
            }
        ).sort_values("importance", ascending=False)

        st.bar_chart(factors_df.set_index("factor"))
        st.caption(urg_exp.nl_explanation)

        # SHAP waterfall plot for aggregated urgency factors
        st.markdown("#### SHAP waterfall plot (urgency factors)")
    feature_names = list(urg_exp.feature_contributions.keys())
    values = np.array([urg_exp.feature_contributions[name] for name in feature_names], dtype=float)

    # Use 0 for text_embedding "data" and the actual structured values for others
    data_values = []
    for name in feature_names:
        if name == "text_embedding":
            data_values.append(0.0)
        else:
            data_values.append(float(urg_exp.structured_features.get(name, 0.0)))
    data_arr = np.array(data_values, dtype=float)

    exp = shap.Explanation(
        values=values,
        base_values=urg_exp.expected_value,
        data=data_arr,
        feature_names=feature_names,
    )

    # Waterfall plot summarising how each factor pushes the urgency
    # prediction higher or lower relative to the expected value.
    fig, ax = plt.subplots(figsize=(6, 4))
    shap.plots.waterfall(exp, max_display=9, show=False)
    st.pyplot(fig, clear_figure=True)

    # Optional SHAP force plot (single prediction) to visualise positive vs
    # negative contributions in a compact horizontal bar layout. We use the
    # legacy ``force_plot`` API with ``matplotlib=True`` so it renders as a
    # static image inside Streamlit.
    st.markdown("#### SHAP force plot (urgency factors)")
    fig_force = plt.figure(figsize=(6, 1.5))
    shap.force_plot(
        exp.base_values,
        exp.values,
        exp.data,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    st.pyplot(fig_force, clear_figure=True)

    st.info(
        "These explanations are generated using SHAP (SHapley Additive Explanations) "
        "to show how different words and factors influenced the AI decisions."
    )

# Shared footer
render_footer()
