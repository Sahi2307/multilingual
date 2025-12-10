from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.complaint_processor import ComplaintProcessor
from utils.database import get_connection, init_db, insert_status_update
from utils.ui import apply_global_styles, init_sidebar_language_selector, render_footer, check_official_access


# Initialize language selector in sidebar
init_sidebar_language_selector()

# Check official access
check_official_access()

apply_global_styles()


# Initialize language selector in sidebar
init_sidebar_language_selector()


@st.cache_resource
def get_complaint_processor() -> ComplaintProcessor:
    """Create and cache a :class:`ComplaintProcessor` for explanations."""
    root = Path(__file__).resolve().parents[1]
    return ComplaintProcessor(project_root=root)


apply_global_styles()
current_lang = st.session_state.get("language", "English")
DASH_LABELS = {
    "English": {
        "title": "Official Dashboard",
        "overview": "Overview",
        "complaint_queue": "Complaint queue",
        "live_tab": "Live (DB)",
        "synthetic_tab": "Synthetic (CSV)",
        "live_complaints": "Live complaints",
        "select_to_inspect": "Select a complaint to inspect",
        "details_and_ai": "Complaint details & AI explanation",
        "update_status": "Update status",
        "new_status": "New status",
        "remarks": "Remarks",
        "remarks_help": "Short note about what action will be taken.",
        "save_update": "Save update",
        "status_saved": "Status update recorded. Reload the page to see it reflected elsewhere.",
        "synthetic_missing": "Synthetic CSV dataset not available.",
        "caption": "Live data comes from the SQLite database, while the synthetic CSV dataset is used for benchmarking and demonstrations.",
    },
    "Hindi": {
        "title": "अधिकारी डैशबोर्ड",
        "overview": "सारांश",
        "complaint_queue": "शिकायत कतार",
        "live_tab": "लाइव (DB)",
        "synthetic_tab": "सिंथेटिक (CSV)",
        "live_complaints": "लाइव शिकायतें",
        "select_to_inspect": "जांच के लिए शिकायत चुनें",
        "details_and_ai": "शिकायत विवरण और एआई व्याख्या",
        "update_status": "स्थिति अपडेट करें",
        "new_status": "नई स्थिति",
        "remarks": "टिप्पणी",
        "remarks_help": "कौन‑सी कार्रवाई की जाएगी उसका छोटा सा विवरण।",
        "save_update": "अपडेट सहेजें",
        "status_saved": "स्थिति अपडेट सहेजा गया। अन्य जगह दिखाने के लिए पेज दोबारा लोड करें।",
        "synthetic_missing": "सिंथेटिक CSV डेटासेट उपलब्ध नहीं है।",
        "caption": "लाइव डेटा SQLite डेटाबेस से आता है, जबकि सिंथेटिक CSV डेटासेट डेमो और बेंचमार्क के लिए उपयोग होता है।",
    },
    "Hinglish": {
        "title": "Official Dashboard",
        "overview": "Overview",
        "complaint_queue": "Complaint queue",
        "live_tab": "Live (DB)",
        "synthetic_tab": "Synthetic (CSV)",
        "live_complaints": "Live complaints",
        "select_to_inspect": "Inspect karne ke liye complaint chunen",
        "details_and_ai": "Complaint details & AI explanation",
        "update_status": "Status update karein",
        "new_status": "Nayi status",
        "remarks": "Remarks",
        "remarks_help": "Kya action liya jayega uska short note.",
        "save_update": "Update save karein",
        "status_saved": "Status update save ho gaya. Changes dekhne ke liye page reload karein.",
        "synthetic_missing": "Synthetic CSV dataset available nahi hai.",
        "caption": "Live data SQLite database se aata hai, synthetic CSV demo aur benchmarking ke liye hai.",
    },
}

DL = DASH_LABELS.get(current_lang, DASH_LABELS["English"])

st.title(DL["title"])

root = Path(__file__).resolve().parents[1]
data_path = root / "data" / "civic_complaints.csv"

# Ensure DB is initialised
init_db(root)

# Load CSV (synthetic) if available
_df = pd.read_csv(data_path) if data_path.exists() else pd.DataFrame()

# Load DB complaints
with get_connection(root) as conn:
    cur = conn.cursor()
    cur.execute("SELECT urgency, COUNT(*) as c FROM complaints GROUP BY urgency")
    rows = cur.fetchall()
    db_urg_counts = {r["urgency"]: int(r["c"]) for r in rows}

st.subheader(DL["overview"])

levels = ["Critical", "High", "Medium", "Low"]

c1, c2, c3, c4 = st.columns(4)
for col, level in zip((c1, c2, c3, c4), levels):
    with col:
        db_val = db_urg_counts.get(level, 0)
        csv_val = (
            int((_df["urgency"].value_counts().reindex(levels, fill_value=0)[level]))
            if not _df.empty
            else 0
        )
        st.metric(f"{level} (DB)", db_val, help="Live complaints in the database")
        st.caption(f"Dataset: {csv_val} from synthetic CSV")

st.markdown("---")

st.subheader(DL["complaint_queue"])

live_tab, synthetic_tab = st.tabs([DL["live_tab"], DL["synthetic_tab"]])

with live_tab:
    with get_connection(root) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, category, urgency, language, status, created_at, text
            FROM complaints
            ORDER BY created_at DESC
            """
        )
        rows = cur.fetchall()
        if not rows:
            st.info("No live complaints found in the database yet.")
        else:
            live_df = pd.DataFrame(rows)
            live_df["title"] = live_df["text"].str.slice(0, 60) + "..."

            left_col, right_col = st.columns([2, 3])

            with left_col:
                st.markdown(f"#### {DL['live_complaints']}")
                st.dataframe(
                    live_df[["id", "title", "category", "urgency", "language", "status", "created_at"]],
                    use_container_width=True,
                )

                selected_id = st.selectbox(
                    DL["select_to_inspect"],
                    options=live_df["id"].astype(str).tolist(),
                )

            with right_col:
                st.markdown(f"#### {DL['details_and_ai']}")

                try:
                    row_sel = live_df[live_df["id"].astype(str) == selected_id].iloc[0]
                except IndexError:
                    st.info("Select a complaint from the list on the left to see details.")
                else:
                    st.markdown(f"**ID:** {row_sel['id']}")
                    st.write(f"**Category (stored):** {row_sel['category']}")
                    st.write(f"**Urgency (stored):** {row_sel['urgency']}")
                    st.write(f"**Status:** {row_sel['status']}")
                    st.write(f"**Created at:** {row_sel['created_at']}")

                    st.markdown("**Full text**")
                    st.write(row_sel["text"])

                    # AI explanations using the same models as the File Complaint page.
                    processor = get_complaint_processor()
                    text = row_sel["text"]

                    try:
                        cat_label, cat_conf, _, cat_exp = processor.predict_category(text)

                        # Use a neutral affected population label for structured
                        # features when recomputing urgency explanations.
                        structured_features = processor._build_structured_features(  # noqa: SLF001
                            text,
                            affected_label="Neighborhood / locality",
                        )
                        urg_label, urg_conf, _, urg_exp = processor.predict_urgency(
                            text,
                            structured_features,
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.warning(
                            "Could not compute AI explanations for this complaint. "
                            f"Error: {exc}"
                        )
                    else:
                        st.markdown("---")
                        st.markdown("##### Category explanation")
                        st.write(
                            f"Predicted category: **{cat_exp.predicted_label}** "
                            f"({cat_exp.confidence * 100:.1f}% confidence)"
                        )
                        st.caption(cat_exp.nl_explanation)

                        st.markdown("##### Urgency explanation")
                        st.write(
                            f"Predicted urgency: **{urg_exp.predicted_label}** "
                            f"({urg_exp.confidence * 100:.1f}% confidence)"
                        )

                        # Factor importance bar chart
                        factors_df = pd.DataFrame(
                            {
                                "factor": list(urg_exp.factor_importance.keys()),
                                "importance": list(urg_exp.factor_importance.values()),
                            }
                        ).sort_values("importance", ascending=False)
                        st.bar_chart(factors_df.set_index("factor"))
                        st.caption(urg_exp.nl_explanation)

                        # Waterfall plot for aggregated urgency factors
                        feature_names = list(urg_exp.feature_contributions.keys())
                        values = np.array(
                            [urg_exp.feature_contributions[name] for name in feature_names],
                            dtype=float,
                        )

                        data_values = []
                        for name in feature_names:
                            if name == "text_embedding":
                                data_values.append(0.0)
                            else:
                                data_values.append(
                                    float(urg_exp.structured_features.get(name, 0.0))
                                )
                        data_arr = np.array(data_values, dtype=float)

                        exp = shap.Explanation(
                            values=values,
                            base_values=urg_exp.expected_value,
                            data=data_arr,
                            feature_names=feature_names,
                        )

                        st.markdown("###### SHAP waterfall plot (urgency factors)")
                        fig, _ = plt.subplots(figsize=(6, 4))
                        shap.plots.waterfall(exp, max_display=9, show=False)
                        st.pyplot(fig, clear_figure=True)

                        st.markdown("###### SHAP force plot (urgency factors)")
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

                    st.markdown("---")
                    st.markdown(f"##### {DL['update_status']}")
                    new_status = st.selectbox(
                        DL["new_status"],
                        ["Registered", "In Progress", "Resolved"],
                        index=["Registered", "In Progress", "Resolved"].index(
                            row_sel["status"]
                        )
                        if row_sel["status"] in ["Registered", "In Progress", "Resolved"]
                        else 0,
                    )
                    remarks = st.text_area(
                        DL["remarks"],
                        "",
                        help=DL["remarks_help"],
                    )
                    if st.button(DL["save_update"], key=f"update_{selected_id}"):
                        insert_status_update(
                            complaint_id=str(row_sel["id"]),
                            status=new_status,
                            remarks=remarks,
                            official_id=None,
                            project_root=root,
                        )
                        st.success(DL["status_saved"])

with synthetic_tab:
    if _df.empty:
        st.info(DL["synthetic_missing"])
    else:
        queue_df = _df[["complaint_id", "category", "urgency", "language", "text"]].copy()
        queue_df["title"] = queue_df["text"].str.slice(0, 50) + "..."
        st.dataframe(
            queue_df[["complaint_id", "title", "category", "urgency", "language"]].head(50),
            use_container_width=True,
        )

st.caption(DL["caption"])

# Shared footer
render_footer()
