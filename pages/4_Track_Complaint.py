from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.database import get_complaint, list_status_updates_for_complaint, init_db
from utils.ui import apply_global_styles, init_sidebar_language_selector, render_footer, check_citizen_access


# Initialize language selector in sidebar
init_sidebar_language_selector()

# Check citizen access
check_citizen_access()


TRACK_LABELS = {
    "English": {
        "title": "Track Complaint",
        "enter_id": "Enter Complaint ID",
        "enter_help": "Example: C0001",
        "btn_search": "Search",
        "error_no_id": "Please enter a Complaint ID.",
        "info_live": "Complaint information (live DB)",
        "info_synth": "Complaint information (synthetic dataset)",
        "full_text": "Full complaint text",
        "progress_timeline": "Progress timeline",
        "no_updates": "No status updates recorded yet.",
        "queue_info_live": "Queue position and ETA were estimated when the complaint was registered. Live recalculation and display will be added when the full queue management UI is implemented.",
        "dept_updates": "Department updates",
        "dept_info_live": "Department remarks and escalation controls can be wired to the status updates and notifications tables in the next iteration.",
        "no_data": "No complaint data available (DB or CSV).",
        "no_found": "No complaint found with that ID in DB or CSV dataset.",
        "urgency_label": "Urgency (label)",
        "location_na": "Location: (not stored in synthetic data)",
        "progress": "Progress",
    },
    "Hindi": {
        "title": "शिकायत ट्रैक करें",
        "enter_id": "शिकायत आईडी दर्ज करें",
        "enter_help": "उदाहरण: C0001",
        "btn_search": "खोजें",
        "error_no_id": "कृपया शिकायत आईडी दर्ज करें।",
        "info_live": "शिकायत जानकारी (लाइव डेटाबेस)",
        "info_synth": "शिकायत जानकारी (सिंथेटिक डेटासेट)",
        "full_text": "पूरी शिकायत",
        "progress_timeline": "प्रगति समयरेखा",
        "no_updates": "अभी तक कोई स्थिति अपडेट दर्ज नहीं है।",
        "queue_info_live": "कतार स्थिति और अनुमानित समय शिकायत दर्ज करते समय निकाले गए थे। आगे के संस्करण में इन्हें लाइव अपडेट किया जाएगा।",
        "dept_updates": "विभागीय अपडेट",
        "dept_info_live": "अगले चरण में विभागीय टिप्पणियाँ और एस्केलेशन नियंत्रण यहां दिखाए जा सकते हैं।",
        "no_data": "कोई शिकायत डेटा उपलब्ध नहीं है (DB या CSV)।",
        "no_found": "DB या CSV डेटासेट में इस आईडी की कोई शिकायत नहीं मिली।",
        "urgency_label": "तात्कालिकता (लेबल)",
        "location_na": "स्थान: (सिंथेटिक डेटा में उपलब्ध नहीं)",
        "progress": "प्रगति",
    },
    "Hinglish": {
        "title": "Complaint track karein",
        "enter_id": "Complaint ID daalein",
        "enter_help": "Example: C0001",
        "btn_search": "Search",
        "error_no_id": "Please enter a Complaint ID.",
        "info_live": "Complaint information (live DB)",
        "info_synth": "Complaint information (synthetic dataset)",
        "full_text": "Full complaint text",
        "progress_timeline": "Progress timeline",
        "no_updates": "Abhi tak koi status updates nahi hain.",
        "queue_info_live": "Queue position aur ETA complaint register karte waqt estimate kiye gaye the.",
        "dept_updates": "Department updates",
        "dept_info_live": "Agle iteration mein department remarks aur escalation controls yahan dikhenge.",
        "no_data": "No complaint data available (DB or CSV).",
        "no_found": "DB ya CSV dataset mein is ID ki complaint nahi mili.",
        "urgency_label": "Urgency (label)",
        "location_na": "Location: (not stored in synthetic data)",
        "progress": "Progress",
    },
}

current_lang = st.session_state.get("language", "English")
T = TRACK_LABELS.get(current_lang, TRACK_LABELS["English"])

st.title(T["title"])
apply_global_styles()


root = Path(__file__).resolve().parents[1]
data_path = root / "data" / "civic_complaints.csv"

# Ensure DB exists
init_db(root)

complaint_id = st.text_input(T["enter_id"], help=T["enter_help"])

if st.button(T["btn_search"]):
    if not complaint_id.strip():
        st.error(T["error_no_id"])
        st.stop()

    # 1) Try live DB first
    db_rec = get_complaint(complaint_id.strip(), project_root=root)

    if db_rec is not None:
        st.markdown("---")
        st.subheader(T["info_live"])
        st.write(f"**Complaint ID:** {db_rec['id']}")
        st.write(f"**Category:** {db_rec['category']}")
        st.write(f"**Urgency:** {db_rec['urgency']}")
        st.write(f"**Language:** {db_rec['language']}")
        st.write(f"**Location:** {db_rec.get('location', '') or '(not specified)'}")
        st.write(f"**Status:** {db_rec.get('status', 'Unknown')}")
        st.write(f"**Created at:** {db_rec.get('created_at', '')}")
        st.write(f"**Last updated:** {db_rec.get('updated_at', '')}")

        st.markdown(f"### {T['full_text']}")
        st.write(db_rec.get("text", ""))

        st.markdown(f"### {T['progress_timeline']}")
        updates = list_status_updates_for_complaint(complaint_id.strip(), project_root=root)
        if not updates:
            st.info(T["no_updates"])
        else:
            for upd in updates:
                ts = upd.get("timestamp", "")
                status = upd.get("status", "")
                remarks = upd.get("remarks", "")
                st.markdown(f"- **{status}** at {ts}  ")
                if remarks:
                    st.caption(remarks)

        st.markdown("### Queue information")
        st.write(T["queue_info_live"])

        st.markdown(f"### {T['dept_updates']}")
        st.info(T["dept_info_live"])

    else:
        # 2) Fallback: synthetic CSV dataset
        if not data_path.exists():
            st.error(T["no_data"])
            st.stop()

        df = pd.read_csv(data_path)
        row = df[df["complaint_id"] == complaint_id.strip()].head(1)

        if row.empty:
            st.warning(T["no_found"])
        else:
            rec = row.iloc[0]
            st.markdown("---")
            st.subheader(T["info_synth"])
            st.write(f"**Complaint ID:** {rec['complaint_id']}")
            st.write(f"**Category:** {rec['category']}")
            st.write(f"**{T['urgency_label']}:** {rec['urgency']}")
            st.write(f"**Language:** {rec['language']}")
            st.write(f"**{T['location_na']}")

            st.markdown(f"### {T['full_text']}")
            st.write(rec["text"])

            st.markdown(f"### {T['progress']}")
            # Simple synthetic timeline
            steps = ["Registered", "Under Review", "In Progress", "Resolved"]
            progress_index = 2
            for i, step in enumerate(steps):
                if i <= progress_index:
                    st.markdown(f"- ✅ {step}")
                else:
                    st.markdown(f"- ☐ {step}")

            st.markdown("### Queue information")
            st.write(
                "Queue position and reasoning are not stored for the synthetic "
                "dataset. Live complaints use the queue management logic in the "
                "backend."
            )

            st.markdown("### Department updates")
            st.info(
                "Department updates and remarks will appear here once the "
                "database-backed workflow is fully wired to this view."
            )

# Shared footer
render_footer()
