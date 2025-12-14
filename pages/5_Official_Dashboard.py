from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
import time

from utils.database import (
    get_connection, init_db, list_open_complaints_for_department,
    update_complaint_assignment, insert_status_update,
    get_complaint, get_department_id_for_category, execute_query
)
from utils.ui import apply_global_styles, init_sidebar_language_selector, render_footer, check_official_access
from utils.auth import change_password

# Initialize language selector in sidebar
init_sidebar_language_selector()
# Check official access
check_official_access()

apply_global_styles()
current_lang = st.session_state.get("language", "English")

# Labels
OL = {
    "English": {
        "title": "Official Dashboard",
        "welcome": "Welcome, Official",
        "tabs": ["Overview", "Department Queue", "My Complaints", "Settings"],
        "overview": {
            "pending_dept": "Pending in Dept",
            "my_tasks": "Assigned to Me",
            "resolved_today": "Resolved Today (Me)",
            "avg_time": "Avg. Resolution Time"
        },
        "queue": {
            "header": "Department Queue (Unassigned)",
            "assign_me": "Assign to Me",
            "no_items": "No unassigned complaints in department."
        },
        "my_work": {
            "header": "My Active Cases",
            "status_update": "Update Status",
            "new_status": "New Status",
            "remarks": "Remarks",
            "update_btn": "Submit Update",
            "no_items": "No active cases assigned to you."
        },
        "settings": {
            "header": "My Settings",
            "change_pwd": "Change Password"
        }
    }
}
L = OL.get(current_lang, OL["English"])

st.title(L["title"])

user = st.session_state.get("user")
if not user:
    st.error("Session expired.")
    st.stop()

dept_id = user.get("department_id")
off_id = user.get("id")

st.markdown(f"**{L['welcome']} {user['name']}** (Dept ID: {dept_id})")

# Navigation
tab_over, tab_queue, tab_my, tab_set = st.tabs(L["tabs"])

# ---- OVERVIEW TAB ----
with tab_over:
    st.header("Dashboard")
    
    with get_connection() as conn:
         pending_dept_count = conn.execute("SELECT COUNT(*) FROM complaints WHERE department_id = ? AND status NOT IN ('Resolved', 'Closed')", (dept_id,)).fetchone()[0]
         assigned_me_count = conn.execute("SELECT COUNT(*) FROM complaints WHERE assigned_to = ? AND status NOT IN ('Resolved', 'Closed')", (off_id,)).fetchone()[0]
         # Heuristic for resolved today
         resolved_me_today = conn.execute("SELECT COUNT(*) FROM complaints WHERE assigned_to = ? AND status='Resolved' AND date(updated_at) = date('now')", (off_id,)).fetchone()[0]

    c1, c2, c3 = st.columns(3)
    c1.metric(L["overview"]["pending_dept"], pending_dept_count)
    c2.metric(L["overview"]["my_tasks"], assigned_me_count)
    c3.metric(L["overview"]["resolved_today"], resolved_me_today)


# ---- QUEUE TAB (Unassigned in Dept) ----
with tab_queue:
    st.subheader(L["queue"]["header"])
    
    # Get all open complaints for dept
    # We need to filter for those that are UNASSIGNED (assigned_to is NULL or 0)
    # The list_open_complaints_for_department gets everything. Let's do a direct query or filter.
    
    # Using direct query for precision
    with get_connection() as conn:
        q_df = pd.read_sql_query("""
            SELECT * FROM complaints 
            WHERE department_id = ? 
            AND (assigned_to IS NULL OR assigned_to = 0)
            AND status NOT IN ('Resolved', 'Closed')
            ORDER BY CASE urgency
                WHEN 'Critical' THEN 4
                WHEN 'High' THEN 3
                WHEN 'Medium' THEN 2
                WHEN 'Low' THEN 1
                ELSE 0
            END DESC, created_at ASC
        """, conn, params=(dept_id,))
        
    if q_df.empty:
        st.info(L["queue"]["no_items"])
    else:
        st.dataframe(q_df[["complaint_id", "category", "urgency", "created_at", "location"]], use_container_width=True)
        
        # Removed assign to me functionality per user request


# ---- MY COMPLAINTS TAB (Assigned to Me) ----
with tab_my:
    st.subheader(L["my_work"]["header"])
    
    with get_connection() as conn:
        my_df = pd.read_sql_query("""
            SELECT * FROM complaints 
            WHERE assigned_to = ? 
            AND status NOT IN ('Resolved', 'Closed')
            ORDER BY CASE urgency
                WHEN 'Critical' THEN 4
                WHEN 'High' THEN 3
                WHEN 'Medium' THEN 2
                WHEN 'Low' THEN 1
                ELSE 0
            END DESC
        """, conn, params=(off_id,))
        
    if my_df.empty:
        st.info(L["my_work"]["no_items"])
    else:
        # Detailed View
        sel_my_id = st.selectbox("Select Case to Manage", options=my_df["complaint_id"].tolist(), key="my_sel_id")
        
        if sel_my_id:
            row = my_df[my_df["complaint_id"] == sel_my_id].iloc[0]
            
            with st.expander("Case Details", expanded=True):
                st.write(f"**Category:** {row['category']}")
                st.write(f"**Description:** {row['text']}")
                st.write(f"**Location:** {row['location']}")
                st.write(f"**Urgency:** {row['urgency']}")
            
            st.markdown("### Update Status")
            with st.form("status_update_form"):
                new_status = st.selectbox(L["my_work"]["new_status"], 
                    options=["Service Person Allotted", "Service on Process", "Completed"])
                remarks = st.text_area(L["my_work"]["remarks"])
                submitted = st.form_submit_button(L["my_work"]["update_btn"])
                
                if submitted:
                    # Update status in DB
                    try:
                        # insert_status_update handles both the status_updates table AND updating the complaint status
                        insert_status_update(sel_my_id, new_status, remarks, official_id=off_id)
                        
                        st.success(f"Status updated to {new_status}")
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Error updating status: {e}")

# ---- SETTINGS TAB ----
with tab_set:
    st.subheader(L["settings"]["change_pwd"])
    with st.form("change_pwd_off"):
        old_p = st.text_input("Current Password", type="password")
        new_p = st.text_input("New Password", type="password")
        conf_p = st.text_input("Confirm New Password", type="password")
        sub_p = st.form_submit_button("Update Password")
        
        if sub_p:
            if new_p != conf_p:
                st.error("New passwords do not match")
            else:
                s, m = change_password(off_id, old_p, new_p)
                if s: st.success(m)
                else: st.error(m)

# Footer
render_footer()
