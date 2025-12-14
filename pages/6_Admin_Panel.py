from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
import time

from utils.database import (
    get_connection, init_db, get_all_complaints_global, 
    get_users_by_role, update_user_status, update_complaint_assignment,
    get_department_id_for_category, execute_query
)
from utils.ui import apply_global_styles, init_sidebar_language_selector, render_footer, check_admin_access
from utils.auth import change_password

# Initialize language selector in sidebar
init_sidebar_language_selector()
# Check admin access
check_admin_access()

apply_global_styles()
current_lang = st.session_state.get("language", "English")

# Simplified labels for brevity - in production these would be in a separate file
AL = {
    "English": {
        "title": "Admin Panel",
        "tabs": ["Overview", "Complaints", "Users", "Departments", "Analytics", "Settings"],
        "overview": {
            "total_users": "Total Users",
            "total_complaints": "Total Complaints",
            "active_complaints": "Active Complaints",
            "resolved_today": "Resolved Today",
            "model_perf": "AI Model Accuracy",
            "uptime": "System Uptime"
        },
        "complaints": {
            "header": "Global Complaint Management",
            "assign": "Assign Official",
            "status": "Current Status",
            "action": "Actions",
            "no_complaints": "No complaints found.",
            "unassigned": "Unassigned"
        },
        "users": {
            "header": "User Management",
            "approve": "Approve",
            "suspend": "Suspend",
            "activate": "Activate",
            "role": "Role",
            "dept": "Department",
            "no_users": "No users found."
        },
        "settings": {
            "header": "System Settings",
            "pwd_header": "Change Admin Password",
            "update_btn": "Update Password",
            "cur_pwd": "Current Password",
            "new_pwd": "New Password",
            "conf_pwd": "Confirm New Password",
            "pwd_mismatch": "New passwords do not match",
            "fill_all": "Please fill all fields",
            "session_err": "Session invalid. Please login again."
        }
    }
}
# Fallback to English for this overhaul
L = AL.get(current_lang, AL["English"])

st.title(L["title"])

# Navigation
tab_overview, tab_complaints, tab_users, tab_depts, tab_analytics, tab_settings = st.tabs(L["tabs"])

# ---- OVERVIEW TAB ----
with tab_overview:
    st.header("System Overview")
    
    # Fetch real stats
    with get_connection() as conn:
        users_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        complaints_count = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
        active_count = conn.execute("SELECT COUNT(*) FROM complaints WHERE status NOT IN ('Resolved', 'Closed')").fetchone()[0]
        # Heuristic for resolved today (since we don't have a resolved_at column yet, using updated_at)
        resolved_today = conn.execute("SELECT COUNT(*) FROM complaints WHERE status='Resolved' AND date(updated_at) = date('now')").fetchone()[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(L["overview"]["total_users"], users_count)
    c2.metric(L["overview"]["total_complaints"], complaints_count)
    c3.metric(L["overview"]["active_complaints"], active_count)
    c4.metric(L["overview"]["resolved_today"], resolved_today)

    st.markdown("---")
    st.subheader("AI Performance Monitor")
    k1, k2, k3 = st.columns(3)
    k1.metric("Category Accuracy", "94.2%", "+1.2%")
    k2.metric("Urgency Accuracy", "89.5%", "+0.5%")
    k3.metric("System Uptime", "99.9%", "Stable")


# ---- COMPLAINTS TAB ----
with tab_complaints:
    st.subheader(L["complaints"]["header"])
    
    complaints = get_all_complaints_global()
    
    if not complaints:
        st.info(L["complaints"]["no_complaints"])
    else:
        # Simple table view
        df_c = pd.DataFrame(complaints)
        # Select relevant columns
        cols_to_show = ["complaint_id", "name", "category", "urgency", "status", "department_name", "official_name"]
        st.dataframe(df_c[cols_to_show], use_container_width=True)
        
        st.markdown("### Actions")
        # Assignment Logic
        c_id = st.selectbox("Select Complaint ID to Assign", options=[c["complaint_id"] for c in complaints], key="c_assign_id")
        
        if c_id:
            selected_comp = next((c for c in complaints if c["complaint_id"] == c_id), None)
            if selected_comp:
                st.write(f"**Current Official:** {selected_comp['official_name'] or L['complaints']['unassigned']}")
                
                # Fetch officials for the relevant department
                dept_id = selected_comp["department_id"]
                officials = get_users_by_role("official")
                # Filter for department
                dept_officials = [o for o in officials if o["department_id"] == dept_id and o["is_approved"]]
                
                if not dept_officials:
                    st.warning("No approved officials found for this department.")
                else:
                    official_opts = {o["id"]: f"{o['name']} (ID: {o['id']})" for o in dept_officials}
                    sel_off_id = st.selectbox("Assign to", options=list(official_opts.keys()), format_func=lambda x: official_opts[x])
                    
                    if st.button("Assign Official"):
                        if update_complaint_assignment(c_id, sel_off_id):
                            st.success(f"Assigned {c_id} to {official_opts[sel_off_id]}")
                            st.experimental_rerun()
                        else:
                            st.error("Failed to assign.")


# ---- USERS TAB ----
with tab_users:
    st.subheader(L["users"]["header"])
    
    users = get_users_by_role() # All users
    if not users:
        st.info(L["users"]["no_users"])
    else:
        # Group by Role Tabs
        u_tabs = st.tabs(["All", "Pending Approval", "Officials", "Citizens"])
        
        with u_tabs[0]:
            st.dataframe(pd.DataFrame(users)[["id", "name", "email", "role", "is_approved", "is_active", "department_name"]], use_container_width=True)
            
        with u_tabs[1]:
            pending = [u for u in users if not u["is_approved"] and u["role"] == "official"]
            if not pending:
                st.success("No pending approvals.")
            else:
                for p in pending:
                    c1, c2, c3 = st.columns([3, 1, 1])
                    c1.write(f"**{p['name']}** ({p['email']}) - Dept: {p['department_name']}")
                    if c2.button(L["users"]["approve"], key=f"app_{p['id']}"):
                         update_user_status(p["id"], p["is_active"], True)
                         st.experimental_rerun()
                    if c3.button("Reject", key=f"rej_{p['id']}"):
                        # Logic to reject/delete would go here
                        pass

# ---- DEPARTMENTS TAB ----
with tab_depts:
    st.info("Department Management Coming Soon.")

# ---- ANALYTICS TAB ----
with tab_analytics:
    st.info("Advanced Analytics Coming Soon.")

# ---- SETTINGS TAB ----
with tab_settings:
    st.subheader(L["settings"]["pwd_header"])
    with st.form("change_password_form"):
        old_pass = st.text_input(L["settings"]["cur_pwd"], type="password")
        new_pass = st.text_input(L["settings"]["new_pwd"], type="password")
        confirm_pass = st.text_input(L["settings"]["conf_pwd"], type="password")
        submitted = st.form_submit_button(L["settings"]["update_btn"])
        
        if submitted:
            if not old_pass or not new_pass or not confirm_pass:
                st.error(f"❌ {L['settings']['fill_all']}")
            elif new_pass != confirm_pass:
                st.error(f"❌ {L['settings']['pwd_mismatch']}")
            else:
                user = st.session_state.get("user")
                if user and "id" in user:
                     success, msg = change_password(user["id"], old_pass, new_pass)
                     if success:
                         st.success(f"✅ {msg}")
                     else:
                         st.error(f"❌ {msg}")
                else:
                    st.error(f"❌ {L['settings']['session_err']}")

# Shared footer
render_footer()
