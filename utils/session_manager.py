from __future__ import annotations

"""Session management utilities for Streamlit authentication.

This module provides session management functions for the civic complaint
system including login, logout, and session validation.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import streamlit as st

logger = logging.getLogger(__name__)

# Session configuration
SESSION_TIMEOUT_MINUTES = 30
SESSION_KEYS = ['user_id', 'name', 'email', 'role', 'phone', 'location', 'department_id', 'login_time', 'last_activity']


def create_session(user_data: Dict[str, Any]) -> None:
    """Create a new user session.

    Args:
        user_data: Dictionary containing user information:
            - user_id: User ID
            - name: User's full name
            - email: User's email
            - role: User role (citizen/official/admin)
            - phone: Phone number (optional)
            - location: Location (optional)
            - department_id: Department ID for officials (optional)
    """
    now = datetime.now()

    st.session_state['user_id'] = user_data.get('user_id')
    st.session_state['name'] = user_data.get('name', '')
    st.session_state['email'] = user_data.get('email', '')
    st.session_state['role'] = user_data.get('role', '').lower()
    st.session_state['phone'] = user_data.get('phone', '')
    st.session_state['location'] = user_data.get('location', '')
    st.session_state['department_id'] = user_data.get('department_id')
    st.session_state['login_time'] = now
    st.session_state['last_activity'] = now
    st.session_state['authenticated'] = True

    logger.info(f"Session created for user {user_data.get('email')} with role {user_data.get('role')}")


def is_authenticated() -> bool:
    """Check if user is authenticated.

    Returns:
        True if authenticated, False otherwise.
    """
    if not st.session_state.get('authenticated', False):
        return False

    # Check session timeout
    last_activity = st.session_state.get('last_activity')
    if last_activity:
        elapsed = datetime.now() - last_activity
        if elapsed > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            logger.info(f"Session expired for user {st.session_state.get('email')}")
            clear_session()
            return False

    # Update last activity
    st.session_state['last_activity'] = datetime.now()
    return True


def has_role(required_role: str) -> bool:
    """Check if current user has the required role.

    Args:
        required_role: Role to check ('citizen', 'official', 'admin').

    Returns:
        True if user has the role, False otherwise.
    """
    if not is_authenticated():
        return False

    current_role = st.session_state.get('role', '').lower()
    required_role = required_role.lower()

    # Admin has access to everything
    if current_role == 'admin':
        return True

    return current_role == required_role


def get_user_id() -> Optional[int]:
    """Get current user ID.

    Returns:
        User ID if authenticated, None otherwise.
    """
    if is_authenticated():
        return st.session_state.get('user_id')
    return None


def get_user_name() -> str:
    """Get current user name.

    Returns:
        User name if authenticated, empty string otherwise.
    """
    if is_authenticated():
        return st.session_state.get('name', '')
    return ''


def get_user_email() -> str:
    """Get current user email.

    Returns:
        User email if authenticated, empty string otherwise.
    """
    if is_authenticated():
        return st.session_state.get('email', '')
    return ''


def get_user_role() -> str:
    """Get current user role.

    Returns:
        User role if authenticated, empty string otherwise.
    """
    if is_authenticated():
        return st.session_state.get('role', '')
    return ''


def get_department_id() -> Optional[int]:
    """Get current user's department ID.

    Returns:
        Department ID if user is an official, None otherwise.
    """
    if is_authenticated():
        return st.session_state.get('department_id')
    return None


def clear_session() -> None:
    """Clear the current session (logout)."""
    user_email = st.session_state.get('email', 'unknown')

    for key in SESSION_KEYS:
        if key in st.session_state:
            del st.session_state[key]

    if 'authenticated' in st.session_state:
        del st.session_state['authenticated']

    logger.info(f"Session cleared for user {user_email}")


def require_auth(required_role: Optional[str] = None) -> None:
    """Require authentication to access a page.

    This function should be called at the top of each protected page.
    It checks authentication and optionally role, redirecting to login
    if requirements are not met.

    Args:
        required_role: Optional role required ('citizen', 'official', 'admin').
            If None, any authenticated user can access.
    """
    if not is_authenticated():
        st.error("🔒 Please login to access this page")
        st.info("👉 You will be redirected to the login page.")
        st.stop()

    if required_role and not has_role(required_role):
        st.error(f"⛔ Access Denied: This page requires {required_role.capitalize()} role")
        st.info(f"Your current role: {get_user_role().capitalize()}")
        st.stop()


def show_user_info_sidebar() -> None:
    """Display user information in the sidebar."""
    if is_authenticated():
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 User Info")
        st.sidebar.write(f"**Name:** {get_user_name()}")
        st.sidebar.write(f"**Email:** {get_user_email()}")
        st.sidebar.write(f"**Role:** {get_user_role().capitalize()}")

        if get_user_role() == 'official' and get_department_id():
            st.sidebar.write(f"**Department ID:** {get_department_id()}")

        # Session info
        login_time = st.session_state.get('login_time')
        if login_time:
            elapsed = datetime.now() - login_time
            minutes = int(elapsed.total_seconds() / 60)
            st.sidebar.caption(f"⏱️ Session: {minutes} min")

        st.sidebar.markdown("---")

        if st.sidebar.button("🚪 Logout", use_container_width=True):
            clear_session()
            st.rerun()


__all__ = [
    'create_session',
    'is_authenticated',
    'has_role',
    'get_user_id',
    'get_user_name',
    'get_user_email',
    'get_user_role',
    'get_department_id',
    'clear_session',
    'require_auth',
    'show_user_info_sidebar',
]
