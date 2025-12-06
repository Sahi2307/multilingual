from __future__ import annotations

"""Shared UI helpers for the Streamlit civic complaint app.

This module centralises small visual tweaks so all pages share the same
look-and-feel (background, typography, footer, etc.).
"""

import streamlit as st


def apply_global_styles() -> None:
    """Previously injected global CSS.

    This is now a no-op so the app uses Streamlit's default styling.
    Pages can still call this safely without any visual side effects.
    """

    return None


def render_footer() -> None:
    """Render a simple shared footer for all pages.

    Kept minimal (no extra CSS) per user's request.
    """

    st.markdown("---")
    st.caption("© 2025 Municipal IT Desk")
