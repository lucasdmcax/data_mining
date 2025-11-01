"""
Template page for creating new pages in the dashboard.
Copy this template and modify as needed.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import styles
sys.path.append(str(Path(__file__).parent.parent))
from styles import get_custom_css, get_metric_html, get_info_box_html

# Page configuration
st.set_page_config(
    page_title="Page Title - AIAI Analytics",
    page_icon="📊",
    layout="wide"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 Page Title</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Page subtitle or description</div>', unsafe_allow_html=True)

# Content Section
st.markdown('<div class="section-header">Section Header</div>', unsafe_allow_html=True)

# Your content here
st.write("Add your content here...")

# Example of using info box with centralized colors
st.markdown(
    get_info_box_html("Information Box", "Use this for important information or callouts."),
    unsafe_allow_html=True
)

# Example of using metric containers with centralized colors
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(get_metric_html("Metric 1", "Description"), unsafe_allow_html=True)

with col2:
    st.markdown(get_metric_html("Metric 2", "Description"), unsafe_allow_html=True)

with col3:
    st.markdown(get_metric_html("Metric 3", "Description"), unsafe_allow_html=True)
