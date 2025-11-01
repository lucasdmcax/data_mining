"""
Template page for creating new pages in the dashboard.
Copy this template and modify as needed.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import styles
sys.path.append(str(Path(__file__).parent.parent))
from styles import get_custom_css

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

# Example of using info box
st.markdown("""
    <div class="info-box">
        <h3 style="color: #1f4788;">Information Box</h3>
        <p>Use this for important information or callouts.</p>
    </div>
""", unsafe_allow_html=True)

# Example of using metric containers
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="metric-container">
            <h2 style="color: #1f4788;">Metric 1</h2>
            <p style="color: #5a6c7d;">Description</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-container">
            <h2 style="color: #1f4788;">Metric 2</h2>
            <p style="color: #5a6c7d;">Description</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-container">
            <h2 style="color: #1f4788;">Metric 3</h2>
            <p style="color: #5a6c7d;">Description</p>
        </div>
    """, unsafe_allow_html=True)
