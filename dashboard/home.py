import streamlit as st
from styles import get_custom_css

# Page configuration
st.set_page_config(
    page_title="AIAI Customer Analytics Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header Section
st.markdown('<div class="main-header">✈️ AIAI Customer Loyalty Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data-Driven Customer Segmentation Strategy</div>', unsafe_allow_html=True)

# Hero Section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
        <div class="info-box">
            <h3 style="text-align: center; color: #1f4788;">Welcome to Your Customer Intelligence Platform</h3>
            <p style="text-align: center; font-size: 1.1rem;">
                Transforming three years of loyalty membership and flight activity data into actionable business insights.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.write("")

# Project Overview Section
st.markdown('<div class="section-header">📊 Project Overview</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        ### Objective
        Analyze customer loyalty and flight activity data from a **three-year period** to develop 
        a data-driven customer segmentation strategy for AIAI.
        
        ### Key Deliverables
        - Customer Segmentation Models
        - Actionable Business Insights
        - Strategic Recommendations
    """)

with col2:
    st.markdown("""
        ### Benefits
        - 📈 **Increase Revenue** - Personalized customer targeting
        - 🎯 **Improve Retention** - Identify at-risk customers
        - 💡 **Optimize Marketing** - Efficient resource allocation
    """)

st.write("")

# Data Scope Section
st.markdown('<div class="section-header">📁 Data Scope</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="metric-container">
            <h2 style="color: #1f4788;">3 Years</h2>
            <p style="color: #5a6c7d;">Historical Data Period</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-container">
            <h2 style="color: #1f4788;">2 Datasets</h2>
            <p style="color: #5a6c7d;">Customer & Flight Records</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-container">
            <h2 style="color: #1f4788;">360° View</h2>
            <p style="color: #5a6c7d;">Comprehensive Customer Profile</p>
        </div>
    """, unsafe_allow_html=True)

st.write("")

# Methodology Section
st.markdown('<div class="section-header">🔬 Our Approach</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        #### 1️⃣ Exploration
        Analyze customer demographics and flight patterns
    """)

with col2:
    st.markdown("""
        #### 2️⃣ Engineering
        Create meaningful customer value indicators
    """)

with col3:
    st.markdown("""
        #### 3️⃣ Segmentation
        Identify distinct customer groups
    """)

with col4:
    st.markdown("""
        #### 4️⃣ Strategy
        Develop actionable recommendations
    """)

st.write("")

# Team Section
st.markdown('<div class="section-header">👥 Our Team</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class="team-card-compact">
            <div class="placeholder-img-compact">👤</div>
            <div class="student-name-compact">Student Name 1</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="team-card-compact">
            <div class="placeholder-img-compact">👤</div>
            <div class="student-name-compact">Student Name 2</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="team-card-compact">
            <div class="placeholder-img-compact">👤</div>
            <div class="student-name-compact">Student Name 3</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="team-card-compact">
            <div class="placeholder-img-compact">👤</div>
            <div class="student-name-compact">Student Name 4</div>
        </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# Navigation Section
st.markdown('<div class="section-header">🚀 Explore the Dashboard</div>', unsafe_allow_html=True)

st.markdown("""
    Use the **sidebar navigation** to explore different sections:
    
    - **Exploratory Data Analysis** - Customer and flight data patterns
    - **Customer Segmentation** - Distinct customer groups and characteristics
    - **Business Recommendations** - Strategic actions per segment
""")

st.write("")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
        <p style="text-align: center; color: #5a6c7d;">
            <strong>AIAI Customer Analytics Project</strong><br>
            Powered by Data Mining & Machine Learning<br>
            © 2025 AIAI Analytics Team
        </p>
    """, unsafe_allow_html=True)