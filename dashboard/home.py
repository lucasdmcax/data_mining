import streamlit as st
from styles import get_custom_css, get_metric_html
import os
import base64
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AIAI Customer Analytics Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Function to convert image to base64
def get_image_base64(image_path):
    """Convert image to base64 for HTML embedding"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Function to create team member card
def create_team_card(name, student_number, image_filename=None):
    """
    Create a team member card HTML.
    
    Args:
        name (str): Student name
        student_number (str): Student number
        image_filename (str): Filename of image in pics folder (optional)
    
    Returns:
        str: HTML for the team member card
    """
    # Check if image exists and convert to base64
    img_path = Path(__file__).parent / "pics" / image_filename if image_filename else None
    
    if img_path and img_path.exists():
        img_base64 = get_image_base64(img_path)
        if img_base64:
            img_html = f'<img src="data:image/png;base64,{img_base64}" style="width: 80px; height: 80px; border-radius: 50%; object-fit: cover; margin: 0 auto 0.5rem auto; display: block;">'
        else:
            img_html = '<div class="placeholder-img-compact">👤</div>'
    else:
        img_html = '<div class="placeholder-img-compact">👤</div>'
    
    return f"""
        <div class="team-card-compact">
            {img_html}
            <div class="student-name-compact">{name}</div>
            <div class="student-number-compact">{student_number}</div>
        </div>
    """

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
    st.markdown(get_metric_html("3 Years", "Historical Data Period"), unsafe_allow_html=True)

with col2:
    st.markdown(get_metric_html("2 Datasets", "Customer & Flight Records"), unsafe_allow_html=True)

with col3:
    st.markdown(get_metric_html("360° View", "Comprehensive Customer Profile"), unsafe_allow_html=True)

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

# Team members data - UPDATE THIS SECTION WITH YOUR TEAM INFO
team_members = [
    {"name": "Lucas Campos Ferreira", "student_number": "20250448", "image": "lucas.png"},
    {"name": "João Paulo de Avila", "student_number": "20250436", "image": "joao.jpeg"},
    {"name": "Maria Leonor Ribeiro", "student_number": "20221898", "image": "leonor.jpeg"},
    {"name": "Gonçalo Torrão", "student_number": "20250365", "image": "goncalo.jpg"}
]

# Display team cards
col1, col2, col3, col4 = st.columns(4)
columns = [col1, col2, col3, col4]

for idx, member in enumerate(team_members):
    with columns[idx]:
        st.markdown(
            create_team_card(
                member["name"], 
                member["student_number"], 
                member["image"]
            ), 
            unsafe_allow_html=True
        )

st.write("")
st.write("")

# Navigation Section
st.markdown('<div class="section-header">🚀 Explore the Dashboard</div>', unsafe_allow_html=True)

st.markdown("""
    Use the **sidebar navigation** to explore different sections:
    
    - **[Exploratory Data Analysis](exploratory_data_analysis)** - Customer and flight data patterns
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