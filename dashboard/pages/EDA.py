"""
Exploratory Data Analysis page for the dashboard.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import styles
sys.path.append(str(Path(__file__).parent.parent))
from styles import get_custom_css, get_metric_html, get_info_box_html

# Page configuration
st.set_page_config(
    page_title="EDA - AIAI Analytics",
    page_icon="📊",
    layout="wide"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Initial exploration of customer data</div>', unsafe_allow_html=True)

# Multitab Section
tab1, tab2, tab3 = st.tabs(["🔍 Initial Inspection", "🗺️ Geospatial Analysis", "📈 Correlation Analysis"])

# Tab 1: Initial Inspection
with tab1:
    st.markdown("### Dataset Overview")
    
    st.markdown(
        get_info_box_html(
            "About Initial Inspection",
            "This section provides a first look at the customer and flight data, including basic statistics, data quality checks, and initial observations."
        ),
        unsafe_allow_html=True
    )
    
    st.write("")
    
    # Placeholder for data inspection content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            get_metric_html(
                "Customer Data",
                "Dataset dimensions, missing values, and key statistics"
            ),
            unsafe_allow_html=True
        )
        
        st.write("")
        st.write("**Add your customer data inspection here:**")
        st.write("- Data shape and size")
        st.write("- Column types and descriptions")
        st.write("- Missing value analysis")
        st.write("- Basic statistics")
    
    with col2:
        st.markdown(
            get_metric_html(
                "Flight Data",
                "Flight records overview and data quality"
            ),
            unsafe_allow_html=True
        )
        
        st.write("")
        st.write("**Add your flight data inspection here:**")
        st.write("- Data shape and size")
        st.write("- Column types and descriptions")
        st.write("- Missing value analysis")
        st.write("- Basic statistics")

# Tab 2: Geospatial Analysis
with tab2:
    st.markdown("### Geographic Patterns")
    
    st.markdown(
        get_info_box_html(
            "About Geospatial Analysis",
            "Explore geographic distribution of customers and flight routes. Identify regional patterns and opportunities."
        ),
        unsafe_allow_html=True
    )
    
    st.write("")
    
    # Placeholder for geospatial content
    st.write("**Add your geospatial visualizations here:**")
    st.write("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Customer Distribution")
        st.write("- Geographic distribution of customers")
        st.write("- Country/region analysis")
        st.write("- Population density maps")
        st.write("- Market penetration by region")
    
    with col2:
        st.markdown("#### Flight Routes")
        st.write("- Popular flight routes")
        st.write("- Origin-destination patterns")
        st.write("- Route frequency analysis")
        st.write("- International vs domestic flights")

# Tab 3: Correlation Analysis
with tab3:
    st.markdown("### Variable Relationships")
    
    st.markdown(
        get_info_box_html(
            "About Correlation Analysis",
            "Analyze relationships between variables to understand what drives customer loyalty, flight frequency, and spending patterns."
        ),
        unsafe_allow_html=True
    )
    
    st.write("")
    
    # Placeholder for correlation content
    st.write("**Add your correlation analysis here:**")
    st.write("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Customer Attributes")
        st.write("- Age vs loyalty metrics")
        st.write("- Income vs spending")
        st.write("- Membership duration vs activity")
        st.write("- Demographics correlations")
    
    with col2:
        st.markdown("#### Flight Behavior")
        st.write("- Flight frequency vs spending")
        st.write("- Distance vs ticket price")
        st.write("- Seasonality patterns")
        st.write("- Booking patterns vs loyalty")
