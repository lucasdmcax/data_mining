"""
Exploratory Data Analysis page for the dashboard.
"""

import streamlit as st
import sys
import pandas as pd
from pathlib import Path

# Correct paths - data is in the parent directory of dashboard
FLIGHTS_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "DM_AIAI_FlightsDB.csv"
CUSTOMERS_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "DM_AIAI_CustomerDB.csv"

# load datasets
customers_db = pd.read_csv(CUSTOMERS_DATA_PATH, index_col=0).sort_values(by='Loyalty#')
flights_db = pd.read_csv(FLIGHTS_DATA_PATH)

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
        # Calculate duplicated loyalty IDs and store in session state
        if 'duplicated_loyalty_ids' not in st.session_state:
            st.session_state.duplicated_loyalty_ids = customers_db[customers_db['Loyalty#'].duplicated()]['Loyalty#'].unique()

        st.markdown(
            get_metric_html(
                "Customer Data",
                "Customer records overview",
                number_of_customers=customers_db.shape[0],
                duplicated_loyalty_ids=len(st.session_state.duplicated_loyalty_ids),
            ),
            unsafe_allow_html=True
        )

        st.write("")

        customers_number_rows = st.number_input("Number of customer rows to display", value=5, min_value=1, max_value=customers_db.shape[0], step=1)
        st.dataframe(customers_db.head(customers_number_rows))

        customers_col = st.selectbox("Select column to view distribution", options=customers_db.columns.tolist())

        if customers_col != 'Loyalty#':
            st.write("**Distribution:**")
            # Sort the value counts by the index (the column's values)
            distribution = customers_db[customers_col].value_counts()
            st.bar_chart(distribution)

    with col2:
        # calculate flights with duplicated ids:
        if 'flights_duplicated_id' not in st.session_state:
         st.session_state.flights_duplicated_id = flights_db[flights_db['Loyalty#']\
                                                              .isin(st.session_state.duplicated_loyalty_ids)]\
                                                              .shape[0]

        st.markdown(
            get_metric_html(
                "Flight Data",
                "Flight records overview",
                number_of_flights=flights_db.shape[0],
                number_of_flights_from_duplicated_ids=st.session_state.flights_duplicated_id,
                
            ),
            unsafe_allow_html=True
        )
        st.write("")

        flights_number_rows = st.number_input("Number of flight rows to display", value=5, min_value=1, max_value=flights_db.shape[0], step=1)
        
        st.dataframe(flights_db.head(flights_number_rows))

# Tab 2: Geospatial Analysis
with tab2:
    st.markdown("### Geographic Patterns")
    
    st.markdown(
        get_info_box_html(
            "About Geospatial Analysis",
            "Explore geographic distribution of customers and flight routes." \
            "Identify regional patterns and opportunities."
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
