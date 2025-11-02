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
from utils import plot_data_distribution, plot_correlation_analysis, get_data_class, plot_box_plot, plot_canada_map

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
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Initial Inspection", "📈 Correlation Analysis", "⚠️ Outliers", "🗺️ Geographic Analysis"])

# Tab 1: Initial Inspection
with tab1:
    st.markdown("### Dataset Overview")
    
    st.markdown(
        get_info_box_html(
            "About Initial Inspection",
            "This section provides a first look at the customer and flight data, " \
            "you can filter the data by <strong>Loyalty#</strong> to inspect individual customer records and their associated flights. "\
            "Also see distributions of key variables to understand data characteristics."

        ),
        unsafe_allow_html=True
    )
    
    st.write("")
    
    # ==========================================
    # FILTER SECTION
    # ==========================================
    st.markdown("#### 🔍 Filter Data by Loyalty#")
    
    # Loyalty# filter
    loyalty_options = ["All"] + sorted(customers_db['Loyalty#'].unique().tolist())
    selected_loyalty = st.selectbox(
        "", 
        options=loyalty_options,
        help="Select a specific Loyalty# to filter both customer and flight data"
    )
    
    # Apply filter
    if selected_loyalty != "All":
        filtered_customers = customers_db[customers_db['Loyalty#'] == selected_loyalty]
        filtered_flights = flights_db[flights_db['Loyalty#'] == selected_loyalty]
    else:
        filtered_customers = customers_db
        filtered_flights = flights_db
    
    st.write("")
    
    # ==========================================
    # DATA DISPLAY SECTION
    # ==========================================
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate duplicated loyalty IDs
        if 'duplicated_loyalty_ids' not in st.session_state:
            st.session_state.duplicated_loyalty_ids = customers_db[
                customers_db['Loyalty#'].duplicated()
            ]['Loyalty#'].unique()

        st.markdown(
            get_metric_html(
                "Customer Data",
                "Customer records overview",
                number_of_customers=filtered_customers.shape[0],
                duplicated_loyalty_ids=len(st.session_state.duplicated_loyalty_ids),
            ),
            unsafe_allow_html=True
        )

        st.write("")

        customers_number_rows = st.number_input(
            "Number of customer rows to display", 
            value=min(5, filtered_customers.shape[0]), 
            min_value=1, 
            max_value=filtered_customers.shape[0], 
            step=1
        )
        st.dataframe(filtered_customers.head(customers_number_rows))

        customers_col = st.selectbox(
            "Select column to view distribution", 
            options=filtered_customers.columns.tolist()
        )

        if customers_col != 'Loyalty#':
            st.write("**Distribution:**")
            plot_data_distribution(filtered_customers[customers_col], dataset='customers')

    with col2:
        # Calculate flights with duplicated ids
        if 'flights_duplicated_id' not in st.session_state:
            st.session_state.flights_duplicated_id = flights_db[
                flights_db['Loyalty#'].isin(st.session_state.duplicated_loyalty_ids)
            ].shape[0]

        st.markdown(
            get_metric_html(
                "Flight Data",
                "Flight records overview",
                number_of_flights=filtered_flights.shape[0],
                number_of_flights_from_duplicated_ids=st.session_state.flights_duplicated_id,
            ),
            unsafe_allow_html=True
        )
        st.write("")


        flights_number_rows = st.number_input(
            "Number of flight rows to display", 
            value=min(5, max(1, filtered_flights.shape[0])), 
            min_value=1, 
            max_value=max(1, filtered_flights.shape[0]), 
            step=1,
            disabled=(filtered_flights.shape[0] == 0)
        )
        
        if filtered_flights.shape[0] > 0:
            st.dataframe(filtered_flights.head(flights_number_rows))

            flights_col = st.selectbox(
            "Select column to view distribution", 
            options=filtered_flights.columns.tolist()
            )
            if flights_col != 'Loyalty#':
                st.write("**Distribution:**")
                plot_data_distribution(filtered_flights[flights_col], dataset='flights')
        else:
            st.info("No flight records found for this Loyalty#.")
        

        


# Tab 2: Correlation Analysis
with tab2:
    st.markdown("### Variable Relationships")
    
    st.markdown(
        get_info_box_html(
            "About Correlation Analysis",
            "Explore relationships between variables to understand patterns in customer behavior. " \
            "Select two features from a dataset and the system will automatically generate the appropriate visualization: " \
            "<br><br>" \
            "<strong>• Numerical vs Numerical:</strong> Scatter plot with Pearson correlation coefficient<br>" \
            "<strong>• Numerical vs Categorical:</strong> Overlaid histograms colored by category<br>" \
            "<strong>• Categorical vs Categorical:</strong> Heatmap showing cross-tabulation counts"
        ),
        unsafe_allow_html=True
    )
    
    st.write("")
    
    # Dataset selection
    st.markdown("#### Select Dataset")
    dataset_choice = st.radio(
        "Choose dataset",
        options=["Customers", "Flights"],
        horizontal=True,
        label_visibility="collapsed",
        help="Choose which dataset to analyze"
    )
    
    # Get the appropriate dataframe and dataset name
    if dataset_choice == "Customers":
        analysis_df = filtered_customers
        dataset_name = 'customers'
    else:
        analysis_df = filtered_flights
        dataset_name = 'flights'
    
    st.write("")
    
    # Feature selection
    st.markdown("#### Select Features to Compare")
    col1, col2 = st.columns(2)
    
    with col1:
        feature1 = st.selectbox(
            "First Feature",
            options=analysis_df.columns.tolist(),
            help="Select the first variable for comparison"
        )
    
    with col2:
        # Exclude the already selected feature from second dropdown
        available_features = [col for col in analysis_df.columns.tolist() if col != feature1]
        feature2 = st.selectbox(
            "Second Feature",
            options=available_features,
            help="Select the second variable for comparison"
        )
    
    st.write("")
    
    # Display correlation analysis
    if feature1 and feature2:
        if feature1 == feature2:
            st.warning("⚠️ Please select two different features to compare.")
        elif feature1 == 'Loyalty#' or feature2 == 'Loyalty#':
            st.info("ℹ️ Both features must be different from 'Loyalty#' for correlation analysis.")
        else:
            plot_correlation_analysis(analysis_df, feature1, feature2, dataset=dataset_name)


# Tab 3: Outliers Detection
with tab3:
    st.markdown("### Outlier Detection")
    
    st.markdown(
        get_info_box_html(
            "About Outlier Detection",
            "Identify unusual values in numerical variables that may indicate data quality issues or interesting patterns. " \
            "This analysis uses the <strong>Interquartile Range (IQR)</strong> method with box plots for visualization." \
            "<br><br>" \
            "<strong>What is a Box Plot?</strong><br>" \
            "A box plot displays the distribution of data based on five key statistics:<br>" \
            "• <strong>Minimum:</strong> Lowest value within 1.5×IQR below Q1<br>" \
            "• <strong>Q1 (25th percentile):</strong> 25% of data falls below this value<br>" \
            "• <strong>Median (Q2):</strong> Middle value that divides data in half<br>" \
            "• <strong>Q3 (75th percentile):</strong> 75% of data falls below this value<br>" \
            "• <strong>Maximum:</strong> Highest value within 1.5×IQR above Q3<br>" \
            "• <strong>Outliers:</strong> Points beyond 1.5×IQR from Q1 or Q3 (shown as individual dots)"
        ),
        unsafe_allow_html=True
    )
    
    st.write("")
    
    # Dataset selection
    st.markdown("#### Select Dataset")
    outlier_dataset_choice = st.radio(
        "Choose dataset for outlier analysis",
        options=["Customers", "Flights"],
        horizontal=True,
        label_visibility="collapsed",
        help="Choose which dataset to analyze for outliers",
        key="outlier_dataset_radio"
    )
    
    # Get the appropriate dataframe and dataset name
    if outlier_dataset_choice == "Customers":
        outlier_df = filtered_customers
        outlier_dataset_name = 'customers'
    else:
        outlier_df = filtered_flights
        outlier_dataset_name = 'flights'
    
    st.write("")
    
    # Get only numerical columns
    numerical_cols = [col for col in outlier_df.columns if get_data_class(col, outlier_dataset_name) == 'numerical']
    
    if len(numerical_cols) == 0:
        st.warning("No numerical columns available for outlier detection in this dataset.")
    else:
        st.markdown("#### Select Numerical Feature")
        selected_feature = st.selectbox(
            "Choose a numerical variable to analyze",
            options=numerical_cols,
            help="Select a numerical column to detect outliers"
        )
        
        st.write("")
        
        # Plot box plot with outlier detection
        if selected_feature:
            plot_box_plot(outlier_df, selected_feature, dataset=outlier_dataset_name)


# Tab 4: Geographic Analysis
with tab4:
    st.markdown("### Geographic Distribution")
    
    st.markdown(
        get_info_box_html(
            "About Geographic Analysis",
            "Visualize the geographic distribution of AIAI customers across Canada. " \
            "This map shows customer density by province/state, helping identify regional patterns and opportunities. " \
            "<br><br>" \
            "<strong>Key Insights:</strong><br>" \
            "• <strong>Customer Concentration:</strong> Identify provinces with the highest customer base<br>" \
            "• <strong>Regional Patterns:</strong> Understand geographic trends in customer distribution<br>" \
            "• <strong>Market Opportunities:</strong> Discover underserved regions for expansion<br>" \
            "• <strong>Interactive Map:</strong> Hover over provinces to see detailed customer counts"
        ),
        unsafe_allow_html=True
    )
    
    st.write("")
    
    # Display map
    st.markdown("#### Customer Distribution Map")
    plot_canada_map(filtered_customers)
