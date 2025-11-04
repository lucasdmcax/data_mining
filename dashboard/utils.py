"""
Utility functions for the AIAI Customer Analytics Dashboard
"""

import streamlit as st
import pandas as pd
import altair as alt
import json
from pathlib import Path

# Import color for box plot
import sys
sys.path.append(str(Path(__file__).parent))
from styles import PRIMARY_BLUE

# Load column metadata into session state
def load_metadata():
    """Load column metadata from JSON file into session state."""
    if 'column_metadata' not in st.session_state:
        METADATA_PATH = Path(__file__).parent / "column_metadata.json"
        with open(METADATA_PATH, 'r') as f:
            st.session_state.column_metadata = json.load(f)
    return st.session_state.column_metadata


def get_variable_type(column_name: str, dataset: str = 'customers') -> str:
    """
    Get the variable type (discrete/continuous) from metadata.
    
    Args:
        column_name (str): Name of the column
        dataset (str): Dataset name - 'customers' or 'flights' (default: 'customers')
    
    Returns:
        str: 'discrete' or 'continuous', defaults to 'discrete' if not found
    """
    metadata = load_metadata()
    try:
        return metadata[dataset][column_name]['type']
    except KeyError:
        # Default to discrete if column not found in metadata
        return 'discrete'


def get_data_class(column_name: str, dataset: str = 'customers') -> str:
    """
    Get the data class (numerical/categorical) from metadata.
    
    Args:
        column_name (str): Name of the column
        dataset (str): Dataset name - 'customers' or 'flights' (default: 'customers')
    
    Returns:
        str: 'numerical' or 'categorical', defaults to 'categorical' if not found
    """
    metadata = load_metadata()
    try:
        data_class = metadata[dataset][column_name]['data_class']
        return data_class
    except KeyError as e:
        # Default to categorical if column not found in metadata
        st.warning(f"Column '{column_name}' not found in {dataset} metadata. Defaulting to categorical. Error: {e}")
        return 'categorical'


def plot_data_distribution(series: pd.Series, variable_type: str = None, title: str = None, dataset: str = 'customers'):
    """
    Plot data distribution based on variable type using Altair charts.
    Automatically determines type from metadata if not specified.
    
    Args:
        series (pd.Series): The data series to plot
        variable_type (str): Type of variable - 'discrete' or 'continuous'. 
                            If None, will use metadata (default: None)
        title (str): Optional title for the plot. If None, uses series name
        dataset (str): Dataset name - 'customers' or 'flights' for metadata lookup (default: 'customers')
    
    Returns:
        None: Displays the plot in Streamlit
    
    Example:
        # Auto-detect from metadata
        plot_data_distribution(customers_db['Gender'], dataset='customers')
        plot_data_distribution(flights_db['DistanceKM'], dataset='flights')
        
        # Manual override
        plot_data_distribution(df['Age'], variable_type='continuous')
    """
    # Get variable type from metadata if not specified
    if variable_type is None:
        variable_type = get_variable_type(series.name, dataset)
    
    # Display title
    st.write(f"**{title if title else f'Distribution of {series.name}'}**")
    
    # Prepare data - drop nulls
    clean_data = series.dropna()
    
    # Plot based on variable type
    if variable_type == 'discrete':
        # Pre-aggregate for discrete variables - much faster
        value_counts = clean_data.value_counts().reset_index()
        value_counts.columns = [series.name, 'Count']
        
        chart = alt.Chart(value_counts).mark_bar().encode(
            x=alt.X(f'{series.name}:N', sort='-y', title=series.name),
            y=alt.Y('Count:Q', title='Count')
        ).properties(height=400)
        
    elif variable_type == 'continuous':
        # Sample large datasets for faster rendering
        if len(clean_data) > 5000:
            clean_data = clean_data.sample(n=5000, random_state=42)
        
        df = clean_data.to_frame()
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{series.name}:Q',
                    bin=alt.Bin(maxbins=30),
                    title=series.name),
            y=alt.Y('count()', title='Count')
        ).properties(height=400)
        
    else:
        st.error(f"Invalid variable_type: {variable_type}. Must be 'discrete' or 'continuous'.")
        return
    
    # Disable Altair's max rows warning and render
    st.altair_chart(chart.configure_axis(labelLimit=500), width='stretch')


def plot_correlation_analysis(df: pd.DataFrame, col1: str, col2: str, dataset: str = 'customers'):
    """
    Plot correlation analysis between two variables based on their data classes.
    Automatically determines appropriate visualization and calculates correlation when applicable.
    
    Args:
        df (pd.DataFrame): The dataframe containing both columns
        col1 (str): Name of the first column
        col2 (str): Name of the second column
        dataset (str): Dataset name - 'customers' or 'flights' for metadata lookup (default: 'customers')
    
    Returns:
        None: Displays the plot and correlation metric in Streamlit
    
    Cases:
        - numerical vs numerical: Scatter plot + Pearson correlation
        - numerical vs categorical: Distributions colored by category
        - categorical vs categorical: Heatmap/Count matrix
    """
    # Get data classes for both columns
    class1 = get_data_class(col1, dataset)
    class2 = get_data_class(col2, dataset)
    
    # Debug: Show detected classes
    st.caption(f"Detected: {col1} ({class1}) vs {col2} ({class2})")
    
    # Remove rows with missing values in either column
    clean_df = df[[col1, col2]].dropna()
    
    # Sample if dataset is too large
    if len(clean_df) > 5000:
        clean_df = clean_df.sample(n=5000, random_state=42)
        st.info(f"Note: Showing sample of 5,000 points from {len(df)} total records for performance.")
    
    # Case 1: Numerical vs Numerical - Scatter plot + Pearson correlation
    if class1 == 'numerical' and class2 == 'numerical':
        st.markdown(f"### {col1} vs {col2}")
        
        # Calculate Pearson correlation
        correlation = clean_df[col1].corr(clean_df[col2])
        
        # Display correlation metric
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric2:
            st.metric(
                label="Pearson Correlation",
                value=f"{correlation:.3f}",
                help="Ranges from -1 (perfect negative) to +1 (perfect positive)"
            )
        
        st.write("")
        
        # Create scatter plot
        chart = alt.Chart(clean_df).mark_circle(size=60, opacity=0.5).encode(
            x=alt.X(f'{col1}:Q', title=col1),
            y=alt.Y(f'{col2}:Q', title=col2),
            tooltip=[col1, col2]
        ).properties(
            height=400,
            title=f"Scatter Plot: {col1} vs {col2}"
        )
        
        st.altair_chart(chart, width='stretch')
    
    # Case 2: Numerical vs Categorical - Distributions by category
    elif (class1 == 'numerical' and class2 == 'categorical') or \
         (class1 == 'categorical' and class2 == 'numerical'):
        
        # Determine which is numerical and which is categorical
        num_col = col1 if class1 == 'numerical' else col2
        cat_col = col2 if class2 == 'categorical' else col1
        
        st.markdown(f"### {num_col} by {cat_col}")
        
        # Limit categories to top 10 most frequent to avoid clutter
        top_categories = clean_df[cat_col].value_counts().head(10).index
        filtered_df = clean_df[clean_df[cat_col].isin(top_categories)]
        
        if len(top_categories) < len(clean_df[cat_col].unique()):
            st.info(f"Showing top 10 categories out of {len(clean_df[cat_col].unique())} total.")
        
        # Create histogram with color by category
        chart = alt.Chart(filtered_df).mark_bar(opacity=0.7).encode(
            x=alt.X(f'{num_col}:Q', bin=alt.Bin(maxbins=30), title=num_col),
            y=alt.Y('count()', title='Count', stack=None),
            color=alt.Color(f'{cat_col}:N', title=cat_col),
            tooltip=[cat_col, 'count()']
        ).properties(
            height=400,
            title=f"Distribution of {num_col} by {cat_col}"
        )
        
        st.altair_chart(chart, width='stretch')
    
    # Case 3: Categorical vs Categorical - Heatmap
    else:  # Both categorical
        st.markdown(f"### {col1} vs {col2}")
        
        # Limit to top categories for readability
        top_cat1 = clean_df[col1].value_counts().head(10).index
        top_cat2 = clean_df[col2].value_counts().head(10).index
        filtered_df = clean_df[
            (clean_df[col1].isin(top_cat1)) & 
            (clean_df[col2].isin(top_cat2))
        ]
        
        # Create crosstab for heatmap
        crosstab = pd.crosstab(filtered_df[col1], filtered_df[col2])
        
        # Convert to long format for Altair
        heatmap_data = crosstab.reset_index().melt(id_vars=col1, var_name=col2, value_name='Count')
        
        # Create heatmap
        chart = alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X(f'{col2}:N', title=col2),
            y=alt.Y(f'{col1}:N', title=col1),
            color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), title='Count'),
            tooltip=[col1, col2, 'Count']
        ).properties(
            height=400,
            title=f"Heatmap: {col1} vs {col2}"
        )
        
        st.altair_chart(chart, width='stretch')


def plot_box_plot(df: pd.DataFrame, column: str, dataset: str = 'customers'):
    """
    Plot box plot for outlier detection using IQR method.
    
    Args:
        df (pd.DataFrame): The dataframe containing the column
        column (str): Name of the numerical column to analyze
        dataset (str): Dataset name - 'customers' or 'flights' for metadata lookup (default: 'customers')
    
    Returns:
        None: Displays the box plot and outlier statistics in Streamlit
    """
    st.markdown(f"### Outlier Analysis: {column}")
    
    # Remove missing values
    clean_data = df[column].dropna()
    
    if len(clean_data) == 0:
        st.warning(f"No valid data available for {column}")
        return
    
    # Calculate IQR statistics
    Q1 = clean_data.quantile(0.25)
    Q3 = clean_data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
    n_outliers = len(outliers)
    outlier_percentage = (n_outliers / len(clean_data)) * 100
    
    # Display statistics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Values",
            value=f"{len(clean_data):,}",
            help="Total number of non-missing values"
        )
    
    with col2:
        st.metric(
            label="Outliers Detected",
            value=f"{n_outliers:,}",
            help="Number of values outside 1.5×IQR range"
        )
    
    with col3:
        st.metric(
            label="Outlier %",
            value=f"{outlier_percentage:.2f}%",
            help="Percentage of outliers in the data"
        )
    
    with col4:
        st.metric(
            label="IQR",
            value=f"{IQR:.2f}",
            help="Interquartile Range (Q3 - Q1)"
        )
    
    st.write("")
    
    # Display key statistics
    with st.expander("📊 Statistical Summary", expanded=False):
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.write("**Quartiles:**")
            st.write(f"Q1 (25%): {Q1:.2f}")
            st.write(f"Median (50%): {clean_data.median():.2f}")
            st.write(f"Q3 (75%): {Q3:.2f}")
        
        with stats_col2:
            st.write("**Range:**")
            st.write(f"Min: {clean_data.min():.2f}")
            st.write(f"Max: {clean_data.max():.2f}")
            st.write(f"Range: {clean_data.max() - clean_data.min():.2f}")
        
        with stats_col3:
            st.write("**Outlier Bounds:**")
            st.write(f"Lower: {lower_bound:.2f}")
            st.write(f"Upper: {upper_bound:.2f}")
            st.write(f"IQR: {IQR:.2f}")
    
    st.write("")
    
    # Create horizontal box plot using Altair
    # Prepare data for box plot (use all data, no sampling)
    box_df = pd.DataFrame({column: clean_data})
    
    # Create horizontal box plot
    box_chart = alt.Chart(box_df).mark_boxplot(
        extent='min-max',
        size=40
    ).encode(
        x=alt.X(f'{column}:Q', title=column, scale=alt.Scale(zero=False)),
        color=alt.value(PRIMARY_BLUE)
    ).properties(
        height=150,
    )
    
    # Create scatter plot for outliers overlay (horizontal)
    outlier_df = pd.DataFrame({column: outliers})
    if len(outlier_df) > 0:
        outlier_chart = alt.Chart(outlier_df).mark_circle(
            size=80,
            color='red',
            opacity=0.6
        ).encode(
            x=alt.X(f'{column}:Q'),
            tooltip=[alt.Tooltip(f'{column}:Q', format='.2f')]
        )
        
        combined_chart = (box_chart + outlier_chart).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_view(
            strokeWidth=0
        )
    else:
        combined_chart = box_chart.configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_view(
            strokeWidth=0
        )
    
    st.altair_chart(combined_chart, width='stretch')
    
    # Show outlier values if not too many
    if n_outliers > 0 and n_outliers <= 50:
        with st.expander(f"🔍 View Outlier Values ({n_outliers} total)", expanded=False):
            outlier_display = pd.DataFrame({
                column: outliers.values,
                'Index': outliers.index
            }).sort_values(by=column)
            st.dataframe(outlier_display, hide_index=True)
    elif n_outliers > 50:
        st.info(f"ℹ️ {n_outliers} outliers detected. Too many to display individually. Use the box plot for visualization.")


@st.cache_data
def create_canada_map_chart(df: pd.DataFrame):
    """
    Create an interactive Altair map with aggregated customer data.
    Cached to avoid recalculation.
    
    Args:
        df: Customer DataFrame with location columns
        
    Returns:
        Altair chart object
    """
    # Aggregate by city - count customers per location
    map_summary = df.groupby(['City', 'Province or State']).agg({
        'Latitude': 'mean',
        'Longitude': 'mean'
    }).reset_index()
    map_summary['Customers'] = df.groupby(['City', 'Province or State']).size().values
    map_summary.columns = ['City', 'Province', 'Latitude', 'Longitude', 'Customers']
    
    # Load Canada GeoJSON for background
    canada_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson"
    
    # Create background map
    background = alt.Chart(
        alt.Data(url=canada_url, format=alt.DataFormat(property='features', type='json'))
    ).mark_geoshape(
        fill='#e8f4f8',
        stroke='#2c5aa0',
        strokeWidth=1.5
    ).project(
        type='mercator'
    ).properties(
        width=800,
        height=500
    )
    
    # Create scatter map with customer data
    scatter_map = alt.Chart(map_summary).mark_circle(
        opacity=0.85,
        stroke='#1f4788',
        strokeWidth=1
    ).encode(
        longitude='Longitude:Q',
        latitude='Latitude:Q',
        size=alt.Size(
            'Customers:Q',
            scale=alt.Scale(range=[50, 800], type='sqrt'),
            legend=alt.Legend(title='Customers per City')
        ),
        color=alt.Color(
            'Customers:Q',
            scale=alt.Scale(scheme='orangered'),
            legend=alt.Legend(title='Customer Count')
        ),
        tooltip=[
            alt.Tooltip('City:N', title='City'),
            alt.Tooltip('Province:N', title='Province'),
            alt.Tooltip('Customers:Q', title='Customer Count', format=',')
        ]
    ).project(
        type='mercator'
    )
    
    # Combine and make interactive
    combined_map = (background + scatter_map).properties(
        title="Customer Distribution Across Canada"
    ).configure_view(
        strokeWidth=0
    ).interactive()
    
    return combined_map


@st.cache_data
def prepare_table_data(df: pd.DataFrame):
    """
    Prepare data for geographic tables. Cached to avoid recalculation.
    
    Args:
        df: Customer DataFrame
        
    Returns:
        tuple: (city_counts, province_counts)
    """
    province_counts = df['Province or State'].value_counts().reset_index()
    province_counts.columns = ['province', 'count']
    
    city_counts = df.groupby(['City', 'Province or State']).size().reset_index(name='count')
    city_counts = city_counts.sort_values('count', ascending=False)
    
    return city_counts, province_counts


def display_geographic_tables(df: pd.DataFrame, city_counts: pd.DataFrame, province_counts: pd.DataFrame):
    """
    Display interactive tables showing geographic distribution of customers.
    
    Args:
        df: Original customer DataFrame
        city_counts: DataFrame with city-level customer counts
        province_counts: DataFrame with province-level customer counts
    """
    
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        selected_province = st.selectbox(
            "Filter by Province/State",
            options=["All"] + sorted(df['Province or State'].dropna().unique().tolist()),
            help="Select a province to see detailed city breakdown"
        )
    
    # Filter data based on selection
    if selected_province != "All":
        filtered_data = df[df['Province or State'] == selected_province]
        
        with filter_col2:
            cities_in_province = sorted(filtered_data['City'].dropna().unique().tolist())
            selected_city = st.selectbox(
                "Filter by City",
                options=["All"] + cities_in_province,
                help="Select a city to see detailed information"
            )
        
        if selected_city != "All":
            filtered_data = filtered_data[filtered_data['City'] == selected_city]
    else:
        filtered_data = df
        with filter_col2:
            st.selectbox(
                "Filter by City",
                options=["All"],
                disabled=True,
                help="Select a province first"
            )
    
    st.write("")
    
    # Display filtered results in two columns
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### Top 10 Cities by Customer Count")
        
        if selected_province != "All":
            display_cities = city_counts[city_counts['Province or State'] == selected_province].head(10)
            st.caption(f"Showing cities in {selected_province}")
        else:
            display_cities = city_counts.head(10)
            st.caption("Showing all cities")
        
        # Add percentage column
        display_cities_table = display_cities.copy()
        display_cities_table['percentage'] = (display_cities_table['count'] / len(filtered_data) * 100).round(2)
        display_cities_table.columns = ['City', 'Province/State', 'Customer Count', 'Percentage (%)']
        
        st.dataframe(display_cities_table, hide_index=True, use_container_width=True)
    
    with col_right:
        st.markdown("#### Provincial Distribution")
        
        if selected_province != "All":
            st.caption(f"Selected: {selected_province}")
            selected_data = province_counts[province_counts['province'] == selected_province]
            st.metric("Customers in Province", f"{selected_data.iloc[0]['count']:,}")
            st.metric("Percentage of Total", f"{(selected_data.iloc[0]['count'] / len(df) * 100):.2f}%")
        else:
            display_provinces = province_counts.head(10).copy()
            display_provinces['percentage'] = (display_provinces['count'] / len(df) * 100).round(2)
            display_provinces.columns = ['Province/State', 'Customer Count', 'Percentage (%)']
            st.dataframe(display_provinces, hide_index=True, use_container_width=True)


def plot_canada_map(df: pd.DataFrame):
    """
    Display an interactive Altair map showing customer distribution across Canada.
    
    Args:
        df: Customer DataFrame with Latitude, Longitude, Province, and City columns
    """
    # Validate required columns
    required_cols = ['Latitude', 'Longitude', 'Province or State', 'City']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return
    
    # Remove rows with missing coordinates
    clean_df = df.dropna(subset=['Latitude', 'Longitude'])
    
    if len(clean_df) == 0:
        st.warning("No valid coordinate data available for mapping.")
        return
    
    # Prepare table data (cached)
    city_counts, province_counts = prepare_table_data(df)
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}", help="Total number of customers")
    
    with col2:
        st.metric("Provinces/States", f"{len(province_counts)}", help="Number of unique provinces/states")
    
    with col3:
        st.metric("Cities", f"{df['City'].nunique()}", help="Number of unique cities")
    
    with col4:
        top_city = city_counts.iloc[0]
        st.metric("Top City", top_city['City'], delta=f"{top_city['count']:,} customers")
    
    st.write("")
    
    # Display the interactive Altair map
    st.markdown("#### 🗺️ Customer Geographic Distribution")
    
    # Create and display the map (cached)
    map_chart = create_canada_map_chart(clean_df)
    st.altair_chart(map_chart, width='stretch')
    
    st.write("")
    
    # Display interactive tables
    display_geographic_tables(df, city_counts, province_counts)
