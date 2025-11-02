"""
Utility functions for the AIAI Customer Analytics Dashboard
"""

import streamlit as st
import pandas as pd
import altair as alt
import json
from pathlib import Path

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
