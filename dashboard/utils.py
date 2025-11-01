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
    st.altair_chart(chart.configure_axis(labelLimit=500), use_container_width=True)
