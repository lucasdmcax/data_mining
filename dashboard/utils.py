"""
Utility functions for the AIAI Customer Analytics Dashboard
"""

import streamlit as st
import pandas as pd
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
    Plot data distribution based on variable type using Streamlit native plots.
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
    
    # Display title if provided
    if title:
        st.write(f"**{title}**")
    else:
        st.write(f"**Distribution of {series.name}**")
    
    # Plot based on variable type
    if variable_type == 'discrete':
        # Bar chart for discrete variables, sorted by count (descending)
        value_counts = series.value_counts()
        st.bar_chart(value_counts, x_label=series.name, y_label='Count')
        
    elif variable_type == 'continuous':
        # Histogram for continuous variables using 30 bins
        hist_data, bin_edges = pd.cut(series.dropna(), bins=30, retbins=True)
        hist_counts = hist_data.value_counts().sort_index()
        
        # Create DataFrame with proper column name for the histogram
        hist_df = pd.DataFrame({
            series.name: [round((bin_edges[i] + bin_edges[i+1]) / 2, -3) for i in range(len(bin_edges)-1)],
            'Count': hist_counts.values
        })
        hist_df = hist_df.set_index(series.name)
        
        st.bar_chart(hist_df, y_label='Count')
        
    else:
        st.error(f"Invalid variable_type: {variable_type}. Must be 'discrete' or 'continuous'.")
