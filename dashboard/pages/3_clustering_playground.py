"""
Interactive Clustering Analysis Page.
Allows dynamic parameter tuning, 3D visualization, and hierarchical clustering exploration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path to import styles and utils
sys.path.append(str(Path(__file__).parent.parent))
from styles import get_custom_css, get_metric_html, get_info_box_html
from cluster_utils import run_merged_clustering, apply_pca_2d, apply_tsne_2d, apply_umap_2d

# Page configuration
st.set_page_config(
    page_title="Interactive Clustering - AIAI Analytics",
    page_icon="🔮",
    layout="wide"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔮 Interactive Clustering Playground</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dynamic parameter tuning and multi-view cluster visualization</div>', unsafe_allow_html=True)

# Check for data
if 'model_df_clipped' not in st.session_state:
    st.warning("⚠️ No preprocessed data found. Please go to the 'Preprocessing & Features' page and run the pipeline first.")
    st.stop()

# Load data
model_df = st.session_state['model_df_clipped']

# Define Feature Sets
all_behavior_features = [
    'TotalFlights_log', 'TotalDistanceKM_log', 'TotalPointsAccumulated_log', 
    'TotalPointsRedeemed_log', 'MeanPointsUtilization', 'AverageFlightDistance'
]
all_profile_features = ['Age_log', 'DaysSinceEnrollment_log']

# Filter features to only those present in the dataframe
behavior_features = [f for f in all_behavior_features if f in model_df.columns]
profile_features = [f for f in all_profile_features if f in model_df.columns]

# Sidebar for Feature Selection
with st.sidebar:
    st.markdown("### Feature Selection")
    selected_features = st.multiselect(
        "Select Features for Clustering",
        options=model_df.columns.tolist(),
        default=behavior_features + profile_features
    )
    
    if not selected_features:
        st.error("Please select at least one feature.")
        st.stop()
        
    X = model_df[selected_features]

# Tabs
tab1, tab2 = st.tabs(["🎛️ Dynamic Clustering", "📊 Cluster Profiling"])

# ==========================================
# TAB 1: DYNAMIC CLUSTERING
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Parameter Tuning")
        
        algorithm = st.selectbox("Select Algorithm", ["K-Means", "DBSCAN", "MeanShift"])
        
        if algorithm == "K-Means":
            k_value = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=4)
            params = {'n_clusters': k_value}
            
        elif algorithm == "DBSCAN":
            eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=5.0, value=1.9, step=0.1)
            min_samples = st.slider("Min Samples", min_value=5, max_value=50, value=20)
            params = {'eps': eps, 'min_samples': min_samples}
            
        elif algorithm == "MeanShift":
            quantile = st.slider("Bandwidth Quantile", min_value=0.1, max_value=1.0, value=0.2, step=0.05)
            params = {'quantile': quantile}
        
        st.markdown("### Visualization")
        viz_method = st.selectbox("Projection Method", ["PCA", "t-SNE", "UMAP"])
        
        if st.button(f"Run {algorithm}", type="primary"):
            with st.spinner(f"Running {algorithm} and {viz_method}..."):
                # Run Clustering
                if algorithm == "K-Means":
                    model = KMeans(n_clusters=params['n_clusters'], random_state=42, n_init=10)
                    labels = model.fit_predict(X)
                    
                elif algorithm == "DBSCAN":
                    model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
                    labels = model.fit_predict(X)
                    
                elif algorithm == "MeanShift":
                    bandwidth = estimate_bandwidth(X, quantile=params['quantile'], n_samples=500)
                    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                    labels = model.fit_predict(X)
                
                # Calculate Metrics
                if len(set(labels)) > 1:
                    sil_score = silhouette_score(X, labels)
                    db_score = davies_bouldin_score(X, labels)
                else:
                    sil_score = -1
                    db_score = -1
                
                # Store results
                st.session_state['cluster_labels'] = labels
                st.session_state['cluster_algo'] = algorithm
                st.session_state['cluster_params'] = params
                st.session_state['cluster_sil'] = sil_score
                st.session_state['cluster_db'] = db_score
                st.session_state['viz_method'] = viz_method
                
                # Run Projection for 3D Viz
                if viz_method == "PCA":
                    pca = PCA(n_components=3)
                    proj = pca.fit_transform(X)
                    cols = ['Comp1', 'Comp2', 'Comp3']
                elif viz_method == "t-SNE":
                    from sklearn.manifold import TSNE
                    tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
                    proj = tsne.fit_transform(X)
                    cols = ['Comp1', 'Comp2', 'Comp3']
                elif viz_method == "UMAP":
                    from umap import UMAP
                    umap_model = UMAP(n_components=3, random_state=42)
                    proj = umap_model.fit_transform(X)
                    cols = ['Comp1', 'Comp2', 'Comp3']
                
                st.session_state['viz_data'] = pd.DataFrame(proj, columns=cols, index=X.index)
                
                st.success("Clustering Complete!")
            
    with col2:
        if 'cluster_labels' in st.session_state:
            # Display Metrics
            n_clusters = len(set(st.session_state['cluster_labels'])) - (1 if -1 in st.session_state['cluster_labels'] else 0)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Number of Clusters", n_clusters)
            m2.metric("Silhouette Score", f"{st.session_state['cluster_sil']:.3f}")
            m3.metric("Davies-Bouldin", f"{st.session_state['cluster_db']:.3f}")
            
            # 3D Visualization
            st.markdown(f"### 3D Cluster Visualization ({st.session_state.get('viz_method', 'PCA')})")
            
            if 'viz_data' in st.session_state:
                viz_df = st.session_state['viz_data'].copy()
                viz_df['Cluster'] = st.session_state['cluster_labels'].astype(str)
                viz_df['Loyalty#'] = model_df.index
                
                # Plotly 3D Scatter
                fig = px.scatter_3d(
                    viz_df, x='Comp1', y='Comp2', z='Comp3',
                    color='Cluster',
                    hover_data=['Loyalty#'],
                    title=f"3D Projection ({st.session_state.get('viz_method', 'PCA')})",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    opacity=0.7
                )
                fig.update_layout(height=600, margin=dict(l=0, r=0, b=0, t=40))
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("👈 Adjust parameters and click 'Run' to visualize results.")

# ==========================================
# TAB 2: CLUSTER PROFILING
# ==========================================
with tab2:
    st.markdown("### Cluster Profiling")
    
    if 'cluster_labels' in st.session_state:
        labels = st.session_state['cluster_labels']
        
        # Prepare data for profiling
        # Try to use unscaled data if available for better interpretability (integers)
        if 'model_df_unscaled' in st.session_state:
            # Get unscaled data
            raw_df = st.session_state['model_df_unscaled']
            
            # Map selected features (scaled) to raw features
            # e.g. 'TotalFlights_log' -> 'TotalFlights'
            cols_to_profile = []
            for col in selected_features:
                # Direct match
                if col in raw_df.columns:
                    cols_to_profile.append(col)
                # Log match
                elif col.endswith('_log') and col[:-4] in raw_df.columns:
                    cols_to_profile.append(col[:-4])
                # Fallback: keep scaled if no raw equivalent found (e.g. ratios)
                else:
                    pass
            
            # Create profile dataframe with raw values
            # We use the intersection of indices to be safe
            common_index = X.index.intersection(raw_df.index)
            profile_df = raw_df.loc[common_index, cols_to_profile].copy()
            profile_df['Cluster'] = labels
            
            title_suffix = "(Original Values)"
            text_fmt = ".0f" # Integer format for heatmap
        else:
            # Fallback to scaled data
            profile_df = X.copy()
            profile_df['Cluster'] = labels
            title_suffix = "(Scaled)"
            text_fmt = ".2f"
        
        # Calculate means
        cluster_means = profile_df.groupby('Cluster').mean()
        
        # Heatmap of Feature Means
        st.markdown(f"#### Feature Means by Cluster {title_suffix}")
        
        # Create heatmap with text values
        fig_heat = px.imshow(
            cluster_means.T,
            labels=dict(x="Cluster", y="Feature", color="Mean Value"),
            x=cluster_means.index.astype(str),
            y=cluster_means.columns,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            text_auto=text_fmt
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # Parallel Coordinates Plot
        st.markdown("#### Parallel Coordinates Plot")
        
        # Sample for plot clarity
        if len(profile_df) > 500:
            plot_sample = profile_df.sample(500, random_state=42)
        else:
            plot_sample = profile_df
            
        fig_par = px.parallel_coordinates(
            plot_sample, 
            color="Cluster",
            dimensions=selected_features,
            color_continuous_scale=px.colors.diverging.Tealrose,
        )
        st.plotly_chart(fig_par, use_container_width=True)
        
        # Export
        st.markdown("#### Export Results")
        csv = profile_df.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Download Cluster Assignments (CSV)",
            data=csv,
            file_name="cluster_assignments.csv",
            mime='text/csv',
        )
        
    else:
        st.info("Please run a clustering algorithm in the 'Dynamic Clustering' tab first to see profiles.")
