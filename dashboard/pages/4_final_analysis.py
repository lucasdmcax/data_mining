"""
Final Analysis Page.
Displays the results of the final merged clustering model (MeanShift + Hierarchical).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path to import styles and utils
sys.path.append(str(Path(__file__).parent.parent))
from styles import get_custom_css
from cluster_utils import run_merged_clustering

# Page configuration
st.set_page_config(
    page_title="Final Analysis - AIAI Analytics",
    page_icon="📊",
    layout="wide"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 Final Clustering Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Results from the optimized merged clustering model</div>', unsafe_allow_html=True)

# Check for data
if 'model_df_clipped' not in st.session_state:
    # Try to load pre-calculated data
    data_dir = Path(__file__).parent.parent / "data"
    model_df_path = data_dir / "model_df_clipped.csv"
    unscaled_df_path = data_dir / "model_df_unscaled_clipped.csv"
    
    if model_df_path.exists():
        try:
            # Load model_df
            model_df = pd.read_csv(model_df_path, index_col=0)
            st.session_state['model_df_clipped'] = model_df
            
            # Load unscaled df if available
            if unscaled_df_path.exists():
                model_df_unscaled = pd.read_csv(unscaled_df_path, index_col=0)
                st.session_state['model_df_unscaled'] = model_df_unscaled
                
            st.success("✅ Loaded pre-calculated feature data.")
            
        except Exception as e:
            st.warning(f"Could not load pre-calculated data: {e}")
            st.warning("⚠️ Please go to the 'Preprocessing & Features' page and run the pipeline first.")
            st.stop()
    else:
        st.warning("⚠️ No preprocessed data found. Please go to the 'Preprocessing & Features' page and run the pipeline first.")
        st.stop()

# Load data
model_df = st.session_state['model_df_clipped']

# Define Feature Sets (Corrected based on notebook analysis)
all_behavior_features = [
    'TotalFlights_log', 'TotalDistanceKM_log', 'TotalPointsAccumulated_log', 
    'TotalPointsRedeemed_log', 'MeanPointsUtilization', 'AverageFlightDistance'
]
all_profile_features = [
    'Income_log', 'Education', 'TenureMonths', 'CLV_log'
]

# Filter features
behavior_features = [f for f in all_behavior_features if f in model_df.columns]
profile_features = [f for f in all_profile_features if f in model_df.columns]
selected_features = behavior_features + profile_features
X = model_df[selected_features]

# Main Content
st.markdown("""
This page presents the results of the final clustering model derived from our analysis.
The model combines **MeanShift** (on behavior and profile features separately) followed by **Hierarchical Clustering** on the centroids.
""")

# Check if we need to run the model
if 'final_labels' not in st.session_state:
    # Try to load pre-calculated results first
    results_path = Path(__file__).parent.parent / "data" / "final_clustering_results.csv"
    
    if results_path.exists():
        try:
            results_df = pd.read_csv(results_path, index_col=0)
            # Ensure index matches model_df (intersection)
            common_idx = model_df.index.intersection(results_df.index)
            
            if len(common_idx) > 0:
                final_labels = results_df.loc[common_idx, 'Cluster']
                st.session_state['final_labels'] = final_labels
                st.success("✅ Loaded pre-calculated clustering results.")
            else:
                st.warning("⚠️ Pre-calculated results indices do not match current data. Re-running model...")
                raise ValueError("Index mismatch")
                
        except Exception as e:
            st.warning(f"Could not load pre-calculated results: {e}. Running model live...")
            
    # If still not loaded, run the model
    if 'final_labels' not in st.session_state:
        with st.spinner("Running final merged clustering model..."):
            try:
                final_labels = run_merged_clustering(model_df, behavior_features, profile_features)
                st.session_state['final_labels'] = final_labels
                
            except Exception as e:
                st.error(f"Error running merged model: {e}")
                st.stop()

    # Calculate Metrics & Projections (if not already done)
    if 'final_sil' not in st.session_state:
        final_labels = st.session_state['final_labels']
        # Align X with labels
        X_aligned = X.loc[final_labels.index]
        
        sil_score = silhouette_score(X_aligned, final_labels)
        db_score = davies_bouldin_score(X_aligned, final_labels)
        st.session_state['final_sil'] = sil_score
        st.session_state['final_db'] = db_score
        
        # Run PCA for visualization
        pca = PCA(n_components=3)
        proj = pca.fit_transform(X_aligned)
        st.session_state['final_viz_data'] = pd.DataFrame(proj, columns=['Comp1', 'Comp2', 'Comp3'], index=X_aligned.index)

# Display Results
labels = st.session_state['final_labels']
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Metrics Row
m1, m2, m3 = st.columns(3)
m1.metric("Number of Clusters", n_clusters)
m2.metric("Silhouette Score", f"{st.session_state['final_sil']:.3f}")
m3.metric("Davies-Bouldin", f"{st.session_state['final_db']:.3f}")

# Tabs for Visualization and Profiling
tab1, tab2, tab3 = st.tabs(["🔮 3D Visualization", "📈 Cluster Profiling", "📋 Recommendations"])

with tab1:
    st.markdown("### 3D Cluster Projection (PCA)")
    if 'final_viz_data' in st.session_state:
        viz_df = st.session_state['final_viz_data'].copy()
        viz_df['Cluster'] = labels.astype(str)
        viz_df['Loyalty#'] = model_df.index
        
        fig = px.scatter_3d(
            viz_df, x='Comp1', y='Comp2', z='Comp3',
            color='Cluster',
            hover_data=['Loyalty#'],
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.7,
            title="3D PCA Projection of Customer Segments"
        )
        fig.update_layout(height=700, margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Cluster Characteristics")
    
    # Prepare data for profiling (Unscaled if available)
    if 'model_df_unscaled' in st.session_state:
        raw_df = st.session_state['model_df_unscaled']
        
        # Map features
        cols_to_profile = []
        for col in selected_features:
            if col in raw_df.columns:
                cols_to_profile.append(col)
            elif col.endswith('_log') and col[:-4] in raw_df.columns:
                cols_to_profile.append(col[:-4])
        
        common_index = X.index.intersection(raw_df.index)
        profile_df = raw_df.loc[common_index, cols_to_profile].copy()
        profile_df['Cluster'] = labels
        
        title_suffix = "(Original Values)"
        text_fmt = ".0f"
    else:
        profile_df = X.copy()
        profile_df['Cluster'] = labels
        title_suffix = "(Scaled)"
        text_fmt = ".2f"
    
    # Calculate means
    cluster_means = profile_df.groupby('Cluster').mean()
    
    # Heatmap
    st.markdown(f"#### Feature Means by Cluster {title_suffix}")
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
    
    # Parallel Coordinates
    st.markdown("#### Parallel Coordinates")
    
    # Sample for plot clarity
    if len(profile_df) > 500:
        plot_sample = profile_df.sample(500, random_state=42)
    else:
        plot_sample = profile_df
        
    fig_par = px.parallel_coordinates(
        plot_sample, 
        color="Cluster",
        dimensions=profile_df.columns[:-1], # Exclude Cluster column itself
        color_continuous_scale=px.colors.diverging.Tealrose,
    )
    st.plotly_chart(fig_par, use_container_width=True)
    
    # Feature Importance (Random Forest)
    st.markdown("#### Feature Importance (Random Forest)")
    if st.button("Run Feature Importance Analysis"):
        with st.spinner("Training Random Forest Classifier..."):
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, labels)
            
            importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_imp = px.bar(
                importances, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                title="Feature Importance for Predicting Clusters"
            )
            st.plotly_chart(fig_imp, use_container_width=True)

with tab3:
    st.markdown("### Strategic Recommendations")
    
    st.markdown("""
    Based on the analysis of the 6 identified clusters, here are the strategic recommendations.
    *(Note: Cluster IDs below correspond to the notebook analysis. Please verify if the current run matches these IDs by checking the Profiling tab.)*
    """)
    
    st.info("""
    **Cluster Mapping Guide (Check Profiling Tab):**
    *   **Cluster 4 (Core Standard):** Largest group (~60%), average behavior.
    *   **Cluster 2 (Budget Active):** Lowest Income, but high flight activity.
    *   **Cluster 0 (Inactive):** Lowest flight activity, high churn risk.
    *   **Cluster 5 (Affluent Frequent):** High income, high flights, lower education.
    *   **Cluster 1 (Educated Frequent):** High education, high income, high flights.
    *   **Cluster 3 (Churned Elite):** High income/education, but inactive.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟢 Retention & Growth")
        st.markdown("""
        **Core Standard Customers (Cluster 4)**
        *   **Profile:** The backbone of the airline. Steady activity, average income.
        *   **Action:** **Retention is key.** Ensure consistent service, recognize loyalty with milestone rewards, and cross-sell ancillary services (hotels, car rentals).
        
        **Budget Active Flyers (Cluster 2)**
        *   **Profile:** Low income but highly active "Road Warriors".
        *   **Action:** Ensure competitive pricing on long-haul routes. Offer "value-for-money" redemption options to prevent switching to budget carriers.
        """)
        
    with col2:
        st.markdown("#### 💎 High Value & Premium")
        st.markdown("""
        **Educated Frequent Flyers (Cluster 1)**
        *   **Profile:** High education, high income, frequent flyers.
        *   **Action:** Offer premium partnerships (lounge access, business upgrades) and personalized rewards appealing to a sophisticated lifestyle.
        
        **Affluent Frequent Flyers (Cluster 5)**
        *   **Profile:** High income, frequent flyers, practical travelers.
        *   **Action:** Focus on efficiency (fast-track security, priority boarding). They value time and convenience over purely status-based perks.
        """)
        
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 🔴 At Risk / Churn")
        st.markdown("""
        **Inactive / Churned Low-Value (Cluster 0)**
        *   **Profile:** Stopped flying, short trips, high cancellation rate.
        *   **Action:** Focus on exit surveys. Re-engagement may be difficult; prioritize resources elsewhere.
        
        **Churned Elite (Cluster 3)**
        *   **Profile:** High potential (income/edu) but inactive. Highest churn.
        *   **Action:** **Critical.** Investigate why they left. Targeted "We Miss You" campaigns with premium incentives are needed for this small, high-value group.
        """)
        
    with col4:
        st.markdown("#### 📊 Segment Overview")
        st.markdown("""
        | Cluster Label | Key Characteristics | Risk Level |
        | :--- | :--- | :--- |
        | **Core Standard** | Baseline income, steady activity | Low |
        | **Budget Active** | Low income, high activity, long distance | Low |
        | **Educated Frequent** | High education, high redemption | Low |
        | **Affluent Frequent** | High income, practical travelers | Very Low |
        | **Inactive** | Disengaged, short trips | **High** |
        | **Churned Elite** | High value profile, stopped flying | **Critical** |
        """)
