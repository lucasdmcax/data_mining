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
from cluster_utils import run_merged_clustering, load_preprocessed_data, apply_pca_2d, apply_tsne_2d, apply_umap_2d

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
    # Try to load from CSV
    model_df = load_preprocessed_data()
    
    if model_df is not None:
        st.session_state['model_df_clipped'] = model_df
        st.success("✅ Loaded pre-calculated feature data.")
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
                
                # Load Projections if available
                if 'PCA1' in results_df.columns:
                    st.session_state['final_proj_PCA'] = results_df.loc[common_idx, ['PCA1', 'PCA2', 'PCA3']]
                if 'TSNE1' in results_df.columns:
                    st.session_state['final_proj_t-SNE'] = results_df.loc[common_idx, ['TSNE1', 'TSNE2', 'TSNE3']]
                if 'UMAP1' in results_df.columns:
                    st.session_state['final_proj_UMAP'] = results_df.loc[common_idx, ['UMAP1', 'UMAP2', 'UMAP3']]
                    
                st.success("✅ Loaded pre-calculated clustering results and projections.")
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

    # Calculate Metrics (if not already done)
    if 'final_sil' not in st.session_state:
        final_labels = st.session_state['final_labels']
        # Align X with labels
        X_aligned = X.loc[final_labels.index]
        
        sil_score = silhouette_score(X_aligned, final_labels)
        db_score = davies_bouldin_score(X_aligned, final_labels)
        st.session_state['final_sil'] = sil_score
        st.session_state['final_db'] = db_score

# Display Results
labels = st.session_state['final_labels']
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Metrics Row
m1, m2, m3 = st.columns(3)
m1.metric("Number of Clusters", n_clusters)
m2.metric("Silhouette Score", f"{st.session_state['final_sil']:.3f}")
m3.metric("Davies-Bouldin", f"{st.session_state['final_db']:.3f}")

# Tabs for Visualization and Profiling
tab1, tab2, tab3 = st.tabs(["🔮 Cluster Projection", "📈 Cluster Profiling", "📋 Recommendations"])

with tab1:
    st.markdown("### Cluster Projection (3D)")
    
    # Projection Method Selector
    proj_method = st.selectbox("Select Projection Method", ["PCA", "t-SNE", "UMAP"], index=0)
    
    # Check if projection is loaded
    proj_key = f'final_proj_{proj_method}'
    
    if proj_key not in st.session_state:
        st.warning(f"⚠️ {proj_method} projection not found in pre-calculated data. Please run 'generate_results.py' to generate it.")
        
        if proj_method in ["t-SNE", "UMAP"]:
             st.warning(f"⚠️ {proj_method} projection can take a significant amount of time to compute.")
             
        # Fallback to calculating it live (2D or 3D? User asked for 3D)
        with st.spinner(f"Calculating {proj_method} projection (Live)..."):
             # Align X with labels
            X_aligned = X.loc[labels.index]
            if proj_method == "PCA":
                pca = PCA(n_components=3)
                proj = pca.fit_transform(X_aligned)
                cols = ['PCA1', 'PCA2', 'PCA3']
            elif proj_method == "t-SNE":
                tsne = TSNE(n_components=3, random_state=42)
                proj = tsne.fit_transform(X_aligned)
                cols = ['TSNE1', 'TSNE2', 'TSNE3']
            elif proj_method == "UMAP":
                umap_model = UMAP(n_components=3, random_state=42)
                proj = umap_model.fit_transform(X_aligned)
                cols = ['UMAP1', 'UMAP2', 'UMAP3']
            
            st.session_state[proj_key] = pd.DataFrame(proj, columns=cols, index=X_aligned.index)

    # Plot
    viz_df = st.session_state[proj_key].copy()
    viz_df['Cluster'] = labels.astype(str)
    viz_df['Loyalty#'] = model_df.index
    
    cols = viz_df.columns
    x_col, y_col, z_col = cols[0], cols[1], cols[2]
    
    fig = px.scatter_3d(
        viz_df, x=x_col, y=y_col, z=z_col,
        color='Cluster',
        hover_data=['Loyalty#'],
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.7,
        title=f"3D {proj_method} Projection of Customer Segments"
    )
    fig.update_layout(height=700, margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Cluster Characteristics")
    
    # Prepare data for profiling (Scaled)
    profile_df = X.copy()
    profile_df['Cluster'] = labels
    title_suffix = "(Scaled)"
    text_fmt = ".2f"
    
    # Calculate means
    cluster_means = profile_df.groupby('Cluster').mean()
    
    # 1. Absolute Means Heatmap
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
    
    # 2. Non-Metric Features Heatmap
    st.markdown("#### Non-Metric Features Distribution")
    st.markdown("Heatmap showing the percentage of each category within each cluster.")
    
    # Use model_df_clipped for non-metric features (OHE/Binary)
    non_metric_cols = [
        'LoyaltyStatus_Nova', 'LoyaltyStatus_Star', 
        'Location Code_Suburban', 'Location Code_Urban', 
        'CancelledFlag', 'Marital Status', 'Gender'
    ]
    # Filter to those present
    available_non_metric = [c for c in non_metric_cols if c in model_df.columns]
    
    if available_non_metric:
        nm_df = model_df.loc[X.index, available_non_metric].copy()
        nm_df['Cluster'] = labels
        
        # Group by cluster and calculate mean (proportion)
        nm_means = nm_df.groupby('Cluster').mean()
        
        fig_cat = px.imshow(
            nm_means.T,
            labels=dict(x="Cluster", y="Feature Category", color="Proportion"),
            x=nm_means.index.astype(str),
            y=nm_means.columns,
            color_continuous_scale="Blues",
            aspect="auto",
            text_auto=".2f"
        )
        fig_cat.update_layout(height=600)
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("No non-metric features found in data.")
    
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
