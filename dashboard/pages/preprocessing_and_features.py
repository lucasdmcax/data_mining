"""
Preprocessing and Feature Engineering page.
Replicates the preprocessing steps from the notebook.
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path to import styles and utils
sys.path.append(str(Path(__file__).parent.parent))
from styles import get_custom_css, get_metric_html, get_info_box_html
from cluster_utils import create_model_df, detect_outliers

# Page configuration
st.set_page_config(
    page_title="Preprocessing & Features - AIAI Analytics",
    page_icon="⚙️",
    layout="wide"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">⚙️ Preprocessing & Features</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Feature documentation, preprocessing pipeline, and correlation analysis</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📚 Feature Documentation", "🔄 Preprocessing & Correlation", "📋 View Dataframes"])

# ==========================================
# TAB 1: FEATURE DOCUMENTATION
# ==========================================
with tab1:
    st.markdown("### Feature Documentation")
    st.markdown("""
    The `model_df` contains a mix of behavioral (flight activity) and profile (demographic) features. All numeric features have been standardized (Z-score scaling) unless otherwise noted.

    | Feature Name | Formula / Transformation | Rationale |
    | :--- | :--- | :--- |
    | **Education** | `Ordinal Encoding (12-21)` → `Standardized` | Converts categorical education levels into a numerical hierarchy (e.g., Bachelor=16) to reflect the progression of attainment. |
    | **CLV_log** | `log(Customer Lifetime Value + 1)` → `Standardized` | Log-transformation reduces the skewness of monetary values, making the distribution more normal for clustering algorithms. |
    | **Income_log** | `log(Income + 1)` → `Standardized` | Log-transformation handles the wide range and right-skew of income data, minimizing the impact of extreme high earners. |
    | **TenureMonths** | `(2022-01-01 - EnrollmentDate) / 30.44` → `Standardized` | Calculates the duration of the customer relationship in months from a fixed reference date, serving as a proxy for loyalty. |
    | **MeanPointsUtilization** | `Mean(PointsRedeemed / PointsAccumulated)` → `Standardized` | Represents the average propensity of a customer to use their points per flight, indicating engagement with the rewards program. |
    | **TotalFlights_log** | `log(sum(NumFlights) + 1)` → `Standardized` | Aggregates total flight frequency. Log-transform compresses the scale, preventing frequent flyers from dominating the distance metrics. |
    | **TotalDistanceKM_log** | `log(sum(DistanceKM) + 1)` → `Standardized` | Aggregates total distance flown. Log-transform manages the large variance between short-haul and long-haul travelers. |
    | **TotalPointsAccumulated_log** | `log(sum(PointsAccumulated) + 1)` → `Standardized` | Measure of total value earned. Log-transform normalizes the distribution of points earning. |
    | **TotalPointsRedeemed_log** | `log(sum(PointsRedeemed) + 1)` → `Standardized` | Measure of total value burned. Log-transform normalizes the distribution of points usage. |
    | **AverageFlightDistance** | `TotalDistanceKM / TotalFlights` → `Standardized` | Distinguishes between customers who take many short trips vs. those who take fewer long-haul trips. |
    | **LoyaltyStatus_Nova** | `Binary (1 if Nova, 0 otherwise)` | One-hot encoded flag for the 'Nova' tier. (Base category 'Aurora' is dropped). |
    | **LoyaltyStatus_Star** | `Binary (1 if Star, 0 otherwise)` | One-hot encoded flag for the 'Star' tier. |
    | **Location Code_Suburban** | `Binary (1 if Suburban, 0 otherwise)` | One-hot encoded flag for Suburban residence. (Base category 'Rural' is dropped). |
    | **Location Code_Urban** | `Binary (1 if Urban, 0 otherwise)` | One-hot encoded flag for Urban residence. |
    | **CancelledFlag** | `Binary (1 if CancellationDate exists, 0 otherwise)` | Direct indicator of churn. Kept as binary (0/1) without scaling. |
    | **Marital Status** | `Binary (1 if Not Married, 0 if Married)` | Simplified demographic flag to distinguish single/divorced/widowed customers from married ones. |
    | **Gender** | `Binary (1 if Female, 0 if Male)` | Encoded demographic variable. |

    ### Imputation Strategy

    The data preparation pipeline employs a targeted imputation strategy to handle missing values without introducing significant bias. For demographic variables like **Income** and **Customer Lifetime Value**, a **Group-Median Imputation** was performed based on `LoyaltyStatus`. This assumes that customers within the same loyalty tier share similar financial characteristics, providing a more accurate estimate than a global median. For categorical variables, **Education** was imputed with the mode (Bachelor/16 years), and **Gender** was filled with a default value (Male/0) where missing. Finally, for the flight behavior features, any missing values resulting from the aggregation (e.g., customers with no flight history) were filled with **0**, logically reflecting a lack of activity.
    """)

# ==========================================
# TAB 2: PREPROCESSING & CORRELATION
# ==========================================
with tab2:
    st.markdown("### Preprocessing Pipeline & Correlation Analysis")
    
    st.markdown(
        get_info_box_html(
            "Pipeline Instructions",
            "Click the button below to run the full preprocessing pipeline. This will:<br>"
            "1. Load raw data<br>"
            "2. Remove duplicate customers<br>"
            "3. Engineer features and create the model dataframe<br>"
            "4. Detect and remove outliers<br>"
            "5. Perform correlation analysis and drop highly correlated features (> 0.8)"
        ),
        unsafe_allow_html=True
    )
    
    if st.button("🚀 Run Preprocessing Pipeline", type="primary"):
        with st.spinner("Running preprocessing pipeline..."):
            try:
                # 1. Load Data
                FLIGHTS_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "DM_AIAI_FlightsDB.csv"
                CUSTOMERS_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "DM_AIAI_CustomerDB.csv"
                
                customers_db = pd.read_csv(CUSTOMERS_DATA_PATH, index_col=0)
                flights_db = pd.read_csv(FLIGHTS_DATA_PATH)
                
                # 2. Remove Duplicates
                initial_rows = customers_db.shape[0]
                customers_db = customers_db.drop_duplicates(subset=['Loyalty#'])
                dropped_rows = initial_rows - customers_db.shape[0]
                
                st.success(f"✅ Data Loaded. Dropped {dropped_rows} duplicate customers.")
                
                # 3. Create Model DF
                model_df = create_model_df(customers_db, flights_db)
                st.success("✅ Model Dataframe Created.")
                
                # Store intermediate results
                st.session_state['raw_customers_db'] = customers_db
                st.session_state['raw_flights_db'] = flights_db
                st.session_state['initial_model_df'] = model_df.copy()
                
                # 4. Correlation Analysis
                st.markdown("#### 1. Initial Correlation Matrix")
                fig, ax = plt.subplots(figsize=(12, 10))
                initial_corr = model_df.corr()
                sns.heatmap(initial_corr, annot=False, cmap='coolwarm', center=0, ax=ax)
                st.pyplot(fig)
                
                # Identify high correlations
                corr_matrix = model_df.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
                
                st.warning(f"⚠️ Dropping highly correlated features (> 0.8): {to_drop}")
                
                # Drop features
                model_df_final = model_df.drop(columns=to_drop)
                
                # 5. Outlier Detection
                st.markdown("#### 2. Outlier Detection (DBSCAN)")
                model_df_clipped, outliers_df, outlier_count = detect_outliers(model_df_final)
                
                st.write(f"**Outliers detected:** {outlier_count.get(-1, 0)}")
                st.write(f"**Core customers kept:** {len(model_df_clipped):,}")
                
                # Store final results
                st.session_state['model_df'] = model_df_final
                st.session_state['model_df_clipped'] = model_df_clipped
                st.session_state['outliers_df'] = outliers_df
                
                # 6. Final Correlation Matrix (on clipped data)
                st.markdown("#### 3. Final Correlation Matrix (Clipped Data)")
                fig2, ax2 = plt.subplots(figsize=(12, 10))
                sns.heatmap(model_df_clipped.corr(), annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax2)
                st.pyplot(fig2)
                
                st.success("✅ Pipeline Completed! Dataframes are ready for viewing.")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

# ==========================================
# TAB 3: VIEW DATAFRAMES
# ==========================================
with tab3:
    st.markdown("### View Generated Dataframes")
    
    if 'model_df' in st.session_state:
        df_options = {
            "Final Model Dataframe (model_df)": st.session_state['model_df'],
            "Clipped Model Dataframe (No Outliers)": st.session_state.get('model_df_clipped'),
            "Outliers Dataframe": st.session_state.get('outliers_df'),
            "Initial Model Dataframe (Before Drop)": st.session_state.get('initial_model_df'),
            "Raw Customers DB": st.session_state.get('raw_customers_db'),
            "Raw Flights DB": st.session_state.get('raw_flights_db')
        }
        
        selected_df_name = st.selectbox("Select Dataframe to View", list(df_options.keys()))
        
        if selected_df_name:
            df_to_show = df_options[selected_df_name]
            
            st.markdown(f"**Shape:** {df_to_show.shape}")
            st.dataframe(df_to_show)
            
            # Download button
            csv = df_to_show.to_csv(index=True).encode('utf-8')
            st.download_button(
                label=f"Download {selected_df_name} as CSV",
                data=csv,
                file_name=f"{selected_df_name.replace(' ', '_').lower()}.csv",
                mime='text/csv',
            )
    else:
        st.info("⚠️ No dataframes generated yet. Please go to the 'Preprocessing & Correlation' tab and run the pipeline.")
