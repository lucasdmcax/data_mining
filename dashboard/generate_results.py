"""
Script to generate final clustering results and save them to a CSV file.
This avoids re-running the heavy clustering logic every time the dashboard loads.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

# Suppress warnings
warnings.filterwarnings('ignore')

# Add current directory to path to import cluster_utils
sys.path.append(str(Path(__file__).parent))
from cluster_utils import create_model_df, detect_outliers, run_merged_clustering

def main():
    print("🚀 Starting Result Generation...")
    
    # 1. Load Data
    base_path = Path(__file__).parent.parent
    flights_path = base_path / "data" / "DM_AIAI_FlightsDB.csv"
    customers_path = base_path / "data" / "DM_AIAI_CustomerDB.csv"
    
    print(f"📂 Loading data from: {base_path}")
    
    if not flights_path.exists() or not customers_path.exists():
        print("❌ Error: Data files not found!")
        return

    flights_db = pd.read_csv(flights_path)
    customers_db = pd.read_csv(customers_path, index_col=0)
    
    # 2. Preprocessing
    print("🔄 Running Preprocessing Pipeline...")
    customers_db = customers_db.drop_duplicates(subset=['Loyalty#'])
    model_df, model_df_unscaled = create_model_df(customers_db, flights_db)
    
    # Drop correlated features (Logic from notebook/dashboard)
    corr_matrix = model_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    print(f"⚠️ Dropping correlated features: {to_drop}")
    model_df_final = model_df.drop(columns=to_drop)
    
    # Outlier Detection
    print("🕵️ Detecting Outliers...")
    model_df_clipped, _, _ = detect_outliers(model_df_final)
    print(f"✅ Core customers kept: {len(model_df_clipped)}")
    
    # Align unscaled data with clipped data
    model_df_unscaled_clipped = model_df_unscaled.loc[model_df_clipped.index]

    # Save Preprocessed Data
    data_dir = Path(__file__).parent / "data"
    model_df_clipped.to_csv(data_dir / "model_df_clipped.csv")
    model_df_unscaled_clipped.to_csv(data_dir / "model_df_unscaled_clipped.csv")
    print(f"💾 Saved preprocessed data to {data_dir}")
    
    # 3. Define Features (CORRECTED LISTS FROM NOTEBOOK)
    all_behavior_features = [
        'TotalFlights_log', 'TotalDistanceKM_log', 'TotalPointsAccumulated_log', 
        'TotalPointsRedeemed_log', 'MeanPointsUtilization', 'AverageFlightDistance'
    ]
    all_profile_features = [
        'Income_log', 'Education', 'TenureMonths', 'CLV_log'
    ]
    
    # Filter features to those present in the dataframe
    behavior_features = [f for f in all_behavior_features if f in model_df_clipped.columns]
    profile_features = [f for f in all_profile_features if f in model_df_clipped.columns]
    
    print(f"Behavior Features: {behavior_features}")
    print(f"Profile Features: {profile_features}")
    
    if not behavior_features or not profile_features:
        print("❌ Error: Feature lists are empty! Check column names.")
        return

    # 4. Run Merged Clustering
    print("🧠 Running Merged Clustering Model (MeanShift + Hierarchical)...")
    try:
        final_labels = run_merged_clustering(model_df_clipped, behavior_features, profile_features)
        
        # 5. Calculate Projections (3D)
        print("🔮 Calculating 3D Projections (PCA, t-SNE, UMAP)...")
        
        # Prepare data for projection
        X = model_df_clipped[behavior_features + profile_features]
        
        # PCA
        print("   - PCA...")
        pca = PCA(n_components=3, random_state=42)
        pca_res = pca.fit_transform(X)
        
        # t-SNE
        print("   - t-SNE...")
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        tsne_res = tsne.fit_transform(X)
        
        # UMAP
        print("   - UMAP...")
        umap_model = UMAP(n_components=3, random_state=42, n_jobs=1) # n_jobs=1 to avoid issues on some systems
        umap_res = umap_model.fit_transform(X)
        
        # 6. Save Results
        output_path = Path(__file__).parent / "data" / "final_clustering_results.csv"
        
        # Create a dataframe with the index and labels
        results_df = pd.DataFrame(index=model_df_clipped.index)
        results_df['Cluster'] = final_labels
        
        # Add projections
        results_df['PCA1'] = pca_res[:, 0]
        results_df['PCA2'] = pca_res[:, 1]
        results_df['PCA3'] = pca_res[:, 2]
        
        results_df['TSNE1'] = tsne_res[:, 0]
        results_df['TSNE2'] = tsne_res[:, 1]
        results_df['TSNE3'] = tsne_res[:, 2]
        
        results_df['UMAP1'] = umap_res[:, 0]
        results_df['UMAP2'] = umap_res[:, 1]
        results_df['UMAP3'] = umap_res[:, 2]
        
        results_df.to_csv(output_path)
        print(f"✅ Success! Results and projections saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error running clustering: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
