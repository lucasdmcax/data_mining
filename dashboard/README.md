# AIAI Customer Analytics Dashboard

An interactive Streamlit application for visualizing customer segmentation results.

## Quick Start

1. **Generate Data & Projections** (Optional but Recommended)
   Run this script to pre-calculate clustering results and 3D projections (PCA, t-SNE, UMAP). This significantly speeds up the dashboard.
   ```bash
   python generate_results.py
   ```

2. **Launch Dashboard**
   ```bash
   streamlit run home.py
   ```

## Structure

```
dashboard/
├── home.py                         # Main landing page
├── generate_results.py             # Script to pre-calculate models and projections
├── cluster_utils.py                # Shared clustering and data loading logic
├── styles.py                       # Centralized CSS styling
├── pages/
│   ├── 1_exploratory_data_analysis.py  # EDA visualizations
│   ├── 2_preprocessing_and_features.py # Feature engineering documentation
│   ├── 3_clustering_playground.py      # Interactive clustering (K-Means, DBSCAN, etc.)
│   └── 4_final_analysis.py             # Final merged model results & profiling
└── data/                           # Stores preprocessed CSVs and results
```

## 📊 Key Features
- **Interactive Playground**: Test different clustering algorithms and parameters in real-time.
- **3D Visualization**: Explore customer segments using 3D PCA, t-SNE, and UMAP projections.
- **Cluster Profiling**: Analyze clusters using feature heatmaps and categorical distribution plots.
- **Strategic Recommendations**: Actionable business insights based on the final segmentation.
- Data scope metrics
- Methodology overview
- Compact team member display
- Navigation guide

### Update Team Members
Edit student names in:
- `home.py` (lines with "Student Name 1-4")
- `pages/about_us.py` (lines with "Student Name 1-4")

### Add Real Profile Pictures
Replace the placeholder emoji (👤) with:
```python
st.image("path/to/image.jpg", width=150)
```

### Modify Styles
Edit `styles.py` to change colors, fonts, or layouts globally.
