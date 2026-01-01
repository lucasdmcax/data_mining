import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
from itertools import product
from umap import UMAP
from minisom import MiniSom

def winsorize_dataframe(df, columns, limits=(0.01, 0.01)):
    """
    Apply winsorization to each column in `columns`.
    limits=(lower_pct, upper_pct) means: cap values at the 1st and 99th percentile.

    Returns the winsorized copy of df.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            # winsorize returns masked arrays -> convert to normal array
            df[col] = winsorize(df[col], limits=limits).data
    return df

def preprocess_flights(flights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing steps to the FlightsDB:
    - Winsorize outliers
    - Convert YearMonthDate to datetime
    - Round down NumFlights and NumFlightsWithCompanions
    - Set DistanceKM = 0 where NumFlights == 0
    - Drop DollarCostPointsRedeemed
    - Add log-transformed versions of skewed variables
    - Create PointsUtilizationRatio = PointsRedeemed / PointsAccumulated
    """
    df = flights_df.copy()

    # 0. Winsorize outliers (Flights DB outliers are legitimate but skewed)
    outlier_cols = [
        'NumFlights', 'NumFlightsWithCompanions', 'DistanceKM', 
        'PointsAccumulated', 'PointsRedeemed'
    ]
    df = winsorize_dataframe(df, outlier_cols, limits=(0.01, 0.01))

    # 1. YearMonthDate -> datetime
    if 'YearMonthDate' in df.columns:
        df['YearMonthDate'] = pd.to_datetime(df['YearMonthDate'])

    # 2. Round down flight counts and cast to int
    for col in ['NumFlights', 'NumFlightsWithCompanions']:
        if col in df.columns:
            df[col] = np.floor(df[col]).astype(int)

    # 3. Fix logical inconsistency: DistanceKM must be 0 if NumFlights == 0
    if {'NumFlights', 'DistanceKM'}.issubset(df.columns):
        df.loc[df['NumFlights'] == 0, 'DistanceKM'] = 0

    # 4. Drop perfectly correlated variable
    if 'DollarCostPointsRedeemed' in df.columns:
        df = df.drop(columns=['DollarCostPointsRedeemed'])

    # 5. Log transforms for skewed numeric variables
    log_cols = ['DistanceKM', 'PointsAccumulated', 'PointsRedeemed', 'NumFlights']
    for col in log_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])

    # 6. Points utilisation ratio
    if {'PointsRedeemed', 'PointsAccumulated'}.issubset(df.columns):
        denom = df['PointsAccumulated'].replace({0: np.nan})
        df['PointsUtilizationRatio'] = df['PointsRedeemed'] / denom

    return df

def preprocess_customers(customer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing steps to the CustomerDB:
    - Create cancellation flag from CancellationDate
    - Group-median imputation (by LoyaltyStatus) for Income and Customer Lifetime Value
    - Winsorize outliers (Income, CLV)
    - Log transform Customer Lifetime Value and Income
    - Encode Gender as binary
    """
    df = customer_df.copy()

    # 1. Cancellation flag
    if 'CancellationDate' in df.columns:
        df['CancelledFlag'] = df['CancellationDate'].notna().astype(int)

    # 2. Group-median imputation by LoyaltyStatus
    group_col = 'LoyaltyStatus'
    cols_to_impute = ['Income', 'Customer Lifetime Value']
    for col in cols_to_impute:
        if col in df.columns and group_col in df.columns:
            df[col] = df.groupby(group_col)[col].transform(
                lambda x: x.fillna(x.median())
            )

    # 3. Winsorize outliers
    outlier_cols = ['Income', 'Customer Lifetime Value']
    df = winsorize_dataframe(df, outlier_cols, limits=(0.01, 0.01))

    # 4. Log transforms
    if 'Customer Lifetime Value' in df.columns:
        df['CLV_log'] = np.log1p(df['Customer Lifetime Value'])
    if 'Income' in df.columns:
        df['Income_log'] = np.log1p(df['Income'].clip(lower=0))

    # 5. Gender encoding
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'female': 1, 'male': 0}).fillna(0).astype(int)

    # 6. Education to Years (Ordinal Encoding)
    if 'Education' in df.columns:
        edu_map = {
            'High School or Below': 12,
            'College': 14,
            'Bachelor': 16,
            'Master': 18,
            'Doctor': 21
        }
        df['Education'] = df['Education'].map(edu_map)
        df['Education'] = df['Education'].fillna(16)

    # 7. Turn marital status into a flag
    if 'Marital Status' in df.columns:
        df['Marital Status'] = np.where(df['Marital Status'] != 'Married', 1, 0)

    # 8. Tenure
    ref_date = pd.to_datetime('2022-01-01')
    if 'EnrollmentDateOpening' in df.columns:
        df['EnrollmentDateOpening'] = pd.to_datetime(df['EnrollmentDateOpening'])
        df['TenureMonths'] = (ref_date - df['EnrollmentDateOpening']) / pd.Timedelta(days=30.44)

    return df

def build_customer_flight_features(flights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly flight records into customer-level features:
    - TotalFlights, TotalDistanceKM, TotalPointsAccumulated, TotalPointsRedeemed
    - MeanPointsUtilization
    - AverageFlightDistance
    """
    id_col = 'Loyalty#'
    df = flights_df.copy()
    
    agg = (
        df
        .groupby(id_col)
        .agg(
            TotalFlights=('NumFlights', 'sum'),
            TotalDistanceKM=('DistanceKM', 'sum'),
            TotalPointsAccumulated=('PointsAccumulated', 'sum'),
            TotalPointsRedeemed=('PointsRedeemed', 'sum'),
            MeanPointsUtilization=('PointsUtilizationRatio', 'mean')
        )
        .reset_index()
    )

    # Log transforms for aggregated features
    for col in ['TotalFlights', 'TotalDistanceKM', 'TotalPointsAccumulated', 'TotalPointsRedeemed']:
        agg[f'{col}_log'] = np.log1p(agg[col])
    
    # Average flight distance
    agg['AverageFlightDistance'] = agg['TotalDistanceKM'] / agg['TotalFlights'].replace({0: np.nan})

    return agg

def create_model_df(customer_df: pd.DataFrame, flights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the creation of the final modeling dataframe:
    1. Preprocess customers and flights
    2. Build customer-level flight features
    3. Merge datasets (Left Join)
    4. Set Loyalty# as Index
    5. Handle missing values
    6. Encode categorical variables (OneHotEncoder)
    7. Drop unnecessary columns
    8. Scale numeric features (StandardScaler)
    """
    # 1. Preprocess
    cust_clean = preprocess_customers(customer_df)
    flights_clean = preprocess_flights(flights_df)

    # 2. Build flight features
    flight_features = build_customer_flight_features(flights_clean)

    # 3. Merge
    model_df = cust_clean.merge(flight_features, on='Loyalty#', how='left')

    # 4. Set Loyalty# as Index
    if 'Loyalty#' in model_df.columns:
        model_df.set_index('Loyalty#', inplace=True)

    # 5. Handle Missing Values (Numeric)
    numeric_cols_to_fill = model_df.select_dtypes(include=[np.number]).columns
    model_df[numeric_cols_to_fill] = model_df[numeric_cols_to_fill].fillna(0)

    # 6. Drop unnecessary columns
    cols_to_drop = [
        'First Name', 'Last Name', 'CancellationDate', 'Customer Name',
        'Country', 'Province or State', 'City', 'Postal Code',
        'Latitude', 'Longitude', 'EnrollmentDateOpening', 'EnrollmentType',
        'TotalFlights', 'TotalDistanceKM', 'TotalPointsAccumulated', 'TotalPointsRedeemed',
        'Customer Lifetime Value', 'Income'
    ]
    model_df = model_df.drop(columns=[c for c in cols_to_drop if c in model_df.columns], errors='ignore')

    # 7. Separate Numeric and Categorical
    categorical_cols = ['LoyaltyStatus', 'Location Code']
    categorical_cols = [c for c in categorical_cols if c in model_df.columns]
       
    numeric_cols = model_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude binary/ordinal from scaling
    unscaled_cols = []
    for col in ['CancelledFlag', 'Marital Status', 'Gender']:
        if col in numeric_cols:
            numeric_cols.remove(col)
            unscaled_cols.append(col)

    # 8. OneHotEncoding
    ohe = OneHotEncoder(sparse_output=False, drop='first', dtype=int)
    encoded_data = ohe.fit_transform(model_df[categorical_cols])
    encoded_cols = ohe.get_feature_names_out(categorical_cols)
    
    df_encoded = pd.DataFrame(encoded_data, columns=encoded_cols, index=model_df.index)
    
    # 9. Scale Numeric Features
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(model_df[numeric_cols])
    df_numeric_scaled = pd.DataFrame(scaled_numeric, columns=numeric_cols, index=model_df.index)
    
    # 10. Combine
    dfs_to_concat = [df_numeric_scaled, df_encoded]
    if unscaled_cols:
        dfs_to_concat.append(model_df[unscaled_cols])
        
    df_final = pd.concat(dfs_to_concat, axis=1)
    
    return df_final

def evaluate_clustering(algorithm_cls, X, param_grid, verbose = True, **kwargs):
    algo_name = algorithm_cls.__name__
    best_labels = None
    best_metrics = (-np.inf, np.inf, -np.inf) 
    
    # Store original index to return later
    original_index = getattr(X, 'index', None)
    
    # Convert to numpy for stable math calculations
    X_np = X.values if isinstance(X, pd.DataFrame) else X

    keys = list(param_grid.keys())
    for values in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, values))
        model = algorithm_cls(**params, **kwargs)
        labels = model.fit_predict(X_np)
        
        n_clusters = len(np.unique(labels))
        if n_clusters <= 1:
            continue

        # Calculate Metrics
        sil = round(silhouette_score(X_np, labels), 2)
        db = round(davies_bouldin_score(X_np, labels), 2)
        
        # R2 calculation (Variance Explained)
        # Using X_np ensures total_var is a single float, not a Series
        overall_mean = X_np.mean(axis=0)
        total_var = np.sum((X_np - overall_mean) ** 2)
        
        between_var = 0
        for k in np.unique(labels):
            cluster_data = X_np[labels == k]
            cluster_mean = cluster_data.mean(axis=0)
            n_k = len(cluster_data)
            between_var += n_k * np.sum((cluster_mean - overall_mean) ** 2)
            
        r2 = round(between_var / total_var, 2) if total_var > 0 else 0.0

        # Tie-breaking logic
        current_metrics = (sil, db, r2)
        is_better = False
        if sil > best_metrics[0]:
            is_better = True
        elif sil == best_metrics[0]:
            if db < best_metrics[1]:
                is_better = True
            elif db == best_metrics[1]:
                if r2 > best_metrics[2]:
                    is_better = True

        if is_better:
            best_metrics = current_metrics
            best_labels = labels

        param_str = ', '.join([f'{k}={v}' for k, v in params.items()])

        if verbose:
            print(f"{algo_name} ({param_str}): Clusters={n_clusters}, Sil={sil}, DB={db}, R2={r2}")

    # Return as Series to preserve the index
    if original_index is not None:
        return pd.Series(best_labels, index=original_index, name=f"{algo_name}_labels")
    return best_labels

def plot_k_distance(df, title, ax=None):
    # Rule of thumb: k = 2 * number of dimensions
    k = 2 * len(df.columns)
    
    # Fit Nearest Neighbors
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(df)
    distances, _ = neigh.kneighbors(df)
    
    # Sort distances to the k-th neighbor
    k_distances = np.sort(distances[:, -1])
    
    # Plotting logic
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
    ax.plot(k_distances, color='steelblue', linewidth=2)
    ax.set_title(f"K-Distance: {title} (k={k})", fontsize=12)
    ax.set_xlabel("Points sorted by distance", fontsize=10)
    ax.set_ylabel(f"Distance to {k}-th neighbor", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

def apply_pca_2d(X, random_state=42):
    """Return a pandas DataFrame with a 2D PCA embedding for X."""
    pca = PCA(n_components=2, random_state=random_state)
    X_np = X.values if isinstance(X, pd.DataFrame) else X
    embedding = pca.fit_transform(X_np)
    idx = X.index if isinstance(X, pd.DataFrame) else None
    cols = ['PCA1', 'PCA2']
    return pd.DataFrame(embedding, columns=cols, index=idx)

def apply_tsne_2d(X, random_state=42):
    """Return a pandas DataFrame with a 2D t-SNE embedding for X."""
    tsne = TSNE(n_components=2, random_state=random_state, n_jobs=-1)
    X_np = X.values if isinstance(X, pd.DataFrame) else X
    embedding = tsne.fit_transform(X_np)
    idx = X.index if isinstance(X, pd.DataFrame) else None
    cols = ['TSNE1', 'TSNE2']
    return pd.DataFrame(embedding, columns=cols, index=idx)

# UMAP helper: compute 2D embedding and plotting function
def apply_umap_2d(X, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42):
    """Return a pandas DataFrame with a 2D UMAP embedding for X."""
    umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric=metric, random_state=random_state)
    X_np = X.values if isinstance(X, pd.DataFrame) else X
    embedding = umap_model.fit_transform(X_np)
    idx = X.index if isinstance(X, pd.DataFrame) else None
    cols = ['UMAP1', 'UMAP2']
    return pd.DataFrame(embedding, columns=cols, index=idx)

def plot_cluster(umap_df, pca_df, tsne_df, labels, main_title, palette='tab10', figsize=(24, 7), alpha=0.8, s=30):
    """
    Plots UMAP, PCA, and t-SNE embeddings side-by-side in a single figure,
    colored by cluster labels.

    Args:
        umap_df (pd.DataFrame): DataFrame with 2D UMAP embedding.
        pca_df (pd.DataFrame): DataFrame with 2D PCA embedding.
        tsne_df (pd.DataFrame): DataFrame with 2D t-SNE embedding.
        labels (array-like): Labels for coloring the points (e.g., from a clustering algorithm).
        main_title (str): The main title for the entire figure.
        palette (str, optional): Color palette for the plot.
        figsize (tuple, optional): Figure size for the entire figure.
        alpha (float, optional): Opacity of the points.
        s (int, optional): Size of the points.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(main_title, fontsize=16, y=0.98)

    embeddings = {
        'UMAP': umap_df,
        'PCA': pca_df,
        't-SNE': tsne_df
    }

    # Align labels once, assuming all embedding_dfs share the same index.
    plot_labels = labels.reindex(umap_df.index) if isinstance(labels, pd.Series) else pd.Series(labels, index=umap_df.index)

    for i, (name, df) in enumerate(embeddings.items()):
        ax = axes[i]
        x_col, y_col = df.columns
        
        sns.scatterplot(
            x=x_col, 
            y=y_col, 
            data=df, 
            hue=plot_labels, 
            palette=palette, 
            s=s, 
            alpha=alpha, 
            legend='full' if i == 2 else False, # Show legend only on the last plot
            ax=ax
        )
        ax.set_title(name)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, linestyle='--', alpha=0.4)

    if axes[2].get_legend() is not None:
        axes[2].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Helper: train a MiniSom and return codebook DataFrame + BMU series
def som_codebook_for_evaluation(X, x=10, y=10, sigma=1.0, learning_rate=0.5, n_iter=2000, random_seed=42):
    """Train a MiniSom on DataFrame X and return (som, codebook_df, bmu_series)."""
    X_np = X.values
    som = MiniSom(x, y, X_np.shape[1], sigma=sigma, learning_rate=learning_rate, random_seed=random_seed)
    som.random_weights_init(X_np)
    som.train_random(X_np, n_iter)
    # get weights (x, y, features) and reshape to (x*y, features)
    weights = som.get_weights()
    codebook = weights.reshape(x * y, -1)
    feature_names = X.columns.tolist()
    codebook_df = pd.DataFrame(codebook, columns=feature_names)
    # Map each sample to BMU node id (0..x*y-1)
    bmus = [som.winner(xi) for xi in X_np]
    node_ids = [i * y + j for (i, j) in bmus]
    bmu_series = pd.Series(node_ids, index=X.index, name='SOM_node')
    return som, codebook_df, bmu_series

def detect_outliers(model_df, eps=1.9, min_samples=20):
    """
    Detect and remove multivariate outliers using DBSCAN.
    Returns:
        model_df_clipped: DataFrame without outliers
        outliers_df: DataFrame with outliers
        outlier_count: Counter object with counts of outliers and core points
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
    dbscan_labels = dbscan.fit_predict(model_df)
    
    outlier_count = Counter(dbscan_labels)
    core_mask = (dbscan_labels != -1)
    
    model_df_clipped = model_df[core_mask]
    outliers_df = model_df[dbscan_labels == -1]
    
    return model_df_clipped, outliers_df, outlier_count