# AIAI Customer Loyalty Analytics

A data mining project analyzing three years of customer loyalty and flight activity data to develop a data-driven segmentation strategy.

## Project Structure

### Jupyter Notebooks
- **`Group20_EDA_Code.ipynb`**: Comprehensive Exploratory Data Analysis (EDA) of customer demographics and flight patterns.
- **`Group20_Clustering_Code.ipynb`**: Implementation of various clustering algorithms (K-Means, DBSCAN, MeanShift, SOM) and the final merged clustering solution (MeanShift + Hierarchical).

### Interactive Dashboard
The `dashboard/` folder contains a Streamlit application to visualize the results and explore the data interactively.

## Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**
   Navigate to the dashboard directory and launch the app:
   ```bash
   cd dashboard
   streamlit run home.py
   ```

## Data
- `DM_AIAI_CustomerDB.csv`: Customer demographic and loyalty program data.
- `DM_AIAI_FlightsDB.csv`: Flight activity records.
