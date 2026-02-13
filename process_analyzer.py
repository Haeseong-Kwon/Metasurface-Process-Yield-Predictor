import os
import json
import pandas as pd
import numpy as np
from supabase import create_client, Client
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import uuid
import random

# 1. Supabase Configuration
SUPABASE_URL = "https://hpamcuncbbdrlyvupssl.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhwYW1jdW5jYmJkcmx5dnVwc3NsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDk4MDA0MSwiZXhwIjoyMDg2NTU2MDQxfQ.tzVhOq4NfKXraSIuFXn26wAxEFLFkRtU1tU5NP4qWE8"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def fetch_data(table_name: str):
    """Fetches all data from a specified Supabase table with enhanced error reporting."""
    print(f"Attempting to fetch data from table: {table_name}")
    try:
        response = supabase.from_(table_name).select("*").execute()
        print(f"Supabase response for {table_name}: {response}")
        if response and hasattr(response, 'data'):
            if not response.data:
                print(f"No data returned for table: {table_name}")
            return response.data
        else:
            print(f"Unexpected response structure for {table_name}: {response}")
            return None
    except Exception as e:
        print(f"Error fetching data from {table_name}: {e}")
        return None

def generate_synthetic_data(num_samples=100):
    """Generates synthetic process_runs and yield_results data."""
    synthetic_process_runs = []
    synthetic_yield_results = []

    for _ in range(num_samples):
        run_id = str(uuid.uuid4())
        created_at = pd.Timestamp.now() - pd.Timedelta(days=random.randint(1, 365))

        # Generate recipe_params
        temperature = round(random.uniform(150, 250), 2)
        pressure = round(random.uniform(5, 15), 2)
        duration = random.randint(30, 120)
        catalyst_type = random.choice(['A', 'B', 'C'])
        equipment_id = random.choice(['EQ001', 'EQ002', 'EQ003'])
        
        recipe_params = {
            "temperature": temperature,
            "pressure": pressure,
            "duration": duration,
            "catalyst_type": catalyst_type,
            "equipment_id": equipment_id
        }

        synthetic_process_runs.append({
            "id": run_id,
            "created_at": created_at.isoformat(),
            "recipe_params": recipe_params
        })

        # Generate yield_results
        # Make efficiency somewhat dependent on params for demonstration
        base_efficiency = 70 + (temperature - 200)/10 + (pressure - 10)*2 - (duration - 75)/5
        efficiency = round(max(0, min(100, base_efficiency + random.uniform(-10, 10))), 2)

        synthetic_yield_results.append({
            "yield_result_id": str(uuid.uuid4()),
            "created_at": created_at.isoformat(),
            "process_run_id": run_id,
            "efficiency": efficiency
        })

    return pd.DataFrame(synthetic_process_runs), pd.DataFrame(synthetic_yield_results)


def process_and_analyze_data():
    """
    Fetches, processes, analyzes data, and detects outliers.
    """
    process_runs_data = fetch_data("process_runs")
    yield_results_data = fetch_data("yield_results")

    if not process_runs_data or not yield_results_data:
        print("Supabase data is empty or failed to fetch. Generating synthetic data for analysis.")
        df_runs, df_yield = generate_synthetic_data()
        # Convert recipe_params dictionary to string for consistent handling later
        df_runs['recipe_params'] = df_runs['recipe_params'].apply(json.dumps)
    else:
        df_runs = pd.DataFrame(process_runs_data)
        df_yield = pd.DataFrame(yield_results_data)

    # Ensure 'id' in df_runs is the same type as 'process_run_id' in df_yield
    df_runs['id'] = df_runs['id'].astype(str)
    df_yield['process_run_id'] = df_yield['process_run_id'].astype(str)

    # Merge dataframes
    df_merged = pd.merge(df_runs, df_yield, left_on="id", right_on="process_run_id", how="inner")

    if df_merged.empty:
        print("Merged DataFrame is empty. Check data fetching and merging conditions.")
        return json.dumps({"error": "No common data after merging process_runs and yield_results"})

    # Flatten recipe_params
    # Check if 'recipe_params' column exists and contains dictionaries/JSON strings
    if 'recipe_params' in df_merged.columns and not df_merged['recipe_params'].empty:
        # Check if the first non-null element is a dict or a string that looks like JSON
        first_rp = df_merged['recipe_params'].dropna().iloc[0] if not df_merged['recipe_params'].dropna().empty else None
        
        if isinstance(first_rp, dict):
            recipe_params_flat = pd.json_normalize(df_merged['recipe_params'])
        elif isinstance(first_rp, str):
            try:
                # Attempt to parse strings as JSON
                recipe_params_flat = pd.json_normalize(df_merged['recipe_params'].apply(json.loads))
            except json.JSONDecodeError:
                print("recipe_params column contains strings but they are not valid JSON. Skipping flattening.")
                recipe_params_flat = pd.DataFrame() # Empty DataFrame if not valid JSON
        else:
            print("recipe_params column is not in dictionary or string JSON format. Skipping flattening.")
            recipe_params_flat = pd.DataFrame() # Empty DataFrame if unknown format

        if not recipe_params_flat.empty:
            # Ensure unique column names if there are overlaps
            original_cols = set(df_merged.columns)
            new_cols = []
            for col in recipe_params_flat.columns:
                if col in original_cols:
                    new_cols.append(f"rp_{col}") # Prefix to avoid collision
                else:
                    new_cols.append(col)
            recipe_params_flat.columns = new_cols
            df_merged = pd.concat([df_merged.drop('recipe_params', axis=1), recipe_params_flat], axis=1)
        else:
            print("recipe_params flattening resulted in an empty DataFrame.")
            df_merged = df_merged.drop('recipe_params', axis=1, errors='ignore')
    else:
        print("No 'recipe_params' column found or it is empty.")


    # Convert efficiency to numeric, handling potential errors
    df_merged['efficiency'] = pd.to_numeric(df_merged['efficiency'], errors='coerce')
    df_merged.dropna(subset=['efficiency'], inplace=True) # Drop rows where efficiency couldn't be converted

    if df_merged.empty:
        print("DataFrame is empty after processing efficiency. No valid efficiency data.")
        return json.dumps({"error": "No valid efficiency data after preprocessing"})


    # Identify numerical and categorical columns for processing
    # Exclude Supabase specific IDs and timestamps
    exclude_cols = ['id', 'created_at', 'process_run_id', 'yield_result_id']
    all_cols = [col for col in df_merged.columns if col not in exclude_cols and col != 'efficiency']

    numerical_cols = df_merged[all_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_merged[all_cols].select_dtypes(include=['object', 'category']).columns.tolist()

    # 3. Data Preprocessing: One-hot Encoding for categorical variables
    df_processed = pd.get_dummies(df_merged, columns=categorical_cols, drop_first=True)

    # Prepare data for correlation and outlier detection
    features = [col for col in df_processed.columns if col not in exclude_cols and col != 'efficiency']
    X = df_processed[features].copy() # Create a copy to avoid SettingWithCopyWarning
    y = df_processed['efficiency']

    # Ensure all feature columns are numeric, coerce errors
    for col in X.columns:
        X.loc[:, col] = pd.to_numeric(X[col], errors='coerce') # Use .loc for explicit assignment
    X = X.dropna(axis=1, how='all') # Assign result back to X
    X = X.fillna(X.mean()) # Assign result back to X

    if X.empty or y.empty:
        print("Features or target is empty after preprocessing. Cannot perform analysis.")
        return json.dumps({"error": "Insufficient data for analysis after preprocessing"})


    # 1. Correlation Analyzer
    correlations = {}
    if not X.empty and not y.empty:
        for col in X.columns:
            # Drop NaNs specific to the pair for correlation calculation
            temp_df = pd.DataFrame({'feature': X[col], 'efficiency': y}).dropna()
            if len(temp_df) > 1 and temp_df['feature'].nunique() > 1: # Pearsonr requires at least 2 data points and variance
                try:
                    corr, _ = pearsonr(temp_df['feature'], temp_df['efficiency'])
                    correlations[col] = corr
                except ValueError as ve:
                    print(f"Could not calculate correlation for {col}: {ve}")
                    correlations[col] = None # Or handle as appropriate
            else:
                correlations[col] = None
    else:
        print("Not enough data to calculate correlations.")

    # 2. Outlier Detector
    outlier_results = []
    if not y.empty and y.std() > 0: # Ensure there's variance to calculate std_efficiency
        mean_efficiency = y.mean()
        std_efficiency = y.std()

        # Define outliers as efficiency outside 2 standard deviations
        lower_bound = mean_efficiency - 2 * std_efficiency
        upper_bound = mean_efficiency + 2 * std_efficiency

        outliers_df = df_processed[(y < lower_bound) | (y > upper_bound)]

        for index, row in outliers_df.iterrows():
            outlier_details = {
                "process_run_id": row['process_run_id'],
                "efficiency": row['efficiency'],
                "deviation_from_mean": row['efficiency'] - mean_efficiency,
                "anomalous_params": {}
            }

            # Identify parameters that are unusual for this outlier run
            # Compare parameters of the outlier run against the average of all runs
            for param in X.columns:
                if param in row and pd.notna(row[param]) and pd.notna(X[param].mean()):
                    param_mean = X[param].mean()
                    param_std = X[param].std()
                    
                    # Avoid division by zero if std is 0 (constant feature)
                    if param_std > 0:
                        # For simplicity, if a param deviates by more than 1.5 standard deviations from its mean
                        # it's considered anomalous for this specific outlier process run.
                        if abs(row[param] - param_mean) > 1.5 * param_std:
                            outlier_details["anomalous_params"][param] = {
                                "value": row[param],
                                "mean_across_all_runs": param_mean,
                                "std_across_all_runs": param_std
                            }
                    elif row[param] != param_mean: # If std is 0 but value is different, it's anomalous
                        outlier_details["anomalous_params"][param] = {
                            "value": row[param],
                            "mean_across_all_runs": param_mean,
                            "std_across_all_runs": param_std
                        }
            outlier_results.append(outlier_details)
    else:
        print("Not enough efficiency data or no variance in efficiency to detect outliers.")

    final_results = {
        "correlation_coefficients": {k: v for k, v in correlations.items() if v is not None},
        "outlier_processes": outlier_results
    }

    return json.dumps(final_results, indent=4)

if __name__ == "__main__":
    analysis_output = process_and_analyze_data()
    print(analysis_output)