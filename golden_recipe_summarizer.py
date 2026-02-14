import os
import json
import pandas as pd
import numpy as np
from supabase import create_client, Client
import uuid
import random

# Supabase Configuration (reused)
SUPABASE_URL = "https://hpamcuncbbdrlyvupssl.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhwYW1jdW5jYmJkcmx5dnVwc3NsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDk4MDA0MSwiZXhwIjoyMDg2NTYwMDQxfQ.tzVhOq4NfKXraSIuFXn26wAxEFLFkRtU1tU5NP4qWE8"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def fetch_data(table_name: str):
    """Fetches all data from a specified Supabase table with enhanced error reporting."""
    # print(f"Attempting to fetch data from table: {table_name}") # Suppress verbose output
    try:
        response = supabase.from_(table_name).select("*").execute()
        # print(f"Supabase response for {table_name}: {response}") # Suppress verbose output
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

def generate_synthetic_data(num_samples=200):
    """Generates synthetic process_runs and yield_results data for golden recipe analysis."""
    synthetic_process_runs = []
    synthetic_yield_results = []

    for _ in range(num_samples):
        run_id = str(uuid.uuid4())
        created_at = pd.Timestamp.now() - pd.Timedelta(days=random.randint(1, 365))

        # Generate recipe_params, some values clustered around a "golden" range
        # Golden range for demonstration: temp around 210-230, pressure around 11-13, duration around 60-70
        temperature = round(random.uniform(150, 250), 2)
        pressure = round(random.uniform(5, 15), 2)
        duration = random.randint(30, 120)
        catalyst_type = random.choice(['A', 'B', 'C'])
        equipment_id = random.choice(['EQ001', 'EQ002', 'EQ003'])
        
        # Introduce some "golden" runs
        if random.random() < 0.3: # 30% chance to be a "golden" run
            temperature = round(random.uniform(210, 230), 2)
            pressure = round(random.uniform(11, 13), 2)
            duration = random.randint(60, 70)
            catalyst_type = 'A' # 'A' is best catalyst for golden runs

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
        # Make efficiency dependent on parameters, especially favoring the "golden" range
        base_efficiency = 70 + (temperature - 200)/5 + (pressure - 10)*3 - (duration - 75)/3 
        if catalyst_type == 'A':
            base_efficiency += 10 # Catalyst A is better
        elif catalyst_type == 'B':
            base_efficiency += 5
        
        # Add a bonus for being in the golden range
        # Corrected: ensure it's a single logical line or explicitly group conditions
        if (210 <= temperature <= 230) and (11 <= pressure <= 13) and \
           (60 <= duration <= 70) and (catalyst_type == 'A'):
            base_efficiency += random.uniform(5, 15) # High bonus for golden runs

        efficiency = round(max(0, min(100, base_efficiency + random.uniform(-5, 5))), 2)

        synthetic_yield_results.append({
            "yield_result_id": str(uuid.uuid4()),
            "created_at": created_at.isoformat(),
            "process_run_id": run_id,
            "efficiency": efficiency
        })

    return pd.DataFrame(synthetic_process_runs), pd.DataFrame(synthetic_yield_results)

def summarize_golden_recipe():
    """
    Analyzes process data to identify parameter ranges for "Golden Recipes" (highest yield).
    """
    # Fetch data from Supabase or generate synthetic data
    process_runs_data = fetch_data("process_runs")
    yield_results_data = fetch_data("yield_results")

    if not process_runs_data or not yield_results_data:
        print("Supabase data is empty or failed to fetch. Generating synthetic data for golden recipe analysis.")
        df_runs, df_yield = generate_synthetic_data()
        df_runs['recipe_params'] = df_runs['recipe_params'].apply(json.dumps) # Ensure JSON string format
    else:
        df_runs = pd.DataFrame(process_runs_data)
        df_yield = pd.DataFrame(yield_results_data)

    # Ensure 'id' in df_runs is the same type as 'process_run_id' in df_yield
    df_runs['id'] = df_runs['id'].astype(str)
    df_yield['process_run_id'] = df_yield['process_run_id'].astype(str)

    # Merge dataframes
    df_merged = pd.merge(df_runs, df_yield, left_on="id", right_on="process_run_id", how="inner")

    if df_merged.empty:
        return json.dumps({"error": "No common data after merging process_runs and yield_results"})

    # Flatten recipe_params
    if 'recipe_params' in df_merged.columns and not df_merged['recipe_params'].empty:
        first_rp = df_merged['recipe_params'].dropna().iloc[0] if not df_merged['recipe_params'].dropna().empty else None
        if isinstance(first_rp, dict):
            recipe_params_flat = pd.json_normalize(df_merged['recipe_params'])
        elif isinstance(first_rp, str):
            try:
                recipe_params_flat = pd.json_normalize(df_merged['recipe_params'].apply(json.loads))
            except json.JSONDecodeError:
                recipe_params_flat = pd.DataFrame()
        else:
            recipe_params_flat = pd.DataFrame()

        if not recipe_params_flat.empty:
            original_cols = set(df_merged.columns)
            new_cols = []
            for col in recipe_params_flat.columns:
                if col in original_cols:
                    new_cols.append(f"rp_{col}")
                else:
                    new_cols.append(col)
            recipe_params_flat.columns = new_cols
            df_merged = pd.concat([df_merged.drop('recipe_params', axis=1), recipe_params_flat], axis=1)
        else:
            df_merged = df_merged.drop('recipe_params', axis=1, errors='ignore')
    else:
        print("No 'recipe_params' column found or it is empty.")

    df_merged['efficiency'] = pd.to_numeric(df_merged['efficiency'], errors='coerce')
    df_merged.dropna(subset=['efficiency'], inplace=True)

    if df_merged.empty:
        return json.dumps({"error": "No valid efficiency data after preprocessing"})

    # Define "Golden Recipe" as top X% of yields
    top_percentile = 0.10 # Top 10%
    threshold_efficiency = df_merged['efficiency'].quantile(1 - top_percentile)
    
    golden_runs_df = df_merged[df_merged['efficiency'] >= threshold_efficiency]

    if golden_runs_df.empty:
        return json.dumps({"message": "No 'golden' runs identified above the threshold."})

    # Summarize parameter ranges for golden runs
    golden_recipe_summary = {}
    
    # Exclude IDs, timestamps and efficiency itself from feature selection
    exclude_from_params = ['id', 'created_at', 'process_run_id', 'yield_result_id', 'efficiency', 'created_at_x', 'created_at_y']
    
    # Dynamically determine relevant columns after flattening recipe_params
    param_cols = [col for col in golden_runs_df.columns if col not in exclude_from_params and not col.startswith('rp_created_at_')]

    numerical_cols = golden_runs_df[param_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = golden_runs_df[param_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Process numerical parameters
    for col in numerical_cols:
        param_values = golden_runs_df[col].dropna()
        if not param_values.empty:
            golden_recipe_summary[col] = {
                "min": round(param_values.min(), 2),
                "max": round(param_values.max(), 2),
                "mean": round(param_values.mean(), 2),
                "std": round(param_values.std(), 2) if len(param_values) > 1 else 0.0
            }

    # Process categorical parameters
    for col in categorical_cols:
        param_values = golden_runs_df[col].dropna()
        if not param_values.empty:
            value_counts = param_values.value_counts(normalize=True)
            golden_recipe_summary[col] = value_counts.apply(lambda x: round(x * 100, 2)).to_dict()

    final_summary = {
        "golden_recipe_threshold_efficiency": round(threshold_efficiency, 2),
        "number_of_golden_runs": len(golden_runs_df),
        "parameter_ranges": golden_recipe_summary
    }

    return json.dumps(final_summary, indent=4)

if __name__ == "__main__":
    golden_summary = summarize_golden_recipe()
    print(golden_summary)
