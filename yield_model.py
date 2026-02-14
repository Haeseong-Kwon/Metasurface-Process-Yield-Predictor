import os
import json
import pandas as pd
import numpy as np
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import uuid
import random
from collections import defaultdict
import joblib # For saving/loading the model and preprocessor if needed

# 1. Supabase Configuration
SUPABASE_URL = "https://hpamcuncbbdrlyvupssl.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhwYW1jdW5jYmJkcmx5dnVwc3NsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDk4MDA0MSwiZXhwIjoyMDg2NTYwMDQxfQ.tzVhOq4NfKXraSIuFXn26wAxEFLFkRtU1tU5NP4qWE8" # CORRECTED KEY

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Define default recipe parameters for data integrity audit and handling
DEFAULT_RECIPE_PARAMS = {
    "temperature": 200.0,
    "pressure": 10.0,
    "duration": 60,
    "catalyst_type": "UNKNOWN", # Use a specific UNKNOWN category
    "equipment_id": "UNKNOWN" # Use a specific UNKNOWN category
}

# --- Helper Functions (reused from process_analyzer.py) ---
def fetch_data(table_name: str):
    """Fetches all data from a specified Supabase table with enhanced error reporting."""
    # print(f"Attempting to fetch data from table: {table_name}") # Suppress verbose output
    try:
        response = supabase.from_(table_name).select("*").execute()
        # print(f"Supabase response for {table_name}: {response}") # Suppress verbose output
        if response and hasattr(response, 'data'):
            if not response.data:
                # print(f"No data returned for table: {table_name}") # Keep only for critical errors
                pass
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
        
        # Introduce some missing parameters for testing data integrity
        if random.random() < 0.05: # 5% chance to miss temperature
            del recipe_params["temperature"]
        if random.random() < 0.05: # 5% chance to miss catalyst_type
            del recipe_params["catalyst_type"]

        synthetic_process_runs.append({
            "id": run_id,
            "created_at": created_at.isoformat(),
            "recipe_params": recipe_params
        })

        # Generate yield_results
        # Make efficiency dependent on parameters, especially favoring the "golden" range
        # Use default values for missing params when calculating efficiency
        eff_temp = recipe_params.get("temperature", DEFAULT_RECIPE_PARAMS["temperature"])
        eff_pressure = recipe_params.get("pressure", DEFAULT_RECIPE_PARAMS["pressure"])
        eff_duration = recipe_params.get("duration", DEFAULT_RECIPE_PARAMS["duration"])
        eff_catalyst = recipe_params.get("catalyst_type", DEFAULT_RECIPE_PARAMS["catalyst_type"])


        base_efficiency = 70 + (eff_temp - 200)/5 + (eff_pressure - 10)*3 - (eff_duration - 75)/3 
        if eff_catalyst == 'A':
            base_efficiency += 10 # Catalyst A is better
        elif eff_catalyst == 'B':
            base_efficiency += 5
        
        # Add a bonus for being in the golden range
        if (210 <= eff_temp <= 230) and (11 <= eff_pressure <= 13) and \
           (60 <= eff_duration <= 70) and (eff_catalyst == 'A'):
            base_efficiency += random.uniform(5, 15) # High bonus for golden runs

        efficiency = round(max(0, min(100, base_efficiency + random.uniform(-5, 5))), 2)

        synthetic_yield_results.append({
            "yield_result_id": str(uuid.uuid4()),
            "created_at": created_at.isoformat(),
            "process_run_id": run_id,
            "efficiency": efficiency
        })

    return pd.DataFrame(synthetic_process_runs), pd.DataFrame(synthetic_yield_results)

# --- Data Preparation Function ---
def prepare_data_for_model(df_runs: pd.DataFrame, df_yield: pd.DataFrame, is_inference=False, preprocessor=None, training_cols=None):
    """
    Prepares and preprocesses data for model training or inference.
    Handles merging, flattening recipe_params, one-hot encoding.
    
    Args:
        df_runs (pd.DataFrame): DataFrame from 'process_runs' or synthetic equivalent.
        df_yield (pd.DataFrame): DataFrame from 'yield_results' or synthetic equivalent.
        is_inference (bool): True if preparing data for inference.
        preprocessor (object): Fitted OneHotEncoder for inference.
        training_cols (list): List of columns from training data to align inference data.
        
    Returns:
        tuple: X (features DataFrame), y (target Series), preprocessor (fitted OneHotEncoder for training)
    """
    # Ensure 'id' in df_runs is the same type as 'process_run_id' in df_yield
    df_runs['id'] = df_runs['id'].astype(str)
    
    if not is_inference: # For training, df_yield has process_run_id
        df_yield['process_run_id'] = df_yield['process_run_id'].astype(str)
        
        # --- Data Integrity Audit: Check for mismatches before merging ---
        runs_with_yield = set(df_yield['process_run_id'].unique())
        yields_with_run = set(df_runs['id'].unique())
        
        missing_yield_for_run = list(yields_with_run - runs_with_yield)
        missing_run_for_yield = list(runs_with_yield - yields_with_run)

        if missing_yield_for_run:
            print(f"Warning: {len(missing_yield_for_run)} process runs have no matching yield results. Example IDs: {missing_yield_for_run[:5]}")
        if missing_run_for_yield:
            print(f"Warning: {len(missing_run_for_yield)} yield results have no matching process runs. Example IDs: {missing_run_for_yield[:5]}")

        # Merge dataframes
        df_merged = pd.merge(df_runs, df_yield, left_on="id", right_on="process_run_id", how="inner")
    else: # For inference, df_runs is a single recipe, no merging needed with df_yield
        df_merged = df_runs.copy()


    if df_merged.empty:
        raise ValueError("Merged DataFrame is empty. Cannot prepare data for model.")

    # --- Data Integrity Audit & Handling: Fill missing recipe_params with defaults ---
    # Ensure recipe_params is a column of dicts first for easy manipulation
    if 'recipe_params' in df_merged.columns:
        # Convert JSON strings to dicts if necessary
        df_merged['recipe_params'] = df_merged['recipe_params'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        
        # Fill missing keys within each recipe_params dictionary with DEFAULT_RECIPE_PARAMS
        def fill_missing_recipe_keys(params_dict):
            if not isinstance(params_dict, dict): # Handle cases where it might not be a dict
                return DEFAULT_RECIPE_PARAMS.copy()
            filled_params = DEFAULT_RECIPE_PARAMS.copy()
            filled_params.update(params_dict)
            return filled_params

        df_merged['recipe_params'] = df_merged['recipe_params'].apply(fill_missing_recipe_keys)
    else:
        # If 'recipe_params' column itself is missing, create it with defaults
        df_merged['recipe_params'] = [DEFAULT_RECIPE_PARAMS.copy() for _ in range(len(df_merged))]


    # Flatten recipe_params
    if 'recipe_params' in df_merged.columns and not df_merged['recipe_params'].empty:
        # After filling missing keys, all recipe_params are guaranteed to be dicts with consistent keys
        recipe_params_flat = pd.json_normalize(df_merged['recipe_params'])

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
        # This case should be less likely now due to filling missing 'recipe_params' column
        pass


    # Convert efficiency to numeric for target, drop NaNs if training
    if not is_inference:
        df_merged['efficiency'] = pd.to_numeric(df_merged['efficiency'], errors='coerce')
        df_merged.dropna(subset=['efficiency'], inplace=True)
        if df_merged.empty:
            raise ValueError("DataFrame is empty after processing efficiency. No valid efficiency data.")

    # Identify features for the model
    # Explicitly exclude metadata and timestamp columns
    exclude_cols = ['id', 'created_at', 'process_run_id', 'yield_result_id', 'efficiency', 'created_at_x', 'created_at_y']
    
    # Drop excluded columns from df_merged to form df_features
    df_features = df_merged.drop(columns=[col for col in exclude_cols if col in df_merged.columns], errors='ignore')

    # Separate numerical and categorical columns for preprocessing
    numerical_cols = df_features.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_features.select_dtypes(include=['object', 'category']).columns.tolist() # Exclude 'bool' from here

    # One-hot encode categorical features
    if not is_inference: # For training, fit and transform
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        # Ensure only categorical columns are passed to OHE
        if not df_features[categorical_cols].empty:
            encoded_features = ohe.fit_transform(df_features[categorical_cols])
            encoded_feature_names = ohe.get_feature_names_out(categorical_cols)
            df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_features.index)
            # Combine numerical and encoded features
            X = pd.concat([df_features[numerical_cols], df_encoded], axis=1)
        else: # No categorical columns to encode
            X = df_features[numerical_cols]
        fitted_preprocessor = ohe
    else: # For inference, transform using the provided preprocessor
        if preprocessor is None:
            raise ValueError("Preprocessor (fitted OneHotEncoder) must be provided for inference.")
        
        # Create a temporary DataFrame for OHE transformation with same columns as training's categorical input
        temp_categorical_df = pd.DataFrame(columns=preprocessor.feature_names_in_)
        for col in preprocessor.feature_names_in_:
            if col in df_features.columns:
                temp_categorical_df[col] = df_features[col]
            else:
                # Fill missing categorical columns with a value that OHE can handle (e.g., empty string or 'UNKNOWN')
                # Make sure 'UNKNOWN' is handled by the preprocessor's handle_unknown='ignore'
                temp_categorical_df[col] = DEFAULT_RECIPE_PARAMS.get(col.replace('rp_',''), 'UNKNOWN') # If it was a flattened param, use default for original

        # Transform the prepared categorical data
        encoded_features = preprocessor.transform(temp_categorical_df)
        encoded_feature_names = preprocessor.get_feature_names_out(preprocessor.feature_names_in_) # Use feature_names_in_ to get consistent output names
        df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_features.index)
        
        X = pd.concat([df_features[numerical_cols], df_encoded], axis=1)
        fitted_preprocessor = preprocessor # Return the same preprocessor

    # Ensure all feature columns are numeric, coerce errors, fill NaNs
    for col in X.columns:
        X.loc[:, col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill remaining NaNs for numerical features with their default value, not mean
    # This is more robust for inference where a mean might not be available
    for col in X.columns:
        original_col_name = col.replace('rp_','') # If it was a flattened param
        if original_col_name in DEFAULT_RECIPE_PARAMS and isinstance(DEFAULT_RECIPE_PARAMS[original_col_name], (int, float)):
            X.loc[:, col] = X[col].fillna(DEFAULT_RECIPE_PARAMS[original_col_name])
        else: # Fallback to mean if no specific default or not in default_recipe_params
            X.loc[:, col] = X[col].fillna(X[col].mean())


    # Align columns between training and inference data (critical for OHE consistency)
    if is_inference and training_cols is not None:
        missing_cols = set(training_cols) - set(X.columns)
        for c in missing_cols:
            X[c] = 0 # Add missing columns as 0
        X = X[training_cols] # Reorder columns to match training
    
    y = df_merged['efficiency'] if not is_inference else None # Only return y for training

    return X, y, fitted_preprocessor

# --- Model Training Function ---
def train_prediction_model(X: pd.DataFrame, y: pd.Series):
    """
    Trains a RandomForestRegressor model and returns the model and feature importances.
    """
    if X.empty or y.empty:
        raise ValueError("No data to train the model.")

    # print(f"Training model with {len(X)} samples and {len(X.columns)} features.") # Suppress verbose output

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model (optional, but good practice)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # print(f"Model trained. MAE: {mae:.2f}, R2: {r2:.2f}") # Suppress verbose output

    # Feature Importance
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    return model, feature_importances

# --- Virtual Inference API Function ---
def predict_yield_for_recipe(model: RandomForestRegressor, recipe_params: dict, preprocessor, training_cols: list):
    """
    Predicts yield for a given set of recipe parameters.
    """
    # Convert single recipe_params dict to DataFrame
    df_single_run = pd.DataFrame([{"recipe_params": recipe_params, "id": "virtual_run"}])
    df_single_run['recipe_params'] = df_single_run['recipe_params'].apply(json.dumps) # Match data type

    # Preprocess the single recipe data using the trained preprocessor and align features
    # Pass training_cols to ensure alignment
    X_inference, _, _ = prepare_data_for_model(df_single_run, pd.DataFrame(), is_inference=True, preprocessor=preprocessor, training_cols=training_cols)
    
    predicted_yield = model.predict(X_inference)[0]
    
    # For confidence score, a simple approach for RandomForest is std of predictions across trees
    # Or for a more robust prediction interval, one might use a quantile regressor or more advanced techniques
    # For now, let's use a simple heuristic for demonstration.
    # In a real scenario, this would be more sophisticated.
    # Predict all trees, then calculate std
    all_tree_preds = []
    for estimator in model.estimators_:
        all_tree_preds.append(estimator.predict(X_inference)[0])
    
    if len(all_tree_preds) > 1:
        prediction_std = np.std(all_tree_preds)
        # Convert std to a "confidence" score (higher std = lower confidence)
        # This is a heuristic: smaller std relative to prediction suggests higher confidence.
        # Normalize by typical range of yields (e.g., 100), or by prediction itself.
        # Let's use a simple inverse relationship, clamped to 0-100.
        # A "good" std might be 2-3, so if std is 0, confidence is 100.
        # If std is 10, confidence is 0. This is just illustrative.
        confidence_score = max(0, 100 - (prediction_std * 10)) # Heuristic scaling
    else:
        confidence_score = 100 # If only one tree, max confidence as no variance info
    
    confidence_score = max(0, min(100, confidence_score)) # Clamp between 0-100

    return predicted_yield, confidence_score

# --- Data Sync Function ---
def record_prediction(process_run_id: str, predicted_yield: float, confidence_score: float):
    """
    Simulates recording predicted yield and confidence score to Supabase 'yield_predictions' table.
    """
    prediction_data = {
        "id": str(uuid.uuid4()),
        "created_at": pd.Timestamp.now().isoformat(),
        "process_run_id": process_run_id,
        "predicted_yield": predicted_yield,
        "confidence_score": confidence_score
    }
    
    # print("\n--- Simulating data recording to 'yield_predictions' ---") # Suppress verbose output
    # print(json.dumps(prediction_data, indent=4)) # Suppress verbose output
    
    # In a real application, you would do:
    # try:
    #     response = supabase.from_("yield_predictions").insert([prediction_data]).execute()
    #     print(f"Prediction recorded successfully: {response.data}")
    # except Exception as e:
    #     print(f"Error recording prediction: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # Fetch data from Supabase
        process_runs_data = fetch_data("process_runs")
        yield_results_data = fetch_data("yield_results")

        # Use synthetic data if Supabase data is empty
        if not process_runs_data or not yield_results_data:
            print("Supabase data is empty or failed to fetch. Generating synthetic data for model training.")
            df_runs_raw, df_yield_raw = generate_synthetic_data(num_samples=200) # More samples for training
            df_runs_raw['recipe_params'] = df_runs_raw['recipe_params'].apply(json.dumps) # Ensure JSON string format
        else:
            df_runs_raw = pd.DataFrame(process_runs_data)
            df_yield_raw = pd.DataFrame(yield_results_data)

        # 2. Data Preparation for Model Training
        X_train_processed, y_train, preprocessor = prepare_data_for_model(df_runs_raw, df_yield_raw, is_inference=False)
        
        # Save training feature names and preprocessor for consistent inference
        joblib.dump(X_train_processed.columns.tolist(), 'trained_features.pkl')
        joblib.dump(preprocessor, 'ohe_preprocessor.pkl') # Save the fitted OneHotEncoder

        # 1. Yield Prediction Model Training
        model, feature_importances = train_prediction_model(X_train_processed, y_train)

        # 2. Feature Importance Ranking
        print("\n--- Top 3 Feature Importances ---")
        print(feature_importances.head(3).to_string())

        # Save the trained model
        joblib.dump(model, 'yield_prediction_model.pkl')
        print("\nModel, preprocessor, and feature names saved for future use.")

        # 3. Virtual Inference API Demonstration
        print("\n--- Demonstrating Virtual Inference ---")
        # Example virtual recipe parameters
        virtual_recipe_params = {
            "temperature": 220,
            "pressure": 12,
            "duration": 60,
            "catalyst_type": "B",
            "equipment_id": "EQ001"
        }
        
        # Load the saved components for inference (simulating a separate inference service)
        loaded_model = joblib.load('yield_prediction_model.pkl')
        loaded_preprocessor = joblib.load('ohe_preprocessor.pkl')
        loaded_training_features = joblib.load('trained_features.pkl')

        predicted_yield, confidence = predict_yield_for_recipe(
            loaded_model, 
            virtual_recipe_params, 
            loaded_preprocessor,
            loaded_training_features
        )
        print(f"Virtual Recipe Parameters: {virtual_recipe_params}")
        print(f"Predicted Yield: {predicted_yield:.2f}")
        print(f"Confidence Score: {confidence:.2f}")

        # 4. Data Sync Demonstration (simulated)
        # print(f"Simulating recording prediction for virtual_run_id_123: Predicted Yield={predicted_yield:.2f}, Confidence={confidence:.2f}") # Suppress verbose output
        record_prediction("virtual_run_id_123", predicted_yield, confidence)

    except Exception as e:
        print(f"An error occurred: {e}")
