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
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhwYW1jdW5jYmJkcmx5dnVwc3NsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImiWFCI6MTc3MDk4MDA0MSwiZXhwIjoyMDg2NTYwMDQxfQ.tzVhOq4NfKXraSIuFXn26wAxEFLFkRtU1tU5NP4qWE8" # CORRECTED KEY

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --- Helper Functions (reused from process_analyzer.py) ---
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
        # Adjusted formula to create clearer correlation for modeling
        efficiency = round(
            max(0, min(100, 
                60 + (temperature - 200)/5 + (pressure - 10)*3 - (duration - 75)/3 
                + (5 * (1 if catalyst_type == 'A' else (2 if catalyst_type == 'B' else 3)))
                + (random.uniform(-5, 5)) # Add some noise
            )), 2)

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
        df_merged = pd.merge(df_runs, df_yield, left_on="id", right_on="process_run_id", how="inner")
    else: # For inference, df_runs is a single recipe, no merging needed with df_yield
        df_merged = df_runs.copy()


    if df_merged.empty:
        raise ValueError("Merged DataFrame is empty. Cannot prepare data for model.")

    # Flatten recipe_params
    if 'recipe_params' in df_merged.columns and not df_merged['recipe_params'].empty:
        first_rp = df_merged['recipe_params'].dropna().iloc[0] if not df_merged['recipe_params'].dropna().empty else None
        
        if isinstance(first_rp, dict):
            recipe_params_flat = pd.json_normalize(df_merged['recipe_params'])
        elif isinstance(first_rp, str):
            try:
                recipe_params_flat = pd.json_normalize(df_merged['recipe_params'].apply(json.loads))
            except json.JSONDecodeError:
                print("recipe_params column contains strings but they are not valid JSON. Skipping flattening.")
                recipe_params_flat = pd.DataFrame()
        else:
            print("recipe_params column is not in dictionary or string JSON format. Skipping flattening.")
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
        
        # Create a DataFrame with all possible categorical features, filled with 0s
        # Then fill in the actual values from df_features
        # This helps ensure consistency when new categorical values appear in inference
        # or when certain categories are missing.
        inference_categorical_data = pd.DataFrame(0, index=df_features.index, columns=preprocessor.get_feature_names_out(categorical_cols))
        for col in categorical_cols:
            if col in df_features.columns:
                unique_vals = df_features[col].unique()
                for val in unique_vals:
                    ohe_col_name = f"{col}_{val}"
                    if ohe_col_name in inference_categorical_data.columns:
                        inference_categorical_data.loc[df_features[col] == val, ohe_col_name] = 1

        # Use the preprocessor to transform only the relevant categorical columns
        # This part needs careful handling if the input df_features for inference is very sparse.
        # A more robust approach often involves a Pipeline. For simple OHE, we ensure input
        # columns match or are prepared to match training data's categorical columns.

        # Create a temporary DataFrame for OHE transformation with same columns as training's categorical input
        temp_categorical_df = pd.DataFrame(columns=preprocessor.feature_names_in_)
        for col in preprocessor.feature_names_in_:
            if col in df_features.columns:
                temp_categorical_df[col] = df_features[col]
            else:
                temp_categorical_df[col] = '' # Fill missing categorical columns with empty string or a default value
        
        # Transform the prepared categorical data
        encoded_features = preprocessor.transform(temp_categorical_df)
        encoded_feature_names = preprocessor.get_feature_names_out(preprocessor.feature_names_in_) # Use feature_names_in_ to get consistent output names
        df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_features.index)
        
        X = pd.concat([df_features[numerical_cols], df_encoded], axis=1)
        fitted_preprocessor = preprocessor # Return the same preprocessor

    # Ensure all feature columns are numeric, coerce errors, fill NaNs
    for col in X.columns:
        X.loc[:, col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(X.mean()) # Impute remaining NaNs with column mean

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

    print(f"Training model with {len(X)} samples and {len(X.columns)} features.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model (optional, but good practice)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model trained. MAE: {mae:.2f}, R2: {r2:.2f}")

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
    
    print("\n--- Simulating data recording to 'yield_predictions' ---")
    print(json.dumps(prediction_data, indent=4))
    
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
        record_prediction("virtual_run_id_123", predicted_yield, confidence)

    except Exception as e:
        print(f"An error occurred: {e}")