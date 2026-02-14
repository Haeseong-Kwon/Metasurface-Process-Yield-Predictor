import os
import json
import pandas as pd
import numpy as np
import cv2
from supabase import create_client, Client
import uuid
import random
import base64
import multiprocessing as mp
import joblib
from yield_model import predict_yield_for_recipe # Import the function directly

# Supabase Configuration (reused)
SUPABASE_URL = "https://hpamcuncbbdrlyvupssl.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhwYW1jdW5jYmJkcmx5dnVwc3NsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDk4MDA0MSwiZXhwIjoyMDg2NTYwMDQxfQ.tzVhOq4NfKXraSIuFXn26wAxEFLFkRtU1tU5NP4qWE8"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --- Helper Functions (reused) ---
def fetch_data(table_name: str):
    """Fetches all data from a specified Supabase table with enhanced error reporting."""
    print(f"Attempting to fetch data from table: {table_name}")
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

# --- Image Placeholder Generation ---
def generate_placeholder_sem_image(image_id: str, size=(256, 256), defect_type=None, save_dir="./sem_images"):
    """
    Generates a placeholder SEM-like image with simulated defects.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_path = os.path.join(save_dir, f"sem_image_{image_id}.png")
    
    # Create a blank grayscale image
    image = np.full(size, 200, dtype=np.uint8) # Light gray background

    # Simulate basic nano-pattern (e.g., lines)
    for i in range(0, size[0], 10):
        cv2.line(image, (i, 0), (i, size[1]), 100, 1) # Vertical lines
    for i in range(0, size[1], 10):
        cv2.line(image, (0, i), (size[0], i), 100, 1) # Horizontal lines

    # Simulate defects
    if defect_type == "Bridge":
        # Draw a brighter, thicker line connecting two patterns
        p1 = (random.randint(50, size[0]-50), random.randint(50, size[1]-50))
        p2 = (p1[0] + random.randint(-20, 20), p1[1] + random.randint(-20, 20))
        cv2.line(image, p1, p2, 255, 3)
    elif defect_type == "Missing":
        # Draw a dark square to simulate a missing part
        center = (random.randint(50, size[0]-50), random.randint(50, size[1]-50))
        square_size = random.randint(10, 20)
        cv2.rectangle(image, (center[0]-square_size, center[1]-square_size), 
                      (center[0]+square_size, center[1]+square_size), 0, -1)
    elif defect_type == "Distortion":
        # Add random noise or wavy pattern
        for _ in range(20):
            center = (random.randint(0, size[0]), random.randint(0, size[1]))
            radius = random.randint(1, 5)
            cv2.circle(image, center, radius, random.randint(50, 200), -1)

    cv2.imwrite(image_path, image)
    return image_path

# --- Defect Detection Engine ---
def detect_defects(image_path: str):
    """
    Simulated defect detection. In a real scenario, this would use
    OpenCV advanced features or a trained ML model (e.g., YOLO).
    Returns counts for Bridge, Missing, Distortion defects.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # For demonstration, we'll simulate defect counts based on the 'defect_type'
    # specified during image generation, and add some randomness.
    defects = {"Bridge": 0, "Missing": 0, "Distortion": 0}
    
    # Extract defect_type from filename if it was generated with it
    filename = os.path.basename(image_path)
    if "Bridge" in filename:
        defects["Bridge"] = random.randint(1, 3)
    elif "Missing" in filename:
        defects["Missing"] = random.randint(1, 2)
    elif "Distortion" in filename:
        defects["Distortion"] = random.randint(1, 5)
    else: # Random defects for images without explicit type
        if random.random() < 0.3: defects["Bridge"] = random.randint(0, 1)
        if random.random() < 0.3: defects["Missing"] = random.randint(0, 1)
        if random.random() < 0.3: defects["Distortion"] = random.randint(0, 2)

    return defects

# --- Structural Uniformity Scorer ---
def calculate_uniformity(image_path: str):
    """
    Simulated structural uniformity scoring. In a real scenario, this would
    involve Fourier analysis (FFT) to detect periodicity and quantify deviations.
    Returns a uniformity score sigma (e.g., 0-1, lower is more uniform).
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Simple simulation: higher defect count -> lower uniformity
    defects = detect_defects(image_path) # Reuse defect detection to influence uniformity
    total_defects = sum(defects.values())
    
    # Simulate sigma: 0 (perfect) to 1 (highly non-uniform)
    # Scale total_defects to a sigma value
    sigma = min(1.0, total_defects * 0.1 + random.uniform(0.01, 0.1)) # Add some random noise
    return round(sigma, 3)

# --- Recipe Recommender ---
def recommend_recipe_changes(defects: dict, current_recipe_params: dict, 
                             yield_model, ohe_preprocessor, training_features):
    """
    Recommends optimal parameter changes (Delta P) to improve yield based on defect types.
    This is a rule-based system for demonstration.
    
    Args:
        defects (dict): Dictionary of detected defect counts.
        current_recipe_params (dict): The recipe parameters that led to this defect image.
        yield_model: The trained yield prediction model.
        ohe_preprocessor: The fitted OneHotEncoder for feature transformation.
        training_features: List of feature names from model training.
        
    Returns:
        dict: Recommended changes to recipe parameters (Delta P).
    """
    recommendations = {}
    
    # Deep copy current params to modify
    proposed_recipe_params = current_recipe_params.copy()

    # Simple rule-based recommendations for demonstration
    if defects.get("Bridge", 0) > 0:
        # If bridging, suggest lowering temperature and increasing pressure slightly
        recommendations["temperature"] = -5.0 # Decrease by 5 units
        recommendations["pressure"] = 0.5 # Increase by 0.5 units
        proposed_recipe_params["temperature"] = max(150.0, proposed_recipe_params.get("temperature", 200) + recommendations["temperature"])
        proposed_recipe_params["pressure"] = min(15.0, proposed_recipe_params.get("pressure", 10) + recommendations["pressure"])
        
    if defects.get("Missing", 0) > 0:
        # If missing, suggest increasing duration or temperature
        recommendations["duration"] = 10 # Increase by 10 units
        recommendations["temperature"] = 2.0 # Increase by 2 units
        proposed_recipe_params["duration"] = min(120, proposed_recipe_params.get("duration", 60) + recommendations["duration"])
        proposed_recipe_params["temperature"] = min(250.0, proposed_recipe_params.get("temperature", 200) + recommendations["temperature"])

    if defects.get("Distortion", 0) > 0:
        # If distortion, suggest adjusting catalyst or pressure
        recommendations["catalyst_type"] = random.choice(['A', 'B', 'C']) # Try a different catalyst
        recommendations["pressure"] = -1.0 # Decrease by 1 unit
        proposed_recipe_params["pressure"] = max(5.0, proposed_recipe_params.get("pressure", 10) + recommendations["pressure"])

    # Predict yield for proposed changes (if a model is provided)
    predicted_yield_current = None
    predicted_yield_proposed = None
    if yield_model and ohe_preprocessor and training_features:
        try:
            # Predict current yield
            predicted_yield_current, _ = predict_yield_for_recipe(
                yield_model, current_recipe_params, ohe_preprocessor, training_features
            )
            
            # Predict proposed yield
            predicted_yield_proposed, _ = predict_yield_for_recipe(
                yield_model, proposed_recipe_params, ohe_preprocessor, training_features
            )
            
            recommendations["predicted_yield_current"] = round(predicted_yield_current, 2)
            recommendations["predicted_yield_proposed"] = round(predicted_yield_proposed, 2)
            recommendations["predicted_yield_improvement"] = round(predicted_yield_proposed - predicted_yield_current, 2)
            
        except Exception as e:
            print(f"Error predicting yield for recommendations: {e}")
            recommendations["prediction_error"] = str(e)
            
    return recommendations

# --- Image Thumbnail Generation ---
def generate_thumbnail(image_path: str, thumbnail_size=(128, 128)):
    """
    Generates a base64 encoded thumbnail of the image.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    resized_image = cv2.resize(image, thumbnail_size, interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.png', resized_image)
    return base64.b64encode(buffer).decode('utf-8')

# --- Data Sync ---
def record_sem_analysis(analysis_data: dict):
    """
    Simulates recording SEM analysis results to Supabase 'sem_analysis_results' table.
    """
    print("\n--- Simulating data recording to 'sem_analysis_results' ---")
    print(json.dumps(analysis_data, indent=4))
    
    # In a real application, you would do:
    # try:
    #     response = supabase.from_("sem_analysis_results").insert([analysis_data]).execute()
    #     print(f"Analysis recorded successfully: {response.data}")
    # except Exception as e:
    #     print(f"Error recording analysis: {e}")

# --- Multiprocessing Image Processor ---
def process_sem_image(args):
    """
    Orchestrates the analysis for a single SEM image.
    Designed to be run in a multiprocessing pool.
    """
    image_path, model_components, recipe_params = args
    image_id = os.path.basename(image_path).replace("sem_image_", "").replace(".png", "")

    print(f"Processing image: {image_path}")

    # 1. Defect Detection
    defects = detect_defects(image_path)

    # 2. Structural Uniformity Scorer
    uniformity_score = calculate_uniformity(image_path)

    # 3. Recipe Recommender
    recommendations = recommend_recipe_changes(
        defects, recipe_params, 
        model_components.get('model'), 
        model_components.get('preprocessor'), 
        model_components.get('training_features')
    )

    # 4. Image Thumbnail Generation
    thumbnail_base64 = generate_thumbnail(image_path)

    analysis_data = {
        "id": str(uuid.uuid4()),
        "created_at": pd.Timestamp.now().isoformat(),
        "image_id": image_id,
        "image_path": image_path,
        "defects": defects,
        "uniformity_score": uniformity_score,
        "recipe_params_at_analysis": recipe_params,
        "recommendations": recommendations,
        "thumbnail_base64": thumbnail_base64 # Store thumbnail in Base64
    }
    
    record_sem_analysis(analysis_data)
    return analysis_data

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # Clean up old images
        if os.path.exists("./sem_images"):
            for f in os.listdir("./sem_images"):
                os.remove(os.path.join("./sem_images", f))
            os.rmdir("./sem_images")

        # Load trained model components
        print("Loading trained yield prediction model components...")
        try:
            model = joblib.load('yield_prediction_model.pkl')
            preprocessor = joblib.load('ohe_preprocessor.pkl')
            training_features = joblib.load('trained_features.pkl')
            model_components = {
                'model': model,
                'preprocessor': preprocessor,
                'training_features': training_features
            }
            print("Model components loaded successfully.")
        except FileNotFoundError:
            print("Yield prediction model components not found. Please run yield_model.py first.")
            print("Proceeding with metrology engine, but recipe recommendations will be limited.")
            model_components = {} # Empty dict if components not found

        # Generate placeholder SEM images with various defect types
        print("Generating placeholder SEM images...")
        image_paths = []
        defect_types = [None, "Bridge", "Missing", "Distortion"]
        # Example recipe parameters associated with these images (for recommendation context)
        example_recipe_params = [
            {"temperature": 200, "pressure": 10, "duration": 60, "catalyst_type": "A", "equipment_id": "EQ001"},
            {"temperature": 210, "pressure": 11, "duration": 65, "catalyst_type": "B", "equipment_id": "EQ002"},
            {"temperature": 190, "pressure": 9, "duration": 55, "catalyst_type": "C", "equipment_id": "EQ003"},
            {"temperature": 220, "pressure": 12, "duration": 70, "catalyst_type": "A", "equipment_id": "EQ001"},
        ] * 3 # Generate more images

        for i in range(12): # Generate 12 images
            defect_type = random.choice(defect_types)
            image_id = f"{i}_{defect_type if defect_type else 'NoDefect'}"
            path = generate_placeholder_sem_image(image_id, defect_type=defect_type)
            image_paths.append((path, example_recipe_params[i % len(example_recipe_params)]))
        print(f"Generated {len(image_paths)} placeholder images.")

        # Prepare arguments for multiprocessing
        mp_args = []
        for img_path, recipe_p in image_paths:
            mp_args.append((img_path, model_components, recipe_p))

        # Use multiprocessing to process images
        print("\nStarting multiprocessing for SEM image analysis...")
        num_processes = mp.cpu_count() # Use all available CPU cores
        print(f"Using {num_processes} processes.")

        with mp.Pool(num_processes) as pool:
            results = pool.map(process_sem_image, mp_args)
        
        print("\nAll SEM images processed.")
        # You can now further process 'results' if needed

    except Exception as e:
        print(f"An error occurred in main execution: {e}")
