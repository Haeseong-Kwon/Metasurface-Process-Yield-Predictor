# Metasurface Predictor Project

This project aims to leverage AI and advanced image processing for the prediction and analysis of metasurface fabrication yields and quality. It integrates a yield prediction model based on process parameters with a simulated SEM image analysis engine for defect detection and structural uniformity assessment. Furthermore, it provides recommendations for optimizing process parameters to improve yield.

## Key Components

1.  **AI-based Yield Prediction Model (`yield_model.py`)**
    *   **Objective**: To predict the manufacturing yield (efficiency) of metasurfaces based on various recipe parameters.
    *   **Methodology**: A Random Forest Regressor model is trained on historical `process_runs` data (recipe parameters) and corresponding `yield_results` (efficiency).
    *   **Physical/Mathematical Basis**:
        *   **Correlation Analysis**: Initial correlation analysis (performed by `process_analyzer.py`) helps understand the linear relationships between individual process parameters (e.g., temperature, pressure, duration) and yield. This informs feature selection and understanding.
        *   **Random Forest Regression**: This ensemble learning method builds multiple decision trees during training. For regression, it averages the predictions of individual trees to produce a final prediction. It is robust to overfitting, handles non-linear relationships, and inherently provides feature importance.
        *   **Feature Engineering**: Categorical parameters (e.g., `catalyst_type`, `equipment_id`) are transformed using One-Hot Encoding to be suitable for the model. Timestamps are excluded from direct modeling but could be engineered into time-based features if relevant.
        *   **Inference & Confidence**: The trained model predicts yield for new sets of recipe parameters. A confidence score is derived from the variance of predictions across the individual trees in the Random Forest, indicating the model's certainty.

2.  **SEM Vision Analysis Engine (`metrology_engine.py`)**
    *   **Objective**: To automatically analyze Scanning Electron Microscope (SEM) images of metasurfaces to detect manufacturing defects and assess structural uniformity.
    *   **Methodology (Simulated)**: Due to the complexity of real-time deep learning model training and hardware requirements, defect detection and uniformity scoring are currently *simulated* for demonstration purposes. In a real-world application, these modules would integrate state-of-the-art computer vision techniques.
    *   **Physical/Mathematical Basis (Conceptual for Real-world Implementation)**:
        *   **CNN-based Defect Detection (e.g., YOLO)**: For actual defect detection (Bridge, Missing, Distortion), a Convolutional Neural Network (CNN) architecture like YOLO (You Only Look Once) would be trained on a large dataset of annotated SEM images. CNNs are highly effective at learning spatial hierarchies of features, allowing them to accurately classify and localize defects within images.
        *   **Image Processing for Periodicity Analysis**: Structural uniformity ($\sigma$) analysis typically involves:
            *   **Fourier Transform (FFT)**: Applying a 2D Fast Fourier Transform to the SEM image can reveal the dominant spatial frequencies and orientations within the nano-pattern. The presence of sharp, distinct peaks in the Fourier spectrum indicates strong periodicity.
            *   **Autocorrelation**: Analyzing the autocorrelation function of the image can also identify repeating patterns and their characteristic lengths.
            *   **Quantification of Uniformity**: Deviations from perfect periodicity (e.g., broadening or multiple peaks in the FFT, irregular decay in autocorrelation) would be quantified to derive a uniformity score ($\sigma$). Lower $\sigma$ values would indicate higher structural uniformity.

3.  **Recipe Recommender**
    *   **Objective**: To suggest adjustments to `recipe_params` that could lead to improved yield or reduced specific defect types.
    *   **Methodology**: Based on detected defect types (from SEM analysis) and current process parameters, a rule-based system recommends changes. These proposed changes are then evaluated using the **AI-based Yield Prediction Model** to estimate the potential yield improvement ($\Delta P$).

## Data Integrity Audit & Handling (`process_analyzer.py`, `yield_model.py`, `metrology_engine.py`)

*   **Relationship Check**: The system relies on linking `process_runs` data (containing `recipe_params` and `id`) with `yield_results` (containing `efficiency` and `process_run_id`). Data is merged using these IDs to ensure each yield is associated with its corresponding process parameters.
*   **Missing Data Handling**: During data preparation for model training and inference, missing numerical values in `recipe_params` are imputed (e.g., by filling with the column's mean). Categorical features are handled by One-Hot Encoding, which can implicitly manage new or missing categories in inference by creating zero-filled columns if `handle_unknown='ignore'` is used. For critical missing primary keys or efficiency data, records are dropped to maintain model integrity.

## Optimization Notes

*   **Inference Speed**: The `yield_model.py` leverages `joblib` for efficient serialization and deserialization of the trained model and preprocessor, minimizing load times for inference.
*   **Multiprocessing**: The `metrology_engine.py` is designed to utilize `multiprocessing.Pool` for concurrent processing of multiple SEM images, significantly speeding up the analysis of large datasets.

## Setup and Usage

(This section would contain instructions on how to set up the environment, run `yield_model.py` to train the model, and then execute `metrology_engine.py` for analysis. It would also detail how to use `golden_recipe_summarizer.py`.)
