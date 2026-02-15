import pandas as pd
import numpy as np
import os
import struct
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your local dataset on the M1
DATA_PATH = '/Users/vishalalimchandani/Downloads/Ridge_2nd Deg_HE_ieee_Fraud_Detection/ieee-fraud-detection_data/train_transaction.csv'

# Output directory for C++ binary files
OUTPUT_DIR = "cpp_inputs"

# FHE Constraints
NUM_FEATURES = 10         # Limit to 10 to fit in 8GB RAM (CKKS Ciphertext size)
SCALING_RANGE = (-0.5, 0.5) # Critical: CKKS explodes if values > 1.0. We center at 0.
RIDGE_ALPHA = 1.0         # L2 Regularization to keep weights small

def get_top_features(df, target_col='isFraud', n=10):
    """
    Selects the top N numerical features most correlated with the target.
    This ensures we spend our expensive FHE compute budget on high-signal data.
    """
    print(f"--- Feature Selection (Top {n}) ---")
    
    # 1. Filter for numerical columns only (exclude object/string types)
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 2. Handle NaN values
    # IEEE-CIS has many NaNs. Filling with 0 is standard for sparse transaction data.
    numeric_df = numeric_df.fillna(0)
    
    # 3. Calculate Correlation
    # We drop the target itself from the features list
    correlations = numeric_df.drop(columns=[target_col]).corrwith(numeric_df[target_col])
    
    # 4. Select Top N by absolute magnitude
    top_features = correlations.abs().sort_values(ascending=False).head(n).index.tolist()
    
    print(f"Selected Features: {top_features}")
    return top_features

def save_to_binary(filename, data):
    """
    Saves a numpy array to a flat binary file.
    C++ will ingest this using: file.read(reinterpret_cast<char*>...)
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Ensure data is standard double precision (64-bit float)
    data = data.astype(np.float64)
    
    with open(filepath, 'wb') as f:
        f.write(data.tobytes())
    
    # Print size for debugging (Useful to catch empty files)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Saved {filename} ({data.shape}): {file_size_mb:.2f} MB")

def main():
    print("Starting Preprocessing Pipeline...")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at: {DATA_PATH}")
    
    print("Loading CSV (this might take a moment)...")
    # Load only the first 20k rows to speed up feature selection (Dataset is huge)
    # We can use more for actual training if needed, but 20k is enough for stable correlation.
    df = pd.read_csv(DATA_PATH, nrows=50000)
    
    # 2. Select Features
    top_features = get_top_features(df, target_col='isFraud', n=NUM_FEATURES)
    
    # Extract X (Features) and y (Target)
    X_raw = df[top_features].fillna(0).values
    y = df['isFraud'].values
    
    # 3. Cryptographic Range Conditioning
    # Scale strictly to [-0.5, 0.5] to prevent CKKS scale overflow during x^2
    print(f"Scaling data to range {SCALING_RANGE}...")
    scaler = MinMaxScaler(feature_range=SCALING_RANGE)
    X_scaled = scaler.fit_transform(X_raw)
    
    # 4. Prepare Polynomial Features for Training
    # The C++ circuit calculates x^2 internally. 
    # To get the correct weights, we must train on [x, x^2] in Python.
    X_sq_scaled = X_scaled ** 2
    X_combined = np.hstack([X_scaled, X_sq_scaled]) # Shape: (Rows, 20)
    
    # 5. Train Ridge Regression
    print("Training Ridge Regression Model...")
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(X_combined, y)
    
    # 6. Extract Weights
    # The model returns 20 weights. First 10 are Linear (w1), Last 10 are Quad (w2).
    all_weights = model.coef_
    w_linear = all_weights[:NUM_FEATURES]
    w_quad   = all_weights[NUM_FEATURES:]
    
    # Intercept is usually handled as a plaintext addition at the end
    intercept = np.array([model.intercept_])
    
    print(f"Linear Weights Sample: {w_linear[:3]}")
    print(f"Quad Weights Sample:   {w_quad[:3]}")
    
    # 7. Export for C++ Ingestion
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # We export a specific "Test Set" for the C++ engine to process.
    # We take the first 4096 rows because 4096 is a common Batch Size (N/2) in CKKS.
    BATCH_SIZE = 4096
    if len(X_scaled) < BATCH_SIZE:
        print(f"Warning: Dataset smaller than batch size {BATCH_SIZE}. Padding might be needed.")
    
    X_test = X_scaled[:BATCH_SIZE]
    
    # Generate Ground Truth predictions to verify C++ later
    # y_pred = w1*x + w2*x^2 + b
    y_ground_truth = model.predict(X_combined[:BATCH_SIZE])
    
    # Save Files
    save_to_binary("x_test.bin", X_test)               # The input data
    save_to_binary("weights_linear.bin", w_linear)     # w1
    save_to_binary("weights_quad.bin", w_quad)         # w2
    save_to_binary("bias.bin", intercept)              # b
    save_to_binary("y_ground_truth.bin", y_ground_truth) # For verification
    
    print("\n--- Pipeline Complete ---")
    print(f"Artifacts generated in folder: '{os.path.abspath(OUTPUT_DIR)}'")
    print("Next Step: Run C++ Ingestion.")

if __name__ == "__main__":
    main()