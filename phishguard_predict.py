import argparse
import json
import joblib
import pandas as pd
import numpy as np
import os
import sys
from util import preprocess_data, ThresholdClassifier


def main():
    parser = argparse.ArgumentParser(description="Run PhishGuard Classification on an email JSON file.")
    parser.add_argument("filename", help="Path to the JSON file containing the email data.")
    args = parser.parse_args()
    
    if not os.path.exists(args.filename):
        print(f"Error: File '{args.filename}' not found.")
        sys.exit(1)
    
    model_path = 'phishguard_deployment.joblib'

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)
        
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    try:
        with open(args.filename, 'r') as f:
            email_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{args.filename}'.")
        sys.exit(1)
        
    email_df = pd.DataFrame([email_data])
    X = preprocess_data(email_df)
    print("Running classification...")
    try:
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        label = "PHISHING" if prediction == 1 else "LEGITIMATE"
        color = "\033[91m" if prediction == 1 else "\033[92m" # Red for Phish, Green for Legit
        reset = "\033[0m"
        
        print("-" * 30)
        print(f"Result: {color}{label}{reset}")
        print(f"Phishing Probability: {probability:.3f}")
        print("-" * 30)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()