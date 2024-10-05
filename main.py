import os
import subprocess

def run_data_preprocessing():
    print("Step 1: Preprocessing the Data...")
    subprocess.run(["python", "scripts/data_preprocessing.py"])
    print("Data Preprocessing Complete.\n")

def train_models():
    print("Step 2: Training the Models...")
    subprocess.run(["python", "scripts/train_autoencoder.py"])
    print("Model Training Complete.\n")

def evaluate_models():
    print("Step 3: Evaluating the Models...")
    subprocess.run(["python", "scripts/evaluate_autoencoder.py"])
    print("Model Evaluation Complete.\n")

if __name__ == "__main__":
    print("===== Starting Anomaly Detection Project =====")
    
    # Ensure data folder exists
    if not os.path.exists("/Users/andresaranguren/cs_project/ml_projects/anomaly_detection/data/online_retail.xlsx"):
        print("Dataset not found. Please ensure the dataset is placed in the 'data/' folder.")
    else:
        # Run each step in sequence
        run_data_preprocessing()
        train_models()
        evaluate_models()

    print("===== Anomaly Detection Project Completed =====")
