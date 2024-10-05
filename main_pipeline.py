import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_and_preprocess_data

# Step 1: Data Exploration
def explore_data(file_path):
    print("Step 1: Exploring the data...")

    # Load the dataset
    df = pd.read_excel(file_path)

    # Basic information about the dataset
    print("\nBasic Information about the Dataset:")
    print(df.info())

    # Summary statistics of the numerical columns
    print("\nSummary Statistics:")
    print(df.describe())

    # Plot distributions of relevant columns
    print("\nPlotting distributions...")
    plt.figure(figsize=(12, 6))
    
    # Plot Quantity distribution
    plt.subplot(1, 3, 1)
    sns.histplot(df['Quantity'], kde=True, bins=30)
    plt.title('Quantity Distribution')

    # Plot UnitPrice distribution
    plt.subplot(1, 3, 2)
    sns.histplot(df['UnitPrice'], kde=True, bins=30)
    plt.title('UnitPrice Distribution')

    # Plot TotalPrice distribution (we'll create this feature in preprocessing)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    plt.subplot(1, 3, 3)
    sns.histplot(df['TotalPrice'], kde=True, bins=30)
    plt.title('TotalPrice Distribution')

    plt.tight_layout()
    plt.show()

# Step 2: Preprocess the data
def preprocess_data():
    print("Step 2: Preprocessing the data...")
    os.system('python data_preprocessing.py')

# Step 3: Train the autoencoder
def train_model():
    print("Step 3: Training the autoencoder...")
    os.system('python train_autoencoder.py')

# Step 4: Evaluate the autoencoder
def evaluate_model():
    print("Step 4: Evaluating the autoencoder...")
    os.system('python evaluate_autoencoder.py')

if __name__ == "__main__":
    # Data file path
    data_file = "data/Online Retail.xlsx"
    
    # Step 1: Data exploration
    explore_data(data_file)

    # Step 2: Preprocess the data
    preprocess_data()

    # Step 3: Train the model
    train_model()

    # Step 4: Evaluate the model and show results
    evaluate_model()

    print("Pipeline complete.")

