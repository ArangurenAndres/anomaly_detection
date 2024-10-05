import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# For standalone execution
if __name__ == "__main__":
    data_file = "/Users/andresaranguren/cs_project/ml_projects/anomaly_detection/data/online_retail.xlsx"
    explore_data(data_file)
