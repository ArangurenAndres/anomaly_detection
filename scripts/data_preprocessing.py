import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_excel(file_path)

    # Clean the data: remove negative quantities and prices, and drop NaN Customer IDs
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df.dropna(subset=['CustomerID'], inplace=True)
    
    # Create new features
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Select relevant columns for anomaly detection
    df = df[['CustomerID', 'Quantity', 'TotalPrice']]

    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Split the data into training and test sets
    train_data, test_data = train_test_split(df_scaled, test_size=0.2, random_state=42)

    return train_data, test_data, scaler

if __name__ == "__main__":
    train_data, test_data, scaler = load_and_preprocess_data("/Users/andresaranguren/cs_project/ml_projects/anomaly_detection/data/online_retail.xlsx")
    print("Data preprocessing complete.")
