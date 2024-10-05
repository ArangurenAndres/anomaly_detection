import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_excel(file_path)
    
    # Data cleaning
    df = df[df['Quantity'] > 0]  # Remove negative quantities
    df = df[df['UnitPrice'] > 0]  # Remove negative prices
    df.dropna(subset=['CustomerID'], inplace=True)  # Remove rows without CustomerID
    
    # Create features for anomaly detection
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']  # Create a total price column
    df['TransactionHour'] = df['InvoiceDate'].dt.hour  # Extract hour of transaction
    
    # Select relevant columns
    df = df[['CustomerID', 'Quantity', 'TotalPrice', 'TransactionHour']]
    
    # Encode CustomerID
    le = LabelEncoder()
    df['CustomerID'] = le.fit_transform(df['CustomerID'])
    
    # Normalize features
    scaler = StandardScaler()
    df[['Quantity', 'TotalPrice', 'TransactionHour']] = scaler.fit_transform(
        df[['Quantity', 'TotalPrice', 'TransactionHour']]
    )
    
    # Split the data
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    return train, test

if __name__ == "__main__":
    train_data, test_data = load_and_preprocess_data("/Users/andresaranguren/cs_project/ml_projects/anomaly_detection/data/online_retail.xlsx")
    print("Data preprocessing complete. Training and testing sets created.")
