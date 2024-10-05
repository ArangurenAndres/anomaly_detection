import torch
from torch.utils.data import DataLoader, TensorDataset
from models.lstm_anomaly import LSTMAnomalyDetector
from models.transformer_anomaly import TransformerAnomalyDetector
from data_preprocessing import load_and_preprocess_data

def train_model(model, train_loader, epochs=10, learning_rate=0.001):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs = data[0]
            
            # Reshape inputs for Transformer (batch_first=True is required for LSTM and Transformer compatibility)
            inputs = inputs.unsqueeze(1)  # Reshape to (batch_size, 1, input_dim)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss with outputs (anomaly score) and inputs
            loss = criterion(outputs, inputs.squeeze(1))  # Remove extra dimension for loss calculation
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')

if __name__ == "__main__":
    # Load preprocessed data
    train_data, _ = load_and_preprocess_data("/Users/andresaranguren/cs_project/ml_projects/anomaly_detection/data/online_retail.xlsx")
    
    # Prepare DataLoader
    train_loader = DataLoader(TensorDataset(torch.tensor(train_data.values, dtype=torch.float32)), batch_size=32, shuffle=True)
    
    # Initialize models
    lstm_model = LSTMAnomalyDetector(input_size=4, hidden_size=64, num_layers=2)
    transformer_model = TransformerAnomalyDetector(embed_dim=12, num_heads=4, num_layers=2, hidden_dim=128)  # Note: embed_dim is used here, and embed_dim must be divisible by num_heads
    
    # Train LSTM
    print("Training LSTM model...")
    train_model(lstm_model, train_loader, epochs=10)
    
    # Train Transformer
    print("Training Transformer model...")
    train_model(transformer_model, train_loader, epochs=10)
