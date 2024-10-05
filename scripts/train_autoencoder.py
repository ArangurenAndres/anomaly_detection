import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.autoencoder_model import Autoencoder
from data_preprocessing import load_and_preprocess_data

def train_autoencoder(model, train_loader, epochs=20, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs = data[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    # Save the trained model after training is complete
    torch.save(model.state_dict(), "autoencoder.pth")
    print("Model saved successfully.")

if __name__ == "__main__":
    # Load preprocessed data
    train_data, test_data, scaler = load_and_preprocess_data("/Users/andresaranguren/cs_project/ml_projects/anomaly_detection/data/online_retail.xlsx")

    # Convert data to PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    # Create DataLoader
    train_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=True)

    # Initialize and train the autoencoder
    model = Autoencoder(input_dim=train_data.shape[1])
    train_autoencoder(model, train_loader, epochs=20)
