import torch
import torch.nn as nn

class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output: anomaly score

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Only use last time step output
        return output

# Instantiate the model
if __name__ == "__main__":
    input_size = 3  # Number of features (CustomerID, Quantity, TotalPrice, TransactionHour)
    hidden_size = 64
    num_layers = 2
    
    model = LSTMAnomalyDetector(input_size, hidden_size, num_layers)
    print("LSTM Anomaly Detector model initialized.")
