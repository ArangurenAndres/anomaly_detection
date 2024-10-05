import torch
import matplotlib.pyplot as plt
from lstm_anomaly import LSTMAnomalyDetector
from transformer_anomaly import TransformerAnomalyDetector
from data_preprocessing import load_and_preprocess_data

def evaluate_model(model, test_loader):
    model.eval()
    anomaly_scores = []
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            outputs = model(inputs)
            anomaly_scores.append(outputs.numpy())
    
    return anomaly_scores

def plot_anomaly_scores(anomaly_scores, model_name):
    plt.figure(figsize=(10,6))
    plt.plot(anomaly_scores, label=f'{model_name} Anomaly Scores')
    plt.title(f'Anomaly Detection using {model_name}')
    plt.xlabel('Test Samples')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load preprocessed data
    _, test_data = load_and_preprocess_data("/Users/andresaranguren/cs_project/ml_projects/anomaly_detection/data/online_retail.xlsx")
    test_loader = DataLoader(TensorDataset(torch.tensor(test_data.values, dtype=torch.float32)), batch_size=32, shuffle=False)
    
    # Load models
    lstm_model = LSTMAnomalyDetector(input_size=3, hidden_size=64, num_layers=2)
    transformer_model = TransformerAnomalyDetector(input_size=3, num_heads=4, num_layers=2, hidden_dim=128)
    
    # Evaluate LSTM
    lstm_scores = evaluate_model(lstm_model, test_loader)
    plot_anomaly_scores(lstm_scores, 'LSTM')
    
    # Evaluate Transformer
    transformer_scores = evaluate_model(transformer_model, test_loader)
    plot_anomaly_scores(transformer_scores, 'Transformer')
