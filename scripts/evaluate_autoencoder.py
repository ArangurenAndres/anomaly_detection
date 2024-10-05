import torch
import matplotlib.pyplot as plt
import numpy as np
from models.autoencoder_model import Autoencoder
from data_preprocessing import load_and_preprocess_data

def evaluate_model(model, test_data, threshold=0.05):
    model.eval()
    criterion = torch.nn.MSELoss(reduction='none')

    # Calculate reconstruction errors
    reconstructions = model(test_data)
    reconstruction_errors = criterion(reconstructions, test_data).mean(dim=1).detach().numpy()

    # Identify anomalies
    anomalies = reconstruction_errors > threshold

    return reconstructions.detach().numpy(), reconstruction_errors, anomalies

def plot_reconstruction_errors(reconstruction_errors, anomalies):
    plt.figure(figsize=(10, 6))
    plt.hist(reconstruction_errors, bins=50, alpha=0.75, label='Reconstruction Errors')
    plt.axvline(x=0.05, color='r', linestyle='--', label='Anomaly Threshold')
    plt.title('Reconstruction Errors and Anomalies')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def show_examples(test_data, reconstructions, reconstruction_errors, anomalies, num_examples=5):
    fig, axes = plt.subplots(num_examples, 2, figsize=(12, num_examples * 3))
    
    for i in range(num_examples):
        index = i  # You can randomize this if you want

        original = test_data[index].numpy()
        reconstructed = reconstructions[index]
        error = reconstruction_errors[index]
        is_anomaly = anomalies[index]

        # Show original input
        axes[i, 0].bar(range(len(original)), original, color='blue', alpha=0.6)
        axes[i, 0].set_title(f"Original (Index: {index})")

        # Show reconstruction
        axes[i, 1].bar(range(len(reconstructed)), reconstructed, color='green', alpha=0.6)
        axes[i, 1].set_title(f"Reconstruction\nError: {error:.4f} | Anomaly: {is_anomaly}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load preprocessed data
    train_data, test_data, scaler = load_and_preprocess_data("/Users/andresaranguren/cs_project/ml_projects/anomaly_detection/data/online_retail.xlsx")

    # Convert data to PyTorch tensors
    test_data = torch.tensor(test_data, dtype=torch.float32)

    # Load the trained model
    model = Autoencoder(input_dim=test_data.shape[1])
    model.load_state_dict(torch.load("/Users/andresaranguren/cs_project/ml_projects/anomaly_detection/autoencoder.pth"))
    print("Model loaded successfully.")

    # Evaluate the model
    reconstructions, reconstruction_errors, anomalies = evaluate_model(model, test_data, threshold=0.05)

    # Show histogram of reconstruction errors
    plot_reconstruction_errors(reconstruction_errors, anomalies)

    # Show examples of input and reconstruction
    show_examples(test_data, reconstructions, reconstruction_errors, anomalies, num_examples=5)
