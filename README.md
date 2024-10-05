# Anomaly Detection Using Autoencoder

## Project Overview

This project demonstrates an anomaly detection system using an autoencoder model on a marketplace dataset (e.g., retail transactions). The pipeline includes data exploration, preprocessing, training the autoencoder, and evaluating the model for anomaly detection. Anomalies are detected based on reconstruction errors from the autoencoder model, where higher errors are classified as anomalous behaviors.

## Data Exploration

The data exploration step involves analyzing the dataset for basic statistics, distributions, and key features like `Quantity`, `UnitPrice`, and `TotalPrice`. Visualizations are generated to provide insights into the structure and characteristics of the data.

### Data Preprocessing

During the preprocessing step:
- The dataset is cleaned by removing negative quantities and prices.
- A new feature, `TotalPrice`, is created by multiplying `Quantity` and `UnitPrice`.
- The data is scaled using standard normalization.
- The preprocessed data is split into training and testing sets.

## Autoencoder Model for Anomaly Detection

An **Autoencoder** is a type of neural network used for unsupervised learning, typically for data compression or anomaly detection. In this project, we use the autoencoder to reconstruct input data, and anomalies are detected based on how well the model can reconstruct the input.

### Model Architecture:
The autoencoder consists of two main parts:
1. **Encoder**: Compresses the input data into a lower-dimensional latent space.
   - Input layer: Takes in the input features (3 features in this case).
   - Hidden layers: Compress the input data into a latent representation.
   - Latent space: The compressed representation of the input data.

2. **Decoder**: Reconstructs the input data from the latent space.
   - Hidden layers: Expands the latent representation back into the original input dimension.
   - Output layer: Reconstructs the input data.

#### Autoencoder Architecture Used:
- **Encoder**:
  - Input -> Linear (input_dim, 16) -> ReLU
  - Linear (16, 8) -> ReLU
  - Linear (8, 4) -> Latent space
- **Decoder**:
  - Latent space -> Linear (4, 8) -> ReLU
  - Linear (8, 16) -> ReLU
  - Linear (16, input_dim) -> Output

### Anomaly Detection:

The autoencoder is trained to minimize the reconstruction error (mean squared error between the original input and the reconstructed output). If an input cannot be reconstructed well (i.e., the reconstruction error is high), it is flagged as an anomaly.

### Training and Evaluation Pipeline:
1. **Data Exploration**: Visualize and understand the data.
2. **Preprocessing**: Clean, normalize, and prepare the data.
3. **Training**: Train the autoencoder model on normal data to learn the normal patterns.
4. **Evaluation**: Evaluate the model by computing reconstruction errors on the test set. Data points with high reconstruction errors are classified as anomalies.

## Pipeline Execution

To run the entire pipeline, including data exploration, preprocessing, model training, and evaluation, execute the following command:

```bash
python main_pipeline.py
