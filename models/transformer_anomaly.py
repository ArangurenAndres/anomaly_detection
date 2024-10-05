import torch
import torch.nn as nn

class TransformerAnomalyDetector(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, hidden_dim):
        super(TransformerAnomalyDetector, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.transformer = nn.Transformer(d_model=embed_dim, nhead=num_heads, num_encoder_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)  # Output: anomaly score

    def forward(self, x):
        x = self.transformer(x)
        output = self.fc(x[-1, :, :])  # Use the output from the last time step
        return output


# Correct model initialization (use embed_dim, not input_size)
embed_dim = 12  # Ensure embed_dim is divisible by num_heads
num_heads = 4
num_layers = 2
hidden_dim = 128

transformer_model = TransformerAnomalyDetector(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)

