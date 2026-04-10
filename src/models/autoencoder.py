import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        # Latent representation z
        z = hidden[-1] 
        return z

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=256, output_dim=128, seq_length=64, num_layers=2):
        super(LSTMDecoder, self).__init__()
        self.seq_length = seq_length
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Initialize weights to help break symmetry
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.0)

    def forward(self, z):
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_length, 1)
        lstm_out, _ = self.lstm(z_repeated)
        
        # NO SIGMOID: We output raw linear values to prevent vanishing gradients
        reconstruction = self.fc(lstm_out)
        return reconstruction

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_length=64, input_dim=128, hidden_dim=256):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        # Note: latent_dim matches hidden_dim (256) based on our encoder output
        self.decoder = LSTMDecoder(latent_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=input_dim, seq_length=seq_length)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z