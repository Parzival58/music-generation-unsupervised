import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(LSTMAutoencoder, self). __init__()
        # Encoder: Compresses sequence to a single vector
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.latent_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: Reconstructs sequence from the latent vector
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Encoding
        _, (hidden, _) = self.encoder(x)
        z = self.latent_layer(hidden[-1]) # Latent vector z [cite: 41]
        
        # Decoding
        d_in = self.decoder_input(z).unsqueeze(1).repeat(1, seq_len, 1)
        d_out, _ = self.decoder(d_in)
        reconstruction = torch.sigmoid(self.output_layer(d_out)) # Output between 0 and 1
        
        return reconstruction