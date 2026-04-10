import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    """
    Encodes the input sequence X into a fixed-size latent vector z.
    Algorithm 1, Line 4: z = f_phi(X)
    """
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        
        # We take the hidden state of the LAST layer as our latent representation z
        z = hidden[-1] 
        return z

class LSTMDecoder(nn.Module):
    """
    Decodes the latent vector z back into the reconstructed sequence X_hat.
    Algorithm 1, Line 5: X_hat = g_theta(z)
    """
    def __init__(self, latent_dim=256, hidden_dim=256, output_dim=128, seq_length=64, num_layers=2):
        super(LSTMDecoder, self).__init__()
        self.seq_length = seq_length
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        # Initialization to prevent dead neurons (silence collapse)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)

    def forward(self, z):
        # Repeat the static latent vector for the entire sequence length
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_length, 1)
        
        lstm_out, _ = self.lstm(z_repeated)
        reconstruction = self.fc(lstm_out)
        
        # Apply sigmoid to match the normalized [0, 1] input data
        return self.sigmoid(reconstruction)

class LSTMAutoencoder(nn.Module):
    """
    The full Task 1 Model connecting the Encoder and Decoder.
    """
    def __init__(self, seq_length=64, input_dim=128, hidden_dim=256):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.decoder = LSTMDecoder(
            latent_dim=hidden_dim, 
            hidden_dim=hidden_dim, 
            output_dim=input_dim, 
            seq_length=seq_length
        )

    def forward(self, x):
        z = self.encoder(x)       
        x_hat = self.decoder(z)  
        return x_hat, z