import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    """
    Encodes the input sequence X into a fixed-size latent vector z.
    Formula: z = f_phi(X)
    """
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2):
        super(LSTMEncoder, self).__init__()
        # batch_first=True means inputs are (batch_size, seq_length, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x shape: (batch_size, seq_length, 128)
        _, (hidden, _) = self.lstm(x)
        
        # We take the hidden state of the LAST layer as our latent representation z
        # hidden shape: (num_layers, batch_size, hidden_dim)
        z = hidden[-1] 
        return z

class LSTMDecoder(nn.Module):
    """
    Decodes the latent vector z back into the reconstructed sequence X_hat.
    Formula: X_hat = g_theta(z)
    """
    def __init__(self, latent_dim=256, hidden_dim=256, output_dim=128, seq_length=64, num_layers=2):
        super(LSTMDecoder, self).__init__()
        self.seq_length = seq_length
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        
        # Maps the LSTM hidden states back to the 128 MIDI pitches
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Sigmoid activation to keep note velocities bounded between [0.0, 1.0]
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        # We repeat the static latent vector for the entire sequence length
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_length, 1)
        # z_repeated shape: (batch_size, seq_length, latent_dim)
        
        lstm_out, _ = self.lstm(z_repeated)
        # lstm_out shape: (batch_size, seq_length, hidden_dim)
        
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
        return x_hat
import torch.nn as nn

class LSTMEncoder(nn.Module):
    """
    Encodes the input sequence X into a fixed-size latent vector z.
    Formula: z = f_phi(X)
    """
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2):
        super(LSTMEncoder, self).__init__()
        # batch_first=True means inputs are (batch_size, seq_length, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x shape: (batch_size, seq_length, 128)
        _, (hidden, _) = self.lstm(x)
        
        # We take the hidden state of the LAST layer as our latent representation z
        # hidden shape: (num_layers, batch_size, hidden_dim)
        z = hidden[-1] 
        return z

class LSTMDecoder(nn.Module):
    """
    Decodes the latent vector z back into the reconstructed sequence X_hat.
    Formula: X_hat = g_theta(z)
    """
    def __init__(self, latent_dim=256, hidden_dim=256, output_dim=128, seq_length=64, num_layers=2):
        super(LSTMDecoder, self).__init__()
        self.seq_length = seq_length
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        
        # Maps the LSTM hidden states back to the 128 MIDI pitches
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Sigmoid activation to keep note velocities bounded between [0.0, 1.0]
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        # We repeat the static latent vector for the entire sequence length
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_length, 1)
        # z_repeated shape: (batch_size, seq_length, latent_dim)
        
        lstm_out, _ = self.lstm(z_repeated)
        # lstm_out shape: (batch_size, seq_length, hidden_dim)
        
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