import torch
import torch.nn as nn

class LSTMVAE(nn.Module):
    def __init__(self, seq_length=64, input_dim=128, hidden_dim=256, latent_dim=256):
        super(LSTMVAE, self).__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim

        # Encoder q_phi(z|X)
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) # Using log(sigma^2) for numerical stability

        # Decoder p_theta(X|z)
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

        # Initialize to prevent dead neurons
        nn.init.xavier_uniform_(self.fc_out.weight)
        self.fc_out.bias.data.fill_(0.0)

    def encode(self, x):
        """Algorithm 2, Line 4: (mu, sigma) = Encoder_phi(X)"""
        _, (hidden, _) = self.encoder_lstm(x)
        h_last = hidden[-1]
        
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Algorithm 2, Line 5: z = mu + sigma * epsilon, epsilon ~ N(0, I)"""
        std = torch.exp(0.5 * logvar) # Convert logvar back to standard deviation (sigma)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z):
        """Algorithm 2, Line 6: X_hat = Decoder_theta(z)"""
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_length, 1)
        lstm_out, _ = self.decoder_lstm(z_repeated)
        
        # Linear output to prevent vanishing MSE gradients
        X_hat = self.fc_out(lstm_out)
        return X_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        X_hat = self.decode(z)
        return X_hat, mu, logvar