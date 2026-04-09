import torch
import torch.nn as nn

class LSTMVAE(nn.Module):
    def __init__(self, seq_length=64, input_dim=128, hidden_dim=256, latent_dim=128):
        super(LSTMVAE, self).__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        
        # Encoder: Collapses the sequence into a hidden state
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # The VAE "Bottleneck": mu and logvar
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: Expands the latent z back into a sequence
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def reparameterize(self, mu, logvar):
        """
        The Reparameterization Trick: z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 1. Encode
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1] # Take the last layer's hidden state
        
        # 2. Map to Gaussian parameters
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        
        # 3. Sample latent vector z
        z = self.reparameterize(mu, logvar)
        
        # 4. Decode
        z_projected = self.decoder_fc(z).unsqueeze(1).repeat(1, self.seq_length, 1)
        recon_out, _ = self.decoder_lstm(z_projected)
        x_hat = self.sigmoid(self.output_fc(recon_out))
        
        return x_hat, mu, logvar