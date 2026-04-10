import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class MusicTransformer(nn.Module):
    def __init__(self, input_dim=128, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(MusicTransformer, self).__init__()
        self.d_model = d_model
        
        # Embed the 128-key piano roll into the Transformer dimension
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # For pure autoregressive generation without a separate "source" sequence,
        # a masked TransformerEncoder acts as a GPT-style Decoder.
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Project back to 128 keys
        self.fc_out = nn.Linear(d_model, input_dim)

    def generate_square_subsequent_mask(self, sz):
        """
        Algorithm 3 Line 5: Enforces p(x_t | x_{<t}) by hiding future tokens.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask=None):
        # Embed and add positional context
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Pass through masked transformer
        output = self.transformer(src, mask=src_mask)
        
        # Output raw logits for numerical stability in the loss function
        logits = self.fc_out(output)
        return logits