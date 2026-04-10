import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.vae import LSTMVAE
from src.preprocessing.midi_parser import load_midi, midi_to_piano_roll
from src.preprocessing.piano_roll import LazyMIDIDataset

def vae_loss_function(recon_x, x, mu, logvar, beta):
    # Use BCE instead of MSE to force the model to capture sparse drum hits
    # Binary Cross Entropy is much more sensitive to "missed" notes
    bce_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL Divergence
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss: We keep Beta low (0.001) so reconstruction is the priority
    return bce_loss + (beta * kld_loss), bce_loss, kld_loss

def train_vae(epochs=20, batch_size=64, learning_rate=5e-4, num_files=100):
    print(f"Task 2: VAE Training")
    midi_files = glob.glob('data/raw_midi/**/*.mid', recursive=True)
    all_piano_rolls = []
    processed_count = 0
    
    for i, file in enumerate(midi_files):
        if processed_count >= num_files: break
        midi_data = load_midi(file)
        piano_roll = midi_to_piano_roll(midi_data)
        if piano_roll is not None and piano_roll.shape[0] > 64:
            # Threshold the piano roll to 0 or 1 for BCE compatibility
            piano_roll = (piano_roll > 0).astype(np.float32)
            all_piano_rolls.append(piano_roll)
            processed_count += 1
            
    dataset = LazyMIDIDataset(all_piano_rolls, seq_length=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMVAE(seq_length=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"--- Training with Binary Cross-Entropy ---")
    for epoch in range(epochs):
        model.train()
        beta = min(0.001, (epoch / 25.0) * 0.001) 
        
        t_loss_accum, bce_accum = 0, 0
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(inputs)
            loss, bce_val, kld_val = vae_loss_function(recon, inputs, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            
            t_loss_accum += loss.item()
            bce_accum += bce_val.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] Total Loss: {t_loss_accum/len(dataloader):.2f} | BCE (Recon): {bce_accum/len(dataloader):.2f}")

    os.makedirs('outputs/generated_midis', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/generated_midis/lstm_vae.pth')
    print("--- VAE Fix Complete ---")

if __name__ == "__main__":
    train_vae()