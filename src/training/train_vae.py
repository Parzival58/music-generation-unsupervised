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
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + (beta * kld_loss), recon_loss, kld_loss

def train_vae(epochs=50, batch_size=64, learning_rate=1e-3, num_files=100):
    print(f"--- Preparing VAE Dataset ({num_files} files) ---")
    midi_files = glob.glob('data/raw_midi/**/*.mid', recursive=True)
    all_piano_rolls = []
    processed_count = 0
    
    for i, file in enumerate(midi_files):
        if processed_count >= num_files: break
        midi_data = load_midi(file)
        piano_roll = midi_to_piano_roll(midi_data)
        if piano_roll is not None and piano_roll.shape[0] > 64:
            all_piano_rolls.append(piano_roll)
            processed_count += 1
            
    dataset = LazyMIDIDataset(all_piano_rolls, seq_length=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMVAE(seq_length=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'total': [], 'recon': [], 'kld': []}

    print(f"--- Starting VAE Training Loop ---")
    for epoch in range(epochs):
        model.train()
        # Using a much smaller beta (max 0.01) to prioritize reconstruction
        beta = min(0.01, (epoch / 25.0) * 0.01) 
        
        t_loss_accum, r_loss_accum, k_loss_accum = 0, 0, 0
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(inputs)
            loss, recon_val, kld_val = vae_loss_function(recon, inputs, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            
            t_loss_accum += loss.item()
            r_loss_accum += recon_val.item()
            k_loss_accum += kld_val.item()
            
        avg_t = t_loss_accum / len(dataloader)
        history['total'].append(avg_t)
        print(f"Epoch [{epoch+1}/{epochs}] Beta: {beta:.4f} | Total: {avg_t:.2f} | Recon: {r_loss_accum/len(dataloader):.2f}")

    os.makedirs('outputs/generated_midis', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/generated_midis/lstm_vae.pth')
    print("--- VAE Training Complete ---")

if __name__ == "__main__":
    train_vae()