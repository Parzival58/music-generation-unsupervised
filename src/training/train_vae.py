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

def extract_active_rolls(file_list, seq_length, max_files):
    rolls = []
    print(f"Scanning {len(file_list)} files for active multi-genre tracks...")
    for file in file_list:
        if len(rolls) >= max_files: break
        try:
            pr = midi_to_piano_roll(load_midi(file), seq_length=seq_length)
            if pr is not None and pr.shape[0] >= seq_length:
                rolls.append(pr)
        except Exception:
            continue
    return rolls

def train_vae(epochs=30, batch_size=64, learning_rate=1e-3, seq_length=64, num_files=200):
    print("--- Task 2: Training VAE (Strict Algorithm 2) ---")
    
    # Require: Multi-genre MIDI dataset D = {Xi}
    # Notice we don't filter by a specific genre like we did in Task 1
    multi_genre_files = glob.glob('data/raw_midi/**/*.mid', recursive=True)
    all_rolls = extract_active_rolls(multi_genre_files, seq_length, num_files)
    
    if not all_rolls:
        print("CRITICAL ERROR: No active tracks extracted.")
        return

    print(f"Success: Loaded {len(all_rolls)} active multi-genre tracks.")
    
    # Use our intelligent dataset to drop silent slices
    dataset = LazyMIDIDataset(all_rolls, seq_length=seq_length, min_notes=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMVAE(seq_length=seq_length).to(device)
    
    # Line 1: Initialize parameters phi, theta
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    # Line 2: for epoch = 1 to E do
    print("Starting VAE training loop...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Require: KL weight beta (We slowly anneal it to prioritize reconstruction first)
        beta = min(0.05, (epoch / (epochs * 0.5)) * 0.05)
        
        # Line 3: for each batch X in D do
        for X, _ in dataloader:
            X = X.to(device)
            optimizer.zero_grad()
            
            # Lines 4, 5, 6
            X_hat, mu, logvar = model(X)
            
            # Line 7: Reconstruction loss (MSE)
            L_recon = nn.functional.mse_loss(X_hat, X, reduction='sum')
            
            # Line 8: KL divergence
            L_KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Line 9: Total VAE objective
            L_VAE = L_recon + (beta * L_KL)
            
            # Line 10: Update parameters
            L_VAE.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += L_VAE.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] | Beta: {beta:.4f} | L_VAE: {avg_loss:.4f}")
        
    os.makedirs('outputs/generated_midis', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    
    # Save Model Weights
    torch.save(model.state_dict(), 'outputs/generated_midis/lstm_vae.pth')
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), loss_history, marker='o', color='green', label='VAE Loss')
    plt.title('Task 2: VAE Training Loss (Algorithm 2)')
    plt.xlabel('Epochs')
    plt.ylabel('L_VAE (MSE + beta*KL)')
    plt.grid(True)
    plt.legend()
    plt.savefig('outputs/plots/task2_loss_curve.png')
    print("--- VAE Training Complete ---")

if __name__ == "__main__":
    train_vae()