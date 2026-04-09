import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Path setup to ensure internal imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.vae import LSTMVAE
from src.preprocessing.midi_parser import load_midi, midi_to_piano_roll
from src.preprocessing.piano_roll import LazyMIDIDataset

def vae_loss_function(recon_x, x, mu, logvar, beta=0.1):
    """
    VAE Loss = Reconstruction Loss + Beta * KL Divergence
    """
    # Reconstruction Loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence: Forces latent space to be Normal Distribution N(0,1)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + (beta * kld_loss), recon_loss, kld_loss

def train_vae(epochs=20, batch_size=64, learning_rate=1e-3, num_files=50):
    print(f"--- Preparing Memory-Efficient VAE Dataset (Target: {num_files} files) ---")
    
    midi_files = glob.glob('data/raw_midi/**/*.mid', recursive=True)
    if not midi_files:
        print("ERROR: No MIDI files found. Check data/raw_midi/ directory.")
        return

    all_piano_rolls = []
    processed_count = 0
    
    # Extract only full piano rolls to save RAM
    for i, file in enumerate(midi_files):
        if processed_count >= num_files:
            break
            
        print(f"[{processed_count + 1}/{num_files}] Loading into RAM: {os.path.basename(file)}...", end="\r")
        sys.stdout.flush()
        
        midi_data = load_midi(file)
        piano_roll = midi_to_piano_roll(midi_data)
        
        if piano_roll is not None and piano_roll.shape[0] > 64:
            all_piano_rolls.append(piano_roll)
            processed_count += 1
            
    print(f"\nSongs loaded. Initializing LazyMIDIDataset...")
    dataset = LazyMIDIDataset(all_piano_rolls, seq_length=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Total training sequences available: {len(dataset)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = LSTMVAE(seq_length=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'total': [], 'recon': [], 'kld': []}

    print("--- Starting VAE Training Loop ---")
    for epoch in range(epochs):
        model.train()
        t_loss_accum, r_loss_accum, k_loss_accum = 0, 0, 0
        
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            recon, mu, logvar = model(inputs)
            loss, recon_val, kld_val = vae_loss_function(recon, inputs, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            t_loss_accum += loss.item()
            r_loss_accum += recon_val.item()
            k_loss_accum += kld_val.item()
            
        avg_t = t_loss_accum / len(dataloader)
        history['total'].append(avg_t)
        print(f"Epoch [{epoch+1}/{epochs}] Total Loss: {avg_t:.2f} (Recon: {r_loss_accum/len(dataloader):.2f}, KLD: {k_loss_accum/len(dataloader):.2f})")

    # Finalize and Save
    os.makedirs('outputs/generated_midis', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/generated_midis/lstm_vae.pth')
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['total'], label='Total VAE Loss')
    plt.title('Task 2: VAE Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/plots/task2_vae_loss.png')
    print("--- Training Finished ---")

if __name__ == "__main__":
    train_vae()