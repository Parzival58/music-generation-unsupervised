import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Ensure custom modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.vae import LSTMVAE
from src.preprocessing.midi_parser import load_midi, midi_to_piano_roll
from src.preprocessing.piano_roll import MIDIDataset, create_sequences

def vae_loss_function(recon_x, x, mu, logvar, beta=0.1):
    """
    VAE Loss = Reconstruction Loss + Beta * KL Divergence
    Formula follows the assignment requirement for Task 2.
    """
    # Reconstruction Loss (MSE) - using 'sum' to keep values significant
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + (beta * kld_loss), recon_loss, kld_loss

def train_vae(epochs=20, batch_size=64, learning_rate=1e-3, num_files=50):
    print(f"--- Preparing VAE Dataset (Target: {num_files} files) ---")
    
    # Search for MIDI files (covering both local and Colab paths)
    midi_files = glob.glob('data/raw_midi/**/*.mid', recursive=True)
    
    if not midi_files:
        print("ERROR: No MIDI files found. Please check data/raw_midi/ directory.")
        return

    all_sequences = []
    processed_count = 0
    
    # Process files with a 'Heartbeat' print to prevent Cloud Timeouts
    for i, file in enumerate(midi_files):
        if processed_count >= num_files:
            break
            
        # Update progress on the same line
        print(f"[{processed_count + 1}/{num_files}] Extracting: {os.path.basename(file)}...", end="\r")
        sys.stdout.flush()
        
        midi_data = load_midi(file)
        piano_roll = midi_to_piano_roll(midi_data)
        
        if piano_roll is not None and piano_roll.shape[0] > 64:
            seqs = create_sequences(piano_roll, seq_length=64)
            all_sequences.append(seqs)
            processed_count += 1
            
    print(f"\nExtraction complete. Total sequences: {sum(len(s) for s in all_sequences)}")
    
    full_data = np.concatenate(all_sequences, axis=0)
    dataset = MIDIDataset(full_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training VAE on: {device}")
    
    model = LSTMVAE(seq_length=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'total': [], 'recon': [], 'kld': []}

    print("--- Starting VAE Training ---")
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
            
        # Calculate averages for logging
        n_batches = len(dataloader)
        avg_t = t_loss_accum / n_batches
        avg_r = r_loss_accum / n_batches
        avg_k = k_loss_accum / n_batches
        
        history['total'].append(avg_t)
        history['recon'].append(avg_r)
        history['kld'].append(avg_k)
        
        print(f"Epoch [{epoch+1}/{epochs}] Total: {avg_t:.2f} | Recon: {avg_r:.2f} | KLD: {avg_k:.2f}")

    # Ensure output directories exist
    os.makedirs('outputs/generated_midis', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)

    # Save weights
    torch.save(model.state_dict(), 'outputs/generated_midis/lstm_vae.pth')
    
    # Save Loss Curve Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['total'], label='Total VAE Loss')
    plt.plot(history['recon'], label='Reconstruction Loss', linestyle='--')
    plt.title('Task 2: VAE Training Progress (Total vs Recon)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Sum')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/plots/task2_vae_loss.png')
    
    print("--- VAE Training Complete. Weight and Plot saved. ---")

if __name__ == "__main__":
    train_vae()