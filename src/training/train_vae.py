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
from src.preprocessing.piano_roll import MIDIDataset, create_sequences

def vae_loss_function(recon_x, x, mu, logvar, beta=0.1):
    # Reconstruction Loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + (beta * kld_loss), recon_loss, kld_loss

def train_vae(epochs=20, batch_size=64, learning_rate=1e-3):
    print("Preparing VAE dataset...")
    midi_files = glob.glob('data/raw_midi/**/*.mid', recursive=True)
    all_sequences = []
    
    # Let's use more files this time for better diversity
    for file in midi_files[:100]: 
        midi_data = load_midi(file)
        piano_roll = midi_to_piano_roll(midi_data)
        if piano_roll is not None and piano_roll.shape[0] > 64:
            all_sequences.append(create_sequences(piano_roll))
            
    full_data = np.concatenate(all_sequences, axis=0)
    dataset = MIDIDataset(full_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'total': [], 'recon': [], 'kld': []}

    print("Starting VAE training...")
    for epoch in range(epochs):
        model.train()
        t_loss, r_loss, k_loss = 0, 0, 0
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(inputs)
            loss, recon_val, kld_val = vae_loss_function(recon, inputs, mu, logvar)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()
            r_loss += recon_val.item()
            k_loss += kld_val.item()
            
        avg_t = t_loss / len(dataloader)
        history['total'].append(avg_t)
        print(f"Epoch [{epoch+1}/{epochs}] Total Loss: {avg_t:.2f} (Recon: {r_loss/len(dataloader):.2f}, KLD: {k_loss/len(dataloader):.2f})")

    # Save deliverables
    torch.save(model.state_dict(), 'outputs/generated_midis/lstm_vae.pth')
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['total'], label='Total VAE Loss')
    plt.title('Task 2: VAE Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/plots/task2_vae_loss.png')
    print("VAE Training Complete. Deliverables saved.")

if __name__ == "__main__":
    train_vae()