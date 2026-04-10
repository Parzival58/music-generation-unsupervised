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
from src.models.autoencoder import LSTMAutoencoder
from src.preprocessing.midi_parser import load_midi, midi_to_piano_roll
from src.preprocessing.piano_roll import LazyMIDIDataset

def extract_active_rolls(file_list, seq_length, max_files, threshold=20):
    """Helper to extract piano rolls that have at least 'threshold' active notes."""
    rolls = []
    for file in file_list:
        if len(rolls) >= max_files:
            break
        try:
            pr = midi_to_piano_roll(load_midi(file))
            if pr is not None and pr.shape[0] > seq_length:
                # Lowered threshold to 20 to allow sparse drum beats through
                if np.sum(pr) > threshold: 
                    rolls.append(pr)
        except Exception:
            continue
    return rolls

def train_autoencoder(epochs=20, batch_size=64, learning_rate=1e-3, seq_length=64, target_genre="funk", num_files=100):
    print(f"--- Task 1: Training LSTM Autoencoder (Strict Algorithm 1) ---")
    
    # 1. Require: MIDI dataset D = {Xi} (single genre)
    print(f"Scanning for genre: '{target_genre}'...")
    genre_files = glob.glob(f'data/raw_midi/**/*{target_genre}*.mid', recursive=True)
    all_rolls = extract_active_rolls(genre_files, seq_length, num_files)
    
    # SMART FALLBACK: If the target genre has no active tracks, use the whole dataset
    if len(all_rolls) == 0:
        print(f"Warning: No sufficiently active '{target_genre}' tracks found.")
        print("Falling back to general dataset to ensure training can proceed...")
        any_files = glob.glob('data/raw_midi/**/*.mid', recursive=True)
        all_rolls = extract_active_rolls(any_files, seq_length, num_files, threshold=10) # Even lower threshold for fallback

    if len(all_rolls) == 0:
        print("CRITICAL ERROR: No active MIDI tracks could be extracted from your dataset.")
        print("Check if your data/raw_midi/ folder contains valid files.")
        return

    print(f"Success: Loaded {len(all_rolls)} active tracks for dataset D.")
    
    # 2. PyTorch DataLoader
    dataset = LazyMIDIDataset(all_rolls, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Initialize Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = LSTMAutoencoder(seq_length=seq_length).to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    # 4. The Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for X, _ in dataloader:
            X = X.to(device)
            optimizer.zero_grad()
            
            X_hat, z = model(X)
            L_AE = criterion(X_hat, X)
            
            L_AE.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step()
            
            epoch_loss += L_AE.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], L_AE: {avg_loss:.4f}")
        
    # 5. Save the required deliverables
    os.makedirs('outputs/generated_midis', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    
    torch.save(model.state_dict(), 'outputs/generated_midis/lstm_autoencoder.pth')
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), loss_history, marker='o', label='Train Loss')
    plt.title('Task 1: LSTM Autoencoder Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/plots/task1_loss_curve.png')
    print("--- Autoencoder Training Complete ---")

if __name__ == "__main__":
    train_autoencoder()