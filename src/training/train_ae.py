import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Ensure Python can find your custom src modules from any folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.autoencoder import LSTMAutoencoder
from src.preprocessing.midi_parser import load_midi, midi_to_piano_roll
from src.preprocessing.piano_roll import LazyMIDIDataset

def train_autoencoder(epochs=30, batch_size=64, learning_rate=1e-3, seq_length=64, target_genre="funk", num_files=100):
    print(f"--- Task 1: Training LSTM Autoencoder (Strict Algorithm 1) ---")
    
    # 1. Require: MIDI dataset D = {Xi} (single genre)
    print(f"Filtering dataset for genre: '{target_genre}'")
    midi_files = glob.glob(f'data/raw_midi/**/*{target_genre}*.mid', recursive=True)
    
    if not midi_files:
        print(f"Warning: No files found for '{target_genre}'. Falling back to general dataset.")
        midi_files = glob.glob('data/raw_midi/**/*.mid', recursive=True)
        if not midi_files:
            print("ERROR: No MIDI files found at all.")
            return

    all_rolls = []
    for file in midi_files[:num_files]:
        try:
            pr = midi_to_piano_roll(load_midi(file))
            if pr is not None and pr.shape[0] > seq_length:
                # Preprocessing: Ensure dataset D only contains active musical sequences
                if np.sum(pr) > 100: 
                    all_rolls.append(pr)
        except Exception as e:
            continue
            
    print(f"Loaded {len(all_rolls)} highly active tracks for dataset D.")
    
    # 2. PyTorch DataLoader using Memory-Efficient Lazy Loader
    dataset = LazyMIDIDataset(all_rolls, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Initialize Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Require: Encoder f_phi, Decoder g_theta
    model = LSTMAutoencoder(seq_length=seq_length).to(device)
    
    # Line 6 specifies L_AE = ||X - X_hat||^2 (MSE Loss)
    criterion = nn.MSELoss(reduction='sum')
    
    # Line 1: Initialize parameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    # 4. The Training Loop
    # Line 2: for epoch = 1 to E do
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Line 3: for each batch X in D do
        for X, _ in dataloader:
            X = X.to(device)
            optimizer.zero_grad()
            
            # Line 4 & 5: z = f_phi(X) and X_hat = g_theta(z)
            X_hat, z = model(X)
            
            # Line 6: Compute reconstruction loss
            L_AE = criterion(X_hat, X)
            
            # Line 7: Update parameters (phi, theta) <- (phi, theta) - eta * grad(L_AE)
            L_AE.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Stability
            optimizer.step()
            
            epoch_loss += L_AE.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], L_AE: {avg_loss:.4f}")
        
    # 5. Save the required deliverables
    os.makedirs('outputs/generated_midis', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    
    # Save Model Weights
    torch.save(model.state_dict(), 'outputs/generated_midis/lstm_autoencoder.pth')
    print("Model saved to outputs/generated_midis/lstm_autoencoder.pth")
    
    # Save Loss Curve Plot
    plt.figure()
    plt.plot(range(1, epochs+1), loss_history, marker='o', label='Train Loss')
    plt.title('Task 1: LSTM Autoencoder Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/plots/task1_loss_curve.png')
    print("Loss curve saved to outputs/plots/task1_loss_curve.png")

if __name__ == "__main__":
    train_autoencoder()