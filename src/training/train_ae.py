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
from src.preprocessing.piano_roll import MIDIDataset, create_sequences

def train_autoencoder(epochs=20, batch_size=64, learning_rate=1e-3, seq_length=64):
    print("Preparing dataset...")
    # 1. Locate and process the MIDI files
    midi_files = glob.glob('data/raw_midi/**/*.mid', recursive=True)
    if not midi_files:
        print("ERROR: No MIDI files found in data/raw_midi/")
        return
    
    all_sequences = []
    # Note: We use the first 50 files here just to make sure the code works. 
    # Later, you can remove '[:50]' to train on the entire dataset!
    for idx, file in enumerate(midi_files[:50]): 
            print(f"Processing file {idx + 1}/50...") # Add this line!
            midi_data = load_midi(file)
            piano_roll = midi_to_piano_roll(midi_data)
            if piano_roll is not None and piano_roll.shape[0] > seq_length:
                seqs = create_sequences(piano_roll, seq_length=seq_length)
                all_sequences.append(seqs)
            
    # Combine all individual song chunks into one massive matrix
    full_data = np.concatenate(all_sequences, axis=0)
    print(f"Total sequences extracted: {full_data.shape[0]}")
    
    # 2. PyTorch DataLoader for batching
    dataset = MIDIDataset(full_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Initialize Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = LSTMAutoencoder(seq_length=seq_length).to(device)
    # Mean Squared Error Loss matches your assignment's L_AE formula
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    # 4. The Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move data to GPU if available
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()               # Clear old math
            outputs, latent = model(inputs)     # Forward pass (guess)
            loss = criterion(outputs, targets)  # Calculate error
            loss.backward()                     # Backpropagation
            optimizer.step()                    # Update weights
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
    # 5. Save the required deliverables
    os.makedirs('outputs/generated_midis', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    
    # Save Model Weights
    torch.save(model.state_dict(), 'outputs/generated_midis/lstm_autoencoder.pth')
    print("Model saved to outputs/generated_midis/lstm_autoencoder.pth")
    
    # Save Loss Curve Plot for your LaTeX Report
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