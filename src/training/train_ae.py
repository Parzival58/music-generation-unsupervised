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
    rolls = []
    print(f"Found {len(file_list)} matching .mid files on disk.")
    
    for idx, file in enumerate(file_list):
        if len(rolls) >= max_files:
            break
        try:
            midi_data = load_midi(file)
            pr = midi_to_piano_roll(midi_data)
            
            if pr is None:
                print(f"  [Skip] {os.path.basename(file)}: Parser returned None.")
                continue
            if pr.shape[0] <= seq_length:
                print(f"  [Skip] {os.path.basename(file)}: Track too short ({pr.shape[0]} ticks).")
                continue
                
            active_notes = np.sum(pr)
            if active_notes > threshold:
                rolls.append(pr)
            else:
                print(f"  [Skip] {os.path.basename(file)}: Mostly silence (Only {active_notes} active notes).")
                
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(file)} crashed the parser: {str(e)}")
            continue
            
    return rolls

def train_autoencoder(epochs=20, batch_size=64, learning_rate=1e-3, seq_length=64, target_genre="funk", num_files=100):
    print(f"--- Task 1: Diagnostic Data Run ---")
    
    genre_files = glob.glob(f'data/raw_midi/**/*{target_genre}*.mid', recursive=True)
    all_rolls = extract_active_rolls(genre_files, seq_length, num_files)
    
    if len(all_rolls) == 0:
        print(f"\nWarning: '{target_genre}' failed. Trying general dataset...")
        any_files = glob.glob('data/raw_midi/**/*.mid', recursive=True)
        all_rolls = extract_active_rolls(any_files, seq_length, num_files, threshold=10)

    if len(all_rolls) == 0:
        print("\nCRITICAL ERROR: No active MIDI tracks extracted.")
        return

    print(f"\nSuccess: Loaded {len(all_rolls)} active tracks. Starting dataloader...")
    
    dataset = LazyMIDIDataset(all_rolls, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(seq_length=seq_length).to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    print("Starting training loop...")
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
        
    os.makedirs('outputs/generated_midis', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/generated_midis/lstm_autoencoder.pth')
    print("--- Training Complete ---")

if __name__ == "__main__":
    train_autoencoder()