import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)

from src.models.transformer import MusicTransformer
from src.preprocessing.midi_parser import load_midi, midi_to_piano_roll
from src.preprocessing.piano_roll import LazyMIDIDataset

def extract_active_rolls(file_list, seq_length, max_files):
    rolls = []
    print(f"Scanning {len(file_list)} files for Transformer training...")
    for file in file_list:
        if len(rolls) >= max_files: break
        try:
            pr = midi_to_piano_roll(load_midi(file), seq_length=seq_length)
            if pr is not None and pr.shape[0] >= seq_length:
                rolls.append(pr)
        except Exception:
            continue
    return rolls

def train_transformer(epochs=20, batch_size=32, learning_rate=5e-4, seq_length=64, num_files=200):
    print("--- Task 3: Training Autoregressive Transformer ---")
    
    # Require: Tokenized MIDI dataset D
    multi_genre_files = glob.glob('data/raw_midi/**/*.mid', recursive=True)
    all_rolls = extract_active_rolls(multi_genre_files, seq_length, num_files)
    
    if not all_rolls:
        print("CRITICAL ERROR: No active tracks extracted.")
        return

    print(f"Success: Loaded {len(all_rolls)} tracks.")
    dataset = LazyMIDIDataset(all_rolls, seq_length=seq_length, min_notes=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicTransformer().to(device)
    
    # BCEWithLogits computes the negative log likelihood for multi-label categorical distributions
    # This exactly matches Algorithm 3 Line 7: L_TR = - sum(log p(x_t | x_{<t}))
    criterion = nn.BCEWithLogitsLoss()
    
    # Line 1: Initialize parameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    perp_history = []
    
    # Line 2: for epoch = 1 to E do
    print("Starting Transformer training loop...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Line 3: for each sequence X in D do
        for X, _ in dataloader:
            X = X.to(device)
            optimizer.zero_grad()
            
            # Autoregressive shifting: Input is X_{0:T-1}, Target is X_{1:T}
            src = X[:, :-1, :]
            tgt = X[:, 1:, :]
            
            # Generate causal mask to hide future tokens
            src_mask = model.generate_square_subsequent_mask(src.size(1)).to(device)
            
            # Line 5: Predict next token distribution
            logits = model(src, src_mask)
            
            # Line 7: Compute autoregressive loss
            L_TR = criterion(logits, tgt)
            
            # Line 8: Update weights
            L_TR.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += L_TR.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        # Calculate Deliverable Metric: Perplexity = exp(L_TR)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        perp_history.append(perplexity)
        
        print(f"Epoch [{epoch+1}/{epochs}] | L_TR (Loss): {avg_loss:.4f} | Perplexity: {perplexity:.4f}")
        
    os.makedirs('outputs/generated_midis', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    
    torch.save(model.state_dict(), 'outputs/generated_midis/music_transformer.pth')
    
    # Plotting both Loss and Perplexity for your thesis report
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(range(1, epochs+1), loss_history, marker='o', color='b', label='L_TR (Loss)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('BCE Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs+1), perp_history, marker='x', color='r', label='Perplexity')
    ax2.set_ylabel('Perplexity', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Task 3: Transformer Training Metrics')
    fig.tight_layout()
    plt.savefig('outputs/plots/task3_metrics.png')
    print("--- Transformer Training Complete ---")

if __name__ == "__main__":
    train_transformer()