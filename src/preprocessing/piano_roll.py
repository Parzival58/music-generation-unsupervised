import torch
from torch.utils.data import Dataset
import numpy as np

class LazyMIDIDataset(Dataset):
    """
    Memory-efficient dataset that slices piano rolls on-the-fly.
    Filters out silent windows to prevent MSE flatlining.
    """
    def __init__(self, piano_rolls, seq_length=64, min_notes=2):
        self.piano_rolls = [torch.tensor(pr, dtype=torch.float32) for pr in piano_rolls]
        self.seq_length = seq_length
        self.mapping = []
        
        for song_idx, pr in enumerate(self.piano_rolls):
            num_possible_seqs = pr.shape[0] - seq_length + 1
            
            # Slide the window by 16 steps (about 1 beat) instead of 1 step.
            # This drastically speeds up training and removes highly redundant data.
            step_size = 16 
            
            if num_possible_seqs > 0:
                for start_offset in range(0, num_possible_seqs, step_size):
                    # Peek at this specific 64-step slice
                    slice_view = pr[start_offset : start_offset + seq_length, :]
                    
                    # ONLY add the slice if there is actual rhythm inside it!
                    if torch.sum(slice_view) >= min_notes:
                        self.mapping.append((song_idx, start_offset))
                
    def __len__(self):
        return len(self.mapping)
        
    def __getitem__(self, idx):
        song_idx, start = self.mapping[idx]
        sequence = self.piano_rolls[song_idx][start : start + self.seq_length, :]
        return sequence, sequence

def create_sequences(piano_roll, seq_length=64):
    """Legacy helper for Task 1 compatibility."""
    sequences = []
    for i in range(len(piano_roll) - seq_length + 1):
        sequences.append(piano_roll[i : i + seq_length])
    return np.array(sequences)

class MIDIDataset(Dataset):
    """Legacy class to prevent import errors in older scripts."""
    def __init__(self, sequences):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.sequences[idx]