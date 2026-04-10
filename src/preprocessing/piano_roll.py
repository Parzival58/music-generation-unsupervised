import torch
from torch.utils.data import Dataset
import numpy as np

class LazyMIDIDataset(Dataset):
    """
    Memory-efficient dataset that slices piano rolls on-the-fly.
    Prevents Out-of-Memory (OOM) crashes in Colab.
    """
    def __init__(self, piano_rolls, seq_length=64):
        # NOTE: We DO NOT divide by 127.0 here. 
        # The new midi_parser.py already normalizes the data to 0.0-1.0.
        self.piano_rolls = [torch.tensor(pr, dtype=torch.float32) for pr in piano_rolls]
        self.seq_length = seq_length
        
        # Pre-calculate a mapping of (song_index, start_time_step)
        self.mapping = []
        for song_idx, pr in enumerate(self.piano_rolls):
            # +1 ensures that a track of exactly length 64 yields 1 sequence
            num_possible_seqs = pr.shape[0] - seq_length + 1
            if num_possible_seqs > 0:
                for start_offset in range(num_possible_seqs):
                    self.mapping.append((song_idx, start_offset))
                
    def __len__(self):
        return len(self.mapping)
        
    def __getitem__(self, idx):
        song_idx, start = self.mapping[idx]
        # Slice the sequence from the full song on demand
        sequence = self.piano_rolls[song_idx][start : start + self.seq_length, :]
        return sequence, sequence

def create_sequences(piano_roll, seq_length=64):
    """Legacy helper for Task 1 compatibility."""
    sequences = []
    # Fixed the off-by-one error here as well
    for i in range(len(piano_roll) - seq_length + 1):
        sequences.append(piano_roll[i : i + seq_length])
    return np.array(sequences)

class MIDIDataset(Dataset):
    """Legacy class to prevent import errors in older scripts."""
    def __init__(self, sequences):
        # Removed the / 127.0 to prevent double-normalization
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.sequences[idx]