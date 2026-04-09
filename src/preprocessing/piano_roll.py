import numpy as np
import torch
from torch.utils.data import Dataset

def create_sequences(piano_roll, seq_length=64):
    """
    Slices a continuous 2D piano roll into fixed-length sequences using a sliding window.
    
    Args:
        piano_roll (numpy.ndarray): Shape (total_time_steps, 128).
        seq_length (int): The number of time steps per sequence.
        
    Returns:
        numpy.ndarray: Shape (num_sequences, seq_length, 128).
    """
    sequences = []
    # Slide a window of size `seq_length` across the time axis
    for i in range(0, piano_roll.shape[0] - seq_length):
        seq = piano_roll[i : i + seq_length, :]
        sequences.append(seq)
        
    return np.array(sequences)

class MIDIDataset(Dataset):
    """
    PyTorch Dataset wrapper for our segmented MIDI sequences.
    """
    def __init__(self, sequences):
        # Convert numpy arrays to PyTorch float tensors and normalize velocities to [0, 1]
        self.sequences = torch.tensor(sequences, dtype=torch.float32) / 127.0
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        # For an Autoencoder, the input and the target are exactly the same!
        seq = self.sequences[idx]
        return seq, seq