import pretty_midi
import numpy as np

def load_midi(file_path):
    """
    Loads a MIDI file and returns a PrettyMIDI object.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        return midi_data
    except Exception as e:
        print(f"Error loading MIDI file {file_path}: {e}")
        return None

def midi_to_piano_roll(midi_data, sampling_freq=16):
    """
    Converts a PrettyMIDI object into a 2D piano roll array.
    
    Args:
        midi_data: The loaded PrettyMIDI object.
        sampling_freq: Number of time steps per second.
        
    Returns:
        numpy.ndarray: Shape is (time_steps, 128 pitches).
        Values represent note velocity (0 to 127).
    """
    if midi_data is None:
        return None
    
    # Generate the piano roll. Shape: (128 pitches, total time steps)
    piano_roll = midi_data.get_piano_roll(fs=sampling_freq)
    
    # Transpose to (time_steps, 128) which is standard for LSTM inputs
    return piano_roll.T