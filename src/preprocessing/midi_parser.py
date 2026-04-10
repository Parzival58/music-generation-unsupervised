import pretty_midi
import numpy as np

def load_midi(file_path):
    """
    Safely load a MIDI file.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        return midi_data
    except Exception as e:
        # Some MIDI files are corrupted; we silently catch them
        return None

def midi_to_piano_roll(midi_data, fs=16, seq_length=64):
    """
    Extracts the drum track from a MIDI file and converts it to a 2D numpy array.
    """
    if midi_data is None:
        return None
        
    drum_instrument = None
    
    # Standard MIDI protocol: Channel 9 (0-indexed) is the drum kit.
    # However, pretty_midi has an `is_drum` flag we should rely on.
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            drum_instrument = instrument
            break
            
    # If no drum track is explicitly flagged, we assume the first track is drums 
    # (common in loop packs where the whole file is just a drum loop)
    if drum_instrument is None:
        if len(midi_data.instruments) > 0:
            drum_instrument = midi_data.instruments[0]
        else:
            return None

    # Get the piano roll (128 pitches x time_steps)
    # The sampling frequency (fs) determines how many time steps per second.
    piano_roll = drum_instrument.get_piano_roll(fs=fs)
    
    # We want time to go downwards, and pitches across the columns
    # So we transpose from (128, time_steps) -> (time_steps, 128)
    piano_roll = piano_roll.T
    
    # Normalize the velocities (0-127) to (0.0-1.0)
    piano_roll = piano_roll / 127.0
    
    return piano_roll