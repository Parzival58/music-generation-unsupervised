import pretty_midi
import numpy as np
import os
import glob

def midi_to_piano_roll(midi_path, fs=4): 
    """
    Step 1 & 2: Convert MIDI to piano-roll and normalize timing [cite: 31, 32]
    fs=4 provides 16 steps per bar at standard 120 BPM.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        # Standard MIDI has 128 pitches [0-127]
        piano_roll = midi_data.get_piano_roll(fs=fs)
        
        # Binarize: 1 if note is on, 0 if off (Simplifies Task 1)
        piano_roll[piano_roll > 0] = 1
        
        # Transpose to (Time, Pitch) to match Task 1 representation [cite: 14]
        return piano_roll.T 
    except Exception as e:
        print(f"Error parsing {midi_path}: {e}")
        return None

def segment_sequences(piano_roll, window_size=64):
    """
    Step 3: Segment sequences into fixed-length windows 
    """
    sequences = []
    # Using a 50% overlap (stride = window_size // 2) to augment data
    for i in range(0, piano_roll.shape[0] - window_size, window_size // 2):
        segment = piano_roll[i:i + window_size]
        sequences.append(segment)
    return np.array(sequences)

if __name__ == "__main__":
    # Ensure these paths match your VS Code workspace structure [cite: 243]
    RAW_DATA_DIR = "data/raw_midi/"
    PROCESSED_DIR = "data/processed/"
    
    # Create the processed directory if it doesn't exist
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    
    all_segments = []
    
    # RECURSIVE SEARCH: Find all .mid and .midi files in subfolders (e.g., drummer1/)
    midi_files = glob.glob(os.path.join(RAW_DATA_DIR, "**/*.mid"), recursive=True)
    midi_files += glob.glob(os.path.join(RAW_DATA_DIR, "**/*.midi"), recursive=True)
    
    print(f"Found {len(midi_files)} MIDI files in {RAW_DATA_DIR}")

    # Process files (Limiting to 200 files to keep training time low as requested)
    for path in midi_files[:200]:
        roll = midi_to_piano_roll(path)
        if roll is not None:
            segments = segment_sequences(roll)
            if len(segments) > 0:
                all_segments.append(segments)
    
    # Final Step: Save for LSTM Autoencoder Training [cite: 112]
    if all_segments:
        final_data = np.concatenate(all_segments, axis=0)
        save_path = os.path.join(PROCESSED_DIR, "train_data.npy")
        np.save(save_path, final_data)
        print(f"Successfully saved {final_data.shape[0]} sequences of shape {final_data.shape[1:]}")
        print(f"File saved at: {save_path}")
    else:
        print("No sequences were generated. Check if your raw_midi folder contains .mid files.")