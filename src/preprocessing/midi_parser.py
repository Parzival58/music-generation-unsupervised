import pretty_midi
import numpy as np

def load_midi(file_path):
    """Safely load a MIDI file."""
    try:
        return pretty_midi.PrettyMIDI(file_path)
    except Exception:
        return None

def midi_to_piano_roll(midi_data, fs=16, seq_length=64):
    """
    Custom drum-optimized piano roll extractor.
    Prevents ultra-short electronic drum hits from being skipped during sampling.
    """
    if midi_data is None:
        return None
        
    end_time = midi_data.get_end_time()
    if end_time == 0:
        return None
        
    time_steps = int(np.ceil(end_time * fs))
    if time_steps == 0:
        return None
        
    # Initialize a blank canvas (time_steps, 128 pitches)
    piano_roll = np.zeros((time_steps, 128))
    
    # Manually extract every note from every instrument
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start_step = int(note.start * fs)
            # CRITICAL: Force every drum hit to take up AT LEAST 1 time step
            # This prevents 0.001s e-drum hits from disappearing
            end_step = max(start_step + 1, int(np.ceil(note.end * fs)))
            
            # Keep within array bounds
            start_step = min(start_step, time_steps - 1)
            end_step = min(end_step, time_steps)
            
            # Map velocity to 0.0 - 1.0 range
            if start_step < end_step:
                current_vel = piano_roll[start_step:end_step, note.pitch]
                new_vel = note.velocity / 127.0
                # Use maximum velocity if notes overlap
                piano_roll[start_step:end_step, note.pitch] = np.maximum(current_vel, new_vel)

    # CRITICAL: If the track is a short drum fill (e.g., 30 steps), 
    # we "loop" it (tile it) until it hits our required 64 steps for training.
    if piano_roll.shape[0] < seq_length:
        repeats = int(np.ceil(seq_length / piano_roll.shape[0]))
        piano_roll = np.tile(piano_roll, (repeats, 1))
        
    return piano_roll