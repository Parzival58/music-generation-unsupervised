import numpy as np
import pretty_midi

def piano_roll_to_midi(piano_roll, fs=16, threshold=0.05): # Lowered threshold to 0.05
    """
    Converts a 2D numpy array (time_steps, pitches) back into a MIDI file.
    """
    # Print max value to debug why it might be blank
    print(f"Max signal in generated matrix: {np.max(piano_roll):.4f}")
    
    midi = pretty_midi.PrettyMIDI()
    # Using Standard Pro Drum Kit (Channel 10/Program 0 in most cases)
    instrument = pretty_midi.Instrument(program=0, is_drum=True) 
    
    # Transpose to (pitches, time_steps)
    piano_roll = piano_roll.T 
    
    for pitch in range(128):
        # Find steps where model is confident enough
        active_steps = np.where(piano_roll[pitch, :] > threshold)[0]
        
        if len(active_steps) == 0:
            continue
            
        # Group notes to avoid "machine gun" staccato
        start_step = active_steps[0]
        for i in range(1, len(active_steps)):
            if active_steps[i] != active_steps[i-1] + 1:
                # Calculate velocity based on model confidence (0-127)
                velocity = int(np.mean(piano_roll[pitch, start_step:active_steps[i-1]+1]) * 127)
                # Ensure velocity is at least audible
                velocity = max(min(velocity, 127), 40)
                
                note = pretty_midi.Note(
                    velocity=velocity, 
                    pitch=pitch, 
                    start=start_step / fs, 
                    end=(active_steps[i-1] + 1) / fs
                )
                instrument.notes.append(note)
                start_step = active_steps[i]
                
    midi.instruments.append(instrument)
    return midi