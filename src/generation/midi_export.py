import numpy as np
import pretty_midi

def piano_roll_to_midi(piano_roll, fs=16, threshold=0.1):
    """
    Converts a 2D numpy array (time_steps, pitches) back into a MIDI file.
    """
    midi = pretty_midi.PrettyMIDI()
    # Program 114 is a standard Steel Drum/Percussion kit in General MIDI
    instrument = pretty_midi.Instrument(program=114, is_drum=True) 
    
    # Transpose back to (128 pitches, time_steps) for easier iteration
    piano_roll = piano_roll.T 
    
    # Scan through all 128 possible pitches
    for pitch in range(128):
        # Find the time steps where the neural network predicted a note
        active_steps = np.where(piano_roll[pitch, :] > threshold)[0]
        if len(active_steps) == 0:
            continue
            
        # Group consecutive time steps into single sustained notes
        start_step = active_steps[0]
        for i in range(1, len(active_steps)):
            if active_steps[i] != active_steps[i-1] + 1:
                # Note ended, calculate velocity and save it
                velocity = int(np.mean(piano_roll[pitch, start_step:active_steps[i-1]+1]) * 127)
                note = pretty_midi.Note(
                    velocity=velocity, 
                    pitch=pitch, 
                    start=start_step / fs, 
                    end=(active_steps[i-1] + 1) / fs
                )
                instrument.notes.append(note)
                start_step = active_steps[i]
                
        # Catch the final note in the sequence
        velocity = int(np.mean(piano_roll[pitch, start_step:active_steps[-1]+1]) * 127)
        note = pretty_midi.Note(
            velocity=velocity, 
            pitch=pitch, 
            start=start_step / fs, 
            end=(active_steps[-1] + 1) / fs
        )
        instrument.notes.append(note)
        
    midi.instruments.append(instrument)
    return midi