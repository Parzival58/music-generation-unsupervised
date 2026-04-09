import torch
import numpy as np
import pretty_midi
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.autoencoder import LSTMAutoencoder

def piano_roll_to_midi(piano_roll, fs=4, program=0, is_drum=True):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program, is_drum=is_drum)
    
    padded_roll = np.zeros((128, piano_roll.shape[1]))
    padded_roll[:piano_roll.shape[0], :] = piano_roll
    
    for pitch in range(128):
        changes = np.diff(padded_roll[pitch, :])
        starts = np.where(changes > 0)[0] + 1
        ends = np.where(changes < 0)[0] + 1
        
        if padded_roll[pitch, 0] > 0:
            starts = np.insert(starts, 0, 0)
        if padded_roll[pitch, -1] > 0:
            ends = np.append(ends, padded_roll.shape[1])
            
        for start, end in zip(starts, ends):
            # Give the drums a strong velocity
            note = pretty_midi.Note(
                velocity=100, 
                pitch=pitch,
                start=start / fs,
                end=end / fs
            )
            instrument.notes.append(note)
            
    midi.instruments.append(instrument)
    return midi

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    MODEL_PATH = "outputs/models/lstm_ae.pth"
    DATA_PATH = "data/processed/train_data.npy"
    OUTPUT_DIR = "outputs/generated_midis/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    input_dim = 128
    hidden_dim = 256
    latent_dim = 64
    seq_len = 64 
    
    # Load Model
    model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    
    # Load real data to use as a starting point
    dataset = np.load(DATA_PATH)
    dataset_tensor = torch.from_numpy(dataset).float().to(device)
    
    print("Generating 5 distinct AI compositions...")
    
    # FORCED DIVERSITY: Instead of random picking, slice the dataset into 5 distinct chunks
    # This guarantees we start with 5 completely different base drum patterns
    spaced_indices = np.linspace(0, len(dataset_tensor) - 1, 5).astype(int)
    
    with torch.no_grad():
        for i, idx in enumerate(spaced_indices):
            # 1. Get the distinct base sequence
            real_seq = dataset_tensor[idx].unsqueeze(0)
            
            # 2. Encode it
            _, (hidden, _) = model.encoder(real_seq)
            real_z = model.latent_layer(hidden[-1])
            
            # 3. Add MODERATE noise (0.3) to alter the beat without destroying it
            noise = torch.randn_like(real_z) * 0.3  
            novel_z = real_z + noise
            
            # 4. Decode
            d_in = model.decoder_input(novel_z).unsqueeze(1).repeat(1, seq_len, 1)
            d_out, _ = model.decoder(d_in)
            generated_sequence = torch.sigmoid(model.output_layer(d_out))
            generated_np = generated_sequence.squeeze(0).cpu().numpy()
            
            # 5. DYNAMIC THRESHOLD
            max_prob = generated_np.max()
            dynamic_threshold = max_prob * 0.80 
            
            if max_prob < 0.001:
                dynamic_threshold = 0.5 
                
            generated_np = (generated_np > dynamic_threshold).astype(float)
            
            # 6. Save
            midi_obj = piano_roll_to_midi(generated_np.T)
            save_path = os.path.join(OUTPUT_DIR, f"task1_sample_{i+1}.mid")
            midi_obj.write(save_path)
            
            note_count = np.sum(generated_np > 0)
            print(f"Sample {i+1} (Base Index {idx}): Max Conf = {max_prob:.4f} | Notes = {note_count} | Saved to {save_path}")
            
    print("\nTask 1 generation complete! Your files should now sound completely different.")