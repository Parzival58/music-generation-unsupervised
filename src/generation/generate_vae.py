import os
import sys
import torch
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)

from src.models.vae import LSTMVAE
from src.generation.midi_export import piano_roll_to_midi

def generate_vae_samples(num_samples=5, latent_dim=256, seq_length=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LSTMVAE(seq_length=seq_length, latent_dim=latent_dim).to(device)
    weight_path = os.path.join(BASE_DIR, 'outputs/generated_midis/lstm_vae.pth')
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    print("Generating multi-genre music via Algorithm 2, Line 13...")
    out_dir = os.path.join(BASE_DIR, 'outputs/generated_midis')
    os.makedirs(out_dir, exist_ok=True)
    
    with torch.no_grad():
        for i in range(num_samples):
            # Line 13: Generate diverse multi-genre music by sampling z ~ N(0, I)
            z = torch.randn(1, latent_dim).to(device)
            
            # Decode the random latent vector
            X_hat = model.decode(z)
            generated_matrix = X_hat.squeeze(0).cpu().numpy()
            
            # Clip the linear outputs back to standard normalized ranges
            generated_matrix = np.clip(generated_matrix, 0.0, 1.0)
            max_val = np.max(generated_matrix)
            
            # Adaptive threshold to capture the most prominent multi-genre beats
            threshold = max(0.05, max_val * 0.5) if max_val > 0 else 0.1
            
            midi_obj = piano_roll_to_midi(generated_matrix, threshold=threshold)
            
            # Line 14: Output generated MIDI files
            out_file = os.path.join(out_dir, f"task2_algo2_sample_{i+1}.mid")
            midi_obj.write(out_file)
            print(f"Saved: task2_algo2_sample_{i+1}.mid | Max Signal: {max_val:.4f}")

if __name__ == "__main__":
    generate_vae_samples()