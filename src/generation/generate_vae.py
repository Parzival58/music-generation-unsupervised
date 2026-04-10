import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.vae import LSTMVAE
from src.generation.midi_export import piano_roll_to_midi

def generate_vae_samples(num_samples=10, latent_dim=128, seq_length=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMVAE(seq_length=seq_length, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load('outputs/generated_midis/lstm_vae.pth', map_location=device))
    model.eval()
    
    print("Generating Fixed Task 2 Samples (Clustered Sampling)...")
    with torch.no_grad():
        for i in range(num_samples):
            # STRENGTH: Reduce the standard deviation of noise to 0.7 
            # This keeps samples closer to the 'musical' center of the latent space
            z = torch.randn(1, latent_dim).to(device) * 0.7
            
            z_projected = model.decoder_fc(z).unsqueeze(1).repeat(1, seq_length, 1)
            recon_out, _ = model.decoder_lstm(z_projected)
            x_hat = model.sigmoid(model.output_fc(recon_out))
            
            generated_matrix = x_hat.squeeze(0).cpu().numpy()
            max_val = np.max(generated_matrix)
            
            # ADAPTIVE THRESHOLD: If the max is low, use 10% of max to get SOME sound
            current_threshold = max(0.05, max_val * 0.5) if max_val > 0 else 0.1
            
            midi_obj = piano_roll_to_midi(generated_matrix, threshold=current_threshold)
            midi_obj.write(f"outputs/generated_midis/task2_vae_sample_{i+1}.mid")
            print(f"Sample {i+1} | Max: {max_val:.4f} | Threshold: {current_threshold:.4f}")

if __name__ == "__main__":
    generate_vae_samples()