import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.vae import LSTMVAE
from src.generation.midi_export import piano_roll_to_midi

def generate_vae_samples(num_samples=5, latent_dim=128, seq_length=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMVAE(seq_length=seq_length, latent_dim=latent_dim).to(device)
    
    # Load VAE weights
    model.load_state_dict(torch.load('outputs/generated_midis/lstm_vae.pth', map_location=device))
    model.eval()
    
    print(f"Generating Task 2 (VAE) samples...")
    os.makedirs('outputs/generated_midis', exist_ok=True)
    
    with torch.no_grad():
        for i in range(num_samples):
            # Sample from a standard Normal distribution N(0, 1)
            z = torch.randn(1, latent_dim).to(device)
            generated_tensor = model.decoder_fc(z).unsqueeze(1).repeat(1, seq_length, 1)
            recon_out, _ = model.decoder_lstm(generated_tensor)
            x_hat = model.sigmoid(model.output_fc(recon_out))
            
            # Export to MIDI
            generated_matrix = x_hat.squeeze(0).cpu().numpy()
            midi_obj = piano_roll_to_midi(generated_matrix)
            output_path = f"outputs/generated_midis/task2_vae_sample_{i+1}.mid"
            midi_obj.write(output_path)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    generate_vae_samples()