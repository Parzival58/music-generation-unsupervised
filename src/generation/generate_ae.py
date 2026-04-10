import os
import sys
import torch
import numpy as np

# Dynamically find the absolute path to the root of the repository
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)

from src.models.autoencoder import LSTMAutoencoder
from src.generation.midi_export import piano_roll_to_midi

def generate_ae_samples(num_samples=5, latent_dim=256, seq_length=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LSTMAutoencoder(seq_length=seq_length, hidden_dim=latent_dim).to(device)
    
    # Use the absolute path to guarantee we find the weights no matter what directory Colab is in
    weight_path = os.path.join(BASE_DIR, 'outputs/generated_midis/lstm_ae.pth')
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    print("Generating music via Algorithm 1, Line 10...")
    
    # Ensure the absolute output directory exists
    out_dir = os.path.join(BASE_DIR, 'outputs/generated_midis')
    os.makedirs(out_dir, exist_ok=True)
    
    with torch.no_grad():
        for i in range(num_samples):
            # Scale down the noise to stay within bounds of typical encoded features
            z = torch.randn(1, latent_dim).to(device) * 0.5
            
            X_hat = model.decoder(z)
            generated_matrix = X_hat.squeeze(0).cpu().numpy()
            
            # Clip the linear outputs to a valid 0.0 - 1.0 range
            generated_matrix = np.clip(generated_matrix, 0.0, 1.0)
            max_val = np.max(generated_matrix)
            
            threshold = max(0.05, max_val * 0.5) if max_val > 0 else 0.1
            
            midi_obj = piano_roll_to_midi(generated_matrix, threshold=threshold)
            
            # Save using the absolute path
            out_file = os.path.join(out_dir, f"task1_algo1_sample_{i+1}.mid")
            midi_obj.write(out_file)
            print(f"Saved: task1_algo1_sample_{i+1}.mid | Max Signal: {max_val:.4f}")

if __name__ == "__main__":
    generate_ae_samples()