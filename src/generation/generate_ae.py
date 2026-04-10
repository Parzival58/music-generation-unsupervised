import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.autoencoder import LSTMAutoencoder
from src.generation.midi_export import piano_roll_to_midi

def generate_ae_samples(num_samples=5, latent_dim=256, seq_length=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize with hidden_dim=256 to match the saved weights
    model = LSTMAutoencoder(seq_length=seq_length, hidden_dim=latent_dim).to(device)
    model.load_state_dict(torch.load('outputs/generated_midis/lstm_ae.pth', map_location=device))
    model.eval()
    
    print("Generating music via Algorithm 1, Line 10...")
    os.makedirs('outputs/generated_midis', exist_ok=True)
    
    with torch.no_grad():
        for i in range(num_samples):
            # Scale down the noise to stay within bounds of typical encoded features
            z = torch.randn(1, latent_dim).to(device) * 0.5
            
            # Pass z directly to the decoder
            X_hat = model.decoder(z)
            generated_matrix = X_hat.squeeze(0).cpu().numpy()
            
            # Clip the linear outputs to a valid 0.0 - 1.0 range
            generated_matrix = np.clip(generated_matrix, 0.0, 1.0)
            max_val = np.max(generated_matrix)
            
            threshold = max(0.05, max_val * 0.5) if max_val > 0 else 0.1
            
            midi_obj = piano_roll_to_midi(generated_matrix, threshold=threshold)
            midi_obj.write(f"outputs/generated_midis/task1_algo1_sample_{i+1}.mid")
            print(f"Saved: task1_algo1_sample_{i+1}.mid | Max Signal: {max_val:.4f}")

if __name__ == "__main__":
    generate_ae_samples()