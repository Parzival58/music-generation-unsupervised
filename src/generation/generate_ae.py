import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.autoencoder import LSTMAutoencoder
from src.generation.midi_export import piano_roll_to_midi

def generate_ae_samples(num_samples=5, latent_dim=128, seq_length=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(seq_length=seq_length, latent_dim=latent_dim).to(device)
    
    # Load the specific Task 1 weights
    model.load_state_dict(torch.load('outputs/generated_midis/lstm_ae.pth', map_location=device))
    model.eval()
    
    print("Generating music via Algorithm 1, Line 10...")
    os.makedirs('outputs/generated_midis', exist_ok=True)
    
    with torch.no_grad():
        for i in range(num_samples):
            # Line 10: Generate new music by sampling latent codes z
            # We scale the random noise by 0.5 because standard autoencoders 
            # don't have a perfectly smooth N(0,1) latent space like VAEs do.
            z = torch.randn(1, latent_dim).to(device) * 0.5
            
            # Decode
            z_projected = model.decoder_fc(z).unsqueeze(1).repeat(1, seq_length, 1)
            recon_out, _ = model.decoder_lstm(z_projected)
            X_hat = model.sigmoid(model.output_fc(recon_out))
            
            generated_matrix = X_hat.squeeze(0).cpu().numpy()
            
            # Line 11: Output MIDI compositions
            max_val = np.max(generated_matrix)
            
            # Adaptive thresholding guarantees we don't output blank files
            threshold = max(0.05, max_val * 0.5) if max_val > 0 else 0.1
            
            midi_obj = piano_roll_to_midi(generated_matrix, threshold=threshold)
            midi_obj.write(f"outputs/generated_midis/task1_algo1_sample_{i+1}.mid")
            print(f"Saved: task1_algo1_sample_{i+1}.mid | Max Signal: {max_val:.4f}")

if __name__ == "__main__":
    generate_ae_samples()