import os
import sys
import torch

# Ensure Python can find your custom src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.autoencoder import LSTMAutoencoder
from src.generation.midi_export import piano_roll_to_midi

def generate_task1_samples(num_samples=5, latent_dim=256, seq_length=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load the architecture and the trained weights
    model = LSTMAutoencoder(seq_length=seq_length, hidden_dim=latent_dim).to(device)
    model.load_state_dict(torch.load('outputs/generated_midis/lstm_autoencoder.pth', map_location=device))
    model.eval() # Set model to evaluation mode
    
    print(f"Generating {num_samples} MIDI compositions...")
    os.makedirs('outputs/generated_midis', exist_ok=True)
    
    # 2. Hallucinate new music
    for i in range(num_samples):
        # Generate random noise vector z to feed into the decoder
        z_random = torch.randn(1, latent_dim).to(device)
        
        with torch.no_grad():
            generated_tensor = model.decoder(z_random)
        
        # Convert tensor back to numpy matrix
        generated_matrix = generated_tensor.squeeze(0).cpu().numpy()
        
        # 3. Convert matrix to MIDI file
        midi_obj = piano_roll_to_midi(generated_matrix)
        output_path = f"outputs/generated_midis/task1_sample_{i+1}.mid"
        midi_obj.write(output_path)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    generate_task1_samples()