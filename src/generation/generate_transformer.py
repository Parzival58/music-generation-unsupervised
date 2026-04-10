import os
import sys
import torch
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)

from src.models.transformer import MusicTransformer
from src.generation.midi_export import piano_roll_to_midi

def generate_long_composition(num_samples=10, generate_steps=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MusicTransformer().to(device)
    weight_path = os.path.join(BASE_DIR, 'outputs/generated_midis/music_transformer.pth')
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    print(f"Generating {num_samples} long-horizon compositions (Algorithm 3, Line 11)...")
    out_dir = os.path.join(BASE_DIR, 'outputs/generated_midis')
    os.makedirs(out_dir, exist_ok=True)
    
    with torch.no_grad():
        for i in range(num_samples):
            # Seed the sequence with a single active note (e.g., Middle C / Kick Drum)
            # to give the autoregressive model a starting point.
            curr_seq = torch.zeros(1, 1, 128).to(device)
            curr_seq[0, 0, 36] = 1.0 
            
            # Line 11: Iterative Sampling
            for step in range(generate_steps):
                # To prevent memory explosion on extremely long generations, 
                # we only let the model look back at the last 256 steps.
                context = curr_seq[:, -256:, :]
                
                mask = model.generate_square_subsequent_mask(context.size(1)).to(device)
                logits = model(context, mask)
                
                # Get the prediction for the absolute final timestep
                next_step_logits = logits[:, -1:, :]
                next_step_probs = torch.sigmoid(next_step_logits)
                
                # Sample from the probability distribution using Bernoulli
                # This ensures mathematical variance rather than deterministic loops
                next_step_sampled = torch.bernoulli(next_step_probs)
                
                # Append the generated step to the sequence
                curr_seq = torch.cat([curr_seq, next_step_sampled], dim=1)
                
            generated_matrix = curr_seq.squeeze(0).cpu().numpy()
            
            midi_obj = piano_roll_to_midi(generated_matrix, threshold=0.5)
            out_file = os.path.join(out_dir, f"task3_transformer_long_sample_{i+1}.mid")
            midi_obj.write(out_file)
            print(f"Saved: task3_transformer_long_sample_{i+1}.mid | Length: {generated_matrix.shape[0]} steps")

if __name__ == "__main__":
    generate_long_composition()