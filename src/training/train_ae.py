import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os # Make sure os is imported
from src.models.autoencoder import LSTMAutoencoder

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load Processed Data
data = np.load("data/processed/train_data.npy")
data = torch.from_numpy(data).float().to(device)

# 3. Initialize Model & Hyperparameters
input_dim = 128    # Standard MIDI pitch range
hidden_dim = 256
latent_dim = 64
learning_rate = 0.001
epochs = 50

model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
criterion = nn.MSELoss() # Implementation of ||X - X_hat||^2 [cite: 45]
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. Training Loop [cite: 116]
loss_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass: Encode and Decode [cite: 121, 124]
    output = model(data)
    loss = criterion(output, data)
    
    # Backward pass: Update parameters [cite: 130]
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 5. Save Deliverables 
# ---> THIS IS THE FIX: Create the folders before trying to save! <---
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

plt.plot(loss_history)
plt.title("Task 1: LSTM Autoencoder Reconstruction Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("outputs/plots/loss_curve_ae.png")

torch.save(model.state_dict(), "outputs/models/lstm_ae.pth")
print("Model and loss curve saved successfully!")