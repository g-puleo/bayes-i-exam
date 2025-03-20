import torch
import os
import lightning as L
from torch.utils.data import DataLoader
from sbi.dataset import SimulatedDataset
from sbi.simulator import Simulator
from scipy.stats import uniform
from sbi.model import MNRE, lightning_MNRE
from sbi import PROJECT_ROOT
import numpy as np
# Define the simulator
n_timesteps = 100
t = torch.linspace(0, 10, n_timesteps)
sigma = 0.1
simulator = Simulator(sigma, t)

# Define the priors for the parameters, in order of omega, phi, and A
priors = [uniform(0, 4), uniform(0, 2 * np.pi), uniform(0, 5)]

# Create the dataset
N_samples = 10000
dataset = SimulatedDataset(simulator, priors, N_samples)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
val_dataset = SimulatedDataset(simulator, priors, int(N_samples*0.2))
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the model
model = MNRE(n_data=n_timesteps, n_params=len(priors), embed_size=50, hlayer_size=128, tail_hlayer_size=64)
# Define the Lightning model
lr = 1e-3
lightning_model = lightning_MNRE(model, optimizer='adam', lr=lr)
# Define the trainer
trainer = L.Trainer(max_epochs=10)
# Train the model
trainer.fit(lightning_model, dataloader, val_dataloader)
# Save the model
torch.save(lightning_model.state_dict(), os.path.join(PROJECT_ROOT, 'trained_models', 'model.pth'))

