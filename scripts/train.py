import torch
import os
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from sbi.dataset import SimulatedDataset
from sbi.simulator import Simulator
from scipy.stats import uniform
from sbi.model import MNRE, lightning_MNRE
from sbi import PROJECT_ROOT
import numpy as np
import argparse, yaml
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# Define the priors for the parameters, in order of omega, phi, and A
parser = argparse.ArgumentParser(description="Generate data based on a configuration file.")
parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)


# Define the simulator
n_timesteps = config['time_steps']['n']
t = np.linspace(config['time_steps']['min'], config['time_steps']['max'], n_timesteps)
sigma = config["sigma"]
simulator = Simulator(sigma, t)
# Create the dataset
N_train_samples = 1000000
priors = [uniform(config['priors'][param][0], config['priors'][param][1] - config['priors'][param][0]) for param in config['priors']]
dataset = SimulatedDataset(simulator, priors, N_train_samples)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=19)
val_dataset = SimulatedDataset(simulator, priors, int(N_train_samples * 0.2))
val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=19)

# Define the model
model = MNRE(n_data=n_timesteps, n_params=len(priors), embed_size=40, data_hlayer_size=20, tail_hlayer_size=20)
# Define the Lightning model
lr = 1e-3
lightning_model = lightning_MNRE(model, optimizer='adam', lr=lr)
# Define the trainer, train for 50 epochs with early stopping, and the logger
logger = TensorBoardLogger(os.path.join(PROJECT_ROOT, 'lightning_logs'), name=config["name"])
trainer = L.Trainer(max_epochs=50, callbacks=[EarlyStopping(monitor="val_accuracy", mode="max", patience=5, min_delta=0.01)], logger=logger)
# Train the model
trainer.fit(lightning_model, dataloader, val_dataloader)
# Save the model
trainer.save_checkpoint(os.path.join(PROJECT_ROOT, 'trained_models', f'model_{config["name"]}.pth'))
