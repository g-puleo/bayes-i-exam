from sbi import Simulator, PROJECT_ROOT
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import torch
from torch.distributions.uniform import Uniform as uniform
from sbi.utils import p_values_for_grid, DatasetPP
from sbi.dataset import SimulatedDataset
from sbi.model import lightning_MNRE
torch.manual_seed(0)
np.random.seed(0)
# load the configuration file
parser = argparse.ArgumentParser(description="Generate coverage maps based on priors specified in a configuration file.")
parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# load the MNRE model
lightning_model = lightning_MNRE.load_from_checkpoint(os.path.join(PROJECT_ROOT, 'trained_models', f'model_{config["model_name"]}.pth'))
# generate observed data
simulator = Simulator(config['sigma'], torch.linspace(config['time_steps']['min'], config['time_steps']['max'], config['time_steps']['n']))
param_names = list(config['priors'].keys())
# define a 2d grid of (A, omega) values based on prior ranges: 
A = torch.linspace(config['priors']['A'][0], config['priors']['A'][1], 10, dtype=torch.float32,requires_grad=False)
omega = torch.linspace(config['priors']['omega'][0], config['priors']['omega'][1], 10, dtype=torch.float32, requires_grad=False)
# create a meshgrid of A and omega values.
A_grid, omega_grid = torch.meshgrid(A, omega)
print(A_grid.shape, omega_grid.shape)
coverage_map_99 = np.zeros((A_grid.shape[0], omega_grid.shape[1]))
coverage_map_95 = np.zeros((A_grid.shape[0], omega_grid.shape[1]))
coverage_map_68 = np.zeros((A_grid.shape[0], omega_grid.shape[1]))
# for every point in the grid, sample 1000 values of phi from the prior
import torch
import numpy as np
from scipy.stats import uniform

# Define the A and omega grids
A = torch.linspace(config['priors']['A'][0], config['priors']['A'][1], 10, dtype=torch.float32)
omega = torch.linspace(config['priors']['omega'][0], config['priors']['omega'][1], 10, dtype=torch.float32)

# Create a meshgrid and flatten it for batch processing
A_grid, omega_grid = torch.meshgrid(A, omega, indexing="ij")  # Shape: (10, 10)
A_flat = A_grid.flatten()  # Shape: (100,)
omega_flat = omega_grid.flatten()  # Shape: (100,)

# Number of datasets (each grid point is a dataset)
num_datasets = A_flat.shape[0]  # 100
num_obs_per_dataset = 1000  # Number of phi samples per dataset

# Sample `phi` for all datasets at once
phi_samples = torch.tensor(
    uniform(config['priors']['phi'][0], config['priors']['phi'][1]).rvs((num_datasets, num_obs_per_dataset)),
    dtype=torch.float32
)  # Shape: (100, 1000)

# Expand A and omega for all observations
A_expanded = A_flat[:, None].expand(-1, num_obs_per_dataset)  # Shape: (100, 1000)
omega_expanded = omega_flat[:, None].expand(-1, num_obs_per_dataset)  # Shape: (100, 1000)

# Stack to create full parameter tensor
params_inj = torch.stack((A_expanded, omega_expanded, phi_samples), dim=2)  # Shape: (100, 1000, 3)
params_inj_flat = params_inj.view(-1, 3)  # Shape: (100000, 3)
# Simulate observations for all datasets at once
data_obs_flat = simulator.simulate(params_inj_flat).to(torch.float32)  # Shape: (100000, obs_dim)

# **Reshape** back to (num_datasets, num_obs_per_dataset, obs_dim)
obs_dim = data_obs_flat.shape[1]  # Extract observation dimension
data_obs = data_obs_flat.view(num_datasets, num_obs_per_dataset, obs_dim)  # Shape: (100, 1000, obs_dim)
print(data_obs.shape, params_inj.shape)
# Create a dataset object
dataset_instance = DatasetPP(data_obs, params_inj)
dataloader_instance = torch.utils.data.DataLoader(dataset_instance, batch_size=10, shuffle=False)
# Compute p-values for all datasets in one go
wanted_levels = p_values_for_grid(lightning_model, dataloader_instance, config, N_gridpoints=100).detach().numpy()
# reshape wanted_levels to (10, 10, n_wanted_levels)
wanted_levels = wanted_levels.reshape(-1, A_grid.shape[0], A_grid.shape[1])
# colorblind friendly colors
param_colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',]

# plot the coverage maps
alpha_values = [0.68, 0.95, 0.99]
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, coverage_map in enumerate(wanted_levels):
    c = ax[i].pcolormesh(A_grid.numpy(), omega_grid.numpy(), coverage_map, shading='auto', cmap='viridis')
    fig.colorbar(c, ax=ax[i])
    ax[i].set_xlabel('A')
    ax[i].set_ylabel('omega')
    ax[i].set_title(f'Coverage map for {param_names[1]} at {alpha_values[i]} HPD region')
fig.tight_layout()
fig.savefig(os.path.join(PROJECT_ROOT, 'figures', f'coverage_map_{config["model_name"]}.pdf'))
# save also coverage maps
np.save(os.path.join(PROJECT_ROOT, 'data', f'coverage_map_{config["model_name"]}.npy'), coverage_map_95)
np.save(os.path.join(PROJECT_ROOT, 'data', f'coverage_map_99_{config["model_name"]}.npy'), coverage_map_99)
np.save(os.path.join(PROJECT_ROOT, 'data', f'coverage_map_68_{config["model_name"]}.npy'), coverage_map_68)
# save the grid
np.save(os.path.join(PROJECT_ROOT, 'data', f'A_grid_{config["model_name"]}.npy'), A_grid.numpy())
np.save(os.path.join(PROJECT_ROOT, 'data', f'omega_grid_{config["model_name"]}.npy'), omega_grid.numpy())