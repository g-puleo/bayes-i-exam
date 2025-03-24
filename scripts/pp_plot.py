from sbi.dataset import SimulatedDataset
from sbi.model import MNRE, lightning_MNRE
from sbi import Simulator  , PROJECT_ROOT
import os , yaml
from scipy.stats import uniform
import numpy as np
import torch
import argparse


# Load the configuration file
parser = argparse.ArgumentParser(description="Process configuration file path.")
parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
args = parser.parse_args()

config_path = args.config
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

simulator  = Simulator( sigma=config['sigma'], t=np.linspace(config['time_steps']['min'], config['time_steps']['max'], config['time_steps']['n']))  
priors = [uniform(config['priors'][param][0], config['priors'][param][1] - config['priors'][param][0]) for param in config['priors']]
mock_dataset = SimulatedDataset(simulator, priors , 1000)
mock_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=1, shuffle=False)
lightning_model = lightning_MNRE.load_from_checkpoint(os.path.join(PROJECT_ROOT, 'trained_models', f'model_{config["model_name"]}.pth'))

# create a evenly spaced parameter grid for each param
param_grid = {}
N_gridpoints = 100
for param in config['priors']:
    param_grid[param] = torch.linspace(config['priors'][param][0], config['priors'][param][1], N_gridpoints)

model_dict = {'omega': lightning_model.model.tail_1, 
              'phi': lightning_model.model.tail_2, 
              'A': lightning_model.model.tail_3}
credibility_levels = {'omega': [], 'phi': [], 'A': []}
for batch in mock_dataloader:
    obs, par_inj, _ = batch
    # repeat obs along 0th dimension so that it matches the shape of the param_grid
    obs = obs.repeat(N_gridpoints, 1)
    print(obs.shape)
    for idx , param in enumerate( param_grid ) :
        param_in = torch.zeros(N_gridpoints, 3)
        par_grid = param_grid[param]
        param_in[:, idx] = par_grid
        nre_weights = torch.exp(lightning_model.model(obs, param_in))[:, idx]
        nre_weights /= nre_weights.sum()#normalise
        # Sort grid points by decreasing posterior density
        sorted_indices = torch.argsort(nre_weights, descending=True)
        sorted_weights = nre_weights[sorted_indices]
        sorted_params = par_grid[sorted_indices]  # Reorder parameters in the grid by posterior density

        # Compute cumulative sum of posterior mass in this order
        cumulative_mass = torch.cumsum(sorted_weights, dim=0)

        # Find where the injected parameter sits in the original (unsorted) grid
        injected_value = par_inj[0, idx]  # True injected parameter value
        injected_index = torch.argmin(torch.abs(par_grid - injected_value))  # Closest grid point
        
        # Locate the index of this injected parameter in the sorted list
        sorted_rank = (sorted_indices == injected_index).nonzero(as_tuple=True)[0].item()
        
        # Credibility level is the cumulative probability at this index
        credibility_level = cumulative_mass[sorted_rank]  
        
        credibility_levels[param].append(credibility_level.item())
