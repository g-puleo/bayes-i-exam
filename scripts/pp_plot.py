from sbi.utils import p_values
from sbi.model import lightning_MNRE
from sbi.dataset import SimulatedDataset
from sbi import PROJECT_ROOT, Simulator
from scipy.stats import uniform
import matplotlib.pyplot as plt
import argparse, yaml, torch, os
import numpy as np
parser = argparse.ArgumentParser(description="Generate posterior plots based on priors specified in a configuration file.")
parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
args = parser.parse_args()

#import samples from MC simulations:


# load the configuration file
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# load the MNRE model
lightning_model = lightning_MNRE.load_from_checkpoint(os.path.join(PROJECT_ROOT, 'trained_models', f'model_{config["model_name"]}.pth'))
# generate observed data
simulator = Simulator(config['sigma'], torch.linspace(config['time_steps']['min'], config['time_steps']['max'], config['time_steps']['n']))
priors = [uniform(config['priors'][param][0], config['priors'][param][1] - config['priors'][param][0]) for param in config['priors']]
param_names = list(config['priors'].keys())
# for colorblind friendly colors
param_colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',]
dataset = SimulatedDataset(simulator, priors, 100)
observed_data = dataset.observed_data
inj_params = dataset.parameters

credibility_levels = p_values(lightning_model, observed_data, inj_params, config, N_gridpoints=1000).detach().numpy()

# now that credibility levels are computed, we can plot the pp-plot
fig, ax = plt.subplots(figsize=(8, 8))
# plot sorted pvalues 
for idx in range(credibility_levels.shape[1]):
    ax.plot(np.arange(1, len(credibility_levels[:,idx]) + 1) / len(credibility_levels[:,idx]), np.sort(credibility_levels[:,idx]), label=f'{param_names[idx]}', color=param_colors[idx])

ax.grid(visible=True)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('fraction of true parameters falling in the $\\alpha$ HPD region')
ax.legend()

fig.savefig(os.path.join(PROJECT_ROOT, 'figures', f'pp_plot_{config["model_name"]}.pdf'))