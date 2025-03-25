import numpy as np
from sbi import Simulator , Sampler, PROJECT_ROOT
from scipy.stats import uniform
from tqdm import tqdm
import argparse, yaml
import matplotlib.pyplot as plt
import os
np.random.seed(0)
parser = argparse.ArgumentParser(description="Generate data based on a configuration file.")
parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# Load the observed data
fname = os.path.join(PROJECT_ROOT, 'data', f'observed_data_{config["name"]}.npy')
y_obs = np.load(fname)
print(f'Loaded observed data from {fname}')
t = np.linspace(config['time_steps']['min'], config['time_steps']['max'], config['time_steps']['n'])
simulator = Simulator(config['sigma'], t)
prior_bounds = list(config['priors'].values())
uniform_priors = [uniform(prior_bounds[i][0], -prior_bounds[i][0] + prior_bounds[i][1]) for i in range(len(prior_bounds))]
sampler = Sampler(simulator.likelihood_ratio, prior_distributions=uniform_priors, observed_data=y_obs, std=0.1)
N_MC = 100000
burn_in = 50000
burn_in_samples = np.zeros((burn_in, len(uniform_priors)))
samples = np.zeros((N_MC, len(uniform_priors)))
# burn_in samples
print("Burn-in samples...")
for i in range(burn_in):
    burn_in_samples[i] = sampler.state
    sampler.step()
sampler.print_info()
print("Burn-in samples completed.")
print("Sampling...")
accepted_100 = np.empty(100)
for i in range(N_MC):
    samples[i] = sampler.state
    accepted = sampler.step()
    accepted_100[i % 100] = accepted

    #if i % 1000 == 0:
        #print(f"Sample {i}/{N_MC}, Mean Accepted in last 100: {np.mean(accepted_100)}")
    #    sampler.print_info()

# make a plot of the time evolution of each parameter
fig, ax = plt.subplots(1,1, figsize=(5, 5))
param_names = list(config['parameters'].keys())
for i in range(len(uniform_priors)):
    ax.plot(range(burn_in), burn_in_samples[:, i], label=f'{param_names[i]} (Burn-in)')
    ax.plot(range(burn_in, N_MC+burn_in), samples[:, i], label=f'{param_names[i]}')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Parameter Value')
ax.legend()
plt.title('Parameter Sampling')
fig.tight_layout()
fig.savefig(os.path.join(PROJECT_ROOT, 'figures', f'mcmc_sampling_{config["name"]}.pdf'))
print("Sampling completed.")
np.save(os.path.join(PROJECT_ROOT,'data', f'MC_samples_{config["name"]}.npy'), samples)
np.save(os.path.join(PROJECT_ROOT,'data', f'Burn_in_samples_{config["name"]}.npy'), burn_in_samples)

