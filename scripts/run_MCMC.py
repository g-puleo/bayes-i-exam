import numpy as np
from sbi import Simulator , Sampler, PROJECT_ROOT
from scipy.stats import uniform
from tqdm import tqdm
import argparse, yaml
import os
parser = argparse.ArgumentParser(description="Generate data based on a configuration file.")
parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# Load the observed data
y_obs = np.load(os.path.join(PROJECT_ROOT, 'data', f'observed_data_{config["name"]}.npy'))
t = np.linspace(config['time_steps']['min'], config['time_steps']['max'], config['time_steps']['n'])
simulator = Simulator(config['sigma'], t)
prior_bounds = list(config['priors'].values())
uniform_priors = [uniform(prior_bounds[i][0], -prior_bounds[i][0] + prior_bounds[i][1]) for i in range(len(prior_bounds))]
sampler = Sampler(simulator.likelihood_ratio, prior_distributions=uniform_priors, observed_data=y_obs, std=config['sigma'])
N_MC = 100000
burn_in = 50000
samples = np.zeros((N_MC, len(uniform_priors)))
# burn_in samples
print("Burn-in samples...")
for i in tqdm(range(burn_in)):
    sampler.step()
print("Burn-in samples completed.")
print("Sampling...")
for i in tqdm(range(N_MC)):
    samples[i] = sampler.state
    sampler.step()

print("Sampling completed.")
np.save(os.path.join(PROJECT_ROOT,'data', f'MC_samples_{config["name"]}.npy'), samples)
