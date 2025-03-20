import torch
import os
import numpy as np
from sbi import PROJECT_ROOT 
import argparse
import yaml
from scipy.stats import uniform
from sbi.model import MNRE, lightning_MNRE
parser = argparse.ArgumentParser(description="Generate data based on a configuration file.")
parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)
# sample from the prior
N_samples = 10000
priors = config["priors"]
sampled_params_prior  = {}
for param in priors:
    sampled_params_prior[param] = uniform.rvs( *priors[param], size=N_samples)
# put the sampled parameters in a torch tensor
sampled_params_prior = torch.tensor(np.array([sampled_params_prior[param] for param in priors]).T)

# load the MNRE model
lightning_model_state = torch.load(os.path.join(PROJECT_ROOT,'trained_models/model.pth'))
print(lightning_model_state["hyper_parameters"])
lightning_model = lightning_MNRE.load_from_checkpoint(lightning_model_state)
observed_data = torch.Tensor(np.load(os.path.join(PROJECT_ROOT, 'data', 'observed_data.npy'))).repeat(N_samples, 1)
print(observed_data.shape)
print(sampled_params_prior.shape)
weights = lightning_model.model(observed_data, sampled_params_prior)
print(weights.shape)