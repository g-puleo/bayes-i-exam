import torch
import os
import numpy as np
from sbi import PROJECT_ROOT 
import argparse
import yaml
import matplotlib.pyplot as plt
from scipy.stats import uniform
from sbi.model import MNRE, lightning_MNRE
import corner
import seaborn as sns
import pandas as pd
parser = argparse.ArgumentParser(description="Generate posterior plots based on priors specified in a configuration file.")
parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
args = parser.parse_args()

#import samples from MC simulations:


# load the configuration file
with open(args.config, "r") as file:
    config = yaml.safe_load(file)
mc_samples = np.load(os.path.join(PROJECT_ROOT,'data', f'MC_samples_{config["name"]}.npy')) # has shape (N_MC, n_params)
# sample from the prior
N_samples = 10000
priors = config["priors"]
sampled_params_prior  = {}
for param in priors:
    print(f"rvs for param {param} called with inputs")
    print(*priors[param])
    sampled_params_prior[param] = uniform.rvs( loc=priors[param][0], scale=priors[param][1] - priors[param][0], size=N_samples)
    # # print max and min of the sampled parameters
    # print(f"max {param}: {np.max(sampled_params_prior[param])}")
    # print(f"min {param}: {np.min(sampled_params_prior[param])}")
# put the sampled parameters in a torch tensor
sampled_params_prior = torch.tensor(np.array([sampled_params_prior[param] for param in priors]).T, dtype=torch.float32)

# load the MNRE model
lightning_model = lightning_MNRE.load_from_checkpoint(os.path.join(PROJECT_ROOT, 'trained_models', f'model_{config["model_name"]}.pth'))
observed_data = torch.tensor(np.load(os.path.join(PROJECT_ROOT, 'data', f'observed_data_{config["name"]}.npy')), dtype=torch.float32).repeat(N_samples, 1)
print(observed_data.dtype)
print(sampled_params_prior.dtype)
weights_nre = lightning_model.model(observed_data, sampled_params_prior).detach().numpy()
# Create a corner plot of the Monte Carlo samples
#figure = corner.corner(mc_samples, labels=list(priors.keys()), show_titles=True, density=True)
# get the axes of the corner plot
ndim = 3
figure, axes = plt.subplots(ndim, ndim, figsize=(10, 10))
# Convert mc_samples to a DataFrame for seaborn
thinning= 500
print(mc_samples.shape)
mc_samples_df = pd.DataFrame(mc_samples, columns=list(priors.keys()))

# Create a pairplot with KDE contour lines
pairplot = sns.pairplot(mc_samples_df[::100], kind="kde", diag_kind="kde", corner=True)
# Extract the marginal plots (diagonal) and add histograms of weighted prior samples
for i, param in enumerate(priors.keys()):
    ax = pairplot.diag_axes[i]
    weighted_samples = sampled_params_prior[:, i].numpy()
    weights = np.exp(weights_nre[:,i])
    ax.hist(weighted_samples, density=True, bins=30, weights=weights, alpha=0.5, color='orange')
    ax.legend(["MCMC", "MNRE posterior"])

# Extract the off-diagonal plots in a similar way
for idx, ax in np.ndenumerate(pairplot.axes):
    i, j = idx
    if j >= i:
        continue
    print(f'access axis {i}, {j}')
    print(f'done')
    print(type(ax))
    ax.set_title(f"{i}, {j}")
    weighted_samples_i = sampled_params_prior[:, j].numpy()
    weighted_samples_j = sampled_params_prior[:, i].numpy()
    #lnr_1, lnr_2, lnr_3, lnr_12, lnr_13, lnr_23, want the 12, 13, 23 in the right order
    # 1,0 -> 12 == 3
    # 2,0 -> 13 == 4
    # 2,1 -> 23 == 5
    weights_i = np.exp(weights_nre[:, i+2])
    weights_j = np.exp(weights_nre[:, j+i+2])
    # Create a 2D histogram with weights
    sns.kdeplot(
        x=weighted_samples_i,
        y=weighted_samples_j,
        weights=weights_i * weights_j,
        ax=ax,
        cmap='Reds',
        fill=False,
        levels=30
    )
# Adjust the layout and display the plot
# pairplot.suptitle("Pairplot of Monte Carlo Samples", y=1.02)
pairplot.figure.savefig(os.path.join(PROJECT_ROOT, 'figures', 'pairplot_mc_samples.png'))
