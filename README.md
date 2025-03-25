# Simulation based inference by Neural Ratio Estimation

In this project I train a neural network to approximate the posterior $p(\theta | d_{\text{obs}})$ where the observations are generated from the model
$$
d_{\text{obs},i} = A \sin(\omega t_i + \phi) + \epsilon_i
$$
where $A$ is the amplitude, $\omega$ is the frequency, $\phi$ is the phase, and $\epsilon_i$ is Gaussian noise. The goal is to infer the posterior distribution of the parameters $\theta =(\omega, \phi, A)$, given the observations.
What we do here is actually ``marginal neural ratio estimation'', meaning that we address only marginal posteriors.

## Steps to reproduce the results

1. Clone the repo, create a conda environment and install the required packages: 

```bash
conda env create --name sbienv
conda activate sbienv
pip install -r requirements.txt -e  . 
```
2. All scripts take one keyword argument ``--config`` which is a `.yaml` file. Templates are available in the `experiments` folder.  The configuration used for my results is defined in `experiments/config_light5.yaml`.
2. In order to generate an instance of the data and run a MCMC sampler on them, 
```bash
python scripts/generate_data.py --config experiments/config_light5.yaml
python scripts/run_MCMC.py
--config experiments/config_light5.yaml
```
3. A trained network is provided in the `trained_models` folder. If you wish to train your own, run 
```bash
python scripts/train.py --config experiments/config_light5.yaml
```
4. To produce the comparison between the MCMC posterior and the NRE posterior, run 
```bash
python scripts/plot_marginal_posteriors.py --config experiments/config_light5.yaml
```
5. To produce the p-p plots for the NRE posterior, run 
```bash
python scripts/pp_plot.py --config experiments/config_light5.yaml
``` 

6. To produce the coverage maps, run 
```bash 
python scripts/coverage_map.py --config experiments/config_light5.yaml
```