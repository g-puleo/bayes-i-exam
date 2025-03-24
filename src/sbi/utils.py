from sbi.model import lightning_MNRE
import torch
def p_values(lightning_model: lightning_MNRE , data_obs: torch.Tensor, par_inj: torch.Tensor, config:dict, N_gridpoints:int=100):
    """
    compute the p-values associated with each observation data_obs associated to an injected parameter par_inj.
    It means: for every instance of an observation, we evaluate the NRE estimate of the posterior, and we find the associated HPD region. In particular, 
    we find the smallest value of alpha such that the true (injected) parameter is contained within the alpha-HPD region.
    The results are useful to make a pp-plot. 
    """

    all_obs = data_obs
    num_obs = all_obs.shape[0]

    # Stack parameter grids into a tensor of shape (N_gridpoints, num_params)
    param_grid_tensor = torch.stack([torch.linspace(config['priors'][p][0], config['priors'][p][1], N_gridpoints) for p in config['priors']], dim=1)  # Shape: (N_gridpoints, num_params)

    # Expand observations to match param grid shape
    obs_expanded = all_obs.unsqueeze(1).repeat(1, N_gridpoints, 1)  # Shape: (num_obs, N_gridpoints, obs_dim)

    # Expand param grid to match the number of observations
    param_expanded = param_grid_tensor.unsqueeze(0).repeat(num_obs, 1, 1)  # Shape: (num_obs, N_gridpoints, num_params)

    # Flatten for parallel evaluation
    obs_flattened = obs_expanded.view(-1, obs_expanded.shape[-1])  # Shape: (num_obs * N_gridpoints, obs_dim)
    param_flattened = param_expanded.view(-1, param_expanded.shape[-1])  # Shape: (num_obs * N_gridpoints, num_params)

    # evaluate the model
    nre_weights = torch.exp(lightning_model.model(obs_flattened, param_flattened))  # Shape: (num_obs * N_gridpoints, num_params)

    # Reshape back to (num_obs, N_gridpoints, num_params)
    nre_weights = nre_weights.view(num_obs, N_gridpoints, -1)

    # Normalize over the parameter grid for each observation
    nre_weights /= nre_weights.sum(dim=1, keepdim=True)

    credibility_levels = torch.zeros(num_obs, len(config['priors']))  # Store results

    for param_idx, param_name in enumerate(config['priors']):  
        # Extract parameter values and NRE weights for this parameter
        par_grid = param_grid_tensor[:, param_idx]  # Shape: (N_gridpoints,)
        weights = nre_weights[:, :, param_idx]  # Shape: (num_obs, N_gridpoints)

        # Sort grid points **for each observation** by decreasing NRE density
        sorted_indices = torch.argsort(weights, dim=1, descending=True)  # Shape: (num_obs, N_gridpoints)
        
        sorted_weights = torch.gather(weights, 1, sorted_indices)  # Shape: (num_obs, N_gridpoints)
        sorted_params = torch.gather(par_grid.expand(num_obs, -1), 1, sorted_indices)  # Shape: (num_obs, N_gridpoints)

        # Compute cumulative sum of posterior mass
        cumulative_mass = torch.cumsum(sorted_weights, dim=1)  # Shape: (num_obs, N_gridpoints)

        # Find where the injected parameter sits in the original grid
        injected_values = par_inj[:, param_idx]  # Shape: (num_obs,)

        # Find closest grid point to each injected value
        injected_indices = torch.argmin(torch.abs(par_grid.unsqueeze(0) - injected_values.unsqueeze(1)), dim=1)  # Shape: (num_obs,)

        # Locate injected parameter in the sorted list
        sorted_ranks = (sorted_indices == injected_indices.unsqueeze(1)).nonzero(as_tuple=True)[1]  # Shape: (num_obs,)

        # Get credibility level (cumulative mass at this rank)
        credibility_levels[:, param_idx] = cumulative_mass[torch.arange(num_obs), sorted_ranks]

    return credibility_levels