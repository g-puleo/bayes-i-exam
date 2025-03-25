from sbi.model import lightning_MNRE
import torch
from tqdm import tqdm
class DatasetPP(torch.utils.data.Dataset):
    def __init__(self, observed_data, parameters):
        """
        Dataset for observed data and parameters.
        
        args:
        - observed_data: torch.Tensor of shape (num_datasets, num_obs_per_dataset, obs_dim)
        - parameters: torch.Tensor of shape (num_datasets, num_obs_per_dataset, num_params)
        """
        self.observed_data = observed_data
        self.parameters = parameters
        self.num_datasets = observed_data.shape[0]
        self.num_params = parameters.shape[2]

    def __len__(self):
        return self.num_datasets
    
    def __getitem__(self, idx):
        """
        Get the observed data and parameters for a specific dataset.
        
        args:
        - idx: index of the dataset
        
        returns:
        - observed_data: torch.Tensor of shape (num_obs_per_dataset, obs_dim)
        - parameters: torch.Tensor of shape (num_obs_per_dataset, num_params)
        """
        return self.observed_data[idx], self.parameters[idx]
        
    
def p_values_for_grid(lightning_model: lightning_MNRE, 
            loader: torch.utils.data.DataLoader,
             config: dict, 
             N_gridpoints: int = 100,
             levels: list = [0.68, 0.95, 0.99]
             ):
    """
    Compute the p-values for multiple datasets.
    
    args:
    - lightning_model: trained NRE model
    - loader: DataLoader containing the datasets. At each iteration it must return a tuple of (data_obs, par_inj):
        data_obs: torch.Tensor of shape (num_datasets, num_obs_per_dataset, obs_dim) containing the observed data
        par_inj: torch.Tensor of shape (num_datasets, num_obs_per_dataset, num_params) containing the injected parameters
    - config: dict containing the configuration of the priors
    - N_gridpoints: int, number of grid points for the parameter space

    returns:
    - credibility_levels: torch.Tensor of shape (num_datasets, N_gridpoints, num_params)
    """
    wanted_levels = torch.empty(len(levels), len(loader.dataset))
    seen_samples = 0
    for data_obs, par_inj in tqdm(loader): 
        batch_size = data_obs.shape[0]
        num_datasets, num_obs_per_dataset, obs_dim = data_obs.shape
        _, _, num_params = par_inj.shape

        # Create the parameter grid: shape (N_gridpoints, num_params)
        param_grid_tensor = torch.stack(
            [torch.linspace(config['priors'][p][0], config['priors'][p][1], N_gridpoints) 
            for p in config['priors']], dim=1
        )  # (N_gridpoints, num_params)

        # Expand observations to match the parameter grid shape
        obs_expanded = data_obs.unsqueeze(2).expand(-1, -1, N_gridpoints, -1)  # (num_datasets, num_obs_per_dataset, N_gridpoints, obs_dim)

        # Expand parameter grid to match the number of observations
        param_expanded = param_grid_tensor.unsqueeze(0).unsqueeze(0).expand(num_datasets, num_obs_per_dataset, -1, -1)  # (num_datasets, num_obs_per_dataset, N_gridpoints, num_params)

        # Flatten for parallel evaluation
        obs_flattened = obs_expanded.reshape(-1, obs_dim)  # (num_datasets * num_obs_per_dataset * N_gridpoints, obs_dim)
        param_flattened = param_expanded.reshape(-1, num_params)  # (num_datasets * num_obs_per_dataset * N_gridpoints, num_params)

        # Evaluate the model
        nre_weights = torch.exp(lightning_model.model(obs_flattened, param_flattened))[:,:3]  # (num_datasets * num_obs_per_dataset * N_gridpoints, num_params)

        # Reshape back to (num_datasets, num_obs_per_dataset, N_gridpoints, num_params)
        nre_weights = nre_weights.view(num_datasets, num_obs_per_dataset, N_gridpoints, num_params)

        # Normalize over the parameter grid for each observation
        nre_weights /= nre_weights.sum(dim=2, keepdim=True)

        # Store credibility levels: (num_datasets, num_obs_per_dataset, num_params)
        credibility_levels = torch.zeros(num_datasets, num_obs_per_dataset, num_params)

        for param_idx, param_name in enumerate(config['priors']): 

            # Extract parameter values and NRE weights for this parameter
            par_grid = param_grid_tensor[:, param_idx]  # (N_gridpoints,)
            weights = nre_weights[:, :, :, param_idx]  # (num_datasets, num_obs_per_dataset, N_gridpoints)

            # Sort grid points by decreasing NRE density
            sorted_indices = torch.argsort(weights, dim=2, descending=True)  # (num_datasets, num_obs_per_dataset, N_gridpoints)
            sorted_weights = torch.gather(weights, 2, sorted_indices)  # (num_datasets, num_obs_per_dataset, N_gridpoints)
            sorted_params = torch.gather(par_grid.expand(num_datasets, num_obs_per_dataset, -1), 2, sorted_indices)  # (num_datasets, num_obs_per_dataset, N_gridpoints)

            # Compute cumulative sum of posterior mass
            cumulative_mass = torch.cumsum(sorted_weights, dim=2)  # (num_datasets, num_obs_per_dataset, N_gridpoints)

            # Find where the injected parameter sits in the original grid
            injected_values = par_inj[:, :, param_idx]  # (num_datasets, num_obs_per_dataset)

            # Find closest grid point to each injected value
            injected_indices = torch.argmin(torch.abs(par_grid.unsqueeze(0).unsqueeze(0) - injected_values.unsqueeze(-1)), dim=2)  # (num_datasets, num_obs_per_dataset)

            # Locate injected parameter in the sorted list
            sorted_ranks = (sorted_indices == injected_indices.unsqueeze(-1)).nonzero(as_tuple=True)[2].view(num_datasets, num_obs_per_dataset) # (num_datasets, num_obs_per_dataset)

            # Get credibility level (cumulative mass at this rank)
            credibility_levels[:, :, param_idx] = cumulative_mass[ #pippo
                torch.arange(num_datasets).unsqueeze(-1).expand(-1, num_obs_per_dataset),  # (10, 1000)
                torch.arange(num_obs_per_dataset).unsqueeze(0).expand(num_datasets, -1),  # (10, 1000)
                sorted_ranks # (10, 1000)
            ] # has shape (num_datasets, num_obs_per_dataset)
            # whereas credibility_levels has shape (num_datasets, num_obs_per_dataset, num_params)
        
        for i in range(len(levels)):
            wanted_levels[i, seen_samples:seen_samples+batch_size] = torch.mean((credibility_levels < levels[i]).float(), dim=1)[:, 1]  # (num_datasets_in_batch, )
        # Append to the list
        seen_samples += batch_size

    return wanted_levels


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