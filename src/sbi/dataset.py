from sbi import Simulator
from torch.utils.data import Dataset, DataLoader
from scipy.stats import rv_continuous, uniform
import numpy as np
import torch
# set all seeds

class SimulatedDataset(Dataset):
    def __init__(self, simulator, priors:list[rv_continuous], N_samples):
        self.simulator = simulator
        self.priors = priors
        self.N_samples = N_samples
        # generate data-parameters pairs from the joint and from the marginals
        param_instances = []
        param_instances_scrambled = []
        for prior in self.priors:
            x = torch.Tensor(prior.rvs(size=N_samples))
            param_instances.append(x)  # sample from the prior
            # sample also from the marginal by scrambling the parameters
            param_instances_scrambled.append(torch.roll(x, shifts=1))  # avoid sampling twice by scrambling (theta, data) pairs

        self.parameters = torch.stack(param_instances, dim=1)
        self.parameters_scrambled = torch.stack(param_instances_scrambled, dim=1)
        self.observed_data = simulator.simulate(self.parameters).to(torch.float32)
        
    def __len__(self):
        return self.N_samples

    def __getitem__(self, idx):
        obs = self.observed_data[idx]
        par_joint = self.parameters[idx]
        par_scrambled = self.parameters_scrambled[idx]
        return obs, par_joint, par_scrambled


if __name__ == '__main__':
    
    # try to init the dataset
    t = torch.linspace(0, 10, 100)
    sigma = 0.1
    simulator = Simulator(sigma, t)
    priors = [uniform(0, 2*np.pi), uniform(0, 2*np.pi), uniform(0, 1)]
    N_samples = 1000
    dataset = SimulatedDataset(simulator, priors, N_samples)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        data, params_joint, params_scrambled = batch
        print(f"Data shape: {data.shape}")
        print(f"Params joint shape: {params_joint.shape}")
        print(f"Params scrambled shape: {params_scrambled.shape}")
        # print also dtypes
        print(f"Data dtype: {data.dtype}")
        print(f"Params joint dtype: {params_joint.dtype}")
        print(f"Params scrambled dtype: {params_scrambled.dtype}")
        break
    