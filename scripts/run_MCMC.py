import numpy as np
from sbi import Simulator , Sampler
from scipy.stats import uniform


sigma = 0.1
t = np.linspace(0, 1, 100)
# simulate the true data
simul = Simulator(sigma, t)

# Parameters and data generation
omega = 3.14
phi = 1
A = 1
parameters = np.array([omega, phi, A]).reshape(1,-1)

# prior 
uniform_prior_bounds = [   [0,5],      #omega
    [0,6.28],   #phi
    [0,2]]     #A

uniform_prior = [ uniform(*a) for a in uniform_prior_bounds]

y_obs = simul.simulate(parameters)
sampler = Sampler(simul.likelihood_ratio, prior_distributions=uniform_prior, observed_data=y_obs, std=sigma)
N_MC = 100000

samples = np.zeros((N_MC, len(uniform_prior)))
for i in range(N_MC):
    samples[i] = sampler.state
    sampler.step()


np.save('samples.npy', samples)
    