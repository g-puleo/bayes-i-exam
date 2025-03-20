import numpy as np 
from sbi import Simulator , Sampler
from scipy.stats import uniform

def test_mcmc():
    ''' run a small monte carlo simulation to test the simulator and the sampler'''
    sigma = 0.1
    t = np.linspace(0, 1, 100)
    # simulate the true data
    simul = Simulator(sigma, t)

    # Parameters and data generation
    omega = 3.14
    phi = 1
    A = 1
    parameters = np.array([omega, phi, A]).reshape(1,-1)
    simul.simulate( parameters )

    # prior 
    uniform_prior_bounds = [   [0,5],      #omega
        [0,6.28],   #phi
        [0,2]]     #A

    uniform_prior = [ uniform(*a) for a in uniform_prior_bounds]


    y_obs = simul.simulate(parameters)
    sampler = Sampler(simul.likelihood_ratio, prior_distributions=uniform_prior, observed_data=y_obs, std=sigma)
    N_MC = 10
    samples = np.zeros((N_MC, len(uniform_prior)))
    for i in range(N_MC):
        samples[i] = sampler.state
        sampler.step()
        
    
def test_simulator () : 
    ''' test the simulator on a batch with a lot of parameter instances'''
    sigma = 0.1
    t = np.linspace(0, 1, 100)
    # simulate the true data
    simul = Simulator(sigma, t)

    # Parameters and data generation
    omega = np.linspace(0, 5, 10)
    phi = np.random.uniform(0, 6.28, size=10)
    A = np.random.uniform(0, 2, size=10)
    parameters = np.array([omega, phi, A]).reshape(-1, 3)
    obs = simul.simulate( parameters )
    print( obs.shape)