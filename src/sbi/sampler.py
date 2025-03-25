import numpy as np
from .simulator import Simulator
from scipy.stats import rv_continuous
class Sampler: 
    
    def __init__(self, likelihood_ratio: callable, prior_distributions: list[rv_continuous], observed_data, std:np.array):

        self.likelihood_ratio = likelihood_ratio
        self.priors = prior_distributions
        self.state = np.zeros((1,len(self.priors)))
        for i in range(len(self.priors)):
            self.state[0,i] = self.priors[i].rvs()
            print(f"initial state {i}: {self.state[0,i]}")
        self.observation = observed_data
        self.proposed_steps = 0
        self.accepted_steps = 0
        self.std = std

    def proposal(self, par_0) :
        delta = self.std * np.random.normal(size=par_0.shape)
        return par_0 + delta

    def step(self):
        par_new = self.proposal(self.state)
        #print(f"proposal: {par_new}")
        self.proposed_steps +=1

        update_bool = np.prod([ bool(prior.pdf(par)) for par, prior in zip(par_new.T, self.priors)])
        if np.prod(update_bool)==True:
            a = self.likelihood_ratio(self.state, par_new, self.observation)
            if a>1:
                self.accepted_steps+=1
                self.state=par_new
                return True
            else:
                r = np.random.uniform()
                if r<a:
                    self.accepted_steps+=1
                    self.state=par_new
                    return True
                else:
                    return False
        else:
            return False
        
    def print_info(self ):
        print(f"acceptance ratio: {self.accepted_steps/self.proposed_steps}")
    
