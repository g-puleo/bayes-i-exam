import numpy as np

class Simulator: 
    def __init__(self, sigma, t):
        '''
        t: np.array of time points where to evaluate the model
        sigma: noise in the simulator
        '''
        self.sigma = sigma
        self.sigma2 = sigma**2
        self.t = t
    
    def model ( self, parameters):
        '''theoretical model without noise:
        parameters has shape (N,3) where N can be a batch size
        parameters[:,0]: omega
        parameters[:,1]: phi
        parameters[:,2]: A


        A*sin(omega*t+phi)
        
        '''
        #       shape (N,) * (shape(N,T) + shape(N,) ) yields 
        output = parameters[:,2].reshape(-1,1)* np.sin(parameters[:,0].reshape(-1,1) * self.t.reshape(1,-1) + parameters[:,1].reshape(-1, 1))
        return output

    def simulate ( self, parameters) : 
        pure_signal = self.model(parameters)
        data = self.sigma*np.random.normal(size=pure_signal.shape)
        return pure_signal + data

    def likelihood_ratio(self, param_0, param_new, y_observed):
        '''evaluate min(likelihood(param_new)/likelihood(param_0),1) when data y_observed is observed'''
        mu_0 = self.model(param_0)
        mu_new = self.model(param_new)
        logratio = (mu_0-mu_new)*(-2*y_observed + mu_0+mu_new)
        logratio/= (2*self.sigma2)
        summation = np.sum(logratio)
        if summation>0: 
            # when the likelihood ratio is >1 I return 1.1 to avoid numerical issues
            # the exact value does not matter, the important thing is that it is >1
            return 1.1 
        else:
            return np.exp(summation)
    
        
        
if __name__ == '__main__':
    pass