import torch
import numpy as np

class VP():
    def __init__(self, beta_min, beta_max, num_steps):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.num_steps = num_steps
        self.discrete_betas = torch.linspace(beta_min / num_steps, beta_max / num_steps, num_steps)
        self.alphas = 1. - self.discrete_betas

    def _beta_t(self, t):
        # TODO: compute beta(t)
        beta_t = (self.beta_0 + t*(self.beta_1 - self.beta_0))
        #print(beta_t)
        return beta_t
    
    def _c_t(self, t):
        # TODO: compute c(t)
        c_t = -1/4 * t**2 * (self.beta_1 - self.beta_0) - 1/2 * t * self.beta_0
        return c_t

    def marginal_proba(self, x, t):
        """ Compute the mean and standard deviation of the marginal prob p(x_0|x_t)
        """
        # TODO: compute mu and std (std is a scalar)
        mu_t = (torch.exp(self._c_t(t))).view(-1, 1, 1, 1) * x
        std_t = torch.sqrt(1 - torch.exp(2*self._c_t(t)) )
        return mu_t, std_t

    def drift(self, x, t):
        """ Compute the VP drift coefficient f(x, t) 
        """
        batch = x.shape[0]
        # TODO: compute drift coefficient -- make sure to give beta_t the appropriate shape
        #print(self._beta_t(t).shape)
        drift = (-1/2 * self._beta_t(t)) * x
        #print('drift', drift)
        return drift

    def diffusion(self, t):
        """ Compute the VP diffusion coefficient g(t)
        """
        # TODO: compute diffusion coefficient
        #print('beta_t', self._beta_t(t))
        diffusion = torch.sqrt(self._beta_t(t))
        return diffusion
        