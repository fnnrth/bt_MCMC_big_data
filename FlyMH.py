import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

from MetropolisHastings import MetropolisHastings

class FlyMH(MetropolisHastings):

    def __init__(self, dataset, sample_fraction):
        super.__init__(dataset)
        self.sample_fraction = sample_fraction

    def run(self):
        S = torch.zeros((T, theta.size))
        S[0,:] = theta
        for i in range(T-1):
            bright_indx = self.get_bright_indx()
            S[i+1,:] = self.mh_step(S[i,:], subset_data)
        return S

    def get_bright_indx(self):
        numResampledZs = int(np.ceil(N*self.sample_fraction))
        resample_ind = npr.randint(0, self.N, size=numResampledZs)
        bright_indx = np.zeros(self.N)

        bright_indx[resampledInds] = npr.binomial(n=1,p=self.get_dark_probability, size=numResampledZs)

    def get_log_lkhd(self, theta, bright_indx):
            return -(((self.dataset[bright_indx] - theta[0])/theta[1])**2)/2 - np.log(theta[1])

    def get_log_alpha(self, theta, theta_new, bright_indx):
        lkhd = (self.get_log_lkhd(theta_new, bright_indx) - self.get_log_lkhd(theta, bright_indx)) # lkhd_new - lkhd_old
        return np.mean(lkhd)

    def bounding_function(self, theta):
        pass

    def get_dark_probability(self, theta):
        return 0.5 # Not implemented yet
        # 1 - self.get_log_lhd(theta, )