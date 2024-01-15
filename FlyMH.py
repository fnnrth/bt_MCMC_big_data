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
            subset_data = self.subset_data()
            S[i+1,:] = self.mh_step(S[i,:], subset_data)
        return S

    def subset_data(self):
        numResampledZs = int(np.ceil(N*resampleFraction))
        resample_ind = torch.randint(0, self.N, size=numResampledZs)
        subset_data = np.zeros(N)

        subset_data[resampledInds] = npr.binomial(n=1,p=0.5, size=numResampledZs)

    def get_log_lkhd(self, theta, data):
        pass

    def bounding_function(self, theta):
        pass

    def get_dark_probability(self, theta):
        return 0.5 # Not implemented yet