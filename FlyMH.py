import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as npr
import time

from MetropolisHastings import MetropolisHastings

class FlyMH(MetropolisHastings):

    def __init__(self, dataset, sample_fraction):
        super().__init__(dataset)
        self.sample_fraction = sample_fraction

    def run(self, T, theta):
        S = np.zeros((T, theta.size))
        S[0,:] = theta
        for i in range(T-1):
            bright_data = self.get_bright_indx(S[i,:])
            S[i+1,:] = self.mh_step(S[i,:], bright_data)
        return S

    def get_bright_indx(self, theta):
        numResampledZs = int(np.ceil(self.N*self.sample_fraction))
        subset_sample_indx = npr.randint(0, self.N, size=numResampledZs)
        subset_sample_dp = self.dataset[subset_sample_indx]
        p = self.get_bright_prob(theta)
        bernoulli_sample_indx = npr.binomial(n=1,p=p, size=numResampledZs)
        sample_dp = subset_sample_dp[bernoulli_sample_indx == 1]
        return sample_dp

    def get_log_alpha(self, theta, theta_new, data):
        log_lkhd_new = np.log(self.get_lkhd(theta_new, data)/ self.bounding_function(theta_new) -1)
        log_lkhd_old = np.log(self.get_lkhd(theta, data) / self.bounding_function(theta) - 1)
        lkhd = log_lkhd_new - log_lkhd_old
        log_bounding = self.bounding_function(theta_new) - self.bounding_function(theta)
        return np.mean(log_bounding - lkhd)

    def get_lkhd(self, theta, data):
        return 1/(theta[1]*np.sqrt(2*np.pi)) * np.exp(-(((data - theta[0])/theta[1])**2)/2)

    def bounding_function(self, theta):
        return 0.01 # Not implemented yet

    def get_bright_prob(self, theta):
        return 0.5 # Not implemented yet

# x = npr.randn(1000)
# theta = np.array([1,2])
# test = FlyMH(x, 0.1)
# test_run = test.run(100, theta)
# print(test_run)