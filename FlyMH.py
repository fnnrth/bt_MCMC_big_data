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

        self.S = None
        self.alpha = None
        self.u = None
        self.theta = None

    def run(self, T, theta):
        self.S = np.zeros((T, theta.size))
        self.S[0] = theta
        for i in range(T-1):
            bright_data = self.get_bright_indx(S[i])
            S[i+1,:] = self.mh_step(S[i,:], bright_data)
        return S

    def get_bright_indx(self, theta):
        num_sampled_z = int(np.ceil(self.N*self.sample_fraction))
        subset_sample_indx = npr.randint(0, self.N, size=num_sampled_z)
        subset_sample_dp = self.dataset[subset_sample_indx]
        bernoulli_p = self.get_bright_prob(theta)
        bernoulli_sample_indx = npr.binomial(n=1,p=bernoulli_p, size=num_sampled_z)
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

    def get_gradientVec(self, theta, data):
        mu = theta[0]
        sig = theta[1]
        gradient_mu = -(mu - data)/(sig**2)
        gradient_sig = -1/sig + ((data - mu)**2)/(sig**3) 
        return np.array([gradient_mu, gradient_sig])

    def get_hessianVec(self,theta, data):
            hessian_mu_mu = -1/sig**2
            hessian_sig_sig = (1 - 3*(mu - data)**2/(sig**2))/sig**2
            hessian_mu_sig = -2*(data - mu)/sig**3

            return np.array([hessian_mu_mu, hessian_mu_sig],[hessian_mu_sig, hessian_sig_sig])

    def bounding_function(self, theta):
        return 0.01 # Not implemented yet

    def get_bright_prob(self, theta):
        return 1 - bounding_function(theta)/get_lkhd(theta,self.dataset)

# x = npr.randn(1000)
# theta = np.array([1,2])
# test = FlyMH(x, 0.1)
# test_run = test.run(100, theta)
# print(test_run)