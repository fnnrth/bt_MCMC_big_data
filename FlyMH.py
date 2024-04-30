import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as npr
import time

from MetropolisHastings import MetropolisHastings
from likelihood_functions import Norm_lkhd  

class FlyMH(MetropolisHastings):

    def __init__(self, dataset, likelihood_function, sample_fraction):
        super().__init__(dataset, likelihood_function)
        self.sample_fraction = sample_fraction
        self.MAP_mu = np.mean(dataset)
        self.MAP_sigma = np.std(dataset)
        self.thetaMAP = np.array([self.MAP_mu, self.MAP_sigma]) 
        self.avgData = np.mean(dataset)
        self.log_lkhdMAP = likelihood_function.comp_func(self.thetaMAP, np.mean(dataset))
        self.meanGradMAP = likelihood_function.comp_gradient(self.thetaMAP, np.mean(dataset))
        self.meanHessMAP = likelihood_function.comp_hessian(self.thetaMAP, np.mean(dataset))

    def run(self, T, theta):
        self.S = np.zeros((T, theta.size)) # Initialize empty sampleset
        self.bounding = np.zeros(T)
        self.alpha = np.zeros(T)
        self.accept = np.zeros(T)
        self.lkhd = np.zeros(T)
        self.S[0] = theta # First sample is starting point
        self.lkhd[0] = self.get_lkhd(theta, self.dataset) 
        for i in range(T-1):
            bright_data = self.get_data_subset(i)
            self.mh_step(i, bright_data)
        return self.S

    def get_data_subset(self, i):
        num_sampled_z = int(np.ceil(self.N*self.sample_fraction))
        subset_sample_indx = npr.randint(0, self.N, size=num_sampled_z)
        subset_sample_dp = self.dataset[subset_sample_indx]
        #bernoulli_p = self.get_bright_prob(i, subset_sample_dp)
        #bernoulli_sample_indx = npr.binomial(n=1,p=bernoulli_p)
        #sample_dp = subset_sample_dp[bernoulli_sample_indx == 1]
        return subset_sample_dp #Try sampling without lkhd

    def get_alpha(self, i , theta, theta_new, data):
        self.lkhd[i+1] = self.get_lkhd(theta, data) 
        lkhd_proposed = self.lkhd[i+1]
        lkhd_curr = self.lkhd[i]
        self.alpha[i] = np.exp(lkhd_proposed - lkhd_curr)

    def get_lkhd(self, theta, data):
        h = theta - self.thetaMAP
        R = (np.sum(np.abs(h))**3)/6
        avgLogBound =  np.mean(self.log_lkhdMAP) + np.dot(self.meanGradMAP, h) + .5*np.dot(np.dot(self.meanHessMAP.T, h),h) - R
        L = np.exp(self.log_lkhd.comp_func(theta, data))
        B = np.exp(self.log_bounding_function(theta, data))
        proxy = L/B - 1
        lkhd = avgLogBound + np.sum(np.log(np.abs(proxy)))/self.N
        return lkhd

    def log_bounding_function(self, theta, data): #TODO
        thetaMAP = self.thetaMAP
        h = theta - thetaMAP
        R = (np.sum(np.abs(h))**3)/6
        lkhd = self.log_lkhd.comp_func(thetaMAP, data)
        lkhd_gradient = self.log_lkhd.comp_gradient(thetaMAP, data)
        lkhd_hessian = self.log_lkhd.comp_hessian(thetaMAP, data)
        gradient = np.dot(lkhd_gradient.T, h)
        hessian = np.dot(np.dot(lkhd_hessian.T,h), h)
        logB = lkhd + gradient + 0.5*hessian - R
        return logB

    def get_bright_prob(self, i, data):
        mu = self.S[i][0]
        sigma = self.S[i][1]
        L = np.exp(-((mu - data)**2)/(2*sigma**2) - np.log(sigma))
        B = np.exp(self.log_bounding_function(self.S[i], data))
        bright_prob = 1 - L/B
        bright_prob_bounded = np.clip(bright_prob, 0, 1)
        return bright_prob_bounded

x = npr.randn(10000)
theta = np.array([1,2])
norm_lkhd = Norm_lkhd()
test = FlyMH(x, norm_lkhd,0.1)
test_run = test.run(100, theta)
print(test_run)
print(test.accept)