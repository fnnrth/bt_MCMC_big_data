import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as npr
import time

class MetropolisHastings():
    def __init__(self, dataset):
        self.dataset = dataset
        self.N = dataset.size

    def run(self, T, theta):
        S = np.zeros((T, theta.size))
        S[0,:] = theta
        for i in range(T-1):
            S[i+1,:] = self.mh_step(S[i,:], self.dataset)
        return S

    def mh_step(self, theta, data):
        theta_new = self.get_theta_new(theta)
        log_alpha = self.get_log_alpha(theta, theta_new, data)
        log_u = np.log(npr.rand(1))/ data.size
        if log_u < log_alpha:
            theta = theta_new
        return theta

    def get_theta_new(self, theta):
        return theta + self.get_stepsize()*npr.randn(2)

    def get_log_alpha(self, theta, theta_new, data):
        lkhd = self.get_log_lkhd(theta_new, data) - self.get_log_lkhd(theta, data) # lkhd_new - lkhd_old
        return np.mean(lkhd)

    def get_log_lkhd(self, theta, data):
        return -(((data - theta[0])/theta[1])**2)/2 - np.log(theta[1]) # -((data-mu)/sig)**2 - log(sig)

    def get_stepsize(self):
        return 0.5/np.sqrt(self.N) # 0.5/sqrt(N)

