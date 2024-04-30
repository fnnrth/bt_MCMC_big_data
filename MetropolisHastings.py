import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as npr
import time
from likelihood_functions import Norm_lkhd

class MetropolisHastings():
    def __init__(self, dataset, Likelihood_function):
        self.dataset = dataset
        self.N = dataset.size # Number of datapoints
        self.stepsize = 0.01
        self.log_lkhd = Likelihood_function
        self.S = None
        self.alpha = None
        self.accept = None
        self.lkhd = None

    def run(self, T, theta, data):
        self.S = np.zeros((T, theta.size)) # Initialize empty sampleset
        self.alpha = np.zeros(T)
        self.accept = np.zeros(T)
        self.lkhd = np.zeros(T)
        self.S[0] = theta # First sample is starting point
        self.get_log_lkhd(0, theta, data) 
        for i in range(T-1): # Iterate over number of iterations
            self.mh_step(i, data) # New sample computed by mh_step
        return self.S

    def mh_step(self, i , data):
        theta = self.S[i]
        theta_new = self.get_theta_curr(theta) # Draw new sample
        self.get_alpha(i, theta, theta_new, data) #Compute Acceptance Prob alpha
        u = npr.rand(1) # Draw sample from U([0,1])
        self.accept[i] = u < self.alpha[i]
        if self.accept[i]: # Accept step if u < alpha
            self.S[i+1] = theta_new # Theta_new is new sample
        else:
            self.S[i+1] = theta
            self.lkhd[i+1] = self.lkhd[i]

    def get_theta_curr(self, theta):
        return theta + self.stepsize*npr.randn(2)

    def get_alpha(self, i, theta, theta_new, data):
        self.get_log_lkhd(i+1, theta_new, data)
        alpha = np.exp(self.lkhd[i+1] - self.lkhd[i])
        self.alpha[i] = alpha

    def get_log_lkhd(self, i, theta, data):
        self.lkhd[i] =  np.sum(self.log_lkhd.comp_func(theta, data))

