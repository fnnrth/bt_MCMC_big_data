import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as npr
import time

class MetropolisHastings():
    '''
    Class Implementing the Metropolis Hastings Algorithm
    '''
    def __init__(self, dataset):
        self.dataset = dataset
        self.N = dataset.size # Number of datapoints

    def run(self, T, theta, data):
        '''
        Run the algorithm
        Args:
            - T (int): number of iterations
            - theta (np.array): starting point for algorithm

        Returns:
            (np.array) Array with sample for bayesian inference
        '''
        S = np.zeros((T, theta.size)) # Initialize empty sampleset
        S[0,:] = theta # First sample is starting point
        for i in range(T-1): # Iterate over number of iterations
            S[i+1,:] = self.mh_step(S[i,:], data) # New sample computed by mh_step
        return S

    def mh_step(self, theta, data):
        '''
        Determine new sample for one iteration

        Args:
            - theta (np.array): old theta 
            - data (np.arrray): dataset

        Returns:
            - (np.array) new sample for the algorithm
        Notes:
            Accepts a new sample if u < alpha, returns old sample otherwise
        '''
        theta_new = self.set_theta_curr(theta) # Draw new sample
        log_alpha = self.get_log_alpha(theta, theta_new, data) #Compute Acceptance Prob alpha
        log_u = np.log(npr.rand(1))/ data.size # Draw sample from U([0,1])
        if log_u < log_alpha: # Accept step if u < alpha
            theta = theta_new # Theta_new is new sample
        return theta

    def set_theta_curr(self, theta):
        '''
        Get new sample for algorithms
        Args:
            - theta (np.array): old sample
        
        Returns:
            (np.array) new sample for algorthm
        Notes:
         New sample is a random sample drawn from N(theta, stepsize)
        '''
        return theta + self.get_stepsize()*npr.randn(2)

    def get_log_alpha(self, theta, theta_new, data):
        '''
        Compute acceptance prob alpha
        Args:
            - theta (np.array): theta_old based on which to compute alpha
            - theta_new (np.array): theta_new based on which to compute alpha
            - data(np.array): dataset for which to compute alpha
        Returns:
            (float) number between 0 and 1 on how luckily the step is accepted
        Notes:
            log(alpha) = mean(log_likelihood(theta_new) - log_likelihood(theta))
        '''
        lkhd = self.get_log_lkhd(theta_new, data) - self.get_log_lkhd(theta, data)
        return lkhd

    def get_log_lkhd(self, theta, data):
        '''
        Compute log likelihood for normal distribution
        Args:
            theta(np.array): estimate for theta
            data(np.array): dataset for log_likelihood
        
        Returns:
            (np.array) array of log_likelihood for every datapoint
        Notes:
             l(theta) = -(((datapoint-mu)/sig)**2)/2) - log(sig*sqrt(pi*2))
        '''
        mean_diff = np.mean((data - theta[0])**2)
        return -((mean_diff)/theta[1]**2)/2 - np.log(theta[1]*np.sqrt(np.pi*2))

    def get_stepsize(self):
        '''
        Return stepsize
        Returns:
            (float): stepsize for algorithm
        Notes:
            stepsize = 0.5/sqrt(N)
        '''
        return 0.5/np.sqrt(self.N)

