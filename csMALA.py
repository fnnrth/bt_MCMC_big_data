import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import time

from MetropolisHastings import MetropolisHastings

class csMALA(MetropolisHastings):
    '''
    Class implementing the corrected stochastic MALA Algorithm
    '''
    def __init__(self, dataset, batch_percentage):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_percentage = batch_percentage
        self.inv_temp = self.N*(2-self.batch_percentage)
        self.learn_rate = 0.5/np.sqrt(self.N)
        self.std = 0.1
        self.corr_param = 0.000001
        
    def run(self, T, theta):
        '''
        Run the algorithm
        Args:
            - T (int): number of iterations
            - theta (np.array): starting point for algorithm

        Returns:
            (np.array) Array with sample for bayesian inference
        '''
        S = np.zeros((T, theta.size)) # Initialize empty Sample set
        S[0,:] = theta # Set first sample to starting point (theta)
        r = self.get_r(theta, self.take_subset()) # Compute R values for starting point (theta)
        for i in range(T-1): #T Iterations
            batch_data = self.take_subset() # Subset data
            step = self.csMALA_step(S[i,:], r, batch_data) # Do one step of csMALA 
            S[i+1,:] = step[0] # Save new sample in sample set
            r = step[1] # Save new R values
        return S

    def csMALA_step(self, theta, r, data):
        '''
        Run a single step of the algorithm
        Args:
            - theta (np.array): old theta to compute new theta
            - r (list of np.array): values for r
            - data (np.array): dataset
        '''
        theta_new = self.get_theta_new(theta, r) # Sample new theta
        r_new = self.get_r(theta_new, data) # Compute r values for new theta 
        log_alpha = self.get_log_alpha(theta, r, theta_new, r_new) # Compute Acceptance Ratio
        log_u = np.log(npr.rand(1))/ data.size # Draw sample from from U([0,1])
        if log_u < log_alpha:
            # Set new theta and r values
            theta = theta_new
            r = r_new
        return [theta, r]

    def take_subset(self):
        '''
        Subset dataset
        '''
        subset_indx = npr.binomial(n=1,p=self.batch_percentage, size=self.N)
        return self.dataset[subset_indx == 1]

    def get_theta_new(self, theta, r):
        '''
        Compute new sample for theta
        '''
        theta_new = npr.normal(loc=theta - self.learn_rate*r[1], scale = self.std)
        if theta_new[1] < 0: #Filter cases where sig < 0
            theta_new = theta 
        return theta_new

    def get_log_alpha(self, theta, r, theta_new, r_new):
        '''
        Compute Acceptance Ratio
        '''
        r_diff = self.inv_temp*(r[0] - r_new[0])
        old_diff = npl.norm(theta_new - theta + self.learn_rate*r[1]) 
        new_diff = npl.norm(theta - theta_new + self.learn_rate*r_new[1])
        alpha = np.exp(r_diff + (new_diff - old_diff)/(2*self.std**2))
        #print(f"alpha: {alpha}")
        return alpha

    def get_r(self, theta, data):
        '''
        Compute R values
        '''
        correction_term = data.size*self.corr_param * np.log(self.batch_percentage)/self.inv_temp
        r = self.get_log_lkhd(theta, data) + correction_term
        r_delta = self.get_r_delta(theta, data)
        #print(f"r: {[r, r_delta]}")
        return [r, r_delta] 

    def get_r_delta(self, theta, data):
        '''
        Compute delta_r
        Args: 
            theta (np.array): theta bases on which to compute r
            data (np.array): dataset based on which to compute r
        Notes:
            r_delta_mu = mean(data-mu)/(sig**2))
            r_delta_sig = mean(-1/sig + ((data - mu)**2)/sig**3)
        '''
        r_delta_mu = np.mean(data - theta[0])/(theta[1]**2)
        r_delta_sig = -1/theta[1] + np.mean((data - theta[0])**2)/theta[1]**3
        return np.array([r_delta_mu, r_delta_sig])

x = npr.randn(100000)
theta = np.array([0.1,1.1])
test = csMALA(x, 0.5)
test_run = test.run(10000, theta)
print(test_run)

# Print summary
print("Summary of the numpy array:")
print(f"Minimum: {np.min(test_run[:,0])}")
print(f"Maximum: {np.max(test_run[:,0])}")
print(f"Mean: {np.mean(test_run[:,0])}")
print(f"Standard Deviation: {np.std(test_run[:,0])}")
print(f"Quantiles (25th, 50th, 75th percentile): {np.quantile(test_run[:,0], [0.25, 0.5, 0.75])}")