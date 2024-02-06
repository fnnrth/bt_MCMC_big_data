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
        # Dataset
        self.dataset = dataset
        #Hyperparameters
        self.batch_percentage = batch_percentage
        self.inv_temp = self.N*(2-self.batch_percentage)
        self.learn_rate = 0.5/np.sqrt(self.N)
        self.std = 0.1
        self.corr_param = 0.000001
        # Running Information
        self.batch_curr = None
        self.R_curr = None
        self.R_curr_delta = None
        self.theta_curr = None
        # Output Information
        self.S = None
        self.R = None
        self.R_delta = None
        self.alpha = None  
        self.u = None

    def run(self, T, theta):
        '''
        Run the algorithm
        Args:
            - T (int): number of iterations
            - theta (np.array): starting point for algorithm

        Returns:
            (np.array) Array with sample for bayesian inference
        '''
        self.alpha = np.zeros(T-1)
        self.u = np.zeros(T-1)
        self.S = np.zeros((T, theta.size)) # Initialize empty Sample set
        self.S[0,:] = theta # Set first sample to starting point (theta)

        self.subset_batch_curr()

        self.R = np.zeros((T, theta.size))
        self.set_R_curr(0)
        self.R[0] = self.R_curr

        self.R_delta = np.zeros((T,theta.size))
        self.set_R_curr_delta(0)
        self.R_delta[0] = self.R_curr_delta  
        for i in range(1,T): #T Iterations
            step = self.csMALA_step(i) # Do one step of csMALA 

    def csMALA_step(self, i):
        '''
        Run a single step of the algorithm
        Args:
            - theta (np.array): old theta to compute new theta
            - r (list of np.array): values for r
            - data (np.array): dataset
        '''
        self.subset_batch_curr()
        self.set_theta_curr(i-1) # Sample new theta
        self.set_R_curr(i-1) # Compute r values for new theta 
        self.set_R_curr_delta(i-1)
        self.get_log_alpha(i-1) # Compute Acceptance Ratio
        self.u[i-1] = np.log(npr.rand(1))/ data.size # Draw sample from from U([0,1])
        if self.u[i] < self.alpha[i]:
            # Set new theta and r values
            self.S[i] = self.theta_curr
            self.R[i] = self.R_curr
            self.R_delta[i] = self.R_delta_curr

    def subset_batch_curr(self):
        '''
        Subset dataset
        '''
        subset_indx = npr.binomial(n=1,p=self.batch_percentage, size=self.N)
        self.batch_curr = self.dataset[subset_indx == 1]

    def set_theta_curr(self, i):
        '''
        Compute new sample for theta
        '''
        theta = self.S[i-1]
        R = self.R[i-1]
        l_r = self.learn_rate
        theta_new = npr.normal(loc=theta - l_r*R, scale = self.std)
        if theta_new[1] < 0: #Filter cases where sig < 0
            theta_new = theta 
        self.theta_curr = theta_new

    def get_log_alpha(self, i):
        '''
        Compute Acceptance Ratio
        '''
        theta = self.S[i-1]
        theta_new = self.theta_curr
        R = self.R[i-1]
        R_new = self.R_curr
        l_r = self.learn_rate

        r_diff = self.inv_temp*(R - R_new)
        old_diff = npl.norm(thet_new - theta + l_r*R) 
        new_diff = npl.norm(theta - theta_new + l_r*R_new)
        alpha = np.exp(r_diff + (new_diff - old_diff)/(2*self.std**2))
        self.alpha[i-1] = alpha

    def get_log_lkhd(self,i):
        theta = self.S[i]
        data  = self.batch_curr

        mean_diff = np.mean((data - theta[0])**2)
        return -((mean_diff)/theta[1]**2)/2 - np.log(theta[1]*np.sqrt(np.pi*2))

    def set_R_curr(self, i):
        '''
        Compute R values
        '''
        correction_term = self.batch_curr.size*self.corr_param * np.log(self.batch_percentage)/self.inv_temp
        r = self.get_log_lkhd(i) + correction_term
        self.R_curr = r 

    def set_R_curr_delta(self, i):
        '''
        Compute delta_r
        Args: 
            theta (np.array): theta bases on which to compute r
            data (np.array): dataset based on which to compute r
        Notes:
            r_delta_mu = mean(data-mu)/(sig**2))
            r_delta_sig = mean(-1/sig + ((data - mu)**2)/sig**3)
        '''
        theta = self.S[i-1]
        data = self.batch_curr

        r_delta_mu = np.mean(data - theta[0])/(theta[1]**2)
        r_delta_sig = -1/theta[1] + np.mean((data - theta[0])**2)/theta[1]**3
        self.R_curr_delta = np.array([r_delta_mu, r_delta_sig])

    def get_summary(self):
        print("Summary of last run:")
        print(f"Minimum S: {np.min(self.S[:,0])}")
        print(f"Maximum S: {np.max(self.S[:,0])}")
        print(f"Mean S: {np.mean(self.S[:,0])}")
        print(f"Standard Deviation S: {np.std(self.S[:,0])}")
        print(f"Quantiles (25th, 50th, 75th percentile S): {np.quantile(self.S[:,0], [0.25, 0.5, 0.75])}")


x = npr.randn(100000)
theta = np.array([0.1,1.1])
test = csMALA(x, 0.5)
test_run = test.run(1000, theta)
test_data = test_run[0]