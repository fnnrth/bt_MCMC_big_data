import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import time
from likelihood_functions import Norm_lkhd

from MetropolisHastings import MetropolisHastings

class csMALA(MetropolisHastings):
    '''
    Class implementing the corrected stochastic MALA Algorithm
    '''
    def __init__(self, dataset,Likelihood_function, batch_percentage):
        super().__init__(dataset, Likelihood_function)
        self.log_lkhd = Likelihood_function
        self.N = dataset.size
        self.dataset = dataset
        #Hyperparameters
        self.batch_percentage = batch_percentage
        self.inv_temp = 80
        self.learn_rate = 0.01
        self.std = 0.1
        self.corr_param = 0.001
        # Running Information
        self.batch_curr = None
        self.R_curr = None
        self.R_curr_delta = None
        self.theta_curr = None
        self.lkhd = None
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

        Returns:
            (np.array) Array with sample for bayesian inference
        '''
        self.alpha = np.zeros(T)
        self.u = np.zeros(T)
        self.S = np.zeros((T, theta.size)) # Initialize empty Sample set
        self.S[0] = theta # Set first sample to starting point (theta)
        self.theta_curr = theta
        self.lkhd = np.zeros((T))
        self.R = np.zeros((T))
        self.R_delta = np.zeros((T,theta.size))

        self.subset_batch_curr()
        
        self.set_R_curr(0)
        self.R[0] = self.R_curr

        self.set_R_curr_delta(0)
        self.R_delta[0] = self.R_curr_delta  
        for i in range(1,T): #T Iterations
            self.step(i) # Do one step of csMALA 

    def step(self, i):
        '''
        Run a single step of the algorithm
        Args:

        '''
        self.subset_batch_curr()
        self.set_theta_curr(i) # Sample new theta
        self.set_R_curr(i) # Compute r values for new theta 
        self.set_R_curr_delta(i)
        self.set_alpha(i) # Compute Acceptance Ratio
        self.u[i] = npr.rand(1) # Draw sample from from U([0,1])
        if self.u[i] < self.alpha[i]:
            # Set new theta and r values
            self.S[i] = self.theta_curr
            self.R[i] = self.R_curr
            self.R_delta[i] = self.R_curr_delta
        else:
            self.S[i] = self.S[i-1]
            self.R[i] = self.R[i-1]
            self.R_delta[i] = self.R_delta[i-1]


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
        self.theta_curr = theta_new

    def set_alpha(self, i):
        '''
        Compute Acceptance Ratio
        '''
        theta = self.S[i-1]
        theta_new = self.theta_curr
        R = self.R[i-1]
        R_delta = self.R_delta[i-1]
        R_new = self.R_curr
        R_delta_new = self.R_curr_delta
        l_r = self.learn_rate

        r_diff = self.inv_temp*(R - R_new)
        old_diff = npl.norm(theta_new - theta + l_r*R_delta) 
        new_diff = npl.norm(theta - theta_new + l_r*R_delta_new)
        delta_diff = old_diff - new_diff
        alpha = np.exp(r_diff + (delta_diff)/(2*self.std**2))
        self.alpha[i] = alpha

    def set_log_lkhd(self,i):
        theta = self.theta_curr
        data  = self.batch_curr
        self.lkhd[i] = np.sum(self.log_lkhd.comp_func(theta, data))/data.size

    def set_R_curr(self, i):
        '''
        Compute R values
        '''
        correction_term = self.batch_curr.size*self.corr_param * np.log(self.batch_percentage)/self.inv_temp
        self.set_log_lkhd(i)
        r =  np.abs(self.lkhd[i] + correction_term)
        self.R_curr = r 

    def set_R_curr_delta(self, i):
        '''
        Compute delta_r
        Args: 
        Notes:
            r_delta_mu = mean(data-mu)/(sig**2))
            r_delta_sig = mean(-1/sig + ((data - mu)**2)/sig**3)
        '''
        theta = self.theta_curr
        data = self.batch_curr
        self.R_curr_delta = self.log_lkhd.comp_func_grad(theta, data)/data.size

    def get_summary(self):
        print("Summary of last run:")
        print(f"Mean Mu: {np.mean(self.S[:,0])}")
        print(f"Mean Sig: {np.mean(self.S[:,1])}")
        print(f"Acceptance Rate: {np.count_nonzero(self.alpha > self.u) / len(self.alpha)}")
