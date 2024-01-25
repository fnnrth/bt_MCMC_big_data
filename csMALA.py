import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import time

from MetropolisHastings import MetropolisHastings

class csMALA(MetropolisHastings):
    def __init__(self, dataset, batch_percentage):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_percentage = batch_percentage
        self.inv_temp = self.N*(2-self.batch_percentage)
        self.learn_rate = 10**(-4)/self.batch_percentage
        self.std = 0.1
        self.corr_param = 0
        
    def run(self, T, theta):
        S = np.zeros((T, theta.size))
        S[0,:] = theta
        r = self.get_r(theta, self.take_subset())
        for i in range(T-1):
            batch_data = self.take_subset()
            step = self.mh_step(S[i,:], r, batch_data)
            S[i+1,:] = step[0]
            r = step[1]
        return S

    def mh_step(self, theta, r, data):
        theta_new = self.get_theta_new(theta, r)
        r_new = self.get_r(theta_new, data)
        log_alpha = self.get_log_alpha(theta, r, theta_new, r_new)
        log_u = np.log(npr.rand(1))/ data.size
        if log_u < log_alpha:
            theta = theta_new
            r = r_new
        return [theta, r]

    def take_subset(self):
        subset_indx = npr.binomial(n=1,p=self.batch_percentage, size=self.N)
        return self.dataset[subset_indx == 1]

    def get_theta_new(self, theta, r):
        print(f"location: {theta - self.learn_rate*r[1]}")
        print(f"theta_new:{npr.normal(loc=theta - self.learn_rate*r[1], scale = self.std)}")
        return npr.normal(loc=theta - self.learn_rate*r[1], scale = self.std)

    def get_log_alpha(self, theta, r, theta_new, r_new):
        r_diff = self.learn_rate*(r[0] - r_new[0])
        old_diff = npl.norm(theta_new - theta + self.learn_rate*r[1]) 
        new_diff = npl.norm(theta - theta_new + self.learn_rate*r_new[1])
        alpha = np.exp(r_diff + (new_diff - old_diff)/(2*self.std**2))
        print(f"alpha: {alpha}")
        return alpha

    def get_r(self, theta, data):
        correction_term = data.size*self.corr_param * np.log(self.batch_percentage)/self.learn_rate
        print(f"corr_term: {correction_term}") 
        r = np.mean(self.get_log_lkhd(theta, data)) + correction_term
        r_delta = self.get_r_delta(theta, data)
        print(f"r: {np.array([r, r_delta])}")
        return np.array([r, r_delta]) # Not finished

    def get_r_delta(self, theta, data):
        r_delta = (data - theta[0])/(2*theta[1]**2)
        return np.mean(r_delta)

x = npr.randn(1000)
theta = np.array([0.1,1.1])
test = csMALA(x, 0.5)
test_run = test.run(100, theta)
print(test_run)