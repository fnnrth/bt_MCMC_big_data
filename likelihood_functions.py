import numpy as np
class Likelihood_function():
    def comp_func(self):
        pass

    def comp_func_grad(self):
        pass

    def comp_func_hess(self):
        pass


class Norm_lkhd(Likelihood_function):
    def comp_func(self, theta, data):
        mu = theta[0]
        sigma = theta[1]
        lkhd =  -((mu - data)**2)/(2*sigma**2) - np.log(sigma)
        return lkhd
         
    def comp_gradient(self, theta, data):
        mu = theta[0]
        sig = theta[1]
        gradient_mu = -mu - data/(sig**2)
        gradient_sig = -1/sig + ((data - mu)**2)/(sig**3) 
        return np.array([gradient_mu, gradient_sig])

    def comp_hessian(self, theta, data):
        mu = theta[0]
        sig = theta[1]
        num_data = data.size    
        hessian_mu_mu = -1/sig**2 * np.ones(num_data)
        hessian_sig_sig = np.array([(1 - 3*(mu - data)**2/(sig**2))/sig**2])
        hessian_mu_sig = np.array([-2*(data - mu)/sig**3])
        return np.array([[hessian_mu_mu, hessian_mu_sig],[hessian_mu_sig, hessian_sig_sig]])

class Norm_2_d_lkhd(Likelihood_function):
    def comp_func(self, theta, data):
        mu_1 = theta[0]
        mu_2 = theta[1]
        p = 1/4
        lkhd = np.exp(np.sum(-p * ((data - mu_1)**2)/(2) - (1-p) * ((data - mu_2)**2)/(2)))
        return lkhd