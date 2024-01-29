from MetropolisHastings import MetropolisHastings

import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as npr
import time

class ConsensusMH(MetropolisHastings):
    '''
    Class implementing Consensus Metropolis Hastings Algorithm
    '''
    def __init__(self, dataset, num_batches):
        super().__init__(dataset)
        self.num_batches = num_batches

    def run(self, T, theta):
        '''
        Run the algorithm
        Args:
            - T (int): number of iterations
            - theta (np.array): starting point for algorithm

        Returns:
            (np.array) Array with sample for bayesian inference
        Notes: 
            Algorithm runs MH in parallel on batches of the dataset and then combines the returned samples
        '''
        batches_data = self.create_batches() # Create list of batches of data
        args = [(T, theta, batch) for batch in batches_data] # List of arguments to run MH in parallel
        with mp.Pool(self.num_batches) as pool: # Parallel enviorenment        
            batch_sample_list = pool.starmap(super.run(), args) # Run MH in parallel with all batches
        
        return self.combine_batches(batch_sample_list)  # Combine samples returned by batches

    def create_batches(self):
        '''
        Create batches out of the dataset
        '''
        batch_size = self.N // self.num_batches  # Calculate batch size
        indices = npr.permutation(self.N) # Create shuffled indices
        batches = [indices[i*batch_size:(i+1)*batch_size] for i in range(self.num_batches)] # Split shuffled indices into batches
        batches_data = [self.dataset[batch] for batch in batches] # Extract batches from batches using shuffled indices
        return batches_data

    def combine_batches(self, batch_sample_list):
        '''
        '''
        stacked_array = np.stack(batch_sample_list, axis=0) # Compute the mean along the specified dimension (0 in this case)
        average_array = np.mean(stacked_tensor, axis=0)
        return average_tensor