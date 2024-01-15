from MetropolisHastings import MetropolisHastings

import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

class ConsensusMH(MetropolisHastings):
    def __init__(self, dataset, num_batches):
        super().__init__(dataset)
        self.num_batches = num_batches

    def run(self, T, theta):
        S = np.zeros((T, self.num_batches, theta.size))
        S[0,:] = np.repeat(theta,self.num_batches, axis=0)
        batches_data = self.create_batches()
        args = [(T, theta, batch) for batch in batches_data]
        with mp.Pool(self.num_batches) as pool:            
            batch_sample_list = pool.starmap(self.run_batch, args)
        
        return self.combine_batches(batch_sample_list)

    def run_batch(self, T, theta, batch):
        S = np.zeros((T, theta.size))
        S[0,:] = theta
        for i in range(T-1):
            S[i+1,:] = self.mh_step(S[i,:], batch)
        return S

    def create_batches(self):
        batch_size = self.N // self.num_batches  # Calculate batch size

        # Create shuffled indices
        indices = npr.permutation(self.N)

        # Split shuffled indices into batches
        batches = [indices[i*batch_size:(i+1)*batch_size] for i in range(self.num_batches)]

        # Extract batches from input_tensor using shuffled indices
        batches_data = [self.dataset[batch] for batch in batches]
        return batches_data

    def combine_batches(self, batch_sample_list):
        stacked_tensor = np.stack(batch_sample_list, axis=0)
        # Compute the mean along the specified dimension (0 in this case)
        average_tensor = np.mean(stacked_tensor, axis=0)
        return average_tensor