from MetropolisHastings import MetropolisHastings

import torch
import torch.multiprocessing as mp
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

class ConsensusMH(MetropolisHastings):
    def __init__(self, dataset, num_batches):
        super().__init__(dataset)
        self.num_batches = num_batches

    def run(self, T, theta):
        S = torch.zeros(T, self.num_batches, theta.size(0))
        S[0,:] = theta.repeat(self.num_batches, 1)
        batches_data = self.create_batches()
        args = [(T, theta, batch) for batch in batches_data]
        with mp.Pool(self.num_batches) as pool:            
            batch_sample_list = pool.starmap(self.run_batch, args)
        
        return self.combine_batches(batch_sample_list)

    def run_batch(self, T, theta, batch):
        S = torch.zeros(T, theta.size(0))
        S[0,:] = theta
        for i in range(T-1):
            S[i+1,:] = self.mh_step(S[i,:], batch)
        return S

    def create_batches(self):
        batch_size = self.N // self.num_batches  # Calculate batch size

        # Create shuffled indices
        indices = torch.randperm(self.N)

        # Split shuffled indices into batches
        batches = [indices[i*batch_size:(i+1)*batch_size] for i in range(self.num_batches)]

        # Extract batches from input_tensor using shuffled indices
        batches_data = [self.dataset[batch] for batch in batches]
        return batches_data

    def combine_batches(self, batch_sample_list):
        stacked_tensor = torch.stack(batch_sample_list, dim=0)
        # Compute the mean along the specified dimension (0 in this case)
        average_tensor = torch.mean(stacked_tensor, dim=0)
        return average_tensor