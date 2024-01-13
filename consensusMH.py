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
        self.dataset = dataset
        self.N = dataset.size(0)
        self.num_batches = num_batches

    def run(self, T, theta):
        if __name__ == '__main__':
            S = torch.zeros(T, self.num_batches, theta.size(0))
            S[0,:] = theta.repat(self.num_batches, 1)
            batches_data = self.create_batches()
            args = [(T, theta, batch) for batch in batches_data]
            with multiprocessing.Pool(num_processes) as pool:
                
                results = pool.map(self.run_batch, args)


            #     for i in range(T-1):
            #         S[i+1,:] = self.mh_step(S[i,:], data)
            # return S

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

x = torch.randn(1000)
cons = ConsensusMH(dataset = x, num_batches=4)

start_time = time.time()
S = cons.run(10000, map)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
sns.jointplot(x=S[:,0],y=S[:,1])