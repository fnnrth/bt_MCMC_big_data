# import multiprocessing
# import time
# import torch

# def square(x):
#     time.sleep(5)
#     return x

# def worker_function(number):
#     result = square(number)
#     return result

# if __name__ == "__main__":
#     # Number of processes to spawn
#     num_processes = 4

#     # Values to process
#     values_to_process = [torch.tensor([1]) for i in range(4)]

#     # Create a multiprocessing pool
#     with multiprocessing.Pool(num_processes) as pool:
#         # Use the pool.map function to apply the worker function to each value
#         start_time = time.time()
#         results = pool.map(worker_function, values_to_process)
#         end_time = time.time()
#         execution_time = end_time - start_time
#         print(f"Execution Time: {execution_time}")
#     # Print the results
#     print("Original Values:", values_to_process)
#     print("Squared Values:", results)

import torch
import torch.multiprocessing as mp
import time

# Set device to CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def square(x):
    # Simulate a time-consuming task on GPU
    time.sleep(5)
    x_tensor = torch.tensor(x, device=device)
    squared_result = x_tensor * x_tensor
    # Synchronize to ensure timing accuracy
    torch.cuda.synchronize()
    return squared_result.cpu().numpy().item()

def worker_function(number):
    result = square(number)
    return result

if __name__ == "__main__":
    # Record the start time
    start_time = time.time()

    # Number of processes to spawn
    num_processes = 4

    # Values to process
    values_to_process = [1, 2, 3, 4, 5]

    # Use torch.multiprocessing.Pool for CUDA-aware multiprocessing
    with mp.Pool(num_processes) as pool:
        # Use the pool.map function to apply the worker function to each value
        results = pool.map(worker_function, values_to_process)

    # Record the end time
    end_time = time.time()

    # Print the results
    print("Original Values:", values_to_process)
    print("Squared Values:", results)

    # Print the total execution time
    execution_time = end_time - start_time
    print(f"Total Execution Time: {execution_time} seconds")