import multiprocessing
import time
import torch

def square(x):
    time.sleep(5)
    return x

def worker_function(number):
    result = square(number)
    return result

if __name__ == "__main__":
    # Number of processes to spawn
    num_processes = 4

    # Values to process
    values_to_process = [torch.tensor([1]) for i in range(4)]

    # Create a multiprocessing pool
    with multiprocessing.Pool(num_processes) as pool:
        # Use the pool.map function to apply the worker function to each value
        start_time = time.time()
        results = pool.map(worker_function, values_to_process)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time}")
    # Print the results
    print("Original Values:", values_to_process)
    print("Squared Values:", results)