import multiprocessing

def square(x):
    return x * x

def worker_function(number):
    result = square(number)
    return result

if __name__ == "__main__":
    # Number of processes to spawn
    num_processes = 4

    # Values to process
    values_to_process = [1, 2, 3, 4, 5]

    # Create a multiprocessing pool
    with multiprocessing.Pool(num_processes) as pool:
        # Use the pool.map function to apply the worker function to each value
        results = pool.map(worker_function, values_to_process)

    # Print the results
    print("Original Values:", values_to_process)
    print("Squared Values:", results)