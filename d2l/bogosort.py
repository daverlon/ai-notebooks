import numpy as np
from tqdm import tqdm

def bogosort(data):
    iteration = 0
    
    while True:
        iteration += 1
        
        if np.all(x[:-1] <= x[1:]): return
        
        np.random.shuffle(data)
        
        print(iteration, data)
        
x = np.random.randint([10]*50)
bogosort(x)