import numpy as np
import cupy as cp

def log_array(x):
    xp = cp.get_array_module(x)  # cupy or numpy is returned based on the types of the arguments
                                 # If at least one of the arguments is a cupy.ndarray object, the cupy module is returned
    xp.log1p(xp.exp(-abs(x))) 

x_cpu = np.array([1, 2, 3])
x_gpu = cp.array([1, 2, 3, 4, 5])

log_array(x_cpu)
log_array(x_gpu)



