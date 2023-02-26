import numpy as np
import cupy as cp

x_cpu = np.array([1, 2, 3])
y_cpu = np.array([4, 5, 6])

z_cpu = x_cpu + y_cpu

x_gpu = cp.asarray(x_cpu) #copy host data to GPU mem

# x_gpu is located in device memory
# y_cpu is located in host memory
# z_cpu is located in host memory
z_cpu = x_gpu + y_cpu
