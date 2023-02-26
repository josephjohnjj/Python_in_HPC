import numpy as np
import cupy as cp

x_cpu = np.array([1, 2, 3])

x_gpu_0 = cp.asarray(x_cpu)  # move the ndarray from host mem to GPU0 memeory.

with cp.cuda.Device(1):
    x_gpu_1 = cp.asarray(x_gpu_0)  # move the ndarray to GPU0 to GPU1.

# In GPUs D2D transfers are much faster than D2D transfers.
with cp.cuda.Device(1):
    x_cpu = cp.asnumpy(x_gpu_1)  # move the array back to the host.

with cp.cuda.Device(1):
    x_cpu = x_gpu_1.get()
