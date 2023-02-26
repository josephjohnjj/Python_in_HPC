import numpy as np
import cupy as cp

# nvidia-smi will give the CUDA GPUs in a node from command line.
ngpus = cp.cuda.runtime.getDeviceCount()
print('#GPU = ' + str(ngpus))

print('Current Device = ' + str(cp.cuda.runtime.getDevice()))

x_cpu = np.array([1, 2, 3]) # allocate an ndarray in the main memory

x_gpu = cp.array([1, 2, 3]) # alloacte an ndarray in the gpu memory of the current device (0th device)

with cp.cuda.Device(1):  # alloacte an ndarray in the gpu memory of the 1st device
    x_on_gpu1 = cp.array([1, 2, 3, 4, 5])

# CuPy functions expect that the array is on the same device as the current one.
# We can verify the location of the ndarray

print('location of x_gpu = ' + str(x_gpu.device) )
print('location of x_gpu1 = ' + str(x_on_gpu1.device) )

