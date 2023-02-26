import numpy as np
import cupy as cp

# nvidia-smi will give the CUDA GPUs in a node from command line.
ngpus = cp.cuda.runtime.getDeviceCount()
print('#GPU = ' + str(ngpus))

for g in range(ngpus):
    print(cp.cuda.runtime.getDeviceProperties(g))

print('Current Device = ' + str(cp.cuda.runtime.getDevice()))

