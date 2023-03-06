import numpy as np
import cupy as cp

add_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void my_add(const float* x1, const float* x2, float* y) 
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        y[tid] = x1[tid] + x2[tid];
    }''', 'my_add')

x = cp.arange(25, dtype=cp.float32).reshape(5, 5)
y = cp.arange(25, dtype=cp.float32).reshape(5, 5)
z = cp.zeros((5, 5), dtype=cp.float32)

# When calling a raw kernel ypu have to specify  
# how threads are grouped (grids and blocks)
# https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/

add_kernel((5,), (5,), (x, y, z))
print(z)

