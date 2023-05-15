import numpy as np
import cupy as cp

# 4 parts :
#   1. an input argument list
#   2. an output argument list
#   3. loop body
#   4. kernel name

element_diff = cp.ElementwiseKernel('float32 x, float32 y', 'float32 z', 'z = (x - y)', 'element_diff')

x = cp.array([1, 2, 3], dtype=np.float32)
y = cp.array([4, 5, 6], dtype=np.float32)

z = element_diff(x, y)
print(z)

z = cp.empty((1, 3), dtype=np.float32)
element_diff(x, y, z)
print(z)
