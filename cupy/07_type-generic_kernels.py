import numpy as np
import cupy as cp

# 4 parts :
#   1. an input argument list
#   2. an output argument list
#   3. loop body
#   4. kernel name

#type is inferred from the arguments passed
element_diff = cp.ElementwiseKernel('T x, T y', 'T z', 'z = (x - y)', 'element_diff')

x = cp.array([1, 2, 3], dtype=np.float32)
y = cp.array([4, 5, 6], dtype=np.float32)

z = element_diff(x, y)
print(z)

x = cp.array([1, 2, 3], dtype=np.int32)
y = cp.array([4, 5, 6], dtype=np.int32)

z = element_diff(x, y)
print(z)
