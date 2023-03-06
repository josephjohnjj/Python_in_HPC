import numpy as np
import cupy as cp

# 4 parts :
#   1. an input argument list
#   2. an output argument list
#   3. loop body
#   4. kernel name

# i indicates the index within the loop
# _ind.size() indicates total number of elements to apply the elementwise operation
element_sum = cp.ElementwiseKernel('T x, raw T y', 'T z', 'z = x + y[_ind.size() - i - 1]', 'element_sum')

x = cp.array([1, 2, 3], dtype=np.float32)
y = cp.array([4, 5, 6], dtype=np.float32)

z = element_sum(x, y)
print(z)

