import numpy as np
import cupy as cp

a_np = np.arange(10)
s = cp.cuda.Stream()

with s:
   a_cp = cp.asarray(a_np)  # H2D transfer on stream s
   b_cp = cp.sum(a_cp)      # kernel launched on stream s 

# or we can use 'use()'
# if we use 'use() any subsequent CUDA operation will completed
# using the stream we specify, until we make a change 
s.use()

b_np = cp.asnumpy(b_cp)

assert s == cp.cuda.get_current_stream() # run fails is assert condition is false

# go back to the default stream
cp.cuda.Stream.null.use()

assert cp.cuda.Stream.null == cp.cuda.get_current_stream()
#assert s == cp.cuda.get_current_stream()