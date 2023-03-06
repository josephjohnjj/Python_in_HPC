import numpy as np
import cupy as cp

a_cp = cp.arange(10)
b_cp = cp.arange(10)

e1 = cp.cuda.Event() # create an event
e1.record() # Records an event on the stream.
a_cp = b_cp * a_cp + 8
e2 = cp.cuda.get_current_stream().record() # create and record the event

s = cp.cuda.Stream.null
# make sure the stream wait for the second event
s.wait_event(e2)

with s:
   # as the stream is waiting for the second event to complete
   # we can be assured that all the operations ebfror it also has been complete.
   a_np = cp.asnumpy(a_cp)

# Waits for the stream that track an event to complete that event
e2.synchronize()
t = cp.cuda.get_elapsed_time(e1, e2)

print(t)

# https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
