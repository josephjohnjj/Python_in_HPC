import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

sum  = 0

def thread_kernel(thread_index):
    global sum # modify the scope of the variable
    logging.info("I am thread %s", thread_index)
    time.sleep(2)
    sum  =  sum + thread_index
    logging.info("I am thread %s, and I am done", thread_index)
 
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")  

thread_index = [1, 2, 3]

# create an instance of ThreadPoolExecutor
pool = ThreadPoolExecutor(max_workers=8)

# launch the threads
futures = [pool.submit(thread_kernel, i) for i in range(3)] # This is non-blocking

wait(futures) # wait for all results (comment this line to see what happens without wait)

print("Sum = "+ str(sum))

pool.shutdown()