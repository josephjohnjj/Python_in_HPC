import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

def thread_kernel(thread_index):
    logging.info("I am thread %s", thread_index)
    time.sleep(2)
    logging.info("I am thread %s, and I am done", thread_index)
    str1 = "I am thread "+ str(thread_index)
    return str1
 

thread_index = [1, 2, 3]
pool = ThreadPoolExecutor(max_workers=8)

results = pool.map(thread_kernel, thread_index) # does not block

for res in results:
    print(res) # print results as they become available

pool.shutdown()