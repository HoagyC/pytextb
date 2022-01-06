import logging
import time

def enumerateWithEstimate(
    iter,
    desc_str,
    start_ndx=0,
    print_ndx=4,
    backoff=None,
    iter_len=None,
):
    if iter_len is None:
        iter_len = len(iter)
    
    for (current_ndx, item) in enumerate(iter):
        yield (current_ndx, item)
        
        if current_ndx > start_ndx:
            total_to_do = iter_len - start_ndx
            
        elif current_ndx == start_ndx:
            start_time = time.time()
