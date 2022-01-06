import logging
import math
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

    max_pow = math.ceil(math.log(iter_len, 2))
    pow_list = [2 ** i for i in range(max_pow)]

    for (current_ndx, item) in enumerate(iter):
        yield (current_ndx, item)

        if current_ndx == start_ndx:
            start_time = time.time()

        elif (current_ndx in pow_list and current_ndx > start_ndx) or current_ndx == iter_len-  1:
            current_time = time.time()
            elapsed_time = current_time - start_time
            time_per_iter = elapsed_time / (current_ndx - start_ndx)
            remaining_iters = iter_len - current_ndx
            remaining_time = time_per_iter * remaining_iters
            finish_time = current_time + remaining_time

            finish_str = time.strftime(
                "%d %b %Y %H:%M:%S", time.gmtime(finish_time)
            )
            current_str = time.strftime(
                "%d %b %Y %H:%M:%S", time.gmtime(current_time)
            )
            print(
                "{}. Completed {}/{} at {}. Expected finish at {}".format(
                    desc_str, current_ndx, iter_len, current_str, finish_str
                )
            )
