import logging
import logging.handlers
import math
import time

root_logger = logging.getLogger(__name__)
root_logger.setLevel(logging.DEBUG)

logfmt_str = "%(asctime)s %(levelname)-8s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)s %(message)s"
formatter = logging.Formatter(logfmt_str)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("logger.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

root_logger.addHandler(stream_handler)
root_logger.addHandler(file_handler)


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

        elif current_ndx > start_ndx and (
            current_ndx in pow_list or current_ndx == iter_len - 1
        ):
            current_time = time.time()
            elapsed_time = current_time - start_time
            time_per_iter = elapsed_time / (current_ndx - start_ndx)
            remaining_iters = iter_len - current_ndx
            remaining_time = time_per_iter * remaining_iters
            finish_time = current_time + remaining_time

            finish_str = time.strftime("%d %b %Y %H:%M:%S", time.gmtime(finish_time))
            current_str = time.strftime("%d %b %Y %H:%M:%S", time.gmtime(current_time))
            print(
                "{}. Completed {}/{} at {}. Expected finish at {}".format(
                    desc_str, current_ndx, iter_len, current_str, finish_str
                )
            )
