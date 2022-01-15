import argparse
import logging
import os
import sys

import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader

from utils import enumerateWithEstimate
from datasets import LunaDataset
from main import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


class LunaPrepCacheApp:
    @classmethod
    def __init__(self, sys_argv=None):
        if (
            sys_argv is None
        ):  # Option to replace command line arguments with python args
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        # I think batch size can be larger because we're just loading it in, we don't need to do big matrix work?
        parser.add_argument(
            "--batch-size",
            help="Batch size to use for training",
            default=1024,
            type=int,
        )
        parser.add_argument(
            "--num-workers",
            help="Number of worker processes for background data loading",
            default=2,
            type=int,
        )
        parser.add_argument(
            "--data-loc",
            help="Location of the Ct scan data",
            default="/media/hoagy/3666-6361",
            type=str,
        )
        parser.add_argument(
            "--cache-loc",
            help="Location of cached data",
            default="/media/hoagy/3666-6361/diskcache",
            type=str,
        )

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        # Sorting by series_uid to make sure that we have each Ct in the cache
        # when we're getting different subsections of it
        self.prep_dl = DataLoader(
            LunaDataset(
                sortby_str="series_uid",
                data_loc=self.cli_args.data_loc,
                cache_loc=self.cli_args.cache_loc,
            ),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        batch_iter = enumerateWithEstimate(
            self.prep_dl,
            "Stuffing cache",
        )
        for n, _ in enumerate(batch_iter):
            if n % 10 == 0:
                keys, vals = [x.split() for x in os.popen('free -t').readlines()[:2]]
                vals = vals[1:]  # remove 'Mem:'
                mem_dict = dict(zip(keys, vals))
                percent_usage = 100 * int(mem_dict['used']) / int(mem_dict['total'])
                log.info(f"Done {n}, {percent_usage}% memory usage")

if __name__ == "__main__":
    LunaPrepCacheApp().main()
