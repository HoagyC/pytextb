import argparse
import logging
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
        if sys_argv is None: # Option to replace command line arguments with python args
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
            default=8,
            type=int,
        )

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        # Sorting by series_uid to make sure that we have each Ct in the cache when we're getting different subsections of it
        self.prep_dl = DataLoader(
            LunaDataset(
                sortby_str="series_uid",
            ),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        batch_iter = enumerateWithEstimate(
            self.prep_dl,
            "Stuffing cache",
            start_ndx=self.prep_dl.num_workers,
        )
        for _ in batch_iter:
            pass


if __name__ == "__main__":
    LunaPrepCacheApp().main()
