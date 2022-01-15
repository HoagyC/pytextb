import argparse
import datetime
import math
import operator
import os
import sys
import time

from functools import reduce

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from datasets import LunaDataset
from utils import enumerateWithEstimate
from logconf import logging

METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def run(app, *argv):
    argv = list(argv)
    argv.insert(0, "--num-workers=4")
    log.info("Running: {}({!r}).main()")
    # !r calls __repr__ on the object. !s and !a call
    # __str__ and __ascii__ respectively
    # can then also use other string formatting bits

    app_cls = importstr(*app.rsplit(".", 1))
    app_cls(argv).main()

    log.info("Finished: {}({!r}).main()".format(app, argv))


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            conv_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(2, 2)

        # Note that these stacked layers have an effective field of 5x5x5
        # Because each input to the second layer is influenced by a 3x3x3 box around it

    def forward(self, input_batch):
        batch_out = self.conv1(input_batch)
        batch_out = self.relu1(batch_out)
        batch_out = self.conv2(batch_out)
        batch_out = self.relu2(batch_out)

        return self.maxpool(batch_out)


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        # tail is the beginning of the pipeline
        # hence adding different heads to eg. GPT
        self.tail_batchnorm = nn.BatchNorm3d(1)

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        in_dims = [32, 48, 48]  # z, x, y
        out_dims = [a / 16 for a in in_dims]
        linear_outputs = reduce(operator.mul, out_dims, 1) * conv_channels * 8
        self.head_linear = nn.Linear(int(linear_outputs), 2)
        self.head_softmax = nn.Softmax(dim=1)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
            }:
                nn.init.kaiming_normal_(m.weight.data, a=0, nonlinearity="relu")
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
                        m.weight.data
                    )
                    # I think the idea is that you want the total inward bias
                    # to be about 1? or at least smaller when there are lots of inputs
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, batch_input):
        bn_output = self.tail_batchnorm(batch_input)
        block_output = self.block1(bn_output)
        block_output = self.block2(block_output)
        block_output = self.block3(block_output)
        block_output = self.block4(block_output)

        conv_flat = block_output.view(
            block_output.size(0),
            -1,
        )

        linear_output = self.head_linear(conv_flat)
        return linear_output, self.head_softmax(linear_output)


class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        # Important to validate external input carefully,
        # hence the detailed design around parsing args (I think?)
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--num-workers",
            help="Number of workers for background data loading",
            default=2,
            type=int,
        )
        parser.add_argument(
            "--balanced",
            help="Whether to have equal numbers of negative and positive samples",
            default=False,
            type=bool,
        )
        parser.add_argument(
            "--data-loc", help="Location of the raw CT scans", default="data"
        )
        parser.add_argument(
            "--cache-loc", help="Location of the cached CT chunks", default="/media/hoagy/3666-6361/diskcache"
        )
        parser.add_argument(
            "--batch-size",
            help="Batch size to use for training",
            default=2,
            type=int,
        )
        parser.add_argument(
            "--epochs",
            help="Number of epochs of training",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--tb-prefix", default="p2ch11", help="Data prefix for Tensorboard"
        )
        parser.add_argument(
            "--comment",
            help="Comment suffix for Tensorboard run",
            nargs="?",
            default="dlwpt",
        )
        parser.add_argument(
            "--epoch-size",
            help="size of each epoch",
            
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

        self.trn_writer = self.val_writer = None
        self.totalTrainingSamples_count = 0

    def initOptimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)

    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info("Using CUDA, {} devices".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join("runs", self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + "-trn_cls-" + self.cli_args.comment
            )
            self.val_writer = SummaryWriter(
                log_dir=log_dir + "-val_cls-" + self.cli_args.comment
            )
            
    def saveModel(self, model=None, save_loc='saved_models', epochs=0):
        if not model:
            model = self.model
        
        time_format = '%Y-%M-%d-%H:%m:%S'
        time_str = time.strftime(time_format, time.gmtime(time.time()))
        file_name = f'model-{epochs}epochs-{time_str}.pt'
        with open(os.path.join(save_loc, file_name), 'wb') as f:
            torch.save(model, f)

    def initTrainDl(self):
        train_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=False,
            data_loc=self.cli_args.data_loc,
            ratio_int=int(self.cli_args.balanced),
            cache_loc=self.cli_args.cache_loc,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
            
        if not self.cli_args.balanced:
            isNod_mask = [x.isNodule_bool for x in train_ds.candidateInfo_list]
            isNodFraction_float = sum(isNod_mask) / len(isNod_mask)
            weights = [1 if x else isNodFraction_float for x in isNod_mask]
            sampler = WeightedRandomSampler(weights=weights, num_samples=self.cli_args.batch_size)

            train_dl = DataLoader(
                train_ds,
                batch_size=batch_size,
                num_workers=self.cli_args.num_workers,
                pin_memory=self.use_cuda,  # This is a bool, pinned memory transfers fast to GPU
                sampler=sampler,
            )

        else: 
            train_dl = DataLoader(
                train_ds,
                batch_size=batch_size,
                num_workers=self.cli_args.num_workers,
                pin_memory=self.use_cuda,  # This is a bool, pinned memory transfers fast to GPU
                sampler=sampler,
            )

        return train_dl

    def initValDl(self):
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True,
            data_loc=self.cli_args.data_loc,
            cache_loc=self.cli_args.cache_loc
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        # The multiple workers prepare the data in the background
        # So that the data is there immediately when batch is ready to begin
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        return val_dl

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup

        # non-blocking is true allows for some kind of asychronous work
        # with the pinned memory and the GPU
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction="none")  # gets loss per sample
        loss_g = loss_func(logits_g, label_g[:, 1])

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()

        return loss_g.mean()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        for batch_ndx, batch_tup in enumerateWithEstimate(train_dl, "Training"):
            self.optimizer.zero_grad()
            # note that lots of this complexity isn't really much complexity
            # its just that computeBatchLoss is reused between train and validation functions
            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g,
            )

            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to("cpu")

    def doValidation(self, val_dl):
        with torch.no_grad():
            self.model.eval()

            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            for batch_ndx, batch_tup in enumerateWithEstimate(val_dl, "Validation"):
                self.computeBatchLoss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    valMetrics_g,
                )

        return valMetrics_g.to("cpu")

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics_t,
        classificationThreshold=0.5,
    ):
        self.initTensorboardWriters()

        log.info("E{} {}".format(epoch_ndx, type(self).__name__))

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())
        
        precision = pos_correct / posPred_mask.sum()
        recall = pos_correct / posLabel_mask.sum()
        f1score = 2 * ((precision + recall) / (precision * recall))

        metrics_dict = {}
        metrics_dict["loss/all"] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict["loss/neg"] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict["loss/pos"] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict["correct/all"] = ((pos_correct + neg_correct) / float(metrics_t.shape[1]) * 100)
        metrics_dict["correct/neg"] = (neg_correct / max(1e-5, float(neg_count))) * 100
        metrics_dict["correct/pos"] = (pos_correct / max(1e-5, float(pos_count))) * 100
        
        metrics_dict["pr/precision"] = precision
        metrics_dict["pr/recall"] = recall
        metrics_dict["pr/f1score"] = f1score

        log.info(
            (
                "E{} {:8} {loss/all:.4f} loss, "
                + "{correct/all:-5.1f}% correct, "
                + "{pr/precision:.4f} precision, "
                + "{pr/recall:.4f} recall, "
                + "{pr/f1score:.4f} f1 score" 
            ).format(
                epoch_ndx,
                mode_str,

                **metrics_dict,
            )
        )
        log.info(
            (
                "E{} {:8} {loss/pos:.4f} loss, " + 
                "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count})"
            ).format(
                epoch_ndx,
                mode_str,
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )
        log.info(
            (
                "E{} {:8} {loss/neg:.4f} loss, " + 
                "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count})"
            ).format(
                epoch_ndx,
                mode_str,
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + "_writer")

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

    def main(self):
        train_dl = self.initTrainDl()
        val_dl = self.initValDl()
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        # ndx is short for index, didn't work it out :(
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, "trn", trnMetrics_t)

            valMetrics_t = self.doValidation(val_dl)
            self.logMetrics(epoch_ndx, "val", valMetrics_t)
            self.saveModel(epoch_ndx)


if __name__ == "__main__":
    log.info("starting up")
    LunaTrainingApp().main()
