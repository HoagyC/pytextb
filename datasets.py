import functools
import math
import os
import pickle
import csv
import copy
import random

from collections import namedtuple
from glob import glob
from math import sqrt

import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk

from torch.utils.data import Dataset

from caching import getCache

CandidateInfoTuple = namedtuple(
    "CandidateInfoTuple", "isNodule_bool, diameter_mm, series_uid, center_xyz"
)

IrcTuple = namedtuple("IrcTuple", ["index", "row", "col"])
XyzTuple = namedtuple("XyzTuple", ["x", "y", "z"])


def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[
        ::-1
    ]  # we're only talking about a single vector here, so just need to flip it
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = direction_a @ (cri_a * vxSize_a) + origin_a
    return XyzTuple(*coords_xyz)


def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a).astype(int)[::-1]
    return IrcTuple(*cri_a)


# This has basically only one output givne the available data so use lru_cache(1)
@functools.lru_cache(1)
def getCandidateInfoList(data_loc="data", requireOnDisk_bool=True):
    mhd_list = glob(f"{data_loc}/subset*/*.mhd")
    # [:-4] removes the '.mhd' from the end of the filenames
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open(os.path.join(data_loc, "annotations.csv"), "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            # Here, if the series_uid doesn't already exist in the dict we set it to []
            # Otherwise we just append the new data, so each uid has a list of the associated
            # nodules

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )

    candidateInfo_list = []

    with open(os.path.join(data_loc, "candidates.csv"), "r") as f:
        reader = list(csv.reader(f))
        # print(reader[0])
        for row in reader[1:]:
            series_uid = row[0]
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                # second arg in dict.get() is the default
                annotationCenter_xyz, annotation_Diameter_mm = annotation_tup

                delta_mms = [
                    abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    for i in range(3)
                ]

                delta_mm = sqrt(sum([x ** 2 for x in delta_mms]))

                if delta_mm > annotation_Diameter_mm / 2:
                    break
                else:
                    candidateDiameter_mm = annotation_Diameter_mm
                    break

            candidateInfo_list.append(
                CandidateInfoTuple(
                    isNodule_bool, candidateDiameter_mm, series_uid, candidateCenter_xyz
                )
            )
    candidateInfo_list.sort(reverse=True)

    return candidateInfo_list


class Ct:
    def __init__(self, series_uid, data_loc="data"):
        mhd_path = glob(f"{data_loc}/subset*/{series_uid}.mhd")[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        # this call gets data from the .raw file without us having to
        # reference it directly
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        self.hu_a = ct_a
        self.series_uid = series_uid

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        # so we get the center of the candidate in IRC coords
        center_irc = xyz2irc(
            center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a
        )

        slice_list = []
        # we get the center and axis number in IRC coords
        for axis, center_val in enumerate(center_irc):
            # getting the bounding box of the candidate
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])
            slice_list.append(slice(start_ndx, end_ndx))

        # builds up the way that we're eventually going to slice the main CT
        # tuple of slices acts as is we have them comma separated

        ct_chunk = self.hu_a[tuple(slice_list)]

        if not list(ct_chunk.shape) == list(width_irc):
            n_dims = 3
            pad_arr = np.zeros((3, 2))
            centerChanges_list = [0] * 3
            for dim in range(3):
                if slice_list[dim].start < 0:
                    overlap = -slice_list[dim].start
                    pad_arr[dim, 0] = overlap
                    centerChanges_list[dim] -= round(overlap / 2)
                    slice_list[dim] = slice(0, slice_list[dim].stop)

                if slice_list[dim].stop > self.hu_a.shape[dim]:
                    overlap = slice_list[dim].stop - self.hu_a.shape[dim]
                    pad_arr[dim, 1] = overlap
                    centerChanges_list[dim] += round(overlap / 2)

            ct_chunk = self.hu_a[tuple(slice_list)]

            pad_arr = pad_arr.round().astype(np.int32)

            ct_chunk = np.pad(ct_chunk, pad_width=pad_arr)
            center_list = [
                center_irc[dim] + centerChanges_list[dim] for dim in range(n_dims)
            ]
            center_irc = IrcTuple(*center_list)

            assert list(ct_chunk.shape) == list(width_irc)

        # needs to also deal here with cases where the center and width put edges
        # outside the actual CT scan

        return ct_chunk, center_irc


# The way that we go through batches, we go through all of the candidates from a given CT scan at once
# This means there's lots of value of gets
@functools.lru_cache(1, typed=True)
def getCt(series_uid, data_loc="data"):
    return Ct(series_uid, data_loc=data_loc)


@functools.lru_cache(1)
def listdir(cache_loc):
    return os.listdir(cache_loc)
    

# These raw candidates need to be used repeatedly to train the model
# It's therefore worth caching each result on disk so we can run subsequent epochs quickly.

def getCtRawCandidate(series_uid, center_xyz, width_irc, cache_loc, data_loc="data"):
    fname = str(series_uid) + '.'.join([str(x) for x in center_xyz])
    full_fname = os.path.join(cache_loc, fname)
    if fname in listdir(cache_loc):
        ct_chunk, center_irc = pickle.load(open(full_fname, 'rb'))
    else:
        ct = getCt(series_uid, data_loc=data_loc)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
        with open(full_fname, "wb") as f:
            pickle.dump((ct_chunk, center_irc), f)
    return ct_chunk, center_irc


def getCtAugmentedCandidate(
    series_uid,
    center_xyz,
    width_irc,
    augmentation_dict,
    cache_loc,
    use_cache=True,
    data_loc="data",
):
    if use_cache:
        ct_chunk, center_irc = getCtRawCandidate(
            series_uid, center_xyz, width_irc, cache_loc, data_loc,
        )
    else:
        ct = getCt(series_uid, data_loc)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)

    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)
    # two unsqueezes here because we have a batch of one and only want one channel

    transform_t = torch.eye(
        4
    )  # Three spatial dimensions plus one extra for homogeneous coords

    for i in range(3):
        if "flip" in augmentation_dict:
            if random.random() > 0.5:
                transform_t[
                    i, i
                ] *= -1  # as the i'th coord rises, go up in the negative

        if "offset" in augmentation_dict:
            offset_float = augmentation_dict["offset"]
            random_float = (random.random() * 2) - 1
            transform_t[i, 3] = offset_float * random_float
            # this is shifting function of the last column of the affine grid
            # scale factor of it is offset_float

        if "scale" in augmentation_dict:
            scale_float = augmentation_dict["scale"]
            random_float = (random.random() * 2) - 1
            transform_t[i, i] *= 1.0 + scale_float * random_float

        if "rotate" in augmentation_dict:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            # note that we're confining the rotation to the x-y plane
            # because the different width of the voxel in the depth direction
            # so we cant just rotate in.
            # resampling would make the data blurry (though more than rotating
            # which also requires interpolation?) so only rotate in these directions for now

            rotation_t = torch.tensor(
                [
                    [c, -s, 0, 0],
                    [s, c, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )

            transform_t @= rotation_t

    affine_t = F.affine_grid(
        transform_t[:3].unsqueeze(0).to(torch.float32),
        # [:3] because you can't use the last row in an affine matrix
        ct_t.size(),
        align_corners=False,  # if true, treats the corners as the centers of the corner pixels,
        # (rather than the far corners of the corner pixels?)
    )

    # affine grid generates a flow field from a (batch of) affine matrices
    # grid sam[]
    augmented_chunk = F.grid_sample(
        ct_t,
        affine_t,
        padding_mode="border",
        align_corners=False,
    ).to("cpu")

    if "noise" in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_float = augmentation_dict["noise"]

        augmented_chunk += noise_float * noise_t

    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    def __init__(
        self,
        val_stride=0,
        isValSet_bool=None,
        series_uid=None,
        sortby_str="random",
        data_loc="data",
        balance_cats=True,
        ratio_int=0,
        cache_loc="./diskcache"
    ):
        self.data_loc = data_loc
        self.cache_loc = cache_loc
        
        super().__init__()
        self.candidateInfo_list = copy.copy(getCandidateInfoList(data_loc=data_loc))

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]

        if isValSet_bool:
            # If it's a validation set we only take every val_stride'th number
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            # Else, assuming we are making a validation set, delete every val_stride'th
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        if sortby_str == "random":
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == "series_uid":
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == "label_and_size":
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.ratio_int = ratio_int
        self.negative_list = [
            nt for nt in self.candidateInfo_list if not nt.isNodule_bool
        ]
        self.positive_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

    def __len__(self):

        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            if ndx % (self.ratio_int + 1) == 0:
                # if you do the naive way then you only get certain
                pos_ndx = (ndx // self.ratio_int + 1) % len(self.positive_list)
                candidateInfo_tup = self.candidateInfo_list[pos_ndx]
            else:
                neg_ndx = (ndx // self.ratio_int + 1) % len(self.negative_list)
                candidateInfo_tup = self.candidateInfo_list[neg_ndx]

        else:
            candidateInfo_tup = self.candidateInfo_list[ndx % len(self.candidateInfo_list)]
        
        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
            cache_loc=self.cache_loc,
            data_loc=self.data_loc,
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(
            0
        )  # Adding the channel dimension expected for Conv3d

        pos_t = torch.tensor(
            [not candidateInfo_tup.isNodule_bool, candidateInfo_tup.isNodule_bool],
            dtype=torch.long,
        )

        return (
            candidate_t,  # 1((CO10-1))??  This is the input tensor
            pos_t,  # 1((CO10-2))??   This whether it's positive
            candidateInfo_tup.series_uid,  # Series id
            torch.tensor(
                center_irc
            ),  # cander of the candidate within the overall CT scan
        )

    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.positive_list)
