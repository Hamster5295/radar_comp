import os
from math import ceil

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

import lee

PI = 3.14159

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("使用 {}".format(device))


def lim(pic):
    return pic
    min, max = 0.5, 99.5
    nmin = np.percentile(pic, min)
    nmax = np.percentile(pic, max)
    pic = np.clip(pic, nmin, nmax)
    return pic


def ifft(pic):
    pic = np.fft.ifft2(pic)
    pic = np.fft.fftshift(pic)
    pic = np.abs(pic).astype('float32')
    return pic


def split_pics(ori, target_shape=None):
    shape = (ori.shape[0] // 2, ori.shape[1] // 2)
    arr = np.array([])
    for i in range(2):
        sli = ori[i * shape[0]:(i + 1) * shape[0], :]
        sli = ifft(sli)

        if target_shape is not None:
            dshape = (target_shape[0] - sli.shape[0], target_shape[1] - sli.shape[1])
            sli = np.pad(sli, ((dshape[0] // 2, ceil(dshape[0] / 2)), (dshape[1] // 2, ceil(dshape[1] / 2))))
        elif i == 0:
            target_shape = sli.shape

        min, max = 0.5, 99.5
        nmin = np.percentile(sli, min)
        nmax = np.percentile(sli, max)
        sli = np.clip(sli, nmin, nmax)
        sli = lim(sli)
        sli = sli[np.newaxis]
        arr = np.vstack((arr, sli)) if i > 0 else sli
    return arr


def preprocess(path, include_angle, pic_size, split, window):
    mat = loadmat(path[1])

    ag, eg = mat['frame_AzimuthDegree'], mat['frame_ElevationDegree']
    ag = PI * ag / 180
    eg = PI * eg / 180

    eh = mat['frame_Eh']
    ev = mat['frame_Ev']

    if split:
        if include_angle:
            eh = np.vstack((ag, eh, eg))
        eh = lee.apply_window(eh) if window else eh
        eh = split_pics(eh, pic_size)

        if include_angle:
            ev = np.vstack((ag, ev, eg))
        ev = lee.apply_window(ev) if window else ev
        ev = split_pics(ev, pic_size)

    else:
        if include_angle:
            eh = np.vstack((ag, eh, eg))
        eh = lee.apply_window(eh) if window else eh
        #     eh = split_pics(eh, pic_size)
        eh = ifft(eh)
        eh = lim(eh)
        dshape = (pic_size[0] - eh.shape[0], pic_size[1] - eh.shape[1])
        eh = np.pad(eh, ((dshape[0] // 2, ceil(dshape[0] / 2)), (dshape[1] // 2, ceil(dshape[1] / 2))))
        eh = eh[np.newaxis]

        if include_angle:
            ev = np.vstack((ag, ev, eg))
        ev = lee.apply_window(ev) if window else ev
        ev = ifft(ev)
        ev = lim(ev)
        dshape = (pic_size[0] - ev.shape[0], pic_size[1] - ev.shape[1])
        ev = np.pad(ev, ((dshape[0] // 2, ceil(dshape[0] / 2)), (dshape[1] // 2, ceil(dshape[1] / 2))))
        ev = ev[np.newaxis]

    return np.concatenate((eh, ev)), ag, eg


class TrainRadarData(Dataset):
    include_angle = False

    def __init__(self, path, split=True, apply_window=True, normalize=True):
        self.pic_size = (512, 512)
        self.preprocess = split
        self.apply_window = apply_window
        self.normalize = normalize
        self.paths = []
        for path, folders, _ in os.walk(path):
            for cls in folders:
                idx = int(cls) - 1

                cls_path = path + "/" + cls
                for _, _, files in os.walk(cls_path):
                    # 相应类别的各个数据
                    for f in files:
                        # self.paths.append((idx, cls_path + '/' + f))
                        self.paths.append((idx, cls_path + '/' + f, f[:-4].split('_')[-1]))

    def __getitem__(self, item):
        path = self.paths[item]
        uni, ag, eg = preprocess(path, self.include_angle, self.pic_size, self.preprocess, self.apply_window)
        arr = np.zeros(10)
        arr[path[0]] = 1
        return arr, uni, ag, eg

    def __len__(self):
        return len(self.paths)


# 用于验证的数据集
class ValidateRadarData(Dataset):
    include_angle = False

    def __init__(self, path, split=True, apply_window=True):
        if os.path.isabs(path):
            path = os.path.relpath(path).replace('\\', '/')

        self.pic_size = (512, 512)
        self.preprocess = split
        self.apply_window = apply_window
        self.paths = []
        for path, _, files in os.walk(path):
            for f in files:
                self.paths.append((f[:-4], path + "/" + f))

    def __getitem__(self, item):
        path = self.paths[item]
        uni, ag, eg = preprocess(path, self.include_angle, self.pic_size, self.preprocess, self.apply_window)
        return path[0], uni, ag, eg

    def __len__(self):
        return len(self.paths)
