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


def split_pics(ori, target_shape=None):
    shape = (ori.shape[0] // 2, ori.shape[1] // 2)
    arr = np.array([])
    for i in range(2):
        sli = ori[i * shape[0]:(i + 1) * shape[0], :]
        sli = np.fft.ifft2(sli)
        sli = np.fft.fftshift(sli)
        sli = np.abs(sli).astype('float32')

        if target_shape is not None:
            dshape = (target_shape[0] - sli.shape[0], target_shape[1] - sli.shape[1])
            sli = np.pad(sli, ((dshape[0] // 2, ceil(dshape[0] / 2)), (dshape[1] // 2, ceil(dshape[1] / 2))))
        elif i == 0:
            target_shape = sli.shape

        min, max = 0.5, 99.5
        nmin = np.percentile(sli, min)
        nmax = np.percentile(sli, max)
        sli = np.clip(sli, nmin, nmax)
        sli = sli[np.newaxis]
        arr = np.vstack((arr, sli)) if i > 0 else sli
    return arr


def preprocess(path, include_angle, pic_size, pre, window):
    mat = loadmat(path[1])

    ag, eg = mat['frame_AzimuthDegree'], mat['frame_ElevationDegree']
    ag = PI * ag / 180
    eg = PI * eg / 180

    eh = mat['frame_Eh']
    ev = mat['frame_Ev']

    if pre:
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
        eh = np.fft.ifft2(eh)
        eh = np.fft.fftshift(eh)
        eh = np.abs(eh).astype('float32')
        dshape = (pic_size[0] - eh.shape[0], pic_size[1] - eh.shape[1])
        eh = np.pad(eh, ((dshape[0] // 2, ceil(dshape[0] / 2)), (dshape[1] // 2, ceil(dshape[1] / 2))))
        eh = eh[np.newaxis]

        if include_angle:
            ev = np.vstack((ag, ev, eg))
        ev = lee.apply_window(ev) if window else ev
        ev = np.fft.ifft2(ev)
        ev = np.fft.fftshift(ev)
        ev = np.abs(ev).astype('float32')
        dshape = (pic_size[0] - ev.shape[0], pic_size[1] - ev.shape[1])
        ev = np.pad(ev, ((dshape[0] // 2, ceil(dshape[0] / 2)), (dshape[1] // 2, ceil(dshape[1] / 2))))
        ev = ev[np.newaxis]

    return np.concatenate((eh, ev)), ag, eg


class TrainRadarData(Dataset):
    include_angle = False

    def __init__(self, path, preprocess=True, apply_window=True):
        self.pic_size = (512, 512)
        self.preprocess = preprocess
        self.apply_window = apply_window
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

        # if self.save:
        #     p = f"FILTER/{path[0] + 1}"
        #     if not os.path.exists(p):
        #         os.makedirs(p)
        #     ev[0].dump(p + f"/ev_{path[2]}.arr")
        #     eh[0].dump(p + f"/eh_{path[2]}.arr")

        arr = np.zeros(10)
        arr[path[0]] = 1

        return arr, uni, ag, eg

    def __len__(self):
        return len(self.paths)

    # def save_filtered_data(self):
    #     self.save = True
    #     for i in range(len(self)):
    #         self[i]
    #     self.save = False


# 用于验证的数据集
class ValidateRadarData(Dataset):
    include_angle = False

    def __init__(self, path, preprocess=True, apply_window=True):
        if os.path.isabs(path):
            path = os.path.relpath(path).replace('\\', '/')

        self.pic_size = (512, 512)
        self.preprocess = preprocess
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
