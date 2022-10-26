import concurrent.futures
import os
import numpy as np
import torch.utils.data as td
import pandas as pd
import time

class MyDataset(td.Dataset):
    def __init__(self, data_dir: str, statics_file: str, randome_dir: str, _train=True):
        self.data_dir = data_dir
        self.statics_file = statics_file
        self.randome_dir = randome_dir
        self.trainstate = _train
        randome_arry = np.load(randome_dir, allow_pickle=True)
        self.rdarry = randome_arry
        if _train:
            self.sample_count = len(randome_arry)
            print('process is train, nums is ', len(randome_arry))
        else:
            count = 0
            for filename in os.listdir(self.data_dir):
                if filename.endswith('_sample.npy'):
                    count += 1
            self.sample_count = count
            print('process is test, nums is ', count)
        dt = np.load(statics_file)
        self.max = dt[0, :].reshape(-1, 1, 1)
        self.min = dt[1, :].reshape(-1, 1, 1)

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        new_min = 0
        new_max = 1
        if self.trainstate:
            path_x = os.path.join(self.data_dir, '%d_sample.npy' % self.rdarry[index])
            path_y = os.path.join(self.data_dir, '%d_label.npy' % self.rdarry[index])
        else:
            path_x = os.path.join(self.data_dir, '%d_sample.npy' % index)
            path_y = os.path.join(self.data_dir, '%d_label.npy' % index)
        xxx = np.load(path_x)
        yyy = np.load(path_y)
        yy = yyy[0, :, :].squeeze()
        yy[yy == 10] = 2
        yy[yy == 11] = 3
        xxx = (xxx - self.min) / (self.max - self.min) * (new_max - new_min) + new_min
        xx = xxx
        return xx, yy

