import os
import sys
import cv2
import json
import uuid
import tqdm
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
from referit import REFER
import torch.utils.data as data
from referit.refer import mask as cocomask
sys.path.append('.')
import utils
from utils import Corpus
sys.modules['utils'] = utils
cv2.setNumThreads(0)


class DatasetNotFoundError(Exception):
    pass

class ReferDataset(data.Dataset):
    def __init__(self, data_root, dataset='unc', split='train'):
        self.dir_path=osp.join(self.dataset_root, dataset, "{}_batch".format(split))
        self.image_paths = os.listdir(slef.dir_path)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        com_path=osp.join(self.dir_path,self.image_paths)
        content=np.load(com_path)
        return content

def get_ref_dataset(data_root, dataset='referit', split='train'):
    dir_path=osp.join(data_root, dataset, "{}_batch".format(split))
    image_paths = os.listdir(dir_path)
    dataset_dicts=[]
    count=0
    for path in image_paths:
        dataset_dicts.append(osp.join(dir_path,path))
    print("len of {} {} = {}".format(dataset,split,len(dataset_dicts)))
    return dataset_dicts
