import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
from base import BaseDataLoader
from utils import rle_decode


class SaltDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None, train=True):
        super(SaltDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        filename = 'train.csv' if train else 'sample_submission.csv'
        
        self.metadata = pd.read_csv(os.path.join(self.data_dir, filename), usecols=['id'])

    def __len__(self):
        return len(self.metadata.id)

    def _get_image(self, id, mask=False):
        mode = 'masks' if mask else 'images'
        img_dir = 'train' if self.train else 'test'
        img_path = os.path.join(self.data_dir, f'{img_dir}/{mode}/{id}.png')
        return Image.open(img_path)

    def __getitem__(self, idx):
        image_id = self.metadata.id[idx]

        img  = self._get_image(image_id)
        # apply transforms
        if self.transform is not None:
            img = self.transform(img)
        
        if self.train:
            mask = self._get_image(image_id, mask=True)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
                mask = (mask > 0).float()
            return img, mask
        else:
            return img, image_id


class SaltDataLoader(BaseDataLoader):
    def __init__(self, config):
        trsfm = transforms.Compose([
            transforms.Pad((13, 13, 14, 14), padding_mode='reflect'),
            transforms.ToTensor(),
            ])
        self.data_dir = config['data_loader']['data_dir']
        self.dataset = SaltDataset(self.data_dir, transform=trsfm, target_transform=trsfm)
        super(SaltDataLoader, self).__init__(self.dataset, config)


import json
if __name__ == '__main__':
    trsfm = transforms.Compose([
        transforms.ToTensor()
        ])
    
    config = json.load(open('./config.json'))
    loader = SaltDataLoader(config)
    for i, (data, target) in enumerate(loader):
        print(data.shape)
        print(target.shape)
        print(data[0:10])
        print(target[0:10])
        if i == 10:
            break
    
    # dset = SaltDataset('input', transform=trsfm, target_transform=trsfm, train=True)
    # print(len(dset))
    # data, target = dset[len(dset) - 1]
    # print(data.shape)
    # print(target.shape)