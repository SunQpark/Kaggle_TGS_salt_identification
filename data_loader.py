import os
import torch
import numpy as np
import pandas as pd
import random
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
        filename = 'depths.csv' if train else 'sample_submission.csv'
        
        self.metadata = pd.read_csv(os.path.join(self.data_dir, filename), usecols=['id'])

    def __len__(self):
        return len(self.metadata.id)

    def _get_image(self, image_id, mask=False):
        mode = 'masks' if mask else 'images'
        img_dir = 'train' if self.train else 'test'
        img_path = os.path.join(self.data_dir, f'{img_dir}/{mode}/{image_id}.png')
        return Image.open(img_path)

    def __getitem__(self, idx):
        image_id = self.metadata.id[idx]

        img  = self._get_image(image_id)
        seed = np.random.randint(1)
        # apply transforms
        random.seed(seed)
        if self.transform is not None:
            img = self.transform(img)
        
        if self.train:
            mask = self._get_image(image_id, mask=True)
            random.seed(seed)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
                mask = (mask > 0).float()
            return img, mask
        else:
            return img, image_id


class SaltDataLoader(BaseDataLoader):
    def __init__(self, config):
        trsfm_shared = [ 
            transforms.RandomResizedCrop(101, scale=(0.7, 1.0)),# ratio=(0.75, 1.3333333333333333)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad((5, 5, 6, 6), padding_mode='reflect'),
            transforms.ToTensor(),
            ]
        input_trfm = transforms.Compose([transforms.ColorJitter(brightness=0.2)] + trsfm_shared)
        target_trfm = transforms.Compose(trsfm_shared)
        self.data_dir = config['data_loader']['data_dir']
        self.dataset = SaltDataset(self.data_dir, transform=input_trfm, target_transform=target_trfm)
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
    