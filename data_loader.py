import os
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
from base import BaseDataLoader
from utils import rle_decode, blur_measure


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
        trsfm = transforms.Compose([
                BlurAwareCrop(),
            # transforms.RandomResizedCrop(101, scale=(0.7, 1.0)),# ratio=(0.75, 1.3333333333333333)),
                transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.Pad((5, 5, 6, 6), padding_mode='reflect'),
                    transforms.Pad((13, 13, 14, 14), padding_mode='reflect'),
                    # transforms.Pad((1, 1, 2, 2), padding_mode='edge'),
                    transforms.ToTensor()
                    ])
            ]
            )
        self.data_dir = config['data_loader']['data_dir']
        self.dataset = SaltDataset(self.data_dir, transform=trsfm, target_transform=trsfm)
        super(SaltDataLoader, self).__init__(self.dataset, config)

class BlurAwareCrop():
    def __init__(self, prob=0.7, blur_thres=200, min_crop=70, return_size=101):
        self.prob = prob
        self.blur_thres = blur_thres
        self.min_crop = min_crop
        self.return_size = return_size
        self.tr = None

    def __call__(self, img):
        if img.mode == 'RGB':
            if blur_measure(img) > self.blur_thres and np.random.rand() < self.prob:
                crop_size = np.random.randint(self.min_crop, self.return_size)
                self.tr = transforms.Compose([
                    transforms.RandomCrop(crop_size),
                    transforms.Resize(self.return_size)
                ])
            else:
                self.tr = transforms.Compose([])
        return self.tr(img)

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
    