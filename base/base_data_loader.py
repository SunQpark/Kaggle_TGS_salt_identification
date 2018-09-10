import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, config, collate_fn=default_collate):
        self.dataset = dataset
        self.config = config
        self.collate_fn = collate_fn

        self.batch_size = config['data_loader']['batch_size']
        self.validation_split = config['validation']['validation_split']
        self.shuffle = config['data_loader']['shuffle']
        fold = config['validation']['fold']

        self.batch_idx = 0
        self.n_samples = len(self.dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split, fold)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn
            }
        super(BaseDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)

    def __len__(self):
        """
        :return: Total number of batches
        """
        return self.n_samples // self.batch_size

    def _split_sampler(self, split, fold=0, random_seed=0):
        if split == 0.0:
            return None, None

        np.random.seed(random_seed) 
        idx_full = np.random.permutation(self.n_samples)

        len_valid = int(self.n_samples * split)

        start = fold * len_valid
        stop = min((fold + 1) * len_valid, self.n_samples) - 1

        valid_idx = idx_full[start:stop]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler
        
    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
    
