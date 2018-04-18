import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        price_features, targets = sample['price_feature'], sample['target']
        # price_features, volumn_features, targets = sample['price_feature'], sample['volumn_feature'], sample['target']

        return {'price_feature': torch.from_numpy(price_features),
                # 'volumn_feature': torch.from_numpy(volumn_features),
                'target': targets}

class SingleCoinDataset(Dataset):
    def __init__(self, data, transform=ToTensor()):
        self.data = data
        self.transform = transform
        ###############################
        # need to make the multiply factor an argument
        ###############################    
        price_feature_normalize = lambda x: x*150
        self.data["price_feature"] = self.data["price_feature"].apply(price_feature_normalize)

        std = (self.data["volumn_feature"].as_matrix()).reshape(-1).std()
        volumn_feature_normalize = lambda x: x/std
        self.data["volumn_feature"] = self.data["volumn_feature"].apply(volumn_feature_normalize)
        ###############################
        # need to make the multiply factor an argument
        ###############################
        target_normalize = lambda x: torch.sigmoid(torch.from_numpy(np.array([150*x])))
        self.data["target"] = self.data["target"].apply(target_normalize)
        self.data = self.data.drop('volumn_feature', axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = (self.data.iloc[idx,:]).to_dict()

        if self.transform:
            sample = self.transform(sample)

        return sample