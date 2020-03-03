import os
import pandas as pd
import joblib
from torch.utils.data import Dataset
from PIL import Image
import albumentations
import numpy as np
import torch

class BengaliAI(Dataset):
    """Bengali AI dataset for training PyTorch models"""

    def __init__(self, folds, train=True, transform=None, 
                 img_height=137, img_width=236):
        """
        Arguments: 
            train (boolean) -- if true fetches train data else test
            transform (callable) -- Transform to be applied on each sample 
            # using albumenations internally
            fold (list) -- list of folds to be used
        """
        self.train = train
        self.transform = transform

        file = 'train_folds' if self.train else 'test'         
        self.metadata = pd.read_csv(f'{os.path.dirname(os.getcwd())}/input/{file}.csv')
        self.metadata = self.metadata.drop('grapheme', axis=1)
        self.metadata = self.metadata[self.metadata.kfolds.isin(folds)].reset_index(drop=True)

        if train:
            self.grapheme_roots = self.metadata.grapheme_root.values
            self.vowel_diacritics = self.metadata.vowel_diacritic.values
            self.consonant_diacritics = self.metadata.consonant_diacritic.values

        self.image_ids = self.metadata.image_id.values
        

        if len(folds) == 1: # validation
            self.augmentations = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.Normalize(always_apply=True) 
            ])
        else: # train
            self.augmentations = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.ShiftScaleRotate(rotate_limit=10, p=0.9), 
                albumentations.Normalize(always_apply=True) 
            ])



    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Arguments:
            idx (int) -- Dataset class is a map-style dataset (https://pytorch.org/docs/stable/data.html#map-style-datasets) 
                         where idx is the index of a specific sample in the map
        Returns:
            sample (dic) -- Each sample is a dic with keys as image_id, image, grapheme_root, vowel_diacritic, consonant_diacritic
        """
        sample = {'image_id': self.image_ids[idx]}
        image = joblib.load(f'{os.path.dirname(os.getcwd())}/input/image_pickles/{self.image_ids[idx]}.pkl')
        image = image.reshape(137,236).astype(float)
        image = Image.fromarray(image).convert("RGB")
        image = np.array(image)
        image = self.augmentations(image=image)['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        sample['image'] = image
        
        
        if self.train:           
            sample['grapheme_root'] = torch.tensor(self.grapheme_roots[idx], dtype=torch.long)
            sample['vowel_diacritic'] = torch.tensor(self.vowel_diacritics[idx], dtype=torch.long)
            sample['consonant_diacritic'] = torch.tensor(self.consonant_diacritics[idx], dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)
            
        return sample

 

