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
                 img_height=137, img_width=236, test_id = 0,
                 data_root = f'{os.path.dirname(os.getcwd())}/input'):
        """
        Arguments: 
            fold (list) -- list of folds to be used
            train (boolean) -- if true fetches train data else test
            test_id (int) -- parquet file id
            transform (callable) -- Transform to be applied on each sample 
            data_root (string) --  root path to input or dir of data

        """
        self.train = train
        self.transform = transform
        self.data_root = data_root

        file = 'train_folds' if self.train else 'test'         
        self.metadata = pd.read_csv(f'{data_root}/{file}.csv')
        if train:
            self.metadata = self.metadata.drop('grapheme', axis=1)
            self.metadata = self.metadata[self.metadata.k_fold.isin(folds)].reset_index(drop=True)

            self.grapheme_roots = self.metadata.grapheme_root.values
            self.vowel_diacritics = self.metadata.vowel_diacritic.values
            self.consonant_diacritics = self.metadata.consonant_diacritic.values
            self.image_ids = self.metadata.image_id.values

        else:
            df = pd.read_parquet(f'{data_root}/test_image_data_{test_id}.parquet')
            self.image_ids = df.image_id.values # could also get from metadata
            self.image_array = df.iloc[:, 1:].values

        
        if (len(folds) == 1 and train) or not train: # validation and test
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

        if not self.train: # test
            image = self.image_array[idx, :]
        else: # train or val
            image = joblib.load(f'{self.data_root}/image_pickles/{self.image_ids[idx]}.pkl')
        
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

 

