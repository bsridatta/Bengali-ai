import logging
import torch
from dataset import BengaliAI
from torch.utils.data import SubsetRandomSampler

def train_dataloader(params):
    dataset = BengaliAI(folds=params.train_folds, train=True, 
                        data_root=params.data_root)
    
    sampler = SubsetRandomSampler(range(2*params.batch_size)) if params.fast_dev_run else None

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
        sampler=sampler
    )
    logging.info(f'Training data loader called. len - {len(loader)*params.batch_size}')

    return loader

def val_dataloader(params):,
    dataset = BengaliAI(folds=params.val_folds, train=True, 
                        data_root=params.data_root)

    sampler = SubsetRandomSampler(range(params.batch_size)) if params.fast_dev_run else None

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
        sampler=sampler
    )
    logging.info(f'Validation data loader called. len - {len(loader)*params.batch_size}')

    return loader

def test_dataloader(params):
    dataset = BengaliAI(train=False, test_id=params.test_id,
                        data_root=params.data_root)

    sampler = SubsetRandomSampler(range(params.batch_size)) if params.fast_dev_run else None

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
        sampler=sampler
    )
    logging.info(f'Test data loader called. len - {len(loader)*params.batch_size}')

    return loader

