import logging as log
import torch
from dataset import BengaliAI


def train_dataloader(params):
    log.info('Training data loader called')

    dataset = BengaliAI(folds=params.train_folds, train=True, 
                        data_root=params.data_root)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory
    )

    return loader

def val_dataloader(params):
    log.info('Validation data loader called')

    dataset = BengaliAI(folds=params.val_folds, train=True, 
                        data_root=params.data_root)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory
    )

    return loader

def test_dataloader(params):
    log.info('Test data loader called')

    dataset = BengaliAI(train=False, test_id=params.test_id,
                        data_root=params.data_root)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory
    )

    return loader

